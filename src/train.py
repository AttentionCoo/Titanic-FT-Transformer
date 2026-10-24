"""交叉验证训练流程。

* :func:`make_folds` —— 重复分层 K 折，第 0 次重复与 v1 的折划分一致；
* :func:`train_ft_cv` —— FT-Transformer 多折 × 多随机种子训练；
* :func:`train_baseline_cv` —— 互补模型（GBDT / 线性 / 森林）训练。

所有折共享同一套折划分，OOF 预测可直接用于后续融合。
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from .config import CVConfig, FTConfig, FeatureSpec
from .data import TitanicData
from .model import FTTransformer, ModelEMA, set_seed
from .preprocessing import DesignMatrix, FoldPreprocessor

# LightGBM 4.x 在 sklearn 接口下会因传入 numpy 数组产生无害的 feature name 警告。
# 该过滤需要在 worker 进程中也生效，因此放在 train 模块而不是 main 模块。
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


@dataclass(frozen=True)
class FoldSpec:
    repeat: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass
class CVPredictions:
    """一个模型在重复 CV 下的 OOF 与测试集预测。"""

    oof: np.ndarray  # (n_repeats, n_train)，每个元素都是 out-of-fold
    test: np.ndarray  # (n_test,)，已对折 / 重复 / 种子平均
    elapsed_seconds: float = 0.0

    @property
    def oof_mean(self) -> np.ndarray:
        return self.oof.mean(axis=0)


def make_folds(y: np.ndarray, cv: CVConfig) -> list[FoldSpec]:
    folds: list[FoldSpec] = []
    for repeat in range(cv.n_repeats):
        splitter = StratifiedKFold(
            n_splits=cv.n_splits, shuffle=True, random_state=cv.seed + repeat
        )
        for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
            folds.append(FoldSpec(repeat, fold, train_idx, val_idx))
    return folds


# --------------------------------------------------------------------------- #
#                               FT-Transformer                                #
# --------------------------------------------------------------------------- #
def _lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _predict_probs(model: nn.Module, dm: DesignMatrix, batch_size: int = 512) -> np.ndarray:
    model.eval()
    x_num = torch.from_numpy(dm.x_num)
    x_cat = torch.from_numpy(dm.x_cat)
    probs = []
    for start in range(0, len(x_num), batch_size):
        logits = model(x_num[start : start + batch_size], x_cat[start : start + batch_size])
        probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(probs).astype(np.float64)


def fit_predict_ft(
    train: DesignMatrix,
    y_train: np.ndarray,
    eval_sets: list[DesignMatrix],
    cfg: FTConfig,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, torch.Tensor]]:
    """训练单个 FT-Transformer，返回各 eval 集上的概率与 EMA 权重。"""
    set_seed(seed)
    torch.set_num_threads(1)

    model = FTTransformer(
        n_num=train.x_num.shape[1],
        cat_cardinalities=train.cat_cardinalities,
        d_token=cfg.d_token,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        drop_path=cfg.drop_path,
        ff_mult=cfg.ff_mult,
        head_hidden=cfg.head_hidden,
        use_feature_pool=cfg.use_feature_pool,
    )

    x_num = torch.from_numpy(train.x_num)
    x_cat = torch.from_numpy(train.x_cat)
    y = torch.from_numpy(y_train.astype(np.float32))
    y_smooth = y * (1.0 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing

    n_samples = len(y)
    steps_per_epoch = math.ceil(n_samples / cfg.batch_size)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_frac))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_factor(step, total_steps, warmup_steps)
    )
    criterion = nn.BCEWithLogitsLoss()
    ema = ModelEMA(model, decay=cfg.ema_decay)

    model.train()
    step = 0
    # 手动分批：891 行的小数据上，DataLoader 的 collate 开销比前向反向还大。
    for _ in range(cfg.epochs):
        permutation = torch.randperm(n_samples)
        for start in range(0, n_samples, cfg.batch_size):
            batch_idx = permutation[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_num[batch_idx], x_cat[batch_idx])
            loss = criterion(logits, y_smooth[batch_idx])
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update(model, step)
            step += 1

    ema.copy_to(model)
    model.eval()
    probs = [_predict_probs(model, dm) for dm in eval_sets]
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return probs, state


def _fit_ft_fold(
    frame: pd.DataFrame,
    y: np.ndarray,
    n_train: int,
    spec: FeatureSpec,
    fold_spec: FoldSpec,
    cfg: FTConfig,
    keep_state: bool,
) -> dict:
    """单折 + 多随机种子的训练任务（供 joblib 并行调用）。"""
    start = time.time()
    train_frame = frame.iloc[fold_spec.train_idx]
    val_frame = frame.iloc[fold_spec.val_idx]
    test_frame = frame.iloc[n_train:]

    preprocessor = FoldPreprocessor(spec, vocab_source=frame).fit(train_frame)
    train_dm = preprocessor.transform(train_frame)
    val_dm = preprocessor.transform(val_frame)
    test_dm = preprocessor.transform(test_frame)

    val_probs = np.zeros(len(fold_spec.val_idx), dtype=np.float64)
    test_probs = np.zeros(len(test_frame), dtype=np.float64)
    states: dict[int, dict[str, torch.Tensor]] = {}

    for seed in cfg.seeds:
        (seed_val, seed_test), state = fit_predict_ft(
            train_dm, y[fold_spec.train_idx], [val_dm, test_dm], cfg, seed
        )
        val_probs += seed_val / len(cfg.seeds)
        test_probs += seed_test / len(cfg.seeds)
        if keep_state:
            states[seed] = state

    return {
        "repeat": fold_spec.repeat,
        "fold": fold_spec.fold,
        "val_idx": fold_spec.val_idx,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "states": states,
        "elapsed": time.time() - start,
    }


def train_ft_cv(
    data: TitanicData,
    spec: FeatureSpec,
    cv: CVConfig,
    cfg: FTConfig,
    n_jobs: int = 1,
    checkpoint_dir: Path | None = None,
) -> CVPredictions:
    """重复分层 K 折训练 FT-Transformer。"""
    start = time.time()
    folds = make_folds(data.y, cv)
    oof = np.zeros((cv.n_repeats, data.n_train), dtype=np.float64)
    test = np.zeros(len(data.test_ids), dtype=np.float64)
    n_folds = cv.n_splits

    for repeat in range(cv.n_repeats):
        repeat_folds = [f for f in folds if f.repeat == repeat]
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_ft_fold)(
                data.frame,
                data.y,
                data.n_train,
                spec,
                fold_spec,
                cfg,
                keep_state=repeat == 0,
            )
            for fold_spec in repeat_folds
        )
        for result in results:
            oof[repeat, result["val_idx"]] = result["val_probs"]
            test += result["test_probs"] / (n_folds * cv.n_repeats)
        fold_accs = [
            float(np.mean((result["val_probs"] >= 0.5).astype(int) == data.y[result["val_idx"]]))
            for result in results
        ]
        print(
            f"  [FT] repeat {repeat + 1}/{cv.n_repeats} 完成 | "
            f"fold acc = {np.mean(fold_accs):.4f} | "
            f"{sum(r['elapsed'] for r in results):.0f}s(累计并行耗时)"
        )

        if repeat == 0 and checkpoint_dir is not None:
            _save_checkpoints(checkpoint_dir, results, spec, cfg, cv)

    return CVPredictions(oof=oof, test=test, elapsed_seconds=time.time() - start)


def _save_checkpoints(
    checkpoint_dir: Path,
    results: list[dict],
    spec: FeatureSpec,
    cfg: FTConfig,
    cv: CVConfig,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for result in sorted(results, key=lambda r: r["fold"]):
        for seed, state in result["states"].items():
            path = checkpoint_dir / f"fold_{result['fold'] + 1}_seed_{seed}.pt"
            torch.save(state, path)
    meta = {
        "model": "ft_transformer_v2",
        "feature_spec": {"num_cols": list(spec.num_cols), "cat_cols": list(spec.cat_cols)},
        "ft_config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
        "cv": asdict(cv),
        "checkpoint_pattern": "fold_{fold}_seed_{seed}.pt",
        "note": "权重为 EMA 参数，可直接 load_state_dict 推理。",
    }
    (checkpoint_dir / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
#                             Complementary models                            #
# --------------------------------------------------------------------------- #
def _fit_baseline_fold(
    frame: pd.DataFrame,
    y: np.ndarray,
    n_train: int,
    spec: FeatureSpec,
    fold_spec: FoldSpec,
    builders: dict[str, Callable[[], object]],
) -> dict:
    train_frame = frame.iloc[fold_spec.train_idx]
    val_frame = frame.iloc[fold_spec.val_idx]
    test_frame = frame.iloc[n_train:]

    preprocessor = FoldPreprocessor(spec, vocab_source=frame).fit(train_frame)
    train_dm = preprocessor.transform(train_frame)
    val_dm = preprocessor.transform(val_frame)
    test_dm = preprocessor.transform(test_frame)
    y_train = y[fold_spec.train_idx]

    val_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}
    for name, build in builders.items():
        model = build()
        model.fit(train_dm.dense, y_train)
        val_probs[name] = model.predict_proba(val_dm.dense)[:, 1]
        test_probs[name] = model.predict_proba(test_dm.dense)[:, 1]
    return {
        "repeat": fold_spec.repeat,
        "fold": fold_spec.fold,
        "val_idx": fold_spec.val_idx,
        "val_probs": val_probs,
        "test_probs": test_probs,
    }


def train_baseline_cv(
    data: TitanicData,
    spec: FeatureSpec,
    cv: CVConfig,
    builders: dict[str, Callable[[], object]],
    n_jobs: int = 1,
) -> dict[str, CVPredictions]:
    """用与 FT-Transformer 完全相同的折划分训练互补模型。"""
    start = time.time()
    folds = make_folds(data.y, cv)
    names = list(builders)
    oof = np.zeros((cv.n_repeats, len(names), data.n_train), dtype=np.float64)
    test = np.zeros((len(names), len(data.test_ids)), dtype=np.float64)

    for repeat in range(cv.n_repeats):
        repeat_folds = [f for f in folds if f.repeat == repeat]
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_baseline_fold)(data.frame, data.y, data.n_train, spec, fold_spec, builders)
            for fold_spec in repeat_folds
        )
        for result in results:
            for j, name in enumerate(names):
                oof[repeat, j, result["val_idx"]] = result["val_probs"][name]
                test[j] += result["test_probs"][name] / (cv.n_splits * cv.n_repeats)
        print(f"  [baseline] repeat {repeat + 1}/{cv.n_repeats} 完成")

    return {
        name: CVPredictions(oof=oof[:, j, :], test=test[j])
        for j, name in enumerate(names)
    }
