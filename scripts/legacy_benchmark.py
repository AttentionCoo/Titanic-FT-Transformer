"""复现并重新评估 v1（README 中 84.51% 的基线）。

因为 v1 通过"在每个 epoch 上用验证折准确率选最优 epoch"（早停 + 保留最佳
权重），它的交叉验证分数天然偏乐观。本脚本用两种协议量化这一偏差：

* ``published`` —— v1 原协议：预处理拟合于 train+test 全集，按验证折选 epoch；
* ``honest``    —— 固定 80 epoch，不查看验证折，取最终权重（3 次重复 5 折）。

两种协议使用完全相同的模型结构与数据，只差"是否用验证折挑 epoch"。
结果写入 ``reports/legacy_baseline.json``。

用法（项目根目录）::

    python -u scripts/legacy_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import transformer as legacy  # noqa: E402


def build_model(n_cont: int, cat_cardinalities: list[int]) -> nn.Module:
    return legacy.FTTransformer(
        n_cont,
        cat_cardinalities,
        d_token=32,
        n_layers=3,
        n_heads=8,
        dropout=0.45,
        drop_path=0.2,
    )


def train_fold(X_tr, y_tr, X_val, y_val, cat_idx, cat_card, protocol, seed, epochs=80):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_loader = legacy.make_loader(X_tr, y_tr, cat_idx, 64, True)
    val_loader = legacy.make_loader(X_val, y_val, cat_idx, 64, False)
    model = build_model(X_tr.shape[1] - len(cat_idx), cat_card)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    best_acc, best_state, patience = 0.0, None, 0
    start = time.time()
    n_epochs = 200 if protocol == "published" else epochs
    for _ in range(n_epochs):
        legacy.train_epoch(model, tr_loader, optimizer, criterion, 0.1)
        val_acc = legacy.evaluate(model, val_loader)
        scheduler.step(val_acc)
        if protocol == "published":
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 20:
                    break
        else:
            best_acc = val_acc
    if protocol == "published" and best_state is not None:
        model.load_state_dict(best_state)
        best_acc = legacy.evaluate(model, val_loader)
    return best_acc, time.time() - start


def main() -> None:
    X, y, _, cat_idx, cat_card, _ = legacy.load_and_preprocess()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(X, y))

    results: dict = {
        "model": "v1 FT-Transformer (legacy)",
        "published_in_readme": 0.8451,
        "note": "v1 的预处理在 train+test 全集拟合，且按验证折选 epoch，分数偏乐观。",
    }

    # ---- 协议 A：完全复现 v1 的早停 + 选验证折最优 epoch ----
    torch.manual_seed(42)
    np.random.seed(42)
    accs_a: list[float] = []
    t0 = time.time()
    for fold, (tr, val) in enumerate(folds):
        acc, secs = train_fold(X[tr], y[tr], X[val], y[val], cat_idx, cat_card, "published", 42)
        accs_a.append(float(acc))
        print(f"[published] fold {fold + 1}: {acc:.4f} ({secs:.0f}s)")
    results["published_remeasured"] = {
        "fold_accuracy": accs_a,
        "accuracy_mean": float(np.mean(accs_a)),
        "accuracy_std": float(np.std(accs_a)),
        "elapsed_seconds": time.time() - t0,
    }

    # ---- 协议 B：固定 80 epoch，3 次重复 5 折，不查看验证折 ----
    accs_b: list[float] = []
    t0 = time.time()
    for repeat in range(3):
        fold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + repeat)
        repeat_accs: list[float] = []
        for fold, (tr, val) in enumerate(fold_splitter.split(X, y)):
            acc, _ = train_fold(
                X[tr], y[tr], X[val], y[val], cat_idx, cat_card, "honest",
                42 + repeat * 100 + fold,
            )
            repeat_accs.append(float(acc))
        accs_b.append(float(np.mean(repeat_accs)))
        print(f"[honest] repeat {repeat + 1}: {np.mean(repeat_accs):.4f} folds={repeat_accs}")
    results["honest_fixed_80ep"] = {
        "per_repeat_accuracy": accs_b,
        "accuracy_mean": float(np.mean(accs_b)),
        "accuracy_std": float(np.std(accs_b)),
        "n_repeats": 3,
        "elapsed_seconds": time.time() - t0,
    }

    out = ROOT / "reports" / "legacy_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
