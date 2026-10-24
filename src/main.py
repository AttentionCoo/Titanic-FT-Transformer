"""一键训练、评估、融合并生成提交文件。

用法（项目根目录）::

    python -m src.main                 # 完整流程：FT + 互补模型 + 融合
    python -m src.main --quick         # 快速冒烟测试（约 1 分钟）
    python -m src.main --skip-baselines

产物：
    checkpoints/ft_v2/fold_*_seed_*.pt   FT-Transformer 权重（EMA）
    data/submissions/ft_transformer_submission.csv
    reports/metrics.json                 全部指标
    reports/model_comparison.csv         单模型对比
    reports/oof_predictions.csv          各模型 OOF 概率（长表）
    reports/test_predictions.csv         测试集概率
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# LightGBM 4.x 在 sklearn 接口下会因传入 numpy 数组产生无害的 feature name 警告。
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

from .baselines import make_builders
from .config import (
    CHECKPOINT_DIR,
    DEFAULT_SPEC,
    ENSEMBLE_MEMBERS,
    REPORT_DIR,
    SUBMISSION_PATH,
    CVConfig,
    FTConfig,
    FeatureSpec,
)
from .data import load_dataset
from .ensemble import (
    BlenderFactory,
    fit_blender,
    nested_candidate_scores,
    prior_threshold,
    threshold_curve,
)
from .metrics import accuracy, find_best_threshold, safe_auc, safe_log_loss, summarize_oof
from .train import CVPredictions, train_baseline_cv, train_ft_cv

DISPLAY_NAMES = {
    "ft_transformer": "FT-Transformer (v2)",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gbdt": "Hist Gradient Boosting",
    "svc": "RBF-SVM",
    "mlp": "MLP",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}


# --------------------------------------------------------------------------- #
#                                  CLI                                        #
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Titanic 生存预测：训练 + 融合 + 提交")
    parser.add_argument("--folds", type=int, default=CVConfig.n_splits)
    parser.add_argument("--repeats", type=int, default=CVConfig.n_repeats)
    parser.add_argument("--seed", type=int, default=CVConfig.seed)
    parser.add_argument("--epochs", type=int, default=FTConfig.epochs)
    parser.add_argument("--seeds", type=str, default=",".join(map(str, FTConfig.seeds)))
    parser.add_argument("--workers", type=int, default=5, help="并行折数（-1 表示全部核心）")
    parser.add_argument("--dropout", type=float, default=FTConfig.dropout)
    parser.add_argument("--d-token", type=int, default=FTConfig.d_token)
    parser.add_argument("--skip-ft", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--quick", action="store_true", help="1 次重复、1 个种子、40 epoch")
    parser.add_argument("--name", type=str, default="", help="附加的实验名，仅用于打印")
    return parser.parse_args(argv)


def resolve_configs(args: argparse.Namespace) -> tuple[CVConfig, FTConfig]:
    cv = CVConfig(n_splits=args.folds, n_repeats=args.repeats, seed=args.seed)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    if not seeds:
        raise ValueError("--seeds 不能为空")
    ft_cfg = replace(
        FTConfig(),
        epochs=args.epochs,
        seeds=seeds,
        dropout=args.dropout,
        d_token=args.d_token,
    )
    if args.quick:
        cv = replace(cv, n_repeats=1)
        ft_cfg = replace(ft_cfg, epochs=min(ft_cfg.epochs, 40), seeds=(cv.seed,))
    return cv, ft_cfg


# --------------------------------------------------------------------------- #
#                               Reporting                                     #
# --------------------------------------------------------------------------- #
def print_model_table(rows: list[dict]) -> None:
    header = f"{'模型':<28}{'OOF Acc':>16}{'LogLoss':>10}{'AUC':>9}{'耗时(s)':>10}"
    print("\n" + header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: r["accuracy_mean"], reverse=True):
        acc = f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        print(
            f"{row['display_name']:<28}{acc:>16}{row['log_loss_mean']:>10.4f}"
            f"{row['auc_mean']:>9.4f}{row.get('elapsed_seconds', 0.0):>10.1f}"
        )


def save_long_predictions(
    path: Path,
    predictions: dict[str, CVPredictions],
    ids: np.ndarray,
    y: np.ndarray | None,
) -> None:
    rows = []
    for name, pred in predictions.items():
        for repeat in range(pred.oof.shape[0]):
            for idx, passenger_id in enumerate(ids):
                row = {
                    "model": name,
                    "repeat": repeat,
                    "PassengerId": int(passenger_id),
                    "prob": float(pred.oof[repeat, idx]),
                }
                if y is not None:
                    row["y"] = int(y[idx])
                rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
#                                  Main                                       #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    cv, ft_cfg = resolve_configs(args)
    started = time.time()

    print("=" * 78)
    print("Titanic 生存预测 v2 —— FT-Transformer + 互补模型融合")
    if args.name:
        print(f"实验名: {args.name}")
    print(
        f"协议: {cv.n_splits}-fold × {cv.n_repeats} repeats | "
        f"FT seeds={ft_cfg.seeds} epochs={ft_cfg.epochs} | workers={args.workers}"
    )
    print("=" * 78)

    data = load_dataset()
    spec: FeatureSpec = DEFAULT_SPEC
    print(
        f"样本: train={data.n_train} test={len(data.test_ids)} | "
        f"数值特征={len(spec.num_cols)} 类别特征={len(spec.cat_cols)}"
    )

    predictions: dict[str, CVPredictions] = {}

    # ------------------------------ FT-Transformer ------------------------------
    if not args.skip_ft:
        print("\n[1/3] 训练 FT-Transformer ...")
        predictions["ft_transformer"] = train_ft_cv(
            data,
            spec,
            cv,
            ft_cfg,
            n_jobs=args.workers,
            checkpoint_dir=CHECKPOINT_DIR,
        )

    # ------------------------------ complementary -------------------------------
    if not args.skip_baselines:
        print("\n[2/3] 训练互补模型 ...")
        builders = make_builders(seed=cv.seed)
        predictions.update(
            train_baseline_cv(data, spec, cv, builders, n_jobs=args.workers)
        )

    if not predictions:
        raise SystemExit("没有训练任何模型，请检查 --skip-ft / --skip-baselines")

    # ------------------------------ per-model metrics ---------------------------
    model_rows = []
    for name, pred in predictions.items():
        summary = summarize_oof(pred.oof, data.y)
        summary.update(
            {
                "model": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "elapsed_seconds": pred.elapsed_seconds,
            }
        )
        model_rows.append(summary)
    print_model_table(model_rows)

    # ------------------------------ ensemble ------------------------------------
    print("\n[3/3] OOF 融合 ...")
    names = list(predictions)
    oof_matrix = np.stack([predictions[name].oof_mean for name in names])
    test_matrix = np.stack([predictions[name].test for name in names])

    if len(names) >= 2:
        candidates: dict[str, BlenderFactory] = {}
        curated = [name for name in ENSEMBLE_MEMBERS if name in names]
        if len(curated) >= 2:
            curated_weights = np.array([1.0 if name in curated else 0.0 for name in names])
            candidates["curated_avg"] = lambda probs, target, w=curated_weights: fit_blender(
                "fixed", probs, target, fixed_weights=w
            )
        candidates["average_all"] = lambda probs, target: fit_blender("average", probs, target)
        candidates["greedy"] = lambda probs, target: fit_blender("greedy", probs, target)
        candidates["stack"] = lambda probs, target: fit_blender("stack", probs, target)

        nested = nested_candidate_scores(oof_matrix, data.y, candidates, threshold="prior")
        for method, score in nested.items():
            print(
                f"  {method:<12} 嵌套 OOF Acc = {score['accuracy_mean']:.4f} "
                f"± {score['accuracy_std']:.4f}"
            )
        # 一倍标准误内优先选更简单、更稳定的等权融合；只有在明显更优时才
        # 采用贪心权重 / 堆叠，避免在 891 条 OOF 上过拟合融合方式。
        best_score = max(nested.values(), key=lambda score: score["accuracy_mean"])
        standard_error = best_score["accuracy_std"] / math.sqrt(len(best_score["accuracy_per_fold"]))
        cutoff = best_score["accuracy_mean"] - standard_error
        eligible = {name for name, score in nested.items() if score["accuracy_mean"] >= cutoff}
        preference = ("curated_avg", "greedy", "stack", "average_all")
        best_method = next(
            (name for name in preference if name in eligible),
            max(nested, key=lambda name: nested[name]["accuracy_mean"]),
        )
        blender = candidates[best_method](oof_matrix, data.y)
        blend_oof_prob = blender.predict_proba(oof_matrix)
        # 阈值按训练集正类比例校准（prior matching）：只依赖训练标签，
        # 不在 OOF 上按准确率搜索阈值，因此不会有阈值过拟合。
        decision_threshold = prior_threshold(blend_oof_prob, data.y.mean())
        blender.threshold = decision_threshold
        test_prob = blender.predict_proba(test_matrix)
        ensemble_method = best_method
    else:
        nested = {}
        blender = fit_blender("fixed", oof_matrix, data.y, fixed_weights=np.array([1.0]))
        blend_oof_prob = blender.predict_proba(oof_matrix)
        decision_threshold = prior_threshold(blend_oof_prob, data.y.mean())
        blender.threshold = decision_threshold
        test_prob = test_matrix[0]
        ensemble_method = "single"

    oof_acc_05 = accuracy(data.y, blend_oof_prob, 0.5)
    oof_acc_prior = accuracy(data.y, blend_oof_prob, decision_threshold)
    # 仅供诊断：在 OOF 上直接搜阈值会偏乐观，不用于生成预测。
    diag_threshold, diag_accuracy = find_best_threshold(data.y, blend_oof_prob)
    # 测试集概率是多折/多种子平均后的结果，分布比 OOF 更集中；按同一
    # "预测正类比例 = 训练先验" 的规则重新定阈，避免平均带来的漂移。
    test_threshold = prior_threshold(test_prob, data.y.mean())
    test_pred = (test_prob >= test_threshold).astype(int)

    # 单次重复的 OOF 融合准确率（衡量单袋方差；最终提交是多次重复的 bagging）。
    per_repeat_accuracy: list[float] = []
    if blender.weights is not None:
        for repeat in range(cv.n_repeats):
            repeat_matrix = np.stack([predictions[name].oof[repeat] for name in names])
            repeat_prob = np.clip(repeat_matrix.T @ blender.weights, 1e-6, 1 - 1e-6)
            repeat_threshold = prior_threshold(repeat_prob, data.y.mean())
            per_repeat_accuracy.append(accuracy(data.y, repeat_prob, repeat_threshold))

    ensemble_summary = {
        "bagged_oof_accuracy": float(accuracy(data.y, blend_oof_prob, decision_threshold)),
        "bagged_oof_log_loss": safe_log_loss(data.y, blend_oof_prob),
        "bagged_oof_auc": safe_auc(data.y, blend_oof_prob),
        "per_repeat_accuracy": per_repeat_accuracy,
        "per_repeat_accuracy_mean": (
            float(np.mean(per_repeat_accuracy)) if per_repeat_accuracy else None
        ),
        "per_repeat_accuracy_std": (
            float(np.std(per_repeat_accuracy)) if per_repeat_accuracy else None
        ),
    }
    ensemble_summary.update(
        {
            "method": ensemble_method,
            "members": names,
            "curated_members": [name for name in ENSEMBLE_MEMBERS if name in names],
            "threshold": float(blender.threshold),
            "threshold_rule": "prior_matching",
            "test_threshold": float(test_threshold),
            "oof_accuracy_at_0.5": float(oof_acc_05),
            "oof_accuracy_at_threshold": float(oof_acc_prior),
            "diagnostic_best_threshold": float(diag_threshold),
            "diagnostic_best_threshold_accuracy": float(diag_accuracy),
            "nested_scores": nested,
            "weights": (
                {name: float(w) for name, w in zip(names, blender.weights)}
                if blender.weights is not None
                else None
            ),
            "coef": (
                {name: float(c) for name, c in zip(names, blender.coef)}
                if blender.coef is not None
                else None
            ),
            "test_positive_rate": float(test_pred.mean()),
            "test_positive_count": int(test_pred.sum()),
        }
    )

    print(
        f"\n  融合方式: {ensemble_method} | OOF 先验阈值={blender.threshold:.3f} | "
        f"OOF Acc={oof_acc_prior:.4f}（0.5 阈值时 {oof_acc_05:.4f}）"
    )
    print(
        f"  测试集先验阈值={test_threshold:.3f}（OOF 最佳阈值诊断值="
        f"{diag_threshold:.3f} → {diag_accuracy:.4f}，偏乐观，未采用）"
    )
    if nested:
        print(
            f"  嵌套评估（避免融合/阈值过拟合的乐观估计）: "
            f"{nested[ensemble_method]['accuracy_mean']:.4f} ± "
            f"{nested[ensemble_method]['accuracy_std']:.4f}"
        )
    if per_repeat_accuracy:
        print(
            f"  单次重复 OOF 融合: {np.mean(per_repeat_accuracy):.4f} ± "
            f"{np.std(per_repeat_accuracy):.4f}（提交时对重复做 bagging）"
        )
    print(
        f"  测试集预测: 正例 {int(test_pred.sum())} / {len(test_pred)} "
        f"({test_pred.mean():.1%})"
    )

    # ------------------------------ write artifacts -----------------------------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({"PassengerId": data.test_ids, "Survived": test_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)

    save_long_predictions(REPORT_DIR / "oof_predictions.csv", predictions, data.train_ids, data.y)

    test_rows = []
    for j, name in enumerate(names):
        for idx, passenger_id in enumerate(data.test_ids):
            test_rows.append(
                {"model": name, "PassengerId": int(passenger_id), "prob": float(test_matrix[j, idx])}
            )
    for idx, passenger_id in enumerate(data.test_ids):
        test_rows.append(
            {"model": "ensemble", "PassengerId": int(passenger_id), "prob": float(test_prob[idx])}
        )
    pd.DataFrame(test_rows).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    comparison = pd.DataFrame(model_rows).sort_values("accuracy_mean", ascending=False)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    threshold_frame = pd.DataFrame(threshold_curve(data.y, blend_oof_prob))
    threshold_frame.to_csv(REPORT_DIR / "threshold_curve.csv", index=False)

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "n_train": data.n_train,
            "n_test": len(data.test_ids),
            "positive_rate": float(data.y.mean()),
        },
        "protocol": {
            **asdict(cv),
            "fold_wise_preprocessing": True,
            "fixed_epoch_budget": True,
            "ema_weights": True,
            "note": "预处理只在训练折拟合；不使用验证折挑选 epoch。",
        },
        "feature_spec": {"num_cols": list(spec.num_cols), "cat_cols": list(spec.cat_cols)},
        "ft_config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(ft_cfg).items()
        },
        "models": model_rows,
        "ensemble": ensemble_summary,
        "elapsed_seconds": time.time() - started,
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "ensemble_weights.json").write_text(
        json.dumps(ensemble_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n产物已保存:")
    print(f"  - 提交文件: {SUBMISSION_PATH}")
    print(f"  - 指标报告: {REPORT_DIR / 'metrics.json'}")
    print(f"  - 模型对比: {REPORT_DIR / 'model_comparison.csv'}")
    print(f"  - OOF 概率: {REPORT_DIR / 'oof_predictions.csv'}")
    print(f"  - 测试概率: {REPORT_DIR / 'test_predictions.csv'}")
    print(f"  - 阈值曲线: {REPORT_DIR / 'threshold_curve.csv'}")
    print(f"  - 融合权重: {REPORT_DIR / 'ensemble_weights.json'}")
    if not args.skip_ft:
        print(f"  - 模型权重: {CHECKPOINT_DIR}")
    print(f"\n总耗时: {time.time() - started:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
