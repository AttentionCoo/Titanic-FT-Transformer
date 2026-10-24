"""评估指标与汇总工具。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def accuracy(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    return float(accuracy_score(y_true, (np.asarray(prob) >= threshold).astype(int)))


def safe_log_loss(y_true: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    return float(log_loss(y_true, prob, labels=[0, 1]))


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, prob))


def brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=np.float64)
    return float(np.mean((prob - y_true) ** 2))


def summarize_oof(oof: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    """汇总 (n_repeats, n_samples) 形式 OOF 预测。

    每个 repeat 内的预测都是 out-of-fold 的，因此先按 repeat 计算指标，
    再报告均值 ± 标准差，避免把重复折当成独立样本。
    """
    if oof.ndim == 1:
        oof = oof[None, :]
    accs, losses, aucs, briers = [], [], [], []
    for repeat in range(oof.shape[0]):
        prob = oof[repeat]
        accs.append(accuracy(y, prob, threshold))
        losses.append(safe_log_loss(y, prob))
        aucs.append(safe_auc(y, prob))
        briers.append(brier(y, prob))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "accuracy_per_repeat": [float(a) for a in accs],
        "log_loss_mean": float(np.mean(losses)),
        "auc_mean": float(np.mean(aucs)),
        "brier_mean": float(np.mean(briers)),
    }


def find_best_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    low: float = 0.30,
    high: float = 0.70,
    n_steps: int = 161,
) -> tuple[float, float]:
    """在 OOF 概率上搜索使准确率最高的阈值。

    平局时选择更接近 0.5 的阈值，避免在准确率平台区落到极端值。
    """
    best_threshold = 0.5
    best_acc = accuracy(y_true, prob, 0.5)
    for threshold in np.linspace(low, high, n_steps):
        threshold = float(threshold)
        acc = accuracy(y_true, prob, threshold)
        closer = abs(threshold - 0.5) < abs(best_threshold - 0.5) - 1e-12
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and closer):
            best_threshold, best_acc = threshold, acc
    return best_threshold, best_acc
