"""OOF 概率融合与阈值选择。

三种融合方式：

* ``average`` —— 等权平均，零参数、最稳；
* ``greedy``  —— Caruana 贪心集成选择（允许重复选同一模型），在 OOF 上
  以 log-loss 为目标搜索权重；
* ``stack``   —— 以各模型 logit 为输入的逻辑回归，学习非均匀权重。

融合效果用嵌套交叉验证评估：在 OOF 预测上再切分 K 折，融合器与阈值只在
训练部分拟合、在留出部分评估，避免"用同一批 OOF 既调权重又报分数"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .metrics import accuracy, safe_log_loss

EPS = 1e-6


def _logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, EPS, 1 - EPS)
    return np.log(prob / (1 - prob))


@dataclass
class Blender:
    """融合器：权重（average/greedy）或逻辑回归系数（stack）。"""

    method: str
    weights: np.ndarray | None = None
    coef: np.ndarray | None = None
    intercept: float = 0.0
    threshold: float = 0.5

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """probs: (n_models, n_samples) -> (n_samples,)。"""
        probs = np.asarray(probs, dtype=np.float64)
        if self.method == "stack":
            assert self.coef is not None
            logits = _logit(probs).T @ self.coef + self.intercept
            return 1.0 / (1.0 + np.exp(-logits))
        assert self.weights is not None
        return np.clip(probs.T @ self.weights, EPS, 1 - EPS)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return (self.predict_proba(probs) >= self.threshold).astype(int)


BlenderFactory = Callable[[np.ndarray, np.ndarray], Blender]


def greedy_weights(probs: np.ndarray, y: np.ndarray, n_iter: int = 40) -> np.ndarray:
    """贪心集成选择：每轮加入使 OOF log-loss 最低的模型（可重复）。"""
    n_models = probs.shape[0]
    counts = np.zeros(n_models)
    current = np.zeros(probs.shape[1])
    n_selected = 0
    best_loss = np.inf

    for _ in range(n_iter):
        best_j, best_candidate, best_candidate_loss = -1, None, np.inf
        for j in range(n_models):
            candidate = (current * n_selected + probs[j]) / (n_selected + 1)
            loss = safe_log_loss(y, candidate)
            if loss < best_candidate_loss - 1e-12:
                best_j, best_candidate, best_candidate_loss = j, candidate, loss
        if best_j < 0 or best_candidate_loss >= best_loss - 1e-12:
            break
        counts[best_j] += 1
        current = best_candidate
        best_loss = best_candidate_loss
        n_selected += 1

    if n_selected == 0:
        return np.full(n_models, 1.0 / n_models)
    return counts / n_selected


def prior_threshold(prob: np.ndarray, prior: float) -> float:
    """按训练集正类比例确定阈值（prior matching）。

    模型平均后的概率往往整体偏低（尤其多模型平均会收缩方差），固定的 0.5
    会让预测正类比例低于真实比例。这里取使"预测正类比例 = 训练集先验"的
    分位点，只依赖训练标签，不需要在 OOF 上按准确率搜索阈值。
    """
    prior = float(np.clip(prior, 1e-3, 1 - 1e-3))
    return float(np.quantile(prob, 1.0 - prior))


def fit_blender(
    method: str,
    probs: np.ndarray,
    y: np.ndarray,
    fixed_weights: np.ndarray | None = None,
) -> Blender:
    """按指定策略拟合融合器。

    Args:
        method: ``average`` / ``greedy`` / ``stack`` / ``fixed``。
        fixed_weights: 仅 ``fixed`` 使用，为每个模型指定非负权重。
    """
    probs = np.asarray(probs, dtype=np.float64)
    if method == "average":
        return Blender("average", weights=np.full(probs.shape[0], 1.0 / probs.shape[0]))
    if method == "fixed":
        if fixed_weights is None:
            raise ValueError("fixed 融合方式需要 fixed_weights")
        weights = np.asarray(fixed_weights, dtype=np.float64)
        if weights.shape != (probs.shape[0],):
            raise ValueError(f"权重维度 {weights.shape} 与模型数 {probs.shape[0]} 不匹配")
        if weights.sum() <= 0:
            raise ValueError("权重之和必须为正")
        return Blender("average", weights=weights / weights.sum())
    if method == "greedy":
        return Blender("greedy", weights=greedy_weights(probs, y))
    if method == "stack":
        stacker = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        stacker.fit(_logit(probs).T, y)
        return Blender(
            "stack",
            coef=stacker.coef_.ravel(),
            intercept=float(stacker.intercept_[0]),
        )
    raise ValueError(f"未知融合方式: {method}")


def nested_candidate_scores(
    oof: np.ndarray,
    y: np.ndarray,
    candidates: dict[str, "BlenderFactory"],
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    threshold: float | str = "prior",
) -> dict[str, dict]:
    """在 OOF 矩阵上嵌套评估若干候选融合策略。

    每折只在训练部分拟合融合器与决策阈值，在留出部分评估，因此可用于
    选择融合方式而不会用同一批 OOF 既调参又报分。

    Args:
        oof: (n_models, n_samples)，每个模型都是 out-of-fold 预测。
        candidates: 名称 -> ``(probs, y) -> Blender`` 的构造函数。
        threshold: 固定阈值（float）或 ``"prior"``（按训练折正类比例校准）。
    """
    scores: dict[str, list[float]] = {name: [] for name in candidates}

    for repeat in range(n_repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for train_idx, val_idx in splitter.split(np.zeros(len(y)), y):
            for name, build in candidates.items():
                blender = build(oof[:, train_idx], y[train_idx])
                prob_train = blender.predict_proba(oof[:, train_idx])
                if threshold == "prior":
                    cut = prior_threshold(prob_train, y[train_idx].mean())
                else:
                    cut = float(threshold)
                prob_val = blender.predict_proba(oof[:, val_idx])
                scores[name].append(accuracy(y[val_idx], prob_val, cut))

    return {
        name: {
            "accuracy_mean": float(np.mean(values)),
            "accuracy_std": float(np.std(values)),
            "accuracy_per_fold": [float(v) for v in values],
        }
        for name, values in scores.items()
    }


def threshold_curve(
    y: np.ndarray, prob: np.ndarray, low: float = 0.30, high: float = 0.70, n_steps: int = 161
) -> list[dict]:
    """返回阈值-准确率曲线，便于报告中展示。"""
    return [
        {"threshold": float(t), "accuracy": accuracy(y, prob, float(t))}
        for t in np.linspace(low, high, n_steps)
    ]
