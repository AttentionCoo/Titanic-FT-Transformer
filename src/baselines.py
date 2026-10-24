"""与 FT-Transformer 互补的表格模型。

这些小数据上表现稳定的模型提供不同归纳偏置：梯度提升树擅长阈值型
交互，线性模型提供平滑外推，随机森林降低方差。它们的 OOF 概率与
FT-Transformer 融合后通常比任何单一模型更稳。

所有模型都使用固定超参数（不做逐折调参），避免在小验证折上过拟合。
"""

from __future__ import annotations

from functools import partial
from typing import Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def _logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")


def _random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=800,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )


def _extra_trees(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=1000,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=1,
        random_state=seed,
    )


def _hist_gbdt(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _svc(seed: int) -> CalibratedClassifierCV:
    # RBF-SVM 在小样本表格数据上非常稳；Platt 校准后输出概率。
    base = SVC(C=1.0, gamma="scale", kernel="rbf", random_state=seed)
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def _mlp(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        alpha=1e-3,
        learning_rate_init=1e-3,
        batch_size=64,
        max_iter=500,
        early_stopping=False,
        random_state=seed,
    )


def _lightgbm(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=8,
        max_depth=4,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=1,
        verbosity=-1,
        random_state=seed,
    )


def _xgboost(seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=700,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=1,
        eval_metric="logloss",
        random_state=seed,
    )


def _catboost(seed: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        loss_function="Logloss",
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )


def make_builders(seed: int = 42) -> dict[str, Callable[[], object]]:
    """返回 模型名 -> 构造函数。缺少可选依赖时自动跳过该模型。"""
    builders: dict[str, Callable[[], object]] = {
        "logreg": partial(_logreg, seed),
        "random_forest": partial(_random_forest, seed),
        "extra_trees": partial(_extra_trees, seed),
        "hist_gbdt": partial(_hist_gbdt, seed),
        "svc": partial(_svc, seed),
        "mlp": partial(_mlp, seed),
    }
    for name, builder in (("lightgbm", _lightgbm), ("xgboost", _xgboost), ("catboost", _catboost)):
        try:
            builder(seed)
        except ImportError:
            print(f"  [baseline] 未安装 {name}，已跳过")
            continue
        builders[name] = partial(builder, seed)
    return builders
