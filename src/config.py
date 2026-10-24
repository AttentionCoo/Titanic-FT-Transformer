"""全局配置：路径、特征集合、交叉验证协议与超参数。

所有路径都基于项目根目录解析，因此脚本既可以在项目根目录运行
(``python -m src.main``)，也可以从任意工作目录运行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# -------------------- 路径 --------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SUBMISSION_PATH = ROOT / "data" / "submissions" / "ft_transformer_submission.csv"
CHECKPOINT_DIR = ROOT / "checkpoints" / "ft_v2"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class FeatureSpec:
    """一组数值特征 + 类别特征。

    数值特征进入 StandardScaler 后作为连续 token，类别特征进入 Embedding。
    """

    num_cols: tuple[str, ...]
    cat_cols: tuple[str, ...]


# 默认特征集合（v2）：在 v1 的 7 个原始特征之外，补充了家庭结构、
# 票号/舱位信息与年龄缺失标记；经树模型与 FT-Transformer 的消融实验，
# 去掉了噪声较大的 Deck 与 TicketFreq。
DEFAULT_SPEC = FeatureSpec(
    num_cols=(
        "Age",
        "FareLog",
        "FarePerPersonLog",
        "SibSp",
        "Parch",
        "FamilySize",
        "CabinNum",
        "AgeMissing",
    ),
    cat_cols=(
        "Pclass",
        "Sex",
        "Title",
        "Embarked",
        "FamilyType",
        "HasCabin",
        "IsAlone",
        "PclassSex",
    ),
)


@dataclass(frozen=True)
class CVConfig:
    """重复分层 K 折协议。

    同一 ``seed`` 下的第 0 次重复与 v1 的 StratifiedKFold(random_state=42)
    完全一致，方便与历史结果对比。
    """

    n_splits: int = 5
    n_repeats: int = 3
    seed: int = 42


@dataclass(frozen=True)
class FTConfig:
    """FT-Transformer 结构与训练超参数。"""

    d_token: int = 32
    n_layers: int = 3
    n_heads: int = 8
    dropout: float = 0.2
    drop_path: float = 0.1
    ff_mult: float = 2.0
    head_hidden: int = 64
    use_feature_pool: bool = True

    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    epochs: int = 60
    warmup_frac: float = 0.1
    label_smoothing: float = 0.05
    ema_decay: float = 0.995
    seeds: tuple[int, ...] = (42, 2024)


# 默认模型的展示名，报告与提交文件都用它。
MODEL_NAME = "ft_transformer_v2"

# 融合的默认成员：四种独立实现的梯度提升树（XGBoost / LightGBM / CatBoost /
# HistGradientBoosting）覆盖不同的分裂与正则化方式，FT-Transformer 提供完全
# 不同的归纳偏置。等权平均比在 OOF 上搜索权重更稳。若某个模型不可用
# （例如未安装 lightgbm），自动使用其余可用成员。
ENSEMBLE_MEMBERS = ("xgboost", "lightgbm", "catboost", "hist_gbdt", "ft_transformer")
