"""原始数据读取与特征工程。

特征工程只使用乘客属性，不使用任何标签信息，因此训练集与测试集可以
合并后统一构造（票号分组、家庭人数等跨样本特征需要这个全集视角）。
缺失值填充、标准化与类别编码属于"可学习"的预处理，放在
:mod:`src.preprocessing` 中按折拟合，避免验证折信息泄漏。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RAW_DIR

# 称谓映射：保留 Mr / Mrs / Miss / Master，其余合并为 Rare。
# 已婚女性头衔（Lady、Countess、Dona 等）归入 Mrs，可提高该组样本量。
_TITLE_MAP = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Ms": "Miss",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Lady": "Mrs",
    "Countess": "Mrs",
    "Dona": "Mrs",
    "Don": "Rare",
    "Rev": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Sir": "Rare",
    "Col": "Rare",
    "Capt": "Rare",
    "Jonkheer": "Rare",
}

# 出现次数少于该值的票号前缀合并为 RARE。
_RARE_TICKET_PREFIX_MIN = 10


@dataclass
class TitanicData:
    """合并后的特征表 + 训练/测试边界。"""

    frame: pd.DataFrame
    y: np.ndarray
    n_train: int
    train_ids: np.ndarray
    test_ids: np.ndarray

    @property
    def train_frame(self) -> pd.DataFrame:
        return self.frame.iloc[: self.n_train]

    @property
    def test_frame(self) -> pd.DataFrame:
        return self.frame.iloc[self.n_train :]


def _ticket_prefix_one(ticket: str) -> str:
    """提取票号前缀，纯数字票号记为 NUM。"""
    token = str(ticket).strip().split(" ")[0]
    if token.isdigit():
        return "NUM"
    return token.replace(".", "").upper() or "NUM"


def _ticket_prefix(series: pd.Series) -> pd.Series:
    return series.map(_ticket_prefix_one)


def build_feature_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """在 train + test 合并表上构造特征列。"""
    combined = pd.concat([train, test], ignore_index=True, sort=False)

    # ------------------- 称谓 -------------------
    title = combined["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    combined["Title"] = title.map(_TITLE_MAP).fillna("Rare")

    # ------------------- 家庭结构 -------------------
    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["IsAlone"] = (combined["FamilySize"] == 1).astype(np.int64)
    combined["FamilyType"] = pd.cut(
        combined["FamilySize"],
        bins=[0, 1, 4, 100],
        labels=["Alone", "Small", "Large"],
    ).astype(str)
    # 有子女的已婚女性：泰坦尼克号上的优先登艇群体
    combined["IsMother"] = (
        (combined["Sex"] == "female") & (combined["Title"] == "Mrs") & (combined["Parch"] > 0)
    ).astype(np.int64)

    # ------------------- 舱位 -------------------
    # 缺失舱位单独记为 U，同时用 HasCabin 保留"是否登记舱位"这一信号。
    cabin = combined["Cabin"].fillna("U")
    combined["Deck"] = cabin.str[0].fillna("U")
    combined["HasCabin"] = combined["Cabin"].notna().astype(np.int64)
    cabin_first = cabin.str.split(" ").str[0]
    cabin_num = pd.to_numeric(cabin_first.str.extract(r"(\d+)", expand=False), errors="coerce")
    # 未知舱位记为 0，配合 HasCabin 使用。
    combined["CabinNum"] = cabin_num.fillna(0.0).astype(np.float64)

    # ------------------- 票号 -------------------
    combined["TicketPrefix"] = _ticket_prefix(combined["Ticket"])
    prefix_counts = combined["TicketPrefix"].value_counts()
    rare_prefixes = prefix_counts[prefix_counts < _RARE_TICKET_PREFIX_MIN].index
    combined.loc[combined["TicketPrefix"].isin(rare_prefixes), "TicketPrefix"] = "RARE"
    # 同一票号下的乘客人数（家庭或同行友人），不含标签信息。
    combined["TicketFreq"] = combined["Ticket"].map(combined["Ticket"].value_counts()).astype(np.float64)

    # ------------------- 分箱与交互 -------------------
    # 年龄呈非线性关系，分箱后作为类别特征交给模型（含缺失分组）。
    age_bin = pd.cut(
        combined["Age"],
        bins=[-0.1, 5, 12, 18, 30, 45, 60, 100],
        labels=["0-5", "5-12", "12-18", "18-30", "30-45", "45-60", "60+"],
    ).astype("string")
    combined["AgeBin"] = age_bin.fillna("Unknown").astype(str)
    combined["AgeMissing"] = combined["Age"].isna().astype(np.int64)

    # 票价按分位数分箱，捕捉"贵的舱位更可能生还"的非线性阶梯。
    fare_bin = pd.qcut(combined["Fare"], q=8, duplicates="drop").astype("string")
    combined["FareBin"] = fare_bin.fillna("Unknown").astype(str)
    combined["PclassSex"] = combined["Pclass"].astype(str) + "_" + combined["Sex"].astype(str)

    # ------------------- 票价 -------------------
    # 原始 Fare 长尾严重，取对数后更适合数值 token / 线性模型。
    combined["FareLog"] = np.log1p(combined["Fare"])
    combined["FarePerPersonLog"] = np.log1p(combined["Fare"] / combined["FamilySize"])

    keep = [
        "PassengerId",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FareLog",
        "FarePerPersonLog",
        "Embarked",
        "Title",
        "Deck",
        "HasCabin",
        "FamilySize",
        "IsAlone",
        "FamilyType",
        "TicketFreq",
        "TicketPrefix",
        "CabinNum",
        "IsMother",
        "AgeBin",
        "AgeMissing",
        "FareBin",
        "PclassSex",
    ]
    return combined[keep].reset_index(drop=True)


def load_dataset() -> TitanicData:
    """读取原始 CSV 并返回特征表。"""
    train = pd.read_csv(RAW_DIR / "train.csv")
    test = pd.read_csv(RAW_DIR / "test.csv")

    y = train["Survived"].to_numpy(dtype=np.int64)
    train_ids = train["PassengerId"].to_numpy()
    test_ids = test["PassengerId"].to_numpy()

    frame = build_feature_frame(train.drop(columns=["Survived"]), test)
    return TitanicData(frame=frame, y=y, n_train=len(train), train_ids=train_ids, test_ids=test_ids)
