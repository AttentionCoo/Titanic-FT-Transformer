"""折内预处理：缺失值填充、标准化与类别编码。

关键原则：所有"需要从数据中学习"的统计量（中位数、均值、方差、类别词表）
只在**当前训练折**上拟合，验证折和测试集只做 transform。这样交叉验证
结果不会因为提前看到验证折而虚高。

类别词表是唯一例外：它由 train+test 的合并特征表确定，不涉及任何标签，
这样测试集中出现过的类别一定有对应的 embedding，推理时不会退化成 <unk>。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FeatureSpec

UNKNOWN_TOKEN = "<unk>"


@dataclass
class DesignMatrix:
    """模型输入。

    Attributes:
        x_num: 标准化后的数值特征，供 FT-Transformer 的数值 token 使用。
        x_cat: 类别特征整数编码，供 Embedding / 树模型使用。
        cat_cardinalities: 每个类别特征的基数（含 <unk>）。
        dense: 数值 + one-hot 拼接的稠密矩阵，供梯度提升 / 线性模型使用。
    """

    x_num: np.ndarray
    x_cat: np.ndarray
    cat_cardinalities: tuple[int, ...]
    dense: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.x_num.shape[0])


class FoldPreprocessor:
    """在训练折上 fit，在任意折上 transform。"""

    def __init__(self, spec: FeatureSpec, vocab_source: pd.DataFrame):
        self.spec = spec
        self._cat_vocabs: dict[str, tuple[str, ...]] = {}
        self._cat_to_index: dict[str, dict[str, int]] = {}
        for col in spec.cat_cols:
            values = vocab_source[col].fillna(UNKNOWN_TOKEN).astype(str)
            self._cat_vocabs[col] = (UNKNOWN_TOKEN,) + tuple(sorted(set(values) - {UNKNOWN_TOKEN}))
            self._cat_to_index[col] = {v: i for i, v in enumerate(self._cat_vocabs[col])}

        self._age_map: dict[tuple[str, int], float] = {}
        self._age_global: float = 0.0
        self._pclass_maps: dict[str, dict[int, float]] = {}
        self._globals: dict[str, float] = {}
        self._embarked_mode: str = "S"
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, train_frame: pd.DataFrame) -> "FoldPreprocessor":
        num = self.spec.num_cols

        self._age_global = float(train_frame["Age"].median())
        age_median = train_frame.groupby(["Title", "Pclass"])["Age"].median()
        self._age_map = {
            (str(title), int(pclass)): float(value)
            for (title, pclass), value in age_median.items()
            if not np.isnan(value)
        }

        for col in num:
            median = train_frame[col].median()
            self._globals[col] = float(median) if not np.isnan(median) else 0.0
        for col in ("FareLog", "FarePerPersonLog"):
            if col in num:
                med = train_frame.groupby("Pclass")[col].median()
                self._pclass_maps[col] = {
                    int(pclass): float(value) for pclass, value in med.items() if not np.isnan(value)
                }

        embarked = train_frame["Embarked"].mode(dropna=True)
        self._embarked_mode = str(embarked.iloc[0]) if len(embarked) else "S"

        filled = self._fill(train_frame)
        values = filled[list(num)].to_numpy(dtype=np.float64)
        self._mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        self._std = np.where(std > 0, std, 1.0)
        return self

    # -------------------------------------------------------------- helpers
    def _fill(self, frame: pd.DataFrame) -> pd.DataFrame:
        """按训练折统计量填充数值缺失，返回数值列副本。"""
        num = self.spec.num_cols
        filled = frame[list(num)].astype(np.float64).copy()

        if "Age" in num:
            lookup = np.array(
                [self._age_map.get((str(t), int(p)), np.nan) for t, p in zip(frame["Title"], frame["Pclass"])],
                dtype=np.float64,
            )
            age = filled["Age"].fillna(pd.Series(lookup, index=filled.index)).fillna(self._age_global)
            filled["Age"] = age

        for col in ("FareLog", "FarePerPersonLog"):
            if col in num:
                mapping = self._pclass_maps.get(col, {})
                lookup = frame["Pclass"].map(mapping).astype(np.float64)
                filled[col] = filled[col].fillna(pd.Series(lookup, index=filled.index)).fillna(
                    self._globals[col]
                )

        for col in num:
            if col in ("Age", "FareLog", "FarePerPersonLog"):
                continue
            filled[col] = filled[col].fillna(self._globals[col])
        return filled

    def _encode_cats(self, frame: pd.DataFrame) -> np.ndarray:
        n = len(frame)
        codes = np.zeros((n, len(self.spec.cat_cols)), dtype=np.int64)
        for j, col in enumerate(self.spec.cat_cols):
            series = frame[col]
            if col == "Embarked":
                series = series.fillna(self._embarked_mode)
            values = series.fillna(UNKNOWN_TOKEN).astype(str)
            mapping = self._cat_to_index[col]
            codes[:, j] = values.map(mapping).fillna(0).to_numpy(dtype=np.int64)
        return codes

    def _one_hot(self, codes: np.ndarray) -> np.ndarray:
        cards = [len(self._cat_vocabs[col]) for col in self.spec.cat_cols]
        out = np.zeros((codes.shape[0], int(sum(cards))), dtype=np.float32)
        offset = 0
        for j, card in enumerate(cards):
            out[np.arange(codes.shape[0]), offset + codes[:, j]] = 1.0
            offset += card
        return out

    # -------------------------------------------------------------- transform
    def transform(self, frame: pd.DataFrame) -> DesignMatrix:
        if self._mean is None or self._std is None:
            raise RuntimeError("FoldPreprocessor 必须先调用 fit() 再 transform()")

        filled = self._fill(frame)
        x_num = ((filled.to_numpy(dtype=np.float64) - self._mean) / self._std).astype(np.float32)
        x_cat = self._encode_cats(frame)
        cards = tuple(len(self._cat_vocabs[col]) for col in self.spec.cat_cols)
        dense = np.hstack([x_num, self._one_hot(x_cat)]).astype(np.float32)
        return DesignMatrix(x_num=x_num, x_cat=x_cat, cat_cardinalities=cards, dense=dense)

    def fit_transform(self, frame: pd.DataFrame) -> DesignMatrix:
        return self.fit(frame).transform(frame)
