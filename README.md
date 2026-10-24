<div align="center">

# 🚢 Titanic Survival Prediction

**FT-Transformer × 表格数据 | 重复 5 折 CV: 84.74% (嵌套评估)**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 项目简介

基于 Kaggle 经典 [Titanic](https://www.kaggle.com/c/titanic) 数据集，使用
**FT-Transformer**（Feature Tokenizer + Transformer）做乘客生还预测，并与四种
梯度提升树、线性模型、RBF-SVM 等互补模型做 OOF 融合。

v2 重构了特征工程、交叉验证协议与融合策略。核心结论：

| 版本 / 协议 | CV Accuracy | 说明 |
|:---|:---:|:---|
| v1 README 原值 | **84.51%** | 5 折单次；按验证折挑选最优 epoch，分数偏乐观 |
| v1 同协议复测 | 83.39% ± 0.76% | 本机复现 v1 早停协议，数值无法完全复现 |
| v1 诚实复测 | 82.23% ± 0.61% | 固定 80 epoch、不看验证折，3 × 5 折 |
| v2 FT 单模型 | 82.72% ± 0.27% | 3 × 5 折，每折 2 个随机种子 |
| **v2 融合（最终）** | **84.74%** | 嵌套评估（15 个留出折），先验阈值 |
| v2 融合（bagged OOF） | 84.96% | 3 次重复概率平均后的 OOF |
| v2 融合（单次重复） | 83.99% ± 0.28% | bagging 前的单袋估计 |

> ⚠️ **关于 84.51%**：v1 的评估在验证折上挑选最优 epoch，且预处理在
> train+test 全集上拟合。用同一份代码复测只能得到 83.39%，改用"固定 epoch、
> 不查看验证折"的诚实协议则是 82.23%。v2 的 84.74% 是在**不含上述乐观偏差**
> 的协议下得到的，因此与旧数字不是同一把尺子；同尺比较的净提升约 **+2.5 个
> 百分点**。详见 [`scripts/legacy_benchmark.py`](scripts/legacy_benchmark.py)
> 与 [`reports/legacy_baseline.json`](reports/legacy_baseline.json)。
>
> 84.74% 是更保守的嵌套估计（融合方式与阈值都在不同折上拟合）；84.96% 是
> 最终 bagging 模型在其 OOF 上的直接准确率。两个口径都列出，避免只报高值。

---

## 🗂 项目结构

```
Titanic-FT-Transformer/
├── data/
│   ├── raw/                              # 原始数据
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   └── submissions/
│       └── ft_transformer_submission.csv # v2 融合提交文件
├── checkpoints/
│   ├── ft_v2/                            # v2 权重（EMA），fold_{f}_seed_{s}.pt
│   └── legacy/                           # v1 权重（历史基线）
├── figures/                              # EDA 图表
├── reports/
│   ├── metrics.json                      # 全部指标与协议
│   ├── model_comparison.csv              # 单模型 OOF 对比
│   ├── oof_predictions.csv               # 各模型 OOF 概率（长表）
│   ├── test_predictions.csv              # 测试集概率
│   ├── threshold_curve.csv               # 阈值-准确率曲线
│   ├── ensemble_weights.json             # 融合方式、权重、阈值
│   └── legacy_baseline.json              # v1 两种协议复测结果
├── scripts/
│   └── legacy_benchmark.py               # 复现 v1 并与诚实协议对比
├── src/
│   ├── config.py                         # 路径、特征集合、CV 协议、超参数
│   ├── data.py                           # 数据读取 + 特征工程
│   ├── preprocessing.py                  # 折内缺失值填充 / 标准化 / 编码
│   ├── model.py                          # FT-Transformer（Pre-LN + EMA）
│   ├── train.py                          # 重复交叉验证训练
│   ├── baselines.py                      # 互补模型
│   ├── ensemble.py                       # OOF 融合与阈值规则
│   ├── main.py                           # 一键训练 + 评估 + 提交
│   ├── eda.py                            # 探索性数据分析
│   ├── transformer.py                    # v1 基线（保留对照）
│   └── transformer.ipynb                 # v1 Notebook 版本（保留对照）
├── requirements.txt
├── ft_transformer_explanation.md
├── knowledge_summary.md
└── README.md
```

---

## 🧠 模型架构（v2）

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐
│  连续特征 ×8 │   │  类别特征 ×8 │   │  Transformer │   │  MLP Head│
│  Linear+Bias │──▶│  Embedding   │──▶│  Blocks ×3   │──▶│  2 层    │──▶ Survived?
└──────────────┘   └──────────────┘   │  Pre-LN      │   │  GELU    │
        │                 │           └──────────────┘   └──────────┘
        └─────────┬───────┘                   ▲
                  ▼                           │
           ┌──────────────┐            ┌──────────────┐
           │ Feature      │            │  CLS Token   │
           │ Tokenizer    │            │ + Feature 池 │
           └──────────────┘            └──────────────┘
```

| 组件 | v1 | v2 |
|:---|:---|:---|
| 归一化位置 | Post-LN | **Pre-LN**（更稳定，配合 DropPath） |
| FFN | Linear→ReLU→Linear (4/3×) | Linear→GELU→Linear (2×)，带 Dropout |
| 输出表示 | CLS + 特征均值 | CLS + 特征均值（LayerNorm 后拼接） |
| 训练轮数 | 200 + 早停 | **固定 60 epoch**，warmup 10% + 余弦退火 |
| 权重 | 验证折最优 epoch | **EMA**（decay=0.995） |
| 随机性 | 1 个种子 | 每折 **2 个种子**，概率平均 |
| 评估 | 5 折单次 | **5 折 × 3 次重复** |

模型规模约 3.2 万参数（`d_token=32`、3 层、8 头），适合 891 条训练样本的
极小数据场景；性能主要来自特征工程与融合，而不是堆大模型。

---

## ⚙️ 特征工程

### 🆕 v2 新增特征

| 特征 | 构造方式 | 动机 |
|:----:|---------|------|
| `FareLog` | `log1p(Fare)` | 原始票价长尾严重，取对数后更适合数值 token |
| `FarePerPersonLog` | `log1p(Fare / FamilySize)` | 同样票价下，人越多实际支付能力越低 |
| `FamilyType` | Alone / Small(2–4) / Large(5+) | 家庭规模与生还率呈倒 U 型 |
| `IsMother` | 女性 + `Mrs` + `Parch>0` | 泰坦尼克号优先登艇群体 |
| `CabinNum` | `Cabin` 首个数字 | 舱位在船上的位置信息 |
| `HasCabin` | `Cabin` 是否缺失 | 缺失本身与舱位等级相关 |
| `PclassSex` | `Pclass × Sex` 组合 | 显式提供最强的二阶交互 |
| `AgeMissing` | `Age` 是否缺失 | 缺失模式携带信息 |
| `AgeBin` / `FareBin` | 分箱（候选特征） | 捕捉非线性阶梯 |
| `TicketFreq` | 同票号人数（train+test，无标签） | 同行群体规模 |
| `TicketPrefix` | 票号前缀（稀有合并为 RARE） | 票种/舱位等级线索 |

默认使用下面的 8 数值 + 8 类别组合（经消融验证：`Deck` 与 `TicketFreq`
对树模型是噪声，故移出默认集合）：

```python
数值: Age, FareLog, FarePerPersonLog, SibSp, Parch, FamilySize, CabinNum, AgeMissing
类别: Pclass, Sex, Title, Embarked, FamilyType, HasCabin, IsAlone, PclassSex
```

### 🧹 缺失值处理（折内拟合）

| 特征 | 策略 |
|:----:|------|
| `Age` | 按 `Title` + `Pclass` 分组中位数填充 → 全局中位数兜底 |
| `Fare` / `FareLog` | 按 `Pclass` 分组中位数填充 |
| `Embarked` | 训练折众数填充 |
| `Cabin` | 填 `U`，另设 `HasCabin` / `CabinNum` |

> 以上统计量**只在当前训练折上计算**，验证折与测试集只做 transform。

### 🔒 刻意不使用的特征

`FamilySurvived`、`SurnameSurvived` 等由标签构造的"家庭生还"特征在 Kaggle
上能显著提高 LB，但会让交叉验证失去意义（标签泄漏）。本项目只使用乘客属性，
保证报告出的 CV 分数是可复现、可泛化的。

---

## 📊 运行结果

### 单模型 OOF（5 折 × 3 次重复）

| 模型 | OOF Accuracy | LogLoss | AUC |
|:---|:---:|:---:|:---:|
| XGBoost | **84.44%** ± 0.35% | 0.3976 | 0.8829 |
| LightGBM | 83.88% ± 0.43% | 0.3995 | 0.8839 |
| CatBoost | 83.65% ± 0.46% | 0.3986 | **0.8843** |
| Hist Gradient Boosting | 83.28% ± 0.73% | 0.4283 | 0.8818 |
| Logistic Regression | 82.87% ± 0.29% | 0.4120 | 0.8743 |
| **FT-Transformer (v2)** | 82.72% ± 0.27% | 0.4090 | 0.8730 |
| Random Forest | 82.72% ± 0.33% | 0.4083 | 0.8814 |
| RBF-SVM | 82.60% ± 0.09% | 0.4264 | 0.8656 |
| Extra Trees | 82.30% ± 0.21% | 0.4146 | 0.8741 |
| MLP | 81.71% ± 1.14% | 0.6633 | 0.8365 |

### 融合结果

候选融合方式（嵌套 15 折评估）：

| 融合方式 | 嵌套 OOF Acc | 说明 |
|:---|:---:|:---|
| **`curated_avg`（选中）** | **84.74%** ± 1.39% | XGB / LGBM / CatBoost / HGB / FT 等权平均 |
| `greedy` | 84.89% ± 1.34% | OOF 贪心权重，与选中方式在 1 个标准误内 |
| `stack` | 84.47% ± 1.73% | 逻辑回归堆叠 |
| `average_all` | 83.46% ± 1.39% | 10 个模型全部等权 |

> 融合方式按"一倍标准误内优先选更简单方案"的规则确定；`greedy` 的
> 0.15% 优势在噪声范围内，因此采用更稳健的等权融合。

最终决策阈值不固定为 0.5，而是按训练集正类比例做 **prior matching**
（`OOF 阈值 = 0.400`，`测试集阈值 = 0.440`）。多模型平均会让概率向 0.5
收缩，固定阈值会导致预测正类比例偏低；按先验比例定阈在嵌套验证中带来约
+0.8 个百分点的提升，且只依赖训练标签，不按 OOF 准确率搜阈值。测试集的
阈值按同一规则在测试概率上确定（不使用标签），因此预测正类比例仍为 38.5%。

测试集预测：**161 / 418 人为幸存（38.5%）**，与训练集 38.4% 的先验一致。

---

## 🔧 训练配置

<details>
<summary>📋 点击展开完整超参数</summary>

| 参数 | 值 | 说明 |
|:----:|:---:|:---|
| `d_token` | 32 | Token 嵌入维度 |
| `n_layers` | 3 | Pre-LN Transformer 层数 |
| `n_heads` | 8 | 多头注意力头数 |
| `dropout` | 0.2 | Dropout 率 |
| `drop_path` | 0.1 | 随机深度概率 |
| `ff_mult` | 2.0 | FFN 隐层倍率 |
| `optimizer` | AdamW | lr=1e-3, weight_decay=1e-2 |
| `scheduler` | Warmup + Cosine | warmup=10% steps |
| `label_smoothing` | 0.05 | 标签平滑系数 |
| `epochs` | 60 | 固定预算（不按验证折挑 epoch） |
| `ema_decay` | 0.995 | 推理权重使用 EMA |
| `batch_size` | 64 | — |
| `seeds` | (42, 2024) | 每折训练 2 个种子后平均 |
| `cv` | 5 折 × 3 次重复 | StratifiedKFold，第 0 次重复与 v1 折划分一致 |

</details>

---

## 🚀 快速开始

### 1️⃣ 环境安装

```bash
conda create -n Titanic python=3.11
conda activate Titanic
pip install -r requirements.txt
```

`torch` 安装 CPU 版即可，完整流程在 14 核 CPU 上约 4–5 分钟。

### 2️⃣ 运行 EDA

```bash
python -m src.eda
```

图表自动保存至 `figures/` 目录（路径基于项目根目录，无需 `cd src`）。

### 3️⃣ 训练、融合与提交

```bash
# 完整流程（约 4–5 分钟）
python -m src.main --workers 5

# 快速冒烟测试（约 1 分钟）
python -m src.main --quick

# 复现 v1 的 84.51% 并量化其乐观偏差（约 12 分钟）
python -u scripts/legacy_benchmark.py
```

训练完成后自动生成：

- ✅ 提交文件 → `data/submissions/ft_transformer_submission.csv`
- ✅ FT 权重 → `checkpoints/ft_v2/fold_{1-5}_seed_{42,2024}.pt`
- ✅ 指标 → `reports/metrics.json`、`reports/model_comparison.csv` 等

### 4️⃣ 加载已有权重推理

```python
import json
import torch

from src.model import FTTransformer

meta = json.load(open('checkpoints/ft_v2/training_meta.json', encoding='utf-8'))
state = torch.load('checkpoints/ft_v2/fold_1_seed_42.pt', map_location='cpu')
cat_cardinalities = tuple(
    value.shape[0]
    for key, value in state.items()
    if key.startswith('tokenizer.cat_embeddings.') and key.endswith('.weight')
)
model = FTTransformer(
    n_num=len(meta['feature_spec']['num_cols']),
    cat_cardinalities=cat_cardinalities,
    d_token=meta['ft_config']['d_token'],
    n_layers=meta['ft_config']['n_layers'],
    n_heads=meta['ft_config']['n_heads'],
)
model.load_state_dict(state)
model.eval()
```

---

## 🧪 评估协议说明

v2 的所有数字都遵循以下协议：

1. **折内预处理**：缺失值中位数、标准化均值/方差、类别词表只在训练折拟合；
   验证折和测试集只做 transform；
2. **固定训练预算**：不使用验证折挑选 epoch，权重取 EMA，避免"用验证折调早停"；
3. **重复交叉验证**：5 折 × 3 次重复，报告每次重复的均值 ± 标准差；
4. **嵌套融合评估**：在 OOF 预测上再切 15 个留出折，融合方式和阈值只在训练
   部分拟合，在留出部分评估，避免"同一批 OOF 既调权重又报分"；
5. **无标签泄漏特征**：不使用由 `Survived` 构造的家庭/姓氏生还特征。

---

## 📚 参考资料

- [FT-Transformer: Revisiting Deep Learning for Tabular Data](https://arxiv.org/abs/2106.11959)
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Caruana et al., Ensemble Selection from Libraries of Models](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
