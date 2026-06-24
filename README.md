<div align="center">

# 🚢 Titanic Survival Prediction

**FT-Transformer × 表格数据 | 5-Fold CV Accuracy: 84.51%**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 项目简介

基于 Kaggle 经典 [Titanic](https://www.kaggle.com/c/titanic) 数据集，使用 **FT-Transformer**（Feature Tokenizer + Transformer）进行乘客生还预测。项目包含完整的数据探索、特征工程、模型训练与测试集推理流程。

---

## 🗂 项目结构

```
Titanic/
├── 📁 data/
│   ├── 📁 raw/                           # 原始数据
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   └── 📁 submissions/                   # 预测结果
│       └── ft_transformer_submission.csv
├── 📁 figures/                           # EDA 可视化图表
│   ├── survived_distribution.png
│   ├── categorical_vs_survived.png
│   ├── numerical_distribution.png
│   ├── age_group_survival.png
│   ├── family_survival.png
│   ├── correlation_heatmap.png
│   └── missing_values_heatmap.png
├── 📁 checkpoints/                       # 模型权重
│   ├── fold_1.pt
│   ├── fold_2.pt
│   ├── fold_3.pt
│   ├── fold_4.pt
│   └── fold_5.pt
├── 📁 src/
│   ├── eda.py                            # 探索性数据分析
│   ├── transformer.py                    # FT-Transformer 模型与训练
│   └── transformer.ipynb                 # Notebook 版本
├── knowledge_summary.md                  # 热力图与相关系数知识总结
├── requirements.txt
└── README.md
```

---

## 🧠 模型架构

FT-Transformer 是一种专为**表格数据**设计的 Transformer 变体，核心流程如下：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│  Numerical   │     │  Categorical │     │              │     │          │
│  Features    │────▶│  Features    │────▶│  Transformer │────▶│ MLP Head │──▶ Survived?
│  (Linear)    │     │  (Embedding) │     │  Blocks ×3   │     │          │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────┘
                          Feature              CLS Token
                         Tokenizer            + Residual
```

| 组件 | 说明 |
|:---:|------|
| 🔢 **Feature Tokenizer** | 数值特征线性映射为 token，类别特征通过 Embedding 映射为 token |
| 🎯 **CLS Token** | 可学习的分类 token，附加在序列头部，聚合全局信息 |
| 🔄 **Transformer Blocks** | 3 层 Self-Attention + FFN，配合 DropPath 随机深度正则化 |
| 🔗 **特征残差连接** | 原始特征 token 均值与 CLS 输出拼接，补偿深层信息丢失 |
| 📊 **MLP Head** | 二分类输出 (Sigmoid) |

---

## ⚙️ 特征工程

### 🆕 新增特征

| 特征 | 构造方式 | 动机 |
|:----:|---------|------|
| `Title` | 从 `Name` 提取称谓 → 映射为 Mr / Mrs / Miss / Master / Rare | 称谓蕴含性别、年龄、社会地位信息 |
| `FamilySize` | `SibSp + Parch + 1` | 家庭规模影响逃生决策 |
| `IsAlone` | `FamilySize == 1` | 独行者生还率显著偏低 |
| `Deck` | `Cabin` 首字母，缺失填 Unknown | 舱位甲板位置与生还相关 |

### 🧹 缺失值处理

| 特征 | 策略 |
|:----:|------|
| `Age` | 按 `Title` + `Pclass` 分组中位数填充 → 全局中位数兜底 |
| `Fare` | 按 `Pclass` 分组中位数填充 |
| `Embarked` | 众数填充 |
| `Cabin` / `Deck` | 填为 Unknown |

### 🏷 编码与标准化

| 类型 | 特征 | 方法 |
|:----:|------|------|
| 类别 | `Pclass`, `Sex`, `Embarked`, `Title`, `Deck` | LabelEncoder → Embedding |
| 数值 | `Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`, `IsAlone` | StandardScaler → Linear Token |

---

## 📊 运行结果

### 5-Fold 交叉验证

| Fold | 🎯 Val Accuracy | 📈 趋势 |
|:----:|:---------------:|:-------:|
| 1 | 83.24% | ▅▅▅▅▅▅▅▅▅░ |
| 2 | 83.71% | ▅▅▅▅▅▅▅▅▅░ |
| 3 | 83.71% | ▅▅▅▅▅▅▅▅▅░ |
| 4 | **86.52%** | ▅▅▅▅▅▅▅▅▅▅ |
| 5 | 85.39% | ▅▅▅▅▅▅▅▅▅▅ |

> 🏆 **Average CV Accuracy: 84.51% ± 1.24%**

### 测试集推理

采用 **5 折集成**（概率平均 → 阈值 0.5）策略：

| 类别 | 预测人数 |
|:----:|:-------:|
| 未幸存 (0) | 293 |
| 幸存 (1) | 125 |

结果已保存至 `data/submissions/ft_transformer_submission.csv`，可直接提交 Kaggle。

---

## 🔧 训练配置

<details>
<summary>📋 点击展开完整超参数</summary>

| 参数 | 值 | 说明 |
|:----:|:---:|------|
| `d_token` | 32 | Token 嵌入维度 |
| `n_layers` | 3 | Transformer 层数 |
| `n_heads` | 8 | 多头注意力头数 |
| `dropout` | 0.45 | Dropout 率 |
| `drop_path` | 0.2 | 随机深度概率 |
| `optimizer` | AdamW | lr=1e-3, weight_decay=1e-2 |
| `scheduler` | ReduceLROnPlateau | patience=5, factor=0.5 |
| `label_smoothing` | 0.1 | 标签平滑系数 |
| `early_stopping` | 20 epochs | 早停耐心值 |
| `batch_size` | 64 | — |
| `epochs` | 200 | 最大训练轮数 |
| `cv` | 5 折 StratifiedKFold | 分层交叉验证 |

</details>

---

## 🚀 快速开始

### 1️⃣ 环境安装

```bash
conda create -n Titanic python=3.10
conda activate Titanic
pip install -r requirements.txt
pip install torch
```

### 2️⃣ 运行 EDA

```bash
cd src
python eda.py
```

图表自动保存至 `figures/` 目录。

### 3️⃣ 训练与预测

```bash
cd src
python transformer.py
```

训练完成后自动执行：
- ✅ 模型权重 → `checkpoints/fold_1.pt` ~ `fold_5.pt`
- ✅ 提交文件 → `data/submissions/ft_transformer_submission.csv`

### 4️⃣ 加载已有权重推理

```python
import torch
from transformer import FTTransformer

model = FTTransformer(n_cont_features=6, cat_cardinalities=[3, 2, 3, 5, 9])
model.load_state_dict(torch.load('../checkpoints/fold_1.pt'))
model.eval()
```

---

## 🔍 EDA 主要发现

| 发现 | 详情 |
|:----:|------|
| 👩 **性别是最强特征** | 女性生还率 ~74%，男性仅 ~19%，与 Survived 相关系数 r ≈ 0.54 |
| 🏨 **船舱等级影响显著** | 一等舱 > 二等舱 > 三等舱，r ≈ -0.34 |
| 👶 **年龄呈非线性关系** | 儿童生还率最高，但整体线性相关性仅 r ≈ 0.08 |
| 👨‍👩‍👧 **家庭规模呈倒 U 型** | 2-4 人家庭生还率最高，独行/大家庭偏低 |
| 💰 **Fare 与 Pclass 冗余** | r ≈ -0.55，存在信息重叠 |

---

## 📚 参考资料

- [FT-Transformer: Revisiting Deep Learning for Tabular Data](https://arxiv.org/abs/2106.11959)
- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)