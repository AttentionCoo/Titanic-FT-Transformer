# Titanic Survival Prediction

基于 Kaggle 经典 Titanic 数据集，使用 FT-Transformer（Feature Tokenizer + Transformer）进行乘客生还预测。

## 项目结构

```
Titanic/
├── data/
│   ├── raw/                          # 原始数据
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   └── submissions/                  # 预测结果
│       └── ft_transformer_submission.csv
├── figures/                          # EDA 可视化图表
│   ├── survived_distribution.png
│   ├── categorical_vs_survived.png
│   ├── numerical_distribution.png
│   ├── age_group_survival.png
│   ├── family_survival.png
│   ├── correlation_heatmap.png
│   └── missing_values_heatmap.png
├── checkpoints/                      # 模型权重
│   ├── fold_1.pt
│   ├── fold_2.pt
│   ├── fold_3.pt
│   ├── fold_4.pt
│   └── fold_5.pt
├── src/
│   ├── eda.py                        # 探索性数据分析
│   ├── transformer.py                # FT-Transformer 模型与训练
│   └── transformer.ipynb             # Notebook 版本
├── knowledge_summary.md              # 热力图与相关系数知识总结
├── requirements.txt
└── README.md
```

## 模型架构

FT-Transformer 是一种专为表格数据设计的 Transformer 变体：

1. **Feature Tokenizer**：将数值特征线性映射为 token，类别特征通过 Embedding 层映射为 token
2. **CLS Token**：可学习的分类 token，附加在序列头部
3. **Transformer Blocks**：多层 Self-Attention + FFN，配合 DropPath 随机深度正则化
4. **特征残差连接**：将原始特征 token 均值与 CLS 输出拼接，补偿深层信息丢失
5. **MLP Head**：二分类输出

## 特征工程

| 特征 | 处理方式 |
|------|---------|
| `Title` | 从 `Name` 提取称谓，映射为 Mr/Mrs/Miss/Master/Rare |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `FamilySize == 1` |
| `Age` | 按 `Title` + `Pclass` 分组中位数填充缺失值 |
| `Fare` | 按 `Pclass` 分组中位数填充缺失值 |
| `Embarked` | 众数填充缺失值 |
| `Deck` | `Cabin` 首字母，缺失值填为 Unknown |
| 类别特征 | `Pclass`, `Sex`, `Embarked`, `Title`, `Deck` → LabelEncoder |
| 数值特征 | `Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`, `IsAlone` → StandardScaler |

## 训练配置

| 参数 | 值 |
|------|----|
| Token 维度 | 32 |
| Transformer 层数 | 3 |
| 注意力头数 | 8 |
| Dropout | 0.45 |
| DropPath | 0.2 |
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-2) |
| 学习率调度 | ReduceLROnPlateau (patience=5, factor=0.5) |
| 标签平滑 | 0.1 |
| 早停耐心 | 20 epochs |
| 交叉验证 | 5 折 StratifiedKFold |

## 运行结果

5 折交叉验证准确率：

| Fold | Val Accuracy |
|------|-------------|
| 1 | 0.8324 |
| 2 | 0.8371 |
| 3 | 0.8371 |
| 4 | 0.8652 |
| 5 | 0.8539 |
| **平均** | **0.8451 ± 0.0124** |

测试集预测采用 5 折集成（概率平均），结果保存至 `data/submissions/ft_transformer_submission.csv`。

## 快速开始

### 环境安装

```bash
conda create -n Titanic python=3.10
conda activate Titanic
pip install -r requirements.txt
pip install torch
```

### 运行 EDA

```bash
cd src
python eda.py
```

### 训练与预测

```bash
cd src
python transformer.py
```

训练完成后：
- 模型权重保存至 `checkpoints/fold_1.pt` ~ `fold_5.pt`
- 提交文件保存至 `data/submissions/ft_transformer_submission.csv`

### 仅推理（加载已有权重）

```python
import torch
from transformer import FTTransformer

model = FTTransformer(n_cont_features=6, cat_cardinalities=[3, 2, 3, 5, 9])
model.load_state_dict(torch.load('../checkpoints/fold_1.pt'))
model.eval()
```

## EDA 主要发现

- **性别是最强特征**：女性生还率约 74%，男性仅约 19%
- **船舱等级影响显著**：一等舱生还率 > 二等舱 > 三等舱
- **年龄非线性关系**：儿童生还率最高，但整体 Age 与 Survived 线性相关性仅 0.08
- **家庭规模**：2-4 人家庭生还率最高，独自一人或大家庭生还率较低
- **Fare 与 Pclass 强负相关**（r ≈ -0.55），存在信息冗余