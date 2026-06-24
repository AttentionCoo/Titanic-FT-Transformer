import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.font_manager as fm

font_path = r'C:\Windows\Fonts\msyh.ttc'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = [font_name]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'D:\pycharmProject\Titanic\data\raw\train.csv')

print("=" * 60)
print("1. 数据基本信息")
print("=" * 60)
print(f"数据形状: {df.shape[0]} 行 x {df.shape[1]} 列\n")
print("前5行数据:")
print(df.head())
print("\n数据类型:")
print(df.dtypes)
print("\n数值型字段描述统计:")
print(df.describe())
print("\n分类型字段描述统计:")
print(df.describe(include=['object']))

print("\n" + "=" * 60)
print("2. 缺失值分析")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False)
print(missing_df)

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='YlOrRd', ax=ax)
ax.set_title('缺失值热力图')
plt.tight_layout()
plt.savefig(r'D:\pycharmProject\Titanic\figures\missing_values_heatmap.png', dpi=150)
plt.close()
print("→ 缺失值热力图已保存至 figures/missing_values_heatmap.png")

print("\n" + "=" * 60)
print("3. 目标变量分析 (Survived)")
print("=" * 60)
survived_counts = df['Survived'].value_counts()
survived_pct = df['Survived'].value_counts(normalize=True).round(4) * 100
print(f"遇难: {survived_counts[0]} 人 ({survived_pct[0]:.2f}%)")
print(f"生还: {survived_counts[1]} 人 ({survived_pct[1]:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].pie(survived_counts, labels=['遇难', '生还'], autopct='%1.1f%%',
            colors=['#FF6B6B', '#4ECDC4'], startangle=90)
axes[0].set_title('生还比例')
sns.countplot(x='Survived', data=df, palette=['#FF6B6B', '#4ECDC4'], ax=axes[1])
axes[1].set_xticklabels(['遇难', '生还'])
axes[1].set_title('生还人数')
for p in axes[1].patches:
    axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom')
plt.tight_layout()
plt.savefig(r'D:\pycharmProject\Titanic\figures\survived_distribution.png', dpi=150)
plt.close()
print("→ 生还分布图已保存至 figures/survived_distribution.png")

print("\n" + "=" * 60)
print("4. 分类特征与生还率关系")
print("=" * 60)

categorical_cols = ['Sex', 'Pclass', 'Embarked']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col in enumerate(categorical_cols):
    survival_rate = df.groupby(col)['Survived'].mean().round(4) * 100
    print(f"\n{col} 各类别生还率:")
    for val, rate in survival_rate.items():
        print(f"  {val}: {rate:.2f}%")

    sns.countplot(x=col, hue='Survived', data=df, palette=['#FF6B6B', '#4ECDC4'], ax=axes[i])
    axes[i].set_title(f'{col} 与生还关系')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('人数')
    axes[i].legend(['遇难', '生还'])

plt.tight_layout()
plt.savefig(r'D:\pycharmProject\Titanic\figures\categorical_vs_survived.png', dpi=150)
plt.close()
print("\n→ 分类特征图已保存至 figures/categorical_vs_survived.png")

print("\n" + "=" * 60)
print("5. 数值特征分布")
print("=" * 60)

numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='#3498DB', alpha=0.6)
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'均值={df[col].mean():.1f}')
    axes[i].axvline(df[col].median(), color='green', linestyle='--', label=f'中位数={df[col].median():.1f}')
    axes[i].set_title(f'{col} 分布')
    axes[i].legend()

plt.tight_layout()
plt.savefig(r'D:\pycharmProject\Titanic\figures\numerical_distribution.png', dpi=150)
plt.close()
print("→ 数值特征分布图已保存至 figures/numerical_distribution.png")

print("\n" + "=" * 60)
print("6. 年龄与生还率关系（分箱分析）")
print("=" * 60)

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 65, 80],
                        labels=['儿童(0-12)', '青少年(12-18)', '青年(18-35)',
                                '中年(35-50)', '中老年(50-65)', '老年(65+)'])
age_survival = df.groupby('AgeGroup', observed=True)['Survived'].agg(['mean', 'count'])
age_survival['mean'] = (age_survival['mean'] * 100).round(2)
age_survival.columns = ['生还率(%)', '人数']
print(age_survival)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=age_survival.index, y='生还率(%)', data=age_survival,
            palette='RdYlGn', ax=ax)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')
ax.set_title('各年龄段生还率')
ax.set_ylabel('生还率(%)')
plt.tight_layout()
plt.savefig(r'D:\pycharmProject\Titanic\figures\age_group_survival.png', dpi=150)
plt.close()
print("→ 年龄段生还率图已保存至 figures/age_group_survival.png")

print("\n" + "=" * 60)
print("7. 相关性热力图")
print("=" * 60)

df_corr = df.copy()
df_corr['Sex_encoded'] = df_corr['Sex'].map({'male': 0, 'female': 1})
df_corr['Embarked_encoded'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
corr_cols = ['Survived', 'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
corr_matrix = df_corr[corr_cols].corr()

print("与 Survived 的相关系数（按绝对值排序）:")
survived_corr = corr_matrix['Survived'].drop('Survived').abs().sort_values(ascending=False)
for feat, val in survived_corr.items():
    direction = "正相关" if corr_matrix.loc['Survived', feat] > 0 else "负相关"
    print(f"  {feat:20s}: r = {corr_matrix.loc['Survived', feat]:+.4f}  (|r| = {val:.4f}, {direction})")

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax)
ax.set_title('Titanic 特征相关性热力图')
plt.tight_layout()
plt.savefig(r'D:\pycharmProject\Titanic\figures\correlation_heatmap.png', dpi=150)
plt.close()
print("→ 相关性热力图已保存至 figures/correlation_heatmap.png")

print("\n" + "=" * 60)
print("8. 特征组合分析")
print("=" * 60)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
family_survival = df.groupby('FamilySize')['Survived'].agg(['mean', 'count'])
family_survival['mean'] = (family_survival['mean'] * 100).round(2)
family_survival.columns = ['生还率(%)', '人数']
print("家庭规模与生还率:")
print(family_survival)

df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
alone_survival = df.groupby('IsAlone')['Survived'].mean() * 100
print(f"\n独自一人: 生还率 {alone_survival[1]:.2f}%")
print(f"有家人陪伴: 生还率 {alone_survival[0]:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x=family_survival.index, y='生还率(%)', data=family_survival, palette='YlGn', ax=axes[0])
axes[0].set_title('家庭规模与生还率')
axes[0].set_xlabel('家庭规模')
axes[0].set_ylabel('生还率(%)')

sns.barplot(x=['有家人', '独自一人'], y=[alone_survival[0], alone_survival[1]],
            palette=['#4ECDC4', '#FF6B6B'], ax=axes[1])
axes[1].set_title('是否独自一人与生还率')
axes[1].set_ylabel('生还率(%)')
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom')
plt.tight_layout()
plt.savefig(r'D:\pycharmProject\Titanic\figures\family_survival.png', dpi=150)
plt.close()
print("→ 家庭组合分析图已保存至 figures/family_survival.png")

print("\n" + "=" * 60)
print("分析完成！所有图表已保存至 figures/ 目录")
print("=" * 60)