import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import copy
import random
import os

# -------------------- 0. 全局设置 --------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------- 1. 特征工程数据加载 --------------------
def load_and_preprocess():
    train = pd.read_csv('../data/raw/train.csv')
    test = pd.read_csv('../data/raw/test.csv')

    # ------------------- 原始特征 -------------------
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'

    # 合并
    all_data = pd.concat([train[features + ['Name']]], axis=0)  # 暂时先取训练集特征进行拟合，后面单独做特征工程
    # 更规范：分别处理 train 和 test，然后统一编码
    # 下面我们按完整流程做

    train = pd.read_csv('../data/raw/train.csv')
    test = pd.read_csv('../data/raw/test.csv')
    train['Type'] = 'train'
    test['Type'] = 'test'
    all_data = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

    # ---- 特征工程 ----
    # 1. Title
    all_data['Title'] = all_data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
        'Don': 'Rare', 'Rev': 'Rare', 'Dr': 'Rare', 'Mme': 'Mrs',
        'Ms': 'Miss', 'Major': 'Rare', 'Lady': 'Rare', 'Sir': 'Rare',
        'Mlle': 'Miss', 'Col': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare',
        'Jonkheer': 'Rare', 'Dona': 'Rare'
    }
    all_data['Title'] = all_data['Title'].map(title_map).fillna('Rare')

    # 2. FamilySize 和 IsAlone
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
    all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)

    # 3. 缺失值填充
    all_data['Age'] = all_data.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())  # 再全局填充残余缺失
    all_data['Fare'] = all_data['Fare'].fillna(all_data.groupby('Pclass')['Fare'].transform('median'))
    all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])
    all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
    all_data['Deck'] = all_data['Cabin'].str[0]  # 取首字母

    # 4. 类别编码
    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck']
    for col in cat_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))

    # 数值列标准化
    num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
    scaler = StandardScaler()
    all_data[num_cols] = scaler.fit_transform(all_data[num_cols])

    final_features = num_cols + cat_cols
    cat_cols_final = cat_cols

    # 划分
    train_mask = all_data['Type'] == 'train'
    test_mask = all_data['Type'] == 'test'

    X_train = all_data.loc[train_mask, final_features].values.astype(np.float32)
    X_test = all_data.loc[test_mask, final_features].values.astype(np.float32)
    y_train = train['Survived'].values.astype(np.float32)
    test_passenger_ids = test['PassengerId'].values

    # 标记分类特征索引与基数
    cat_indices = [final_features.index(col) for col in cat_cols_final]
    cat_cardinalities = [len(all_data[col].unique()) for col in cat_cols_final]

    return X_train, y_train, X_test, cat_indices, cat_cardinalities, test_passenger_ids

# -------------------- 2. 模型组件 --------------------
class FeatureTokenizer(nn.Module):
    def __init__(self, n_cont_features, cat_cardinalities, d_token):
        super().__init__()
        self.n_cont = n_cont_features
        self.weight = nn.Parameter(torch.empty(n_cont_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_cont_features, d_token))
        nn.init.normal_(self.weight, std=0.01)
        nn.init.zeros_(self.bias)

        if cat_cardinalities:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(card, d_token) for card in cat_cardinalities
            ])
        else:
            self.cat_embeddings = nn.ModuleList()
        self.norm = nn.LayerNorm(d_token)  # 统一 token 尺度

    def forward(self, x_cont, x_cat_list):
        tokens = []
        if self.n_cont > 0:
            cont_tokens = x_cont.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
            tokens.append(cont_tokens)
        for emb, x_cat in zip(self.cat_embeddings, x_cat_list):
            tokens.append(emb(x_cat).unsqueeze(1))
        x = torch.cat(tokens, dim=1)  # (B, n_feat, d_token)
        return self.norm(x)

class DropPath(nn.Module):
    """随机深度（只在训练时生效）"""
    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, drop_path_rate=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model * 4/3)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * 4/3), d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention + DropPath
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.dropout1(attn_out)
        x = x + self.drop_path(attn_out)
        x = self.norm1(x)

        # FFN + DropPath
        ff_out = self.ff(x)
        ff_out = self.dropout2(ff_out)
        x = x + self.drop_path(ff_out)
        x = self.norm2(x)
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_cont_features, cat_cardinalities, d_token=32,
                 n_layers=3, n_heads=8, dropout=0.4, drop_path=0.2, mlp_hidden=64):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_cont_features, cat_cardinalities, d_token)
        n_features = n_cont_features + len(cat_cardinalities)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, dropout, drop_path)
            for _ in range(n_layers)
        ])
        # 将原始特征的均值 token 作为残差连接到 CLS 输出
        self.proj_residual = nn.Linear(d_token, d_token)  # 投影到相同维度
        self.norm_final = nn.LayerNorm(d_token)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_token * 2, mlp_hidden),  # 拼接 CLS 与特征残差
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x_cont, x_cat_list):
        tokens = self.tokenizer(x_cont, x_cat_list)       # (B, N, d)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)  # (B, 1, d)
        x = torch.cat([cls, tokens], dim=1)                # (B, 1+N, d)

        for block in self.blocks:
            x = block(x)

        cls_out = x[:, 0, :]                 # CLS token 最终输出
        # 特征残差：所有特征 token 的均值，作为原始信号补偿
        feat_res = tokens.mean(dim=1)        # (B, d)
        feat_res = self.proj_residual(feat_res)
        combined = torch.cat([cls_out, feat_res], dim=-1)  # (B, 2d)
        logits = self.mlp_head(combined).squeeze(-1)
        return logits

# -------------------- 3. 数据加载与标签平滑 --------------------
class TitanicDataset(Dataset):
    def __init__(self, X, y, cat_indices):
        n_features = X.shape[1]
        cont_mask = np.ones(n_features, dtype=bool)
        cont_mask[cat_indices] = False
        cont_idx = np.where(cont_mask)[0]

        self.x_cont = torch.tensor(X[:, cont_idx], dtype=torch.float32)
        self.x_cat = [torch.tensor(X[:, idx], dtype=torch.long) for idx in cat_indices]
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x_cont[idx], [cat[idx] for cat in self.x_cat], self.y[idx]

def make_loader(X, y, cat_indices, batch_size, shuffle):
    ds = TitanicDataset(X, y, cat_indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class TestDataset(Dataset):
    def __init__(self, X, cat_indices):
        n_features = X.shape[1]
        cont_mask = np.ones(n_features, dtype=bool)
        cont_mask[cat_indices] = False
        cont_idx = np.where(cont_mask)[0]

        self.x_cont = torch.tensor(X[:, cont_idx], dtype=torch.float32)
        self.x_cat = [torch.tensor(X[:, idx], dtype=torch.long) for idx in cat_indices]

    def __len__(self):
        return self.x_cont.size(0)

    def __getitem__(self, idx):
        return self.x_cont[idx], [cat[idx] for cat in self.x_cat]

def make_test_loader(X, cat_indices, batch_size):
    ds = TestDataset(X, cat_indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def smooth_labels(y, smoothing=0.1):
    # BCEWithLogitsLoss 期望浮点标签，平滑后不再精确为0/1
    return y * (1 - smoothing) + 0.5 * smoothing

# -------------------- 4. 训练与评估 --------------------
def train_epoch(model, loader, optimizer, criterion, label_smoothing=0.1):
    model.train()
    total_loss = 0
    for x_cont, x_cat_list, y in loader:
        x_cont, y = x_cont.to(DEVICE), y.to(DEVICE)
        x_cat_list = [x.to(DEVICE) for x in x_cat_list]
        y_smooth = smooth_labels(y, label_smoothing)

        optimizer.zero_grad()
        logits = model(x_cont, x_cat_list)
        loss = criterion(logits, y_smooth)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_cont.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for x_cont, x_cat_list, y in loader:
        x_cont = x_cont.to(DEVICE)
        x_cat_list = [x.to(DEVICE) for x in x_cat_list]
        logits = model(x_cont, x_cat_list)
        prob = torch.sigmoid(logits)
        preds.extend((prob > 0.5).int().cpu().numpy())
        trues.extend(y.cpu().numpy())
    return accuracy_score(trues, preds)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs = []
    for x_cont, x_cat_list in loader:
        x_cont = x_cont.to(DEVICE)
        x_cat_list = [x.to(DEVICE) for x in x_cat_list]
        logits = model(x_cont, x_cat_list)
        prob = torch.sigmoid(logits)
        probs.extend(prob.cpu().numpy())
    return np.array(probs)

# -------------------- 5. 主程序 --------------------
def main():
    X_train, y_train, X_test, cat_indices, cat_cardinalities, test_passenger_ids = load_and_preprocess()
    n_cont = X_train.shape[1] - len(cat_indices)

    D_TOKEN = 32
    N_LAYERS = 3
    N_HEADS = 8
    DROPOUT = 0.45
    DROP_PATH = 0.2
    WEIGHT_DECAY = 1e-2
    LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 200
    PATIENCE = 20
    N_SPLITS = 5
    LABEL_SMOOTHING = 0.1

    SAVE_DIR = '../checkpoints'
    os.makedirs(SAVE_DIR, exist_ok=True)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_accs = []
    fold_models = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f'\n===== Fold {fold+1}/{N_SPLITS} =====')
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        train_loader = make_loader(X_tr, y_tr, cat_indices, BATCH_SIZE, True)
        val_loader = make_loader(X_val, y_val, cat_indices, BATCH_SIZE, False)

        model = FTTransformer(
            n_cont_features=n_cont,
            cat_cardinalities=cat_cardinalities,
            d_token=D_TOKEN,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            dropout=DROPOUT,
            drop_path=DROP_PATH,
        ).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )

        best_acc = 0
        best_state = None
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, LABEL_SMOOTHING)
            val_acc = evaluate(model, val_loader)
            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

            if (epoch+1) % 20 == 0:
                print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}')

        model.load_state_dict(best_state)
        save_path = os.path.join(SAVE_DIR, f'fold_{fold+1}.pt')
        torch.save(best_state, save_path)
        print(f'Fold {fold+1} Best Val Acc: {best_acc:.4f} | Saved to {save_path}')

        fold_accs.append(best_acc)
        fold_models.append(model)

    print(f'\n===============')
    print(f'Average CV Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}')

    # -------------------- 测试集推理（多折集成） --------------------
    print('\n===== Predicting on test set (ensemble of all folds) =====')
    test_loader = make_test_loader(X_test, cat_indices, BATCH_SIZE)

    all_probs = []
    for fold_idx, model in enumerate(fold_models):
        probs = predict(model, test_loader)
        all_probs.append(probs)
        print(f'Fold {fold_idx+1} prediction done.')

    avg_probs = np.mean(all_probs, axis=0)
    test_preds = (avg_probs > 0.5).astype(int)

    submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': test_preds
    })
    submission_path = '../data/submissions/ft_transformer_submission.csv'
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f'Submission saved to {submission_path}')
    print(f'Test predictions distribution: {np.bincount(test_preds)}')

if __name__ == '__main__':
    main()