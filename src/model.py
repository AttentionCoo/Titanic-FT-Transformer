"""FT-Transformer 模型实现（v2）。

相比 v1 的改动：
* Pre-LN 残差结构（训练更稳定，配合 DropPath 不易发散）；
* FFN 隐层按 ``ff_mult`` 扩展，使用 GELU；
* 数值 token 与类别 token 统一 LayerNorm；
* 输出同时使用 CLS token 与特征 token 均值（可选），保留原始特征信号；
* 训练侧配合 EMA 权重，推理更平滑。
"""

from __future__ import annotations

import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """固定随机种子，保证同一配置可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DropPath(nn.Module):
    """Stochastic depth：训练时以一定概率丢弃整个残差分支。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x / keep * mask


class FeatureTokenizer(nn.Module):
    """把每个特征映射成一个 d_token 维 token。"""

    def __init__(self, n_num: int, cat_cardinalities: tuple[int, ...], d_token: int):
        super().__init__()
        self.n_num = n_num
        self.weight = nn.Parameter(torch.empty(n_num, d_token))
        self.bias = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )
        for embedding in self.cat_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
        self.norm = nn.LayerNorm(d_token)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = []
        if self.n_num:
            tokens.append(x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0))
        for j, embedding in enumerate(self.cat_embeddings):
            tokens.append(embedding(x_cat[:, j]).unsqueeze(1))
        return self.norm(torch.cat(tokens, dim=1))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block：Attention + FFN，各自带残差与 DropPath。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        drop_path: float,
        ff_mult: float = 2.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_token={d_model} 必须能被 n_heads={n_heads} 整除")
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(round(d_model * ff_mult))
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path(self.dropout(attn_out))
        ff_out = self.ff(self.norm2(x))
        x = x + self.drop_path(self.dropout(ff_out))
        return x


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer，二分类输出 logit。"""

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: tuple[int, ...],
        d_token: int = 32,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.2,
        drop_path: float = 0.1,
        ff_mult: float = 2.0,
        head_hidden: int = 64,
        use_feature_pool: bool = True,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_token, n_heads, dropout, drop_path, ff_mult)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_token)
        self.use_feature_pool = use_feature_pool
        head_in = d_token * 2 if use_feature_pool else d_token
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        x = torch.cat([self.cls_token.expand(tokens.size(0), -1, -1), tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        cls_out = x[:, 0]
        if self.use_feature_pool:
            pooled = x[:, 1:].mean(dim=1)
            representation = torch.cat([cls_out, pooled], dim=-1)
        else:
            representation = cls_out
        return self.head(representation).squeeze(-1)


class ModelEMA:
    """参数的指数滑动平均，作为最终推理权重。"""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.state: OrderedDict[str, torch.Tensor] = OrderedDict(
            (key, value.detach().clone()) for key, value in model.state_dict().items()
        )

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        # 前期用较小的 decay，避免 EMA 被随机初始化拖住。
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for key, value in model.state_dict().items():
            shadow = self.state[key]
            if shadow.dtype.is_floating_point:
                shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(value)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.state)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
