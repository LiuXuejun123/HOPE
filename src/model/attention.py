"""Transformer 模块, 用来融合多种观测模态的特征(HOPE 中"用 Transformer 融合感知信息和动作掩码"的部分)。

每种模态(雷达、目标、动作掩码、图像)先被编码成一个 128 维向量, 看作一个 token;
多个 token 经过自注意力层相互交换信息, 再拼接后用 MLP 输出动作或价值。
"""


import torch
from torch import nn
from einops import rearrange
from tqdm import trange

class PreNorm(nn.Module):
    """Pre-LayerNorm 包装: 先做 LayerNorm, 再调用内部模块 fn。"""
    def __init__(self, dim, fn):
        """dim: 特征维度; fn: 被包装的模块(Attention 或 FeedForward)。"""
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        """返回 fn(LayerNorm(x))。"""
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    """多头自注意力层。"""
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        """参数:
            dim:      输入特征维度。
            heads:    注意力头数。
            dim_head: 每个头的维度。
            dropout:  dropout 比例。
        """
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x: [b, n, f]
        """参数:
            x: 形状 (batch, n_token, dim)。
        返回:
            形状相同的张量, 每个 token 融合了其他 token 的信息。
        """
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # [b, h, n, n]

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v) # v: [b, h, n, d]   attn: [b, h, n, n]
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) # [b, n, h*d] -> [b, n, f]
    
class FeedForward(nn.Module):
    """Transformer 中的前馈网络: Linear -> Tanh -> Linear。"""
    def __init__(self, dim, hidden_dim, dropout = 0.):
        """dim: 输入/输出维度; hidden_dim: 隐层维度。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(), # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        """对每个 token 独立做前馈变换, 形状不变。"""
        return self.net(x)

class Transformer(nn.Module):
    """由 depth 层 [自注意力 + 前馈] 组成的 Transformer 编码器, 每层带残差连接。"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        """参数含义同 Attention / FeedForward, depth 为层数。"""
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        """x: (batch, n_token, dim) -> (batch, n_token, dim)。"""
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class AttentionNetwork(nn.Module):
    """Transformer 编码器 + 输出 MLP: 把所有 token 展平后映射到 output_dim。"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, n_features, hidden_dim, output_dim):
        """参数:
            dim, depth, heads, dim_head, mlp_dim: Transformer 配置。
            n_features: token 数量(即模态数)。
            hidden_dim: 输出 MLP 的隐层维度。
            output_dim: 输出维度(actor 为 2, critic 为 1)。
        """
        super().__init__()
        self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim,)
        self.output = nn.Sequential(
            nn.Linear(n_features*dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Tanh(),
        )
        self.view_embed = nn.Parameter(torch.zeros(1, n_features, dim))

    def forward(self, x):
        # x = x + self.view_embed
        """x: (batch, n_features, dim) -> (batch, output_dim)。"""
        x = self.encoder(x)
        x = rearrange(x, 'b n d -> b (n d)')
        return self.output(x)


    
