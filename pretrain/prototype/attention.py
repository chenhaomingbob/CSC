"""
from
# 全网最强ViT (Vision Transformer)原理及代码解析 - Chaos万有引力的文章 - 知乎
# https://zhuanlan.zhihu.com/p/427388113
"""
import torch
import torch.nn as nn


class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, *args):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, *args):
        # B, N, C = x.shape
        # N, C = x.shape
        q = self.q(x)  # 3d
        k = self.k(y)  # 2d
        v = self.v(x)  # 3d

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (N,N)
        attn = attn.softmax(dim=-1) # (N,N)
        attn[attn<0.6] = torch.tensor(0,device=attn.device)
        attn = self.attn_drop(attn)  # (N,N)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn @ v  # (N,N) (N,c) -> (N,C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
