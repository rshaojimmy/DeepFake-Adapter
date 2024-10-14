# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=768):
        super().__init__()

        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        # self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        bs = x.shape[0]
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        # c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        # bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, self.embed_dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, self.embed_dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, self.embed_dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, self.embed_dim, -1).transpose(1, 2)  # 32s

        return c2, c3, c4


class Injector(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.0):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)

        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):
        attn = self.self_attn(self.query_norm(query), self.feat_norm(feat), value=self.feat_norm(feat))[0]
        return query + self.gamma * attn


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=0.0, batch_first=True)

    def forward(self, query, feat):
        attn = self.self_attn(self.query_norm(query), self.feat_norm(feat), value=self.feat_norm(feat))[0]
        query = query + attn

        return query


class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.injector = Injector(dim=dim, num_heads=num_heads, norm_layer=norm_layer)
        self.extractor = Extractor(dim=dim, num_heads=num_heads, norm_layer=norm_layer)

    def forward(self, x, c, blocks):
        x = self.injector(query=x, feat=c)
        for idx, blk in enumerate(blocks):
            x = blk(x)
        c = self.extractor(query=c, feat=x)

        return x, c
