# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025-2026 thiliapr <thiliapr@tutanota.com>
# SPDX-Package: melody_extract
# SPDX-PackageHomePage: https://github.com/thiliapr/melody_extract

import math
from typing import TypedDict
import torch
from torch import nn
from torch.nn import functional as F

MAX_INTERVAL = 16
NUM_INTERVAL_BINS = 32
WINDOW_SIZE = 65  # 包含当前时间点在内的前后各 32 个时间点
assert WINDOW_SIZE % 2 == 1, "WINDOW_SIZE 应为奇数以确保有中心点"


class ModelInput(TypedDict):
    """
    模型输入的类型定义

    Attributes:
        piano_roll: 形状为 [batch_size, window_size, 128]
        interval: 形状为 [batch_size, window_size]
    """
    piano_roll: torch.Tensor
    interval: torch.Tensor


class ModelOutput(TypedDict):
    """
    模型输出的类型定义

    Attributes:
        logit: 形状为 [batch_size]
    """
    logit: torch.Tensor


class BottleNeck(nn.Module):
    def __init__(self, num_channels: int, dilation: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels // 4)
        self.bn3 = nn.BatchNorm2d(num_channels // 4)
        self.conv1 = nn.Conv2d(num_channels, num_channels // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(num_channels // 4, num_channels // 4, 3, dilation=dilation, padding="same", bias=False)
        self.conv3 = nn.Conv2d(num_channels // 4, num_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(F.silu(self.bn1(x)))
        x = self.conv2(F.silu(self.bn2(x)))
        x = self.conv3(F.silu(self.bn3(x)))
        return x


class Model(nn.Module):
    def __init__(self, dropout: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 好像很多 bin 都不可能被落到，不过懒得优化了
        self.inverval_embedding = nn.Embedding(NUM_INTERVAL_BINS, 32)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers1 = nn.ModuleList(BottleNeck(32, 2 ** i) for i in range(5))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.layers2 = nn.ModuleList(BottleNeck(64, 2 ** i) for i in range(4))
        self.fc_out = nn.Linear(64, 1)

    def forward(self, inputs: ModelInput) -> ModelOutput:
        # 时间嵌入对数处理，对缩放更友好
        interval = inputs["interval"]
        zero_interval = interval == 0
        interval = interval.log()
        # 在对数空间平移时间间隔（相当于在实际空间除以一个数），模拟时间的缩放不变性
        interval[zero_interval] = torch.inf  # 
        # 注: 零间隔设为 inf，但不会影响 5% 分位数计算，因为需要超过 95% 的间隔为 0 才会使 inf 进入 5% 分位数，在实际音乐数据中这几乎不可能出现
        interval = interval - interval.quantile(torch.tensor(0.05, device=interval.device), dim=1, keepdim=True)
        # 归一化并离散化
        interval = interval / math.log(MAX_INTERVAL)
        interval = interval.clamp(0, 1)
        # 这里减少一个箱，因为 0 时间间隔需要特别分一个箱 
        interval = interval * (NUM_INTERVAL_BINS - 2)
        interval = 1 + interval.round().long()
        # 给零间隔分特别的箱
        interval[zero_interval] = 0

        # 获取时间间隔嵌入，交换维度并添加音高维度，以便作为通道特征与卷积输出相加
        interval_emb = self.inverval_embedding(interval)  # [batch_size, window_size, 32]
        interval_emb = interval_emb.transpose(1, 2).unsqueeze(-1)  # [batch_size, 32, window_size, 1]

        # 在通道维添加维度，经过卷积得到通道特征
        x = self.bn1(self.conv1(inputs["piano_roll"].unsqueeze(1)))  # [batch_size, 32, window_size, 128]

        # 融合 interval embedding，通过广播在音高维（最后一维）匹配并逐元素相加
        x = self.dropout(F.silu(x) + interval_emb)  # [batch_size, 32, window_size, 128]

        # 继续卷积处理
        x = F.avg_pool2d(x, 3, (2, 2), 1)  # [batch_size, 32, ceil(window_size / 2), 64]
        for layer in self.layers1:
            x = x + self.dropout(layer(x))
        x = self.conv2(x)  # [batch_size, 64, ceil(window_size / 4), 32]
        for layer in self.layers2:
            x = x + self.dropout(layer(x))

        # 全局平均池化并预测
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # [batch_size, 64]
        x = self.fc_out(x).squeeze(-1)  # [batch_size]
        return {"logit": x}
