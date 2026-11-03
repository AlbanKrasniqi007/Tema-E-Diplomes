"""SE residual network matching Lc0 NETWORK_SE_WITH_HEADFORMAT.

Topology is taken from https://lczero.org/dev/old/nn/ :

* 3×3 input convolution 112 → FILTERS
* BLOCKS residual units, each two 3×3 convolutions plus a Squeeze-and-Excitation
  unit (global average pool → FC → ReLU → FC → split W,B → sigmoid(W)*x + B)
* skip connection then ReLU
* convolutional policy head (80×8×8 gathered to 1858) or classical FC head
* WDL value head (softmax over win/draw/loss)
* moves-left head (ReLU scalar)

BatchNorm is kept for training; Lc0 folds BN into convolution weights only
for inference in the C++ backends.
"""

from __future__ import annotations

import torch
from torch import nn

from lc0.chess.encoder import NUM_INPUT_PLANES
from lc0.chess.policy_index import POLICY_SIZE
from lc0.chess.policy_map import AZ_GATHER_INDICES
from lc0.config import NetworkConfig


class ResidualSEBlock(nn.Module):
    def __init__(self, filters: int, se_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(filters)
        self.fc1 = nn.Linear(filters, se_channels)
        self.fc2 = nn.Linear(se_channels, 2 * filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        squeezed = x.mean(dim=(2, 3))
        excited = self.fc2(self.relu(self.fc1(squeezed)))
        weight, bias = excited.chunk(2, dim=1)
        scale = torch.sigmoid(weight).unsqueeze(-1).unsqueeze(-1)
        shift = bias.unsqueeze(-1).unsqueeze(-1)
        x = scale * x + shift
        return self.relu(x + residual)


class Lc0SEResNet(nn.Module):
    """Trainable Lc0 SE-ResNet with WDL, policy, and moves-left heads."""

    def __init__(self, cfg: NetworkConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or NetworkConfig()
        self.cfg = cfg
        filters = cfg.filters
        self.input_conv = nn.Conv2d(NUM_INPUT_PLANES, filters, kernel_size=3, padding=1, bias=True)
        self.input_bn = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.residuals = nn.Sequential(
            *[ResidualSEBlock(filters, cfg.se_channels) for _ in range(cfg.blocks)]
        )

        if cfg.policy_head == "convolution":
            self.policy_conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=True)
            self.policy_bn = nn.BatchNorm2d(filters)
            self.policy_conv2 = nn.Conv2d(filters, 80, kernel_size=3, padding=1, bias=True)
            self.policy_fc = None
            gather = torch.from_numpy(AZ_GATHER_INDICES).long()
            self.register_buffer("az_gather", gather, persistent=False)
        else:
            self.policy_conv1 = nn.Conv2d(filters, cfg.policy_conv_size, kernel_size=1, bias=True)
            self.policy_bn = nn.BatchNorm2d(cfg.policy_conv_size)
            self.policy_conv2 = None
            self.policy_fc = nn.Linear(cfg.policy_conv_size * 64, POLICY_SIZE)

        self.value_conv = nn.Conv2d(filters, 32, kernel_size=1, bias=True)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 64, 128)
        if cfg.value_head == "wdl":
            self.value_fc2 = nn.Linear(128, 3)
        else:
            self.value_fc2 = nn.Linear(128, 1)

        self.mlh_conv = nn.Conv2d(filters, cfg.mlh_channels, kernel_size=1, bias=True)
        self.mlh_bn = nn.BatchNorm2d(cfg.mlh_channels)
        self.mlh_fc1 = nn.Linear(cfg.mlh_channels * 64, cfg.mlh_fc)
        self.mlh_fc2 = nn.Linear(cfg.mlh_fc, 1)

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits, wdl_or_q, moves_left)``.

        ``planes`` shape is ``(N, 112, 8, 8)``.
        """
        x = self.relu(self.input_bn(self.input_conv(planes)))
        x = self.residuals(x)

        if self.policy_conv2 is not None:
            p = self.relu(self.policy_bn(self.policy_conv1(x)))
            p = self.policy_conv2(p).reshape(p.size(0), -1)
            policy_logits = p.index_select(1, self.az_gather)
        else:
            p = self.relu(self.policy_bn(self.policy_conv1(x)))
            policy_logits = self.policy_fc(p.reshape(p.size(0), -1))

        v = self.relu(self.value_bn(self.value_conv(x)))
        v = self.relu(self.value_fc1(v.reshape(v.size(0), -1)))
        value = self.value_fc2(v)
        if self.cfg.value_head != "wdl":
            value = torch.tanh(value)

        m = self.relu(self.mlh_bn(self.mlh_conv(x)))
        m = self.relu(self.mlh_fc1(m.reshape(m.size(0), -1)))
        moves_left = self.relu(self.mlh_fc2(m)).squeeze(-1)
        return policy_logits, value, moves_left
