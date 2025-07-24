# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    def __init__(self, reduction: str = "mean", dim: int = 1):
        super(PearsonLoss, self).__init__()
        self.reduction = reduction
        self.dim = dim

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        x = x.transpose(0, self.dim)
        y = y.transpose(0, self.dim)
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)

        x_mean = torch.mean(x, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)
        x = x - x_mean
        y = y - y_mean

        cov = torch.sum(x * y, dim=1)

        x_std = torch.sqrt(torch.sum(x**2, dim=1))
        y_std = torch.sqrt(torch.sum(y**2, dim=1))

        pcc = cov / ((x_std * y_std) + 1e-8)

        loss = 1 - pcc
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
