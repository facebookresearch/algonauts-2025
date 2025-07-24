# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp

import pydantic
import torch
from torch import nn
from torchvision.ops import MLP


class SubjectLayers(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_subjects: int,
        bias: bool = False,
        init_id: bool = False,
        average_subjects: bool = False,
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(n_subjects, out_channels)) if bias else None
        if init_id:
            if in_channels != out_channels:
                raise ValueError(
                    "in_channels and out_channels must be the same for identity initialization."
                )
            self.weights.data[:] = torch.eye(in_channels)[None]
            if self.bias is not None:
                self.bias.data[:] = 0
        else:
            self.weights.data.normal_()
            if self.bias is not None:
                self.bias.data.normal_()
        self.weights.data *= 1 / in_channels**0.5
        if self.bias is not None:
            self.bias.data *= 1 / in_channels**0.5
        self.average_subjects = average_subjects

    def forward(
        self,
        x: torch.Tensor,
        subjects: torch.Tensor,
    ) -> torch.Tensor:

        B, C, T = x.shape
        N, C, D = self.weights.shape
        assert (
            subjects.max() < N
        ), "Subject index higher than number of subjects used to initialize the weights."
        if self.average_subjects:
            weights = self.weights.mean(dim=0).expand(B, C, D)
            if self.bias is not None:
                bias = self.bias.mean(dim=0).view(1, D, 1).expand(B, D, 1)
        else:
            weights = self.weights.index_select(0, subjects.flatten())
            if self.bias is not None:
                bias = self.bias.index_select(0, subjects.flatten()).view(B, D, 1)
        out = torch.einsum("bct,bcd->bdt", x, weights)
        if self.bias is not None:
            out += bias
        return out

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class LayerScale(nn.Module):

    def __init__(self, channels: int, init: float = 0.1, boost: float = 5.0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels))
        self.scale.data[:] = init / boost
        self.boost = boost

    def forward(self, x):
        return (self.boost * self.scale[:, None]) * x


class MlpConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["Mlp"] = "Mlp"

    input_size: int | None = None
    hidden_sizes: list[int] | None = None

    norm_layer: tp.Literal["layer", "batch", "instance", "unit", None] = None
    activation_layer: tp.Literal["relu", "gelu", "elu", "prelu", None] = "relu"

    bias: bool = True
    dropout: float = 0.0

    @staticmethod
    def _get_norm_layer(kind: str | None) -> tp.Type[nn.Module] | None:
        return {
            "batch": nn.BatchNorm1d,
            "layer": nn.LayerNorm,
            "instance": nn.InstanceNorm1d,
            None: None,
        }[kind]

    @staticmethod
    def _get_activation_layer(kind: str | None) -> tp.Type[nn.Module]:
        return {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "prelu": nn.PReLU,
            None: nn.Identity,
        }[kind]

    def build(
        self, input_size: int | None = None, output_size: int | None = None
    ) -> nn.Sequential | nn.Identity:
        input_size = self.input_size if input_size is None else input_size
        assert input_size is not None, "input_size cannot be None."
        if not self.hidden_sizes:
            assert (
                output_size is not None
            ), "output_size cannot be None if hidden_sizes is empty."
            return nn.Linear(input_size, output_size)

        hidden_sizes = self.hidden_sizes
        if output_size is not None:
            hidden_sizes.append(output_size)

        return MLP(
            in_channels=input_size,
            hidden_channels=hidden_sizes,
            norm_layer=self._get_norm_layer(self.norm_layer),
            activation_layer=self._get_activation_layer(self.activation_layer),
            bias=self.bias,
            dropout=self.dropout,
        )


class Mean(nn.Module):
    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)
