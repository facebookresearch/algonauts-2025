# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import typing as tp
from functools import partial

import pydantic
import torch
from torch import nn
from torchvision.ops import MLP

from .common import Mean, MlpConfig, SubjectLayers

logger = logging.getLogger(__name__)


class FmriMlpConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["FmriMlp"] = "FmriMlp"

    hidden: int = 4096
    n_blocks: int = 4
    norm_type: str = "ln"
    act_first: bool = False

    n_repetition_times: int = 1
    time_agg: tp.Literal["in_mean", "in_linear", "out_mean", "out_linear"] = "out_linear"

    use_tr_embeds: bool = False
    tr_embed_dim: int = 16
    use_tr_layer: bool = False

    out_dim: int | None = None

    subject_layers: bool = False
    n_subjects: int = 20
    subject_layers_dim: tp.Literal["input", "hidden"] = "hidden"
    subject_layers_id: bool = False

    output_head_config: MlpConfig | dict[str, MlpConfig] | None = None

    def build(self, n_in_channels: int, n_outputs: int | None) -> nn.Module:
        out_dim = self.out_dim if n_outputs is None else n_outputs
        if out_dim is None:
            raise ValueError("One of n_outputs or config.out_dim must be set.")

        return FmriMlp(
            in_dim=n_in_channels,
            out_dim=out_dim,
            config=self,
        )


class FmriMlp(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: FmriMlpConfig | None = None,
    ):
        super().__init__()
        config = config if config is not None else FmriMlpConfig()

        self.in_time_agg: None | nn.Module = None
        self.out_time_agg: None | nn.Module = None
        self.n_repetition_times = config.n_repetition_times
        if config.time_agg == "in_mean":
            self.in_time_agg = Mean(dim=2, keepdim=True)
            self.n_repetition_times = 1
        elif config.time_agg == "in_linear":
            self.in_time_agg = nn.LazyLinear(1)
            self.n_repetition_times = 1
        elif config.time_agg == "out_mean":
            self.out_time_agg = Mean(dim=2)
        elif config.time_agg == "out_linear":
            self.out_time_agg = nn.LazyLinear(1)

        norm_func = (
            partial(nn.BatchNorm1d, num_features=config.hidden)
            if config.norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=config.hidden)
        )
        act_fn = partial(nn.ReLU, inplace=True) if config.norm_type == "bn" else nn.GELU
        act_and_norm = (act_fn, norm_func) if config.act_first else (norm_func, act_fn)

        self.subject_layers = None
        if config.subject_layers:
            dim = {"hidden": config.hidden, "input": in_dim}[config.subject_layers_dim]
            self.subject_layers = SubjectLayers(
                in_dim,
                dim,
                config.n_subjects,
                config.subject_layers_id,
                mode="for_loop",
            )
            in_dim = dim

        self.tr_embeddings = None
        if config.use_tr_embeds:
            self.tr_embeddings = nn.Embedding(
                self.n_repetition_times, config.tr_embed_dim
            )
            in_dim += config.tr_embed_dim

        self.lin0: nn.Conv1d | nn.Linear
        if config.use_tr_layer:
            self.lin0 = nn.Conv1d(
                in_channels=self.n_repetition_times,
                out_channels=self.n_repetition_times * config.hidden,
                kernel_size=in_dim,
                groups=self.n_repetition_times,
                bias=True,
            )
        else:
            self.lin0 = nn.Linear(in_dim, config.hidden)
        self.post_lin0 = nn.Sequential(
            *[item() for item in act_and_norm], nn.Dropout(0.5)
        )

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden, config.hidden),
                    *[item() for item in act_and_norm],
                    nn.Dropout(0.15),
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.lin1 = nn.Linear(config.hidden, out_dim, bias=True)
        self.n_blocks = config.n_blocks

        self.output_head: None | MLP | dict[str, MLP] = None
        if config.output_head_config is not None:
            if isinstance(config.output_head_config, MlpConfig):
                self.output_head = config.output_head_config.build(input_size=out_dim)
            elif isinstance(config.output_head_config, dict):
                self.output_head = nn.ModuleDict()
                for name, head_config in config.output_head_config.items():
                    self.output_head[name] = head_config.build(
                        input_size=out_dim,
                    )

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.in_time_agg is not None:
            x = self.in_time_agg(x)

        B, _, T = x.shape
        assert (
            T == self.n_repetition_times
        ), f"Mismatch between expected and provided number TRs: {T} != {self.n_repetition_times}"

        if self.subject_layers is not None:
            x = self.subject_layers(x, subject_ids)
        x = x.permute(0, 2, 1)

        if self.tr_embeddings is not None:
            embeds = self.tr_embeddings(torch.arange(T, device=x.device))
            embeds = torch.tile(embeds, dims=(B, 1, 1))
            x = torch.cat([x, embeds], dim=2)

        x = self.lin0(x).reshape(B, T, -1)

        x = self.post_lin0(x)

        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x

        x = x.permute(0, 2, 1)

        if self.out_time_agg is not None:
            x = self.out_time_agg(x)

        x = x.flatten(1)

        x = self.lin1(x)

        if isinstance(self.output_head, MLP):
            out = self.output_head(x)
        elif isinstance(self.output_head, nn.ModuleDict):
            out = {name: head(x) for name, head in self.output_head.items()}
        else:
            out = x

        return out
