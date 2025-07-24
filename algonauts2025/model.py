# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pydantic
import torch
from data_utils.dataloader import SegmentData
from einops import rearrange
from modeling_utils.models.common import MlpConfig, SubjectLayers
from modeling_utils.models.transformer import TransformerEncoderConfig
from torch import nn


class FmriEncoderConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["FmriEncoder"] = "FmriEncoder"
    n_subjects: int | None = None  # LATER
    feature_aggregation: tp.Literal["sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    subject_embedding: bool = False
    modality_dropout: float = 0.0

    def build(
        self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int
    ) -> nn.Module:
        return FmriEncoder(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            config=self,
        )


class FmriEncoder(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoderConfig,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs
        self.projectors = nn.ModuleDict()
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)
        hidden = 3072
        for modality, tup in feature_dims.items():
            if tup is None:
                print(
                    f"Warning: {modality} has no feature dimensions. Skipping projector."
                )
                continue
            else:
                num_layers, feature_dim = tup
            input_dim = (
                feature_dim * num_layers
                if config.layer_aggregation == "cat"
                else feature_dim
            )
            output_dim = (
                hidden // len(feature_dims)
                if config.feature_aggregation == "cat"
                else hidden
            )
            self.projectors[modality] = MlpConfig(
                norm_layer="layer", activation_layer="gelu", dropout=0.0
            ).build(input_dim, output_dim)
        input_dim = (
            (hidden // len(feature_dims)) * len(feature_dims)
            if config.feature_aggregation == "cat"
            else hidden
        )
        self.combiner = nn.Identity()
        self.predictor = SubjectLayers(
            in_channels=hidden,
            out_channels=n_outputs,
            n_subjects=config.n_subjects,
            average_subjects=False,
            bias=True,
        )
        self.time_pos_embed = nn.Parameter(torch.randn(1, 1024, hidden))
        if config.subject_embedding:
            self.subject_embed = nn.Embedding(config.n_subjects, hidden)
        self.encoder = TransformerEncoderConfig(
            attn_dropout=0.0, ff_dropout=0.0, layer_dropout=0.0, depth=8
        ).build(dim=hidden)

    def forward(self, batch: SegmentData, pool_outputs: bool = True) -> torch.Tensor:
        x = self.aggregate_features(batch)  # B, T, H
        subject_id = batch.data.get("subject_id", None)
        x = self.transformer_forward(x, subject_id)
        x = x.transpose(1, 2)  # B, H, T
        x = self.predictor(x, subject_id)  # B, O, T
        if pool_outputs:
            out = self.pooler(x)  # B, O, T'
        else:
            out = x
        return out

    def aggregate_features(self, batch):
        tensors = []
        # get B, T
        for modality in batch.data.keys():
            if modality in self.feature_dims:
                break
        x = batch.data[modality]
        B, T = x.shape[0], x.shape[-1]
        # select the modalities to dropout, keep at least one modality
        modalities_to_dropout = []
        for modality in self.feature_dims.keys():
            if torch.rand(1).item() < self.config.modality_dropout and self.training:
                modalities_to_dropout.append(modality)
        if len(modalities_to_dropout) == len(self.feature_dims):
            modalities_to_dropout = np.random.choice(
                modalities_to_dropout, len(modalities_to_dropout) - 1, replace=False
            )
        for modality in self.feature_dims.keys():
            if modality not in self.projectors:
                data = torch.zeros(B, T, 3072 // len(self.feature_dims)).to(x.device)
            else:
                data = batch.data[modality]  # B, L, H, T
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)
                # mean over layers
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)
                elif self.config.layer_aggregation == "cat":
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)
                assert data.ndim == 3  # B, T, D
                data = self.projectors[modality](data)  # B, T, H
                if modality in modalities_to_dropout:
                    data = torch.zeros_like(data)
            tensors.append(data)
        if self.config.feature_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)
        elif self.config.feature_aggregation == "sum":
            out = sum(tensors)
        return out

    def transformer_forward(self, x, subject_id=None):
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed"):
            x = x + self.subject_embed(subject_id)
        x = self.encoder(x)
        return x
