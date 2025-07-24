# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import pydantic
import torch.nn as nn

logger = logging.getLogger(__name__)


class TransformerEncoderConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["TransformerEncoder"] = "TransformerEncoder"
    heads: int = 8
    depth: int = 12

    cross_attend: bool = False
    causal: bool = False
    attn_flash: bool = False
    attn_dropout: float = 0.1

    ff_mult: int = 4

    ff_dropout: float = 0.0

    use_scalenorm: bool = True
    use_rmsnorm: bool = False

    rel_pos_bias: bool = False
    alibi_pos_bias: bool = False
    rotary_pos_emb: bool = True
    rotary_xpos: bool = False

    residual_attn: bool = False
    scale_residual: bool = True
    layer_dropout: float = 0.0

    def build(self, dim: int) -> nn.Module:
        from x_transformers import Decoder, Encoder

        if dim % self.heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by the number of heads ({self.heads})"
            )
        if dim < 256:
            raise ValueError(
                f"dim ({dim}) is less than 256, which causes weird bug in x-transformers"
            )
        kwargs = self.model_dump()
        kwargs["attn_dim_head"] = dim // self.heads
        del kwargs["name"]
        del kwargs["causal"]
        if self.causal:
            return Decoder(dim=dim, **kwargs)
        else:
            return Encoder(dim=dim, **kwargs)
