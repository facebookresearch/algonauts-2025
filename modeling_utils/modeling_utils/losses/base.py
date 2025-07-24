# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from inspect import isclass

import pydantic
from torch import nn
from torch.nn.modules.loss import _Loss

from data_utils.infra import helpers
from modeling_utils.utils import all_subclasses, convert_to_pydantic

from . import losses

custom_losses = [
    obj for obj in losses.__dict__.values() if isclass(obj) and issubclass(obj, nn.Module)
]

TORCHLOSS_NAMES = [loss_class.__name__ for loss_class in all_subclasses(_Loss)]


class BaseLossConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="forbid")
    name: str

    def build(self) -> nn.Module:
        raise NotImplementedError


for loss_class in custom_losses:
    loss_class_name = loss_class.__name__
    config_cls = convert_to_pydantic(
        loss_class, loss_class_name, parent_class=BaseLossConfig
    )
    locals()[f"{loss_class_name}Config"] = config_cls


class TorchLossConfig(BaseLossConfig):
    name: tp.Literal[tuple(TORCHLOSS_NAMES)]

    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

        helpers.validate_kwargs(getattr(nn, self.name), self.kwargs)

    def build(self, **kwargs: tp.Any) -> nn.Module:
        if overlap := set(self.kwargs) & set(kwargs):
            raise ValueError(
                f"Build kwargs overlap with config kwargs for keys: {overlap}."
            )
        kwargs = self.kwargs | kwargs
        return getattr(nn, self.name)(**kwargs)
