# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pydantic
import torch
from torch import optim
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from data_utils.infra import helpers
from modeling_utils.utils import all_subclasses

TORCH_OPTIMIZER_NAMES = [
    cls.__name__ for cls in all_subclasses(Optimizer) if cls.__name__ != "NewCls"
]
TORCH_LR_SCHEDULER_NAMES = [cls.__name__ for cls in all_subclasses(LRScheduler)]


class BaseOptimizerConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="forbid")
    name: str

    def build(self, params: tp.Iterable[torch.Tensor]) -> Optimizer:
        raise NotImplementedError


class TorchOptimizerConfig(BaseOptimizerConfig):
    name: tp.Literal[tuple(TORCH_OPTIMIZER_NAMES)]

    lr: float
    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        assert (
            "lr" not in self.kwargs
        ), "lr should be defined as a base parameter instead of within kwargs."

        helpers.validate_kwargs(getattr(optim, self.name), self.kwargs | {"params": None})

    def build(self, params: tp.Iterable[torch.Tensor]) -> Optimizer:
        return getattr(optim, self.name)(params, lr=self.lr, **self.kwargs)


class BaseLRSchedulerConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="forbid")
    name: str

    def build(self, optimizer: Optimizer) -> LRScheduler:
        raise NotImplementedError


class TorchLRSchedulerConfig(BaseLRSchedulerConfig):
    name: tp.Literal[tuple(TORCH_LR_SCHEDULER_NAMES)]

    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

        helpers.validate_kwargs(
            getattr(optim.lr_scheduler, self.name), self.kwargs | {"optimizer": None}
        )

    def build(self, optimizer: Optimizer, **build_kwargs: tp.Any) -> LRScheduler:
        return getattr(optim.lr_scheduler, self.name)(
            optimizer, **(self.kwargs | build_kwargs)
        )


class LightningOptimizerConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["LightningOptimizer"] = "LightningOptimizer"
    optimizer: TorchOptimizerConfig
    scheduler: TorchLRSchedulerConfig | None = None
    interval: tp.Literal["step", "epoch"] = "step"

    def build(
        self,
        params: tp.Iterable[torch.Tensor],
        **scheduler_build_kwargs: tp.Any,
    ) -> dict[str, tp.Any]:
        out = {"optimizer": self.optimizer.build(params)}
        if self.scheduler is not None:
            scheduler = self.scheduler.build(out["optimizer"], **scheduler_build_kwargs)
            out["lr_scheduler"] = {"scheduler": scheduler, "interval": self.interval}

        return out
