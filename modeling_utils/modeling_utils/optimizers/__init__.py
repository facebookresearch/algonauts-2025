# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp

import pydantic

from ..utils import all_subclasses
from .base import BaseOptimizerConfig, LightningOptimizerConfig, TorchLRSchedulerConfig

OptimizerConfig = BaseOptimizerConfig


def update_config_optimizer() -> None:
    global OptimizerConfig

    from .base import BaseOptimizerConfig

    OptimizerConfig = tp.Annotated[
        tp.Union[tuple(all_subclasses(BaseOptimizerConfig))],
        pydantic.Field(discriminator="name"),
    ]


update_config_optimizer()
