# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp

import pydantic

from ..utils import all_subclasses
from .base import BaseLossConfig

LossConfig = BaseLossConfig


def update_config_loss() -> None:
    global LossConfig

    from .base import BaseLossConfig

    LossConfig = tp.Annotated[
        tp.Union[tuple(all_subclasses(BaseLossConfig))],
        pydantic.Field(discriminator="name"),
    ]


update_config_loss()
