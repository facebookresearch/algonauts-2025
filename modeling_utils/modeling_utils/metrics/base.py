# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp
from inspect import isclass

import pandas as pd
import pydantic
import torch
import torch.nn as nn
import torchmetrics
from data_utils.infra import helpers
from modeling_utils.metrics import metrics
from modeling_utils.utils import all_subclasses, convert_to_pydantic
from torchmetrics import Metric

custom_metrics = [
    obj for obj in metrics.__dict__.values() if isclass(obj) and issubclass(obj, Metric)
]


class MultidimPearsonCorrCoef(torchmetrics.PearsonCorrCoef):

    def compute(self):
        return super().compute().mean()


TORCHMETRICS_NAMES = {
    metric_class.__name__: metric_class
    for metric_class in all_subclasses(Metric)
    if metric_class not in custom_metrics
}


class GroupedMetric(Metric):
    def __init__(self, metric_name: str, kwargs: dict[str, tp.Any] | None = None) -> None:
        super().__init__()
        if kwargs is None:
            kwargs = {}
        if metric_name in TORCHMETRICS_NAMES:
            self.base_metric_cls = TORCHMETRICS_NAMES[metric_name]
        else:
            assert hasattr(metrics, metric_name), f"Metric {metric_name} not found"
            self.base_metric_cls = getattr(metrics, metric_name)
        self.metric_kwargs = kwargs
        self.metrics = torch.nn.ModuleDict()

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        groups: tp.Optional[torch.Tensor] = None,
    ) -> None:

        if groups is None:
            groups = torch.zeros(preds.shape[0])
        else:
            groups = groups.flatten()
            assert (
                len(groups) == preds.shape[0]
            ), f"Groups must be the same shape as preds/target, got {groups.shape} and {preds.shape}"

        groups_df = pd.DataFrame({"label": groups.tolist()})
        for group_id, group in groups_df.groupby("label", sort=False):
            mask = group.index.to_numpy()
            group_preds = preds[mask]
            group_target = target[mask]

            group_key = str(group_id)
            if group_key not in self.metrics:
                self.metrics[group_key] = self.base_metric_cls(**self.metric_kwargs)
                self.metrics[group_key] = self.metrics[group_key].to(preds.device)

            self.metrics[group_key].update(group_preds, group_target)

    def compute(self) -> dict[str, float]:

        return {
            group_id: metric.compute().item() for group_id, metric in self.metrics.items()
        }

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def __repr__(self) -> str:
        return f"GroupedMetric({self.base_metric_cls.__name__})"


class BaseMetricConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="forbid")

    log_name: str
    name: str

    def build(self) -> nn.Module:
        raise NotImplementedError


for metric_class in custom_metrics + [GroupedMetric]:
    metric_class_name = metric_class.__name__
    config_cls = convert_to_pydantic(
        metric_class,
        metric_class_name,
        parent_class=BaseMetricConfig,
        exclude_from_build=["log_name"],
    )
    locals()[f"{metric_class_name}Config"] = config_cls


class TorchMetricConfig(BaseMetricConfig):
    name: tp.Literal[tuple(TORCHMETRICS_NAMES.keys())]

    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

        helpers.validate_kwargs(TORCHMETRICS_NAMES[self.name], self.kwargs)

    def build(self) -> nn.Module:
        return TORCHMETRICS_NAMES[self.name](**self.kwargs)
