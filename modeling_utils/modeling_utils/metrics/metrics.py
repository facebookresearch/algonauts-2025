# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp
from collections import defaultdict

import numpy as np
import torch
import torchmetrics


class OnlinePearsonCorr(torchmetrics.regression.PearsonCorrCoef):

    def __init__(
        self,
        dim: int,
        reduction: tp.Literal["mean", "sum", "none"] | None = "mean",
    ):

        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self._initialized = False

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:

        if self.dim == 1:
            preds = preds.T
            target = target.T

        if not self._initialized:

            self.num_outputs = preds.shape[1]
            state_names = ["mean_x", "mean_y", "var_x", "var_y", "corr_xy", "n_total"]
            for state_name in state_names:
                self.add_state(
                    state_name,
                    default=torch.zeros(self.num_outputs).to(self.device),
                    dist_reduce_fx=None,
                )
            self._initialized = True

        super().update(preds, target)

    def compute(self):

        corrcoef = super().compute()

        if self.reduction == "mean":
            return torch.mean(corrcoef)
        elif self.reduction == "sum":
            return torch.sum(corrcoef)
        else:

            return corrcoef

    def reset(self) -> None:
        self._initialized = False
        super().reset()


class Rank(torchmetrics.Metric):

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(
        self,
        reduction: tp.Literal["mean", "median", "std"] = "median",
        relative: bool = False,
    ):
        super().__init__()

        self.reduction = reduction
        self.relative = relative
        self.add_state(
            "ranks",
            default=torch.Tensor([]),
            dist_reduce_fx="cat",
        )
        self.rank_count: torch.Tensor

    @classmethod
    def _compute_sim(cls, x, y, norm_kind="y", eps=1e-15):
        if norm_kind is None:
            eq, inv_norms = "b", torch.ones(x.shape[0])
        elif norm_kind == "x":
            eq, inv_norms = "b", 1 / (eps + x.norm(dim=(1), p=2))
        elif norm_kind == "y":
            eq, inv_norms = "o", 1 / (eps + y.norm(dim=(1), p=2))
        elif norm_kind == "xy":
            eq = "bo"
            inv_norms = 1 / (
                eps + torch.outer(x.norm(dim=(1), p=2), y.norm(dim=(1), p=2))
            )
        else:
            raise ValueError(f"norm must be None, x, y or xy, got {norm_kind}.")

        return torch.einsum(f"bc,oc,{eq}->bo", x, y, inv_norms)

    def _compute_ranks(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_labels: None | list[str] = None,
        y_labels: None | list[str] = None,
    ) -> torch.Tensor:
        scores = self._compute_sim(x, y)

        if x_labels is not None and y_labels is not None:

            true_inds = torch.tensor(
                [y_labels.index(x) for x in x_labels],
                dtype=torch.long,
                device=scores.device,
            )[:, None]
            true_scores = torch.take_along_dim(scores, true_inds, dim=1)
        else:

            assert x_labels is None and y_labels is None
            assert x.shape[0] == y.shape[0]
            true_scores = torch.diag(scores)[:, None]

        ranks_gt = (scores > true_scores).nansum(axis=1)
        ranks_ge = (scores >= true_scores).nansum(axis=1) - 1
        ranks = (ranks_gt + ranks_ge) / 2
        ranks[ranks < 0] = len(scores) // 2

        if self.relative:
            ranks /= len(y)

        return ranks

    @torch.inference_mode()
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_labels: None | list[str] = None,
        y_labels: None | list[str] = None,
    ) -> None:

        ranks = self._compute_ranks(x, y, x_labels, y_labels)
        self.ranks = torch.cat([self.ranks, ranks])

    def compute(self) -> torch.Tensor:
        agg_func: tp.Callable
        if self.reduction == "mean":
            agg_func = torch.mean
        elif self.reduction == "median":
            agg_func = torch.median
        elif self.reduction == "std":
            agg_func = torch.std
        else:
            raise ValueError(
                f'Unknown aggregation {self.reduction} for computing metric. Available aggregations are: "mean", "median" or "std".'
            )
        return agg_func(self.ranks)

    def _compute_macro_average(
        self, ranks: torch.Tensor, labels: list[str]
    ) -> tp.Dict[str, float]:

        assert len(ranks) == len(labels)
        groups = defaultdict(list)
        agg_func = np.mean if self.reduction == "mean" else np.median
        for i, label in enumerate(labels):
            groups[label].append(ranks[i])
        return {label: agg_func(ranks) for label, ranks in groups.items()}

    @classmethod
    def _compute_topk_scores(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        y_labels: list[str],
        k: int = 5,
    ) -> tp.Tuple[list[list[str]], list[list[float]]]:

        scores = cls._compute_sim(x, y)
        topk_inds = torch.argsort(scores, dim=1, descending=True)[:, :k]
        topk_labels = [[y_labels[ind] for ind in inds] for inds in topk_inds]
        scores = [
            [scores[i, ind].item() for ind in inds] for i, inds in enumerate(topk_inds)
        ]
        return topk_labels, scores


class TopkAcc(Rank):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(self, topk: int = 5):
        super().__init__(relative=False)
        self.topk = topk

    def _compute_macro_average(
        self, ranks: torch.Tensor, labels: list[str]
    ) -> tp.Dict[str, float]:

        groups = defaultdict(list)
        for i, label in enumerate(labels):
            groups[label].append(ranks[i])
        return {
            label: float(np.mean([r < self.topk for r in ranks]))
            for label, ranks in groups.items()
        }

    def compute(self) -> torch.Tensor:
        ranks = self.ranks
        return (ranks < self.topk).float().mean()
