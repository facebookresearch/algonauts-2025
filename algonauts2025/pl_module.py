# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp
from pathlib import Path

import lightning.pytorch as pl
from data_utils.dataloader import SegmentData
from einops import rearrange
from modeling_utils.optimizers import OptimizerConfig
from torch import nn
from torchmetrics import Metric


class BrainModule(pl.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optim_config: OptimizerConfig,
        metrics: dict[str, Metric],
        max_epochs: int = 100,
        checkpoint_path: Path | None = None,
        config: dict[str, tp.Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.config = config

        # Optimizer
        self.optim_config = optim_config
        self.max_epochs = max_epochs

        self.loss = loss
        self.metrics = metrics

    def forward(self, batch):
        return self.model(batch)

    def _run_step(self, batch: SegmentData, batch_idx, step_name):
        y_true = batch.data["fmri"]  # B, D, T
        y_pred = self.forward(batch)  # B, D, T
        if step_name == "val":
            y_true = y_true[:, :, 0:]
            y_pred = y_pred[:, :, 0:]
        subject_ids_flat = batch.data["subject_id"].repeat_interleave(y_pred.shape[2], 0)

        y_pred_flat = rearrange(y_pred, "b d t -> (b t) d")
        y_true_flat = rearrange(y_true, "b d t -> (b t) d")
        loss = self.loss(y_pred_flat, y_true_flat)
        log_kwargs = {
            "on_step": True if step_name == "train" else False,
            "on_epoch": True,
            "logger": True,
            "prog_bar": True,
            "batch_size": y_pred.shape[0],
        }

        self.log(
            f"{step_name}/loss",
            loss,
            **log_kwargs,
        )

        # Compute metrics
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                if "grouped" in metric.__class__.__name__.lower():
                    metric.update(y_pred_flat, y_true_flat, groups=subject_ids_flat)
                else:
                    if "retrieval" in metric_name:
                        metric.update(y_pred.mean(dim=-1), y_true.mean(dim=-1))
                    else:
                        metric.update(y_pred_flat, y_true_flat)
                    self.log(
                        metric_name,
                        metric,
                        **log_kwargs,
                    )
        return loss, y_pred.detach().cpu(), y_true.detach().cpu()

    def on_val_or_test_epoch_end(self, step_name: str) -> None:
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                if "grouped" in metric.__class__.__name__.lower():
                    metric_dict = {
                        metric_name + "/" + k: v for k, v in metric.compute().items()
                    }
                    self.log_dict(metric_dict)

    def on_validation_epoch_end(self) -> None:
        self.on_val_or_test_epoch_end("val")
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        self.on_val_or_test_epoch_end("test")
        return super().on_test_epoch_end()

    def training_step(self, batch: SegmentData, batch_idx):
        loss, _, _ = self._run_step(batch, batch_idx, step_name="train")
        return loss

    def validation_step(self, batch: SegmentData, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, step_name="val")
        return y_pred, y_true

    def test_step(self, batch: SegmentData, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, step_name="test")
        return y_pred, y_true

    def configure_optimizers(self):
        optim_config = self.optim_config.copy()
        unfrozen_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim_config.build(
            unfrozen_params, total_steps=self.trainer.estimated_stepping_batches
        )
        return optimizer
