# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import typing as tp

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import tqdm
from exca import ConfDict, TaskInfra

from .main import Experiment
from .pl_module import BrainModule


class EnsembleBrainModule(BrainModule):
    def __init__(self, checkpoint_paths: list[str], ensembling_strategy: str = "mean", **kwargs):
        """
        Args:
            checkpoint_paths (list): List of paths to LightningModule checkpoints.
        """
        super().__init__(**kwargs)
        del self.model
        self.models = torch.nn.ModuleList([])
        for path in tqdm.tqdm(checkpoint_paths, desc="Loading models"):
            model = BrainModule.load_from_checkpoint(path, **kwargs).model
            for param in model.parameters():
                param.requires_grad = False
            self.models.append(copy.deepcopy(model))
        self.ensembling_strategy = ensembling_strategy

    def forward(self, batch):
        outputs = [model(batch) for model in self.models]
        stacked = torch.stack(outputs, dim=0)
        if self.ensembling_strategy == "mean":
            return torch.mean(stacked, dim=0)
        else:
            assert self.weights is not None, "Weights must be provided"
            out = torch.einsum("m b d t, m d -> b d t", stacked, self.weights)
            return out


class EnsembleExperiment(Experiment):

    infra: TaskInfra = TaskInfra(version="1")
    ensemble_checkpoint_paths: list[str] | None = None
    ensembling_strategy: tp.Literal["mean", "weighted", "learnt"] = "mean"


    def _init_module(self, model: nn.Module) -> pl.LightningModule:
        metrics = {split + "/" + metric.log_name: metric.build() for metric in self.metrics for split in ["val", "test"]}
        metrics = nn.ModuleDict(metrics)
        pl_module = EnsembleBrainModule(
            model=model,
            loss=self.loss.build(),
            optim_config=self.optim,
            metrics=metrics,
            max_epochs=self.n_epochs,
            config=ConfDict(self.model_dump()),
            checkpoint_paths=self.ensemble_checkpoint_paths,
            ensembling_strategy=self.ensembling_strategy,
        )

        return pl_module

    @infra.apply
    def run(self):
        self.setup_run()
        self._logger = (
            self.wandb_config.build(
                save_dir=self.infra.folder,
                xp_config=self.model_dump(),
                id=f"{self.wandb_config.group}-{self.infra.uid().split('-')[-1]}",
            )
            if self.wandb_config
            else None
        )
        pl.seed_everything(self.seed, workers=True)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        loaders = self.data.get_loaders(split_to_build=["val", "test"])
        self._setup_trainer(next(iter(loaders.values())))

        n_outputs = self._brain_module.models[0].n_outputs
        if self.ensembling_strategy == "weighted":
            scores = torch.zeros(len(self._brain_module.models), n_outputs)
            for i, model in enumerate(self._brain_module.models):
                metrics =self._trainer.validate(
                    model=model,
                    dataloaders=loaders["val"],
                )
                score = metrics["val/pearson"]
                scores[i] = score
                print(f"Model {i} score: {score}")
            weights = torch.zeros_like(scores)
            for i in range(n_outputs):
                weights[torch.argmax(scores[:, i])] = 1
            self._brain_module.weights = weights

        elif self.ensembling_strategy == "learnt":
            weights = torch.nn.Parameter(torch.randn(len(self._brain_module.models), n_outputs))
            # init to uniform
            weights.data = torch.ones_like(weights.data) / len(self._brain_module.models)
            self._brain_module.weights = weights
            post_trainer = pl.Trainer(max_epochs=self.n_epochs)
            print("Fitting post-trainer")
            post_trainer.fit(self._brain_module, train_dataloaders=loaders["val"], val_dataloaders=None)

        self._trainer.validate(self._brain_module, dataloaders=loaders["val"])
        self._trainer.test(self._brain_module, dataloaders=loaders["test"])
        return
