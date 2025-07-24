# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pydantic
import torch
import yaml
from exca import ConfDict, TaskInfra
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

import data_utils as ns
from data_utils.data import StudyLoader
from data_utils.events import EventTypesHelper
from data_utils.features.audio import AudioFeature
from data_utils.features.neuro import Fmri
from data_utils.features.text import LLM
from data_utils.features.video import VideoFeature
from data_utils.helpers import prepare_features
from data_utils.splitting import DeterministicSplitter
from modeling_utils.losses import LossConfig
from modeling_utils.metrics import MetricConfig
from modeling_utils.optimizers.base import LightningOptimizerConfig
from modeling_utils.utils import WandbLoggerConfig

from .callbacks import Benchmark, JitterWindows
from .model import FmriEncoder, FmriEncoderConfig
from .pl_module import BrainModule

dummy = FmriEncoder


# Configure logger
LOGGER = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", "%H:%M:%S")
_handler.setFormatter(_formatter)
if not LOGGER.handlers:
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


class Data(pydantic.BaseModel):
    """Handles configuration and creation of DataLoaders from dataset and features."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: StudyLoader
    neuro: Fmri
    text_feature: LLM | None = None
    audio_feature: AudioFeature | None = None
    video_feature: VideoFeature | None = None
    layers: list[float] | None = None
    layer_aggregation: tp.Literal["group_mean"] | None = None
    num_workers: int = 0

    def model_post_init(self, __context):
        super().model_post_init(__context)
        for modality in ["text", "audio", "video"]:
            feature = getattr(self, f"{modality}_feature")
            if self.layers is not None:
                setattr(feature, "layers", self.layers)
            if self.layer_aggregation is not None:
                setattr(feature, "layer_aggregation", self.layer_aggregation)

    def get_events(self) -> pd.DataFrame:
        events = self.study.build()

        if "split" not in events.columns:
            events["split"] = "train"

        train_sel = events.split == "train"
        splitter = DeterministicSplitter(ratios={"train": 1 - 0.1, "val": 0.1})
        values = events.loc[train_sel]["chunk"].unique()
        splits = [splitter(value) for value in values]
        if splits and "val" not in splits:
            splits[-1] = "val"  # need at least one val split
        events.loc[train_sel, "split"] = events.loc[train_sel]["chunk"].map(
            dict(zip(values, splits))
        )
        # check that all rows have split assigned
        unassigned_events = events[events.split.isna()]
        if len(unassigned_events) > 0:
            msg = f"The following events do not have a split assigned: {unassigned_events.type.unique()}"
            if any(
                [
                    name.capitalize() in unassigned_events.type.unique()
                    for name in ["Fmri", "text", "audio", "video"]
                ]
            ):
                raise ValueError(msg)
            else:
                LOGGER.warning(msg)

        cols = ["index", "subject", "timeline"]
        if "movie" in events.columns:
            cols.append("movie")
        if "chunk" in events.columns:
            cols.append("chunk")
        event_summary = events.reset_index().groupby(["split", "type"])[cols].nunique()
        LOGGER.info("Event summary: \n%s", event_summary)
        return events

    def get_loaders(
        self,
        events: pd.DataFrame | None = None,
        split_to_build: (
            tp.Literal["train", "val", "test", "all"]
            | list[tp.Literal["train", "val", "test", "all"]]
            | None
        ) = None,
    ) -> tuple[dict[str, DataLoader], int]:

        if events is None:
            events = self.get_events()
        features = {}
        for modality in ["text", "audio", "video"]:
            features[modality] = getattr(self, f"{modality}_feature")
        if "Fmri" in events.type.unique():
            features["fmri"] = self.neuro
        subject_id = ns.features.SubjectEncoder()
        features["subject_id"] = subject_id

        features_to_type = {
            "text": "Word",
            "audio": "Sound",
            "video": "Video",
            "fmri": "Fmri",
            "subject_id": "Event",
        }

        features_to_remove = set()
        for feature_name, feature in features.items():
            event_types = EventTypesHelper(features_to_type[feature_name]).names
            if not any(
                [event_type in events.type.unique() for event_type in event_types]
            ):
                features_to_remove.add(feature_name)
        for feature_name in features_to_remove:
            del features[feature_name]
            LOGGER.warning(
                "Removing feature %s as there are no corresponding events", feature_name
            )
        prepare_features(features, events)

        # Prepare dataloaders
        loaders = {}
        if isinstance(split_to_build, list):
            splits = split_to_build
        elif split_to_build is None:
            splits = ["train", "val", "test"]
        else:
            splits = [split_to_build]
        for split in splits:
            LOGGER.info("Building dataloader for split %s", split)
            if split == "all":
                sel = [True] * len(events)
                shuffle = False
            else:
                sel = events.split == split
                shuffle = {
                    "train": "train" in events.split.unique(),
                    "val": "val" in events.split.unique(),
                    "test": False,
                }[split]
            segments = ns.segments.list_segments(
                events[sel],
                stride=149,
                start=-4.47,
                duration=149,
                stride_drop_incomplete=False,
            )
            if len(sel) == 0:
                LOGGER.warning("No events found for split %s", split)
                continue
            dataset = ns.SegmentDataset(
                features=features,
                segments=segments,
            )
            dataloader = dataset.build_dataloader(
                shuffle=shuffle,
                num_workers=self.num_workers,
                batch_size=16,
            )
            loaders[split] = dataloader

        return loaders


class Experiment(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="forbid")

    data: Data
    # Reproducibility
    seed: int | None = 33
    # Model
    brain_model_config: FmriEncoderConfig
    # Loss
    loss: LossConfig
    # Optimization
    optim: LightningOptimizerConfig
    # Metrics
    metrics: list[MetricConfig]
    monitor: str = "val/pearson"
    # Weights & Biases
    wandb_config: WandbLoggerConfig | None = None
    # Hardware
    accelerator: str = "gpu"
    # Optim
    n_epochs: int = 10
    patience: int | None = None
    limit_train_batches: int | None = None
    # Others
    enable_progress_bar: bool = True
    log_every_n_steps: int | None = None
    fast_dev_run: bool = False
    save_checkpoints: bool = True
    # Eval
    checkpoint_path: str | None = None
    test_only: bool = False

    # Internal properties
    _trainer: pl.Trainer | None = None
    _brain_module: BrainModule | None = None
    _logger: WandbLogger | None = None

    # Others
    infra: TaskInfra = TaskInfra(version="1")

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.infra.folder is None:
            msg = "infra.folder needs to be specified to save the results."
            raise ValueError(msg)
        # Update Trainer parameters based on infra
        self.infra.tasks_per_node = self.infra.gpus_per_node
        self.infra.slurm_use_srun = True if self.infra.gpus_per_node > 1 else False
        if self.infra.gpus_per_node > 1:
            self.metrics = [
                m for m in self.metrics if m.name not in ["TopkAcc"]
            ]  # FIXME: TopkAcc is not supported in DDP

        if self.brain_model_config.n_subjects is None:
            self.brain_model_config.n_subjects = (
                self.data.study.study_summary().subject.nunique()
            )

    def _get_checkpoint_path(self) -> Path | None:
        if self.checkpoint_path:
            assert Path(
                self.checkpoint_path
            ).exists(), f"Checkpoint path {self.checkpoint_path} does not exist."
            checkpoint_path = Path(self.checkpoint_path)
        else:
            checkpoint_path = Path(self.infra.folder) / "last.ckpt"
            if not checkpoint_path.exists():
                checkpoint_path = None
        return checkpoint_path

    def _init_module(self, model: nn.Module) -> pl.LightningModule:
        # Setup torch-lightning module
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path is not None:
            LOGGER.info("Loading model from %s", checkpoint_path)
            init_fn = BrainModule.load_from_checkpoint
            init_kwargs = {"checkpoint_path": checkpoint_path, "strict": False}
        else:
            init_fn = BrainModule
            init_kwargs = {}

        metrics = {
            split + "/" + metric.log_name: metric.build()
            for metric in self.metrics
            for split in ["val", "test"]
        }
        metrics = nn.ModuleDict(metrics)
        pl_module = init_fn(
            model=model,
            loss=self.loss.build(),
            optim_config=self.optim,
            metrics=metrics,
            max_epochs=self.n_epochs,
            config=ConfDict(self.model_dump()),
            **init_kwargs,
        )

        return pl_module

    def _setup_trainer(self, train_loader: DataLoader) -> pl.Trainer:
        root_data_dir = Path(self.data.study.path) / "algonauts2025" / "download"
        # Initialize brain model
        batch = next(iter(train_loader))
        feature_dims = {}
        for modality in ["text", "audio", "video"]:
            if modality in batch.data:  # B, L, D, T
                if batch.data[modality].ndim == 4:
                    feature_dims[modality] = (
                        batch.data[modality].shape[1],
                        batch.data[modality].shape[2],
                    )
                elif batch.data[modality].ndim == 3:
                    feature_dims[modality] = (
                        1,
                        batch.data[modality].shape[1],
                    )
                else:
                    raise ValueError(
                        f"Unexpected number of dimensions for modality {modality}: {batch.data[modality].ndim}"
                    )
            else:
                feature_dims[modality] = None
        if "fmri" in batch.data:
            fmri = batch.data["fmri"]
            n_outputs = fmri.shape[1]
            for metric in self.metrics:
                if hasattr(metric, "kwargs") and "num_outputs" in metric.kwargs:
                    metric.kwargs["num_outputs"] = n_outputs
        else:
            n_outputs = 1000
        n_output_timesteps = 100
        brain_model = self.brain_model_config.build(
            feature_dims=feature_dims,
            n_outputs=n_outputs,
            n_output_timesteps=n_output_timesteps,
        )
        # print(brain_model)

        LOGGER.info("Feature dims: %s", feature_dims)
        input_data = brain_model.aggregate_features(batch)
        LOGGER.info("Input shapes: %s", input_data.shape)
        LOGGER.info("Target shapes: %s", n_outputs)
        _ = brain_model(batch)
        total_params = sum(p.numel() for p in brain_model.parameters())
        LOGGER.info(f"Total parameters: {total_params}")
        self._brain_module = self._init_module(brain_model)
        if self.monitor == "val/pearson":
            mode = "max"
        else:
            mode = "min"
        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
            JitterWindows(start_jitter_amount=10.0),
        ]
        if self.patience is not None:
            callbacks.append(
                EarlyStopping(monitor=self.monitor, mode=mode, patience=self.patience)
            )
        annealing_epochs = int(self.n_epochs * (1 - 0.6))
        callbacks.append(
            StochasticWeightAveraging(
                swa_epoch_start=0.6,
                annealing_epochs=annealing_epochs,
                swa_lrs=1e-5,
                annealing_strategy="cos",
            )
        )
        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=1,
                    dirpath=self.infra.folder,
                    filename="best",
                    monitor=self.monitor,
                    mode=mode,
                    save_on_train_epoch_end=True,
                )
            )
        callbacks.append(Benchmark(root_data_dir))

        trainer = pl.Trainer(
            strategy=(
                "auto"
                if self.infra.gpus_per_node == 1
                else "ddp_find_unused_parameters_true"
            ),
            devices=self.infra.gpus_per_node,
            accelerator=self.accelerator,
            max_epochs=self.n_epochs,
            limit_train_batches=self.limit_train_batches,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
            fast_dev_run=self.fast_dev_run,
            callbacks=callbacks,
            logger=self._logger,
            enable_checkpointing=self.save_checkpoints,
        )
        self._trainer = trainer
        return trainer

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self._trainer.fit(
            model=self._brain_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=self._get_checkpoint_path(),
        )

    def test(self, test_loader: DataLoader) -> None:
        if self.infra.gpus_per_node > 1:
            LOGGER.info(
                "Destroying DDP process group to enable testing on single device."
            )
            torch.distributed.destroy_process_group()
            if not self._trainer.is_global_zero:
                return
        if self.checkpoint_path:
            ckpt_path = self.checkpoint_path
        else:
            ckpt_path = None
            self._trainer.test(
                self._brain_module,
                dataloaders=test_loader,
                ckpt_path=ckpt_path,
            )

    def setup_run(self):

        if self.infra.cluster and self.infra.status() != "not submitted":
            for out_type in ["stdout", "stderr"]:
                old_path = Path(getattr(self.infra.job().paths, out_type))
                new_path = Path(self.infra.folder) / f"log.{out_type}"
                try:
                    if new_path.exists():
                        os.remove(new_path)
                    os.symlink(
                        old_path,
                        new_path,
                    )
                except:
                    pass
        config_path = Path(self.infra.folder) / "config.yaml"
        os.makedirs(self.infra.folder, exist_ok=True)
        with open(config_path, "w") as outfile:
            yaml.dump(
                self.model_dump(),
                outfile,
                indent=4,
                default_flow_style=False,
                sort_keys=False,
            )

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

        if self.seed is not None:
            pl.seed_everything(self.seed, workers=True)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        loaders = self.data.get_loaders(split_to_build="test" if self.test_only else None)
        self._setup_trainer(next(iter(loaders.values())))

        if not self.test_only:
            self.fit(loaders["train"], loaders["val"])
        self._trainer.validate(self._brain_module, loaders["val"])

        metrics = self._trainer.callback_metrics
        metrics_df = pd.DataFrame([{k: v.item() for k, v in metrics.items()}])
        metrics_df.to_csv(Path(self.infra.folder) / "metrics.csv", index=False)

        self.test(loaders["test"])
