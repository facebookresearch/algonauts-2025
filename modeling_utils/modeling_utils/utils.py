# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import random
import shutil
import typing as tp
from itertools import product
from pathlib import Path

import pydantic
import wandb
from data_utils.infra import ConfDict, TaskInfra
from pydantic import BaseModel, Field, create_model


def convert_to_pydantic(
    class_to_convert: type,
    name: str,
    parent_class: tp.Any = None,
    exclude_from_build: list[str] | None = None,
) -> BaseModel:

    init = class_to_convert.__init__

    sig = inspect.signature(init)
    empty = inspect.Parameter.empty

    fields = {
        k: (
            v.annotation if v.annotation != empty else tp.Any,
            v.default if v.default != empty else ...,
        )
        for k, v in sig.parameters.items()
        if k != "self" and not k.startswith("_")
    }

    assert "name" not in sig.parameters.items()

    Builder = create_model(
        name,
        name=(tp.Literal[name], Field(default=name)),
        __base__=parent_class,
        **fields,
    )
    Builder._cls = class_to_convert

    if exclude_from_build is None:
        exclude_from_build = []

    def build_method(instance: BaseModel):
        params = dict(
            (field, getattr(instance, field))
            for field in type(instance).model_fields
            if (field != "name" and field not in exclude_from_build)
        )
        return instance._cls(**params)

    setattr(Builder, "build", build_method)

    return Builder


def all_subclasses(cls):

    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def run_grid(
    exp_cls,
    exp_name: str,
    base_config: dict[str, tp.Any],
    grid: dict[str, list],
    n_randomly_sampled: int | None = None,
    job_name_keys: list[str] | None = None,
    combinatorial: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    infra_mode: str = "retry",
) -> list[ConfDict]:

    job_array_kwargs = {}
    if dry_run:
        from importlib.metadata import version

        from pkg_resources import parse_version

        if parse_version(version("exca")) < parse_version("0.4.5"):
            raise ImportError("`dry_run` requires `exca>=0.4.5` to be installed.")
        job_array_kwargs["allow_empty"] = True

    base_config = base_config
    base_config["infra.job_name"] = exp_name
    base_folder = Path(base_config["infra"]["folder"])
    assert all([isinstance(v, list) for v in grid.values()]), "Grid values must be lists."

    task = exp_cls(
        **base_config,
    )

    if combinatorial:
        grid_product = list(dict(zip(grid.keys(), v)) for v in product(*grid.values()))
    else:
        grid_product = [
            {param: value} for param, values in grid.items() for value in values
        ]

    if n_randomly_sampled is not None:
        assert n_randomly_sampled <= len(
            grid_product
        ), "n_randomly_sampled must be less than the number of grid products"
        grid_product = random.sample(grid_product, n_randomly_sampled)

    print(f"Launching {len(grid_product)} tasks")

    out_configs = []
    tmp = task.infra.clone_obj(**{"infra.mode": infra_mode})
    with tmp.infra.job_array(**job_array_kwargs) as tasks:
        for params in grid_product:
            job_name = ConfDict(params).to_uid()

            config = ConfDict(base_config)
            config.update(params)

            folder = base_folder / exp_name / job_name
            if folder.exists():

                print(f"{folder} already exists.")
                if overwrite and not dry_run:

                    print(f"Folder {folder} already exists. Overwrite? (y/n)")
                    answer = input()
                    if answer.lower() != "y":
                        print("Skipping.")
                        continue
                    print(f"Deleting {folder}.")
                    shutil.rmtree(folder)
                    folder.mkdir()

            config["infra.folder"] = str(folder)
            if job_name_keys is not None:
                for key in job_name_keys:
                    config.update({key: str(job_name)})

            if not dry_run:
                task_ = exp_cls(**config)
                tasks.append(task_)

            out_configs.append(config)

    print("Done.")

    return out_configs


class WandbLoggerConfig(pydantic.BaseModel):

    model_config = pydantic.ConfigDict(extra="forbid")

    offline: bool = False
    host: str | None = None
    name: str | None = None
    group: str | None = None
    entity: str | None = None

    version: str | None = None
    dir: Path | None = None
    id: str | None = None
    anonymous: bool | None = None
    project: str | None = None
    log_model: str | bool = False
    experiment: tp.Any | None = None
    prefix: str = ""

    def build(
        self,
        save_dir: str | Path,
        xp_config: dict | pydantic.BaseModel | None = None,
        id: str | None = None,
    ) -> tp.Any:
        if self.offline:
            login_kwargs = {"key": "X" * 40}
        else:
            login_kwargs = {"host": self.host}

        wandb.login(**login_kwargs)

        from lightning.pytorch.loggers import WandbLogger

        if isinstance(xp_config, pydantic.BaseModel):
            xp_config = xp_config.model_dump()
        config = self.model_dump()
        if id is not None:
            config["id"] = id
        del config["host"]
        logger = WandbLogger(**config, save_dir=save_dir, config=xp_config)
        try:
            logger.experiment.config["_dummy"] = None

        except TypeError:
            pass

        return logger
