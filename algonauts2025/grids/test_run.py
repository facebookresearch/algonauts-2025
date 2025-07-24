# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Quick test run on reduced data and number of epochs for CI."""

import logging
import os

from exca import ConfDict

from ..main import Experiment  # type: ignore
from .defaults import default_config

logging.getLogger("exca").setLevel(logging.DEBUG)


update = {
    "save_checkpoints": False,
    "n_epochs": 6,
    "infra.cluster": None,
    "infra.gpus_per_node": 1,
    "infra.mode": "force",
    "data.num_workers": 0,
    "data.study.query": "subject_timeline_index<10",
    "data.study.cache_all_timelines": False,
}


def test_run(config: dict) -> None:
    task = Experiment(**config)
    task.infra.clear_job()
    trainer = task.run()


if __name__ == "__main__":
    updated_config = ConfDict(default_config)
    updated_config.update(update)
    folder = os.path.join(updated_config["infra"]["folder"], "test")
    updated_config["infra"]["folder"] = folder
    if os.path.exists(folder):
        import shutil

        shutil.rmtree(folder)
    test_run(updated_config)
