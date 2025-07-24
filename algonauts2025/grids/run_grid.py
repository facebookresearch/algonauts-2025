# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from data_utils.infra import ConfDict
from modeling_utils.utils import run_grid

from ..main import Experiment  # type: ignore
from .defaults import PROJECT_NAME, SAVEDIR, default_config

GRID_NAME = "grid"

update = {
    "infra": {
        "cluster": "auto",
        "folder": SAVEDIR,
        "slurm_partition": "learnfair",
        "job_name": PROJECT_NAME,
    },
    "wandb_config.group": GRID_NAME,
    "save_checkpoints": False,
}

grid = {
    "data.audio_feature.name": ["Granite", "Wav2VecBert", "Wav2Vec"],
    "data.text_feature.model_name": [
        "meta-llama/Llama-3.2-3B",
        "Qwen/Qwen2.5-1.5B",
        "kyutai/helium-1-2b",
    ],
    "data.video_feature.model_name": [
        "facebook/vjepa2-vitg-fpc64-256",
        "facebook/vjepa2-vitl-fpc64-256",
        "facebook/vjepa2-vith-fpc64-256",
        "facebook/vjepa2-vitb-fpc64-256",
    ],
    "seed": list(range(5)),
}


if __name__ == "__main__":
    updated_config = ConfDict(default_config)
    updated_config.update(update)

    out = run_grid(
        Experiment,
        GRID_NAME,
        updated_config,
        grid,
        job_name_keys=["wandb_config.name", "infra.job_name"],
        combinatorial=True,
        overwrite=False,
        dry_run=False,
        infra_mode="force",
    )
