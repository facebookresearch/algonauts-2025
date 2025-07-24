# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

PROJECT_NAME = "algonauts-2025"


SLURM_PARTITION = "partition"
DATADIR = "save_dir"
BASEDIR = os.path.expandvars("save_dir")

CACHEDIR = os.path.join(BASEDIR, "cache", PROJECT_NAME)
SAVEDIR = os.path.join(BASEDIR, "results", PROJECT_NAME)

for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

text_feature = {
    "name": "LLM",
}
video_feature = {
    "name": "VideoFeature",
    "model_name": "facebook/vjepa2-vitg-fpc64-256",
}
audio_feature = {
    "name": "Wav2VecBert",
}
neuro_feature = {
    "name": "Fmri",
}
for feature in [
    text_feature,
    video_feature,
    audio_feature,
    neuro_feature,
]:
    feature["infra"] = {
        "folder": CACHEDIR,
        "keep_in_ram": True,
        "mode": "cached",
        "version": "final",
    }

default_config = {
    "infra": {
        "cluster": "slurm",  # Run example locally
        "folder": SAVEDIR,
    },
    "data": {
        "num_workers": N_CPUS,
        "study": {
            "path": Path(DATADIR) / "algonauts2025",
            "query": None,
            "infra": {
                "folder": CACHEDIR,
            },
            "enhancers": {
                "addtext": {"name": "AddText"},
                "addsentence": {
                    "name": "AddSentenceToWords",
                    "max_unmatched_ratio": 0.05,
                },
                "addcontext": {
                    "name": "AddContextToWords",
                    "sentence_only": False,
                    "max_context_len": 1024,
                },
                "removemissing": {"name": "RemoveMissing"},
                "extractaudio": {"name": "ExtractAudioFromVideo"},
                "chunkevents": {
                    "name": "ChunkEvents",
                    "event_type_to_chunk": "Sound",
                    "max_duration": 60,
                    "min_duration": 30,
                },
            },
        },
        "neuro": neuro_feature,
        "text_feature": text_feature,
        "video_feature": video_feature,
        "audio_feature": audio_feature,
        "layers": [0.5, 0.75, 1.0],
        "layer_aggregation": "group_mean",
    },
    "wandb_config": {
        "log_model": False,
        "project": "algonauts-2025",
        "group": "default",
        "host": None,
    },
    "brain_model_config": {
        "name": "FmriEncoder",
        "modality_dropout": 0.3,
        "feature_aggregation": "cat",
        "layer_aggregation": "cat",
        "subject_embedding": False,
    },
    "metrics": [
        {
            "log_name": "pearson",
            "name": "MultidimPearsonCorrCoef",
            "kwargs": {"num_outputs": 1000},
        },
        {
            "log_name": "subj_pearson",
            "name": "GroupedMetric",
            "metric_name": "MultidimPearsonCorrCoef",
            "kwargs": {"num_outputs": 1000},
        },
        {
            "log_name": "retrieval_top1",
            "name": "TopkAcc",
            "topk": 1,
        },
    ],
    "loss": {"name": "MSELoss"},
    "optim": {
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "kwargs": {
                "weight_decay": 0.0,
            },
        },
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {
                "max_lr": 1e-4,
                "pct_start": 0.1,
            },
        },
    },
    "n_epochs": 15,
    "limit_train_batches": None,
    "patience": None,
    "enable_progress_bar": True,
    "log_every_n_steps": 5,
    "fast_dev_run": False,
    "seed": 33,
}


if __name__ == "__main__":
    from ..main import Experiment

    exp = Experiment(
        **default_config,
    )

    exp.infra.clear_job()
    out = exp.run()
    print(out)
