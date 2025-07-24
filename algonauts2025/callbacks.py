# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
from data_utils.segments import SegmentCreator, _prepare_strided_windows
from lightning.pytorch import Callback

SUBJECT_MAPPINGS = {0: 1, 1: 2, 2: 3, 3: 5}


class JitterWindows(Callback):
    def __init__(
        self,
        start_jitter_amount: float = 0.0,
        duration_jitter_amount: float = 0.0,
    ):
        self.start_jitter_amount = start_jitter_amount
        self.duration_jitter_amount = duration_jitter_amount

    def on_train_epoch_start(self, trainer, pl_module):
        start_jitter = (np.random.rand() * 2 - 1) * self.start_jitter_amount
        duration_jitter = (np.random.rand() * 2 - 1) * self.duration_jitter_amount
        segments = trainer.train_dataloader.dataset.segments
        new_segments = []
        creators = SegmentCreator.from_obj(segments)
        for creator in creators.values():
            starts, durations = _prepare_strided_windows(
                creator.starts.min() - 4.47 + start_jitter,
                creator.stops.max() - 4.47 + start_jitter,
                149,
                149,
                drop_incomplete=False,
            )
            for start_, duration_ in zip(starts, durations):
                seg = creator.select(start=start_, duration=duration_)
                seg._trigger = start_
                new_segments.append(seg)
        assert len(segments) == len(new_segments)
        trainer.train_dataloader.dataset.segments = new_segments


class Benchmark(Callback):

    def __init__(self, root_data_dir):
        self.root_data_dir = Path(root_data_dir)
        self.submission_dict = {}

    def on_test_epoch_start(self, trainer, pl_module):
        self.submission_dict = {}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        y_pred, _ = outputs  # we do not have the ground truth
        overlap_trs = 0.0
        for i, segment in enumerate(batch.segments):
            subject = segment.events.subject.unique()[0]
            chunk = segment.events.chunk.unique()[0]
            pred = y_pred[i].cpu().numpy()  # 1000, T
            pred = pred.T
            subject = subject.split("/")[1]
            chunk = "s07" + chunk.split(":")[1]
            if not subject in self.submission_dict:
                self.submission_dict[subject] = {}
            if not chunk in self.submission_dict[subject]:
                self.submission_dict[subject][chunk] = []
            else:
                pred = pred[overlap_trs:]  # remove the overlap except on the first chunk
            self.submission_dict[subject][chunk].append(pred)

    def on_test_epoch_end(self, trainer, pl_module):

        for subject in self.submission_dict.keys():
            samples_file = (
                self.root_data_dir
                / f"algonauts_2025.competitors/fmri/{subject}/target_sample_number/{subject}_friends-s7_fmri_samples.npy"
            )
            target_sample_number = np.load(samples_file, allow_pickle=True).item()
            for chunk, sample_number in target_sample_number.items():
                result = np.concatenate(self.submission_dict[subject][chunk], axis=0)
                if len(result) < sample_number:
                    raise ValueError(
                        f"Warning: {len(result)} predictions for {chunk} but expected at least {sample_number}"
                    )
                self.submission_dict[subject][chunk] = result[:sample_number]

        # save
        submission_name = "submission.npy"
        submission_path = Path(trainer.logger.save_dir) / submission_name
        np.save(submission_path, self.submission_dict)
        import zipfile

        try:
            with zipfile.ZipFile(submission_path.with_suffix(".zip"), "w") as zipf:
                zipf.write(submission_path, arcname=submission_name)
            print(f"Saved submission to {submission_path.with_suffix('.zip')}")
        except:
            print(f"Failed to save submission to {submission_path.with_suffix('.zip')}")
