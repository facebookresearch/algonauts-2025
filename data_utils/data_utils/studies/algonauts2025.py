# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp
from itertools import product
from pathlib import Path

import h5py
import nibabel
import numpy as np
import pandas as pd

from data_utils.data import BaseData
from data_utils.download import Datalad


class Algonauts2025(BaseData):
    task: tp.Literal["friends", "movie10"]
    movie: str
    chunk: str
    run: int = 0

    device: tp.ClassVar[str] = "Fmri"

    @classmethod
    def _download(cls, path: Path) -> None:
        Datalad(
            dset_dir=path,
        ).download()

    @classmethod
    def _iter_timelines(
        cls,
        path: str | Path,
    ):
        for subject in ["sub-01", "sub-02", "sub-03", "sub-05"]:
            for task in ["friends", "movie10"]:
                if task == "friends":
                    season_episode_chunk = range(1, 8), range(1, 26), "abcd"
                    for season, episode, chunk in product(*season_episode_chunk):
                        timeline = cls(
                            path=str(path),
                            subject=subject,
                            task=task,
                            movie=str(season),
                            chunk=f"e{episode:02d}{chunk}",
                        )
                        stim_path = timeline._get_transcript_filepath()
                        if (
                            (season == 5 and episode == 20 and chunk == "a")
                            or (season == 4 and episode == 1 and chunk == "a")
                            or (season == 6 and episode == 3 and chunk == "a")
                            or (season == 4 and episode == 13 and chunk == "b")
                            or (season == 4 and episode == 1 and chunk == "b")
                        ):
                            continue
                        if stim_path.exists():
                            yield timeline
                elif task == "movie10":
                    movie_chunk_run = (
                        ["bourne", "wolf", "life", "figures"],
                        range(1, 18),
                        [1, 2],
                    )
                    for movie, chunk, run in product(*movie_chunk_run):

                        if movie in ["bourne", "wolf"] and run == 2:
                            continue
                        timeline = cls(
                            path=str(path),
                            subject=subject,
                            task=task,
                            movie=movie,
                            chunk=str(chunk),
                            run=run,
                        )
                        stim_path = timeline._get_transcript_filepath()
                        if stim_path.exists():
                            yield timeline

    def _get_transcript_filepath(self):
        folder = (
            Path(self.path)
            / "download"
            / "algonauts_2025.competitors"
            / "stimuli"
            / "transcripts"
            / f"{self.task}"
        )
        if self.task == "friends":
            stim_path = (
                folder
                / f"s{self.movie}"
                / f"friends_s{int(self.movie):02d}{self.chunk}.tsv"
            )
        elif self.task == "movie10":
            stim_path = (
                folder
                / f"{self.movie}"
                / f"movie10_{self.movie}{int(self.chunk):02d}.tsv"
            )
        return stim_path

    def _get_movie_filepath(self) -> Path:
        folder = (
            Path(self.path)
            / "download"
            / "algonauts_2025.competitors"
            / "stimuli"
            / "movies"
            / f"{self.task}"
        )
        if self.task == "friends":
            stim_path = (
                folder
                / f"s{self.movie}"
                / f"friends_s{int(self.movie):02d}{self.chunk}.mkv"
            )
        elif self.task == "movie10":
            stim_path = (
                folder / f"{self.movie}" / f"{self.movie}{int(self.chunk):02d}.mkv"
            )
        return stim_path

    def _get_fmri_filepath(self) -> Path:
        folder = Path(self.path) / "download" / "algonauts_2025.competitors" / "fmri"
        subj_dir = folder / self.subject / "func"
        file_stem = f"{self.subject}_task-{self.task}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net"
        if self.task == "friends":
            fmri_file = subj_dir / f"{file_stem}_desc-s123456_bold.h5"
        else:
            fmri_file = subj_dir / f"{file_stem}_bold.h5"
        return fmri_file

    def _load_fmri(self, timeline: str) -> nibabel.Nifti2Image:
        fmri_file = self._get_fmri_filepath()
        fmri = h5py.File(fmri_file, "r")
        if self.task == "friends":
            key = f"{int(self.movie):02d}{self.chunk}"
        else:
            key = f"{self.movie}{int(self.chunk):02d}"
            if self.movie in ["life", "figures"]:
                key += f"_run-{self.run}"
        selected_key = [key_ for key_ in fmri.keys() if key in key_]
        if len(selected_key) != 1:
            print(key, selected_key, list(fmri.keys()))
            raise ValueError(f"Multiple or no keys found, {key}, {list(fmri.keys())}")
        fmri = fmri[selected_key[0]]
        data = fmri[:].astype(np.float32)
        obj = nibabel.Nifti2Image(data.T, affine=np.eye(4))
        return obj

    def _get_split(self) -> str:

        if self.task == "friends":
            if int(self.movie) in range(1, 7):
                return "train"
            elif int(self.movie) == 7:
                return "test"
        else:
            return "train"

    def _load_events(self) -> pd.DataFrame:

        all_events = []
        if not (self.task == "friends" and self.movie == "7"):
            uri = f"method:_load_fmri?timeline={self.timeline}"
            fmri = self._load_fmri(timeline="")
            fmri_duration = fmri.shape[-1] * 1.49
            fmri_event = dict(
                type="Fmri",
                filepath=uri,
                start=0,
                frequency=1 / 1.49,
                duration=fmri_duration,
            )
            all_events.append(fmri_event)

        movie_filepath = self._get_movie_filepath()
        movie_event = dict(type="Video", filepath=movie_filepath, start=0)
        all_events.append(movie_event)

        transcript_path = self._get_transcript_filepath()
        transcript_df = pd.read_csv(transcript_path, sep="\t")
        word_events = []
        for _, row in transcript_df.iterrows():
            words = eval(row["words_per_tr"])
            starts = eval(row["onsets_per_tr"])
            durations = eval(row["durations_per_tr"])
            for word, start, duration in zip(words, starts, durations):
                event = dict(
                    type="Word",
                    text=word,
                    start=start,
                    duration=duration,
                    stop=start + duration,
                    language="english",
                )
                word_events.append(event)
        if word_events:
            word_df = pd.DataFrame(word_events)
            text = " ".join(word_df["text"].tolist())
            text_event = dict(
                type="Text",
                text=text,
                start=word_df["start"].min(),
                duration=word_df["stop"].max() - word_df["start"].min(),
                stop=word_df["stop"].max(),
                language="english",
            )
            all_events.append(text_event)
        all_events.extend(word_events)

        events_df = pd.DataFrame(all_events)
        events_df["split"] = self._get_split()
        events_df["movie"] = "movie:" + str(self.movie)
        events_df["chunk"] = "chunk:" + str(self.chunk)
        return events_df
