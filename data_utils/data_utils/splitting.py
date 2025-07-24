# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import hashlib
import random
import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import events as event_module


@dataclass
class DeterministicSplitter:
    ratios: tp.Dict[str, float]
    seed: float = 0.0

    def __post_init__(self) -> None:

        assert all(ratio > 0 for ratio in self.ratios.values())
        assert np.allclose(
            sum(self.ratios.values()), 1.0
        ), f"the sum of ratios must be equal to 1. got {self.ratios}"

    def __call__(self, uid: str) -> str:
        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        rng = random.Random(hashed + self.seed)
        score = rng.random()

        cdf = np.cumsum(list(self.ratios.values()))
        names = list(self.ratios.keys())

        for idx, cdf_val in enumerate(cdf):
            if score < cdf_val:
                return names[idx]
        raise ValueError


def chunk_events(
    events: pd.DataFrame,
    event_type_to_chunk: tp.Literal["Sound", "Video"],
    event_type_to_use: str | None = None,
    min_duration: float | None = None,
    max_duration: float = np.inf,
):

    added_events: tp.List[tp.Dict] = []
    dropped_rows: tp.List[int] = []
    ns_event_type_to_chunk = getattr(event_module, event_type_to_chunk)
    assert hasattr(
        ns_event_type_to_chunk, "_split"
    ), f"Event type {event_type_to_chunk} is not splittable"
    if event_type_to_use is not None:
        assert "split" in events.columns, "Events must have a split column"

    for _, df in events.groupby("timeline"):
        df.sort_values("start", inplace=True)
        if event_type_to_use is None:

            timepoints: list[float] = np.arange(
                df.start.min(), df.stop.max(), max_duration
            ).tolist()
            if min_duration is not None:
                if df.stop.max() - timepoints[-1] < min_duration:
                    timepoints = timepoints[:-1]
        else:

            timepoints = []
            events_to_use = df.loc[events.type == event_type_to_use].copy()
            previous = events_to_use.copy().shift(1)
            split_change = events_to_use.split.astype(str) != previous.split.astype(str)
            events_to_use["section"] = np.cumsum(split_change.values)

            for _, section in events_to_use.groupby("section"):
                start, end = (
                    section.iloc[0].start,
                    section.iloc[-1].start + section.iloc[-1].duration,
                )
                timepoints.extend(np.arange(start, end, max_duration))

        events_to_chunk = df.loc[events.type == event_type_to_chunk]
        dropped_rows.extend(events_to_chunk.index)
        for row in events_to_chunk.itertuples():
            event_to_chunk = ns_event_type_to_chunk.from_dict(row)
            new_events = event_to_chunk._split(
                [t - event_to_chunk.start for t in timepoints], min_duration
            )

            for new_event in new_events:
                new_event_dict = new_event.to_dict()

                for k, v in row._asdict().items():

                    if k not in new_event_dict:
                        new_event_dict[k] = v
                added_events.append(new_event_dict)

    out_events = events.copy()
    out_events.drop(dropped_rows, inplace=True)
    out_events = pd.concat([out_events, pd.DataFrame(added_events)])
    out_events.reset_index(drop=True, inplace=True)
    return out_events



