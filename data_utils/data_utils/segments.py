# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import collections
import dataclasses
import logging
import typing as tp
import warnings

import numpy as np
import pandas as pd
import tqdm

from .events import Event
from .utils import warn_once

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Segment:

    start: float
    duration: float
    _index: np.ndarray

    ns_events: tp.List[Event] = dataclasses.field(default_factory=list)
    _trigger: float | tp.Dict[str, tp.Any] | None = None

    @property
    def events(self) -> pd.DataFrame:

        if not self.ns_events:
            raise RuntimeError(f"ns_events was not populated in {self}")
        if len(self.ns_events) != len(self._index):
            msg = f"Cannot recreate events dataframe as some rows were not actual Event\n(on segment={self})"
            raise RuntimeError(msg)
        return pd.DataFrame(index=self._index, data=[e.to_dict() for e in self.ns_events])

    def subsegment(self, start: float, duration: float) -> "Segment":

        assert (
            start >= 0
        ), "Start is relative to the segment start and must be non-negative"
        new_start = self.start + start
        new_duration = duration
        new_index, new_ns_events = [], []
        for i, e in enumerate(self.ns_events):
            if e.start <= new_start + new_duration and e.start + e.duration >= new_start:
                new_index.append(self._index[i])
                new_ns_events.append(e)
        new_index = np.array(new_index)
        return Segment(
            start=new_start,
            duration=new_duration,
            _index=new_index,
            ns_events=new_ns_events,
            _trigger=self._trigger,
        )

    @property
    def event_list(self) -> list[Event]:
        raise RuntimeError(
            "segment.event_list is deprecated in favor of segment.ns_events"
        )

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def _to_feature(self) -> dict[str, tp.Any]:

        return {
            "start": self.start,
            "duration": self.duration,
            "events": self.ns_events,
            "trigger": self._trigger,
        }


def _validate_event(event: pd.Series) -> dict[str, tp.Any]:

    event_type = event["type"]
    lower = {x.lower() for x in Event._CLASSES}
    if event_type in Event._CLASSES:
        event_class = Event._CLASSES[event_type]
        event_obj = event_class.from_dict(event).to_dict()

        event_dict = {**event, **event_obj}
    elif event_type in lower:
        raise ValueError(f"Legacy uncapitalized event {event}")
    else:
        warn_once(
            f'Unexpected type "{event["type"]}". Support for new event '
            "types can be added by creating new `Event` classes in "
            "`data_utils.events`."
        )
        event_dict = {**event}

    return event_dict


def validate_events(events: pd.DataFrame) -> pd.DataFrame:

    if events.empty:
        return events.copy()
    msg = 'events DataFrame must have a "type" column with strings'
    if "type" not in events.keys():
        raise ValueError(msg)
    types = events["type"].unique()
    if not all(isinstance(typ, str) for typ in types):
        raise ValueError(msg)

    df = pd.DataFrame(
        events.apply(_validate_event, axis=1).tolist(),
        index=events.index,
    )

    null = df.loc[df.duration <= 0, :]
    if not null.empty:
        types = null["type"].unique()
        msg = f"Found {len(null)} event(s) with null duration (types: {types})"
        warnings.warn(msg)

    dfs = []
    for _, sub in df.groupby(by="timeline", sort=False):
        dfs.append(
            sub.sort_values(
                by=["start", "duration"], ascending=[True, False], ignore_index=True
            )
        )
    important = ["type", "start", "duration", "timeline"]
    df = pd.concat(dfs, ignore_index=True)

    columns = important + [c for c in df.columns if c not in important]
    df = df.loc[:, columns]

    df = df.assign(stop=lambda x: x.start + x.duration)
    return df


def read_events(events: pd.DataFrame) -> tp.Iterable[tp.Any]:
    raise RuntimeError("read_segments is deprecated, use Event.from_dict(row).read()")


def intersection_segments(
    events: pd.DataFrame,
    starts: float | np.ndarray,
    durations: float | np.ndarray,
) -> tp.Generator[Segment, None, None]:

    if events.timeline.nunique() != 1:
        raise RuntimeError("only support a single timeline")
    starts = np.ravel(starts)
    if isinstance(durations, (list, tuple)):
        durations = np.array(durations)
    if not isinstance(durations, np.ndarray):
        durations = durations * np.ones_like(starts)
    stops = np.array(starts + durations)
    starts = starts[:, None]
    stops = stops[:, None]
    estarts = np.array(events.start)[None, :]
    estops = np.array(events.start + events.duration)[None, :]

    select = estarts < stops
    select &= estops > starts

    for k, (start_, duration) in enumerate(zip(starts, durations)):
        start = float(start_.item())
        yield Segment(
            _index=np.array(events.index[select[k]]), start=start, duration=duration
        )


def _prepare_strided_windows(
    start: float,
    stop: float,
    stride: float,
    duration: float,
    drop_incomplete: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    eps = 1e-8
    if drop_incomplete:
        stop -= duration
    starts = np.arange(start, stop + eps, stride)
    durations = np.full_like(starts, fill_value=duration)
    return starts, durations


def iter_segments(
    events: pd.DataFrame,
    idx: int | pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | None = None,
    stride: float | None = None,
    stride_drop_incomplete: bool = True,
) -> tp.Iterator[Segment]:

    df = events
    starts: tp.Any

    durations: tp.Any
    if not hasattr(df, "stop"):

        raise ValueError("Run ns.segments.validate_data on dataframe first")
    if not isinstance(start, (int, float)):
        raise TypeError("start must be int/float")

    creators = SegmentCreator.from_obj(events)

    start = float(start)
    if idx is None and stride is None:

        stride = 2 * (1 + abs(start) + max(c.stops.max() for c in creators.values()))
    if stride is not None:
        if not isinstance(stride, (int, float)):
            raise RuntimeError(
                f"stride can only be None or int/float, got {type(stride)}"
            )
        if not isinstance(duration, (int, float)):
            raise RuntimeError(
                f"duration must be int/float for strided windows, got {duration}"
            )
    if idx is None:
        if stride is None or duration is None:
            raise ValueError("Either stride or idx must be provided")
        stride = float(stride)
        duration = float(duration)
        for creator in creators.values():
            starts, durations = _prepare_strided_windows(
                creator.starts.min() + start,
                creator.stops.max() + start,
                stride,
                duration,
                drop_incomplete=stride_drop_incomplete,
            )
            for start_, duration_ in zip(starts, durations):
                seg = creator.select(start=start_, duration=duration_)
                seg._trigger = start_
                yield seg
        return

    if isinstance(idx, int):
        idx = df.index == idx

    if not np.any(idx):
        avail = pd.unique(df["type"])
        raise ValueError(
            "Empty trigger events provided to list_segments (first argument)\n"
            f"Available events.type: {avail} (did you forget capitalizing the event name?)"
        )

    if "bool" in str(idx.dtype).lower():

        idx = df.loc[idx].index

    df.loc[idx]

    triggers: tp.Generator | list | np.ndarray
    groups = tqdm.tqdm(df.groupby("timeline", sort=False), desc="Creating segments")

    for tl_name, tl in groups:
        if not isinstance(tl_name, str):
            raise TypeError(f"timeline should be a string, got {tl_name!r}")

        if idx is not None:
            j = tl.index.isin(idx)
            if not any(j):

                warn_once(f"No valid events found for timeline {tl_name}.")
                continue

            if stride is None:
                starts = tl.loc[j].start + start

                triggers = (r._asdict() for r in tl.loc[j].itertuples())

                if duration is None:
                    durations = tl.loc[j].duration
                else:
                    durations = np.ones_like(starts) * duration

            else:

                starts, durations, triggers = [], [], []
                for row in tl.loc[j].itertuples():
                    if not isinstance(duration, (int, float)):
                        msg = f"Unsupported type for one of duration {duration}"
                        raise TypeError(msg)
                    _starts, _durations = _prepare_strided_windows(
                        row.start + start,
                        row.stop + start,
                        stride,
                        duration,
                        drop_incomplete=stride_drop_incomplete,
                    )
                    starts.append(_starts)
                    durations.append(_durations)
                    triggers.extend([row._asdict()] * len(_starts))

                starts = np.concatenate(starts)
                durations = np.concatenate(durations)

        creator = creators[tl_name]
        for start_, duration_, trigger_ in zip(starts, durations, triggers):
            seg = creator.select(start=start_, duration=duration_)
            seg._trigger = trigger_
            yield seg


def list_segments(
    events: pd.DataFrame,
    idx: pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | None = None,
    stride: float | None = None,
    stride_drop_incomplete: bool = True,
) -> list[Segment]:

    return list(iter_segments(**locals()))


def find_enclosed(df: pd.DataFrame, start: float, duration: float) -> pd.Series:
    estart = np.array(df.start)
    estop = estart + np.array(df.duration)
    is_enclosed = np.logical_and(estart >= start, estop <= start + duration)
    return pd.Series(df.index[is_enclosed])


def find_overlap(
    events: pd.DataFrame,
    idx: int | pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | np.ndarray | None = None,
) -> pd.Series:

    if idx is None:

        assert duration is not None
        assert events.timeline.nunique() == 1
        has_overlap = (events.start >= start) & (events.start < start + duration)
        has_overlap |= (events.start + events.duration > start) & (
            events.start + events.duration <= start + duration
        )
        has_overlap |= (events.start <= start) & (
            events.start + events.duration >= start + duration
        )

        out = events.index[has_overlap]
        return pd.Series(out)
    else:
        sel: list[int] = []
        for segment in iter_segments(
            events,
            idx=idx,
            start=start,
            duration=duration,
            stride=None,
        ):
            sel.extend(segment._index.tolist())

        return pd.Series(sel)


class SegmentCreator:

    def __init__(self, events: list[Event]) -> None:
        timelines = {e.timeline for e in events}
        if len(timelines) > 1:
            name = self.__class__.__name__
            msg = f"Cannot create {name} on several timelines, got {timelines}"
            raise ValueError(msg)
        self.events = np.array(events)
        self.starts = np.array([e.start for e in events])
        self.indices = np.array([e._index for e in events])
        self.stops = np.array([e.duration for e in events]) + self.starts

    @classmethod
    def from_obj(cls, obj: tp.Any) -> dict[str, "SegmentCreator"]:

        from data_utils import helpers

        timeline_events: dict[str, list[Event]] = collections.defaultdict(list)
        for e in helpers.extract_events(obj):
            timeline_events[e.timeline].append(e)
        timelines = list(timeline_events)
        if isinstance(obj, pd.DataFrame):

            timelines = list(obj.timeline.unique())
        return {tl: cls(timeline_events[tl]) for tl in timelines}

    def select(self, start: float, duration: float) -> Segment:

        select = self.starts < start + duration
        select &= self.stops > start
        events = list(self.events[select])
        index = self.indices[select]
        return Segment(ns_events=events, start=start, duration=duration, _index=index)
