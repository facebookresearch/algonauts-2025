# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import yaml
from typing_extensions import Annotated

PathLike = str | Path


logger = logging.getLogger("data_utils")
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _int_cast(v: tp.Any) -> tp.Any:

    if isinstance(v, int):
        return str(v)
    return v


StrCast = Annotated[str, pydantic.BeforeValidator(_int_cast)]
CACHE_FOLDER = Path.home() / ".cache/data_utils/"
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


class Frequency(float):

    @tp.overload
    def to_ind(self, seconds: float) -> int: ...

    @tp.overload
    def to_ind(self, seconds: np.ndarray) -> np.ndarray: ...

    def to_ind(self, seconds: tp.Any) -> tp.Any:

        if isinstance(seconds, np.ndarray):
            return np.round(seconds * self).astype(int)
        return int(round(seconds * self))

    @tp.overload
    def to_sec(self, index: int) -> float: ...

    @tp.overload
    def to_sec(self, index: np.ndarray) -> np.ndarray: ...

    def to_sec(self, index: tp.Any) -> tp.Any:

        return index / self

    @staticmethod
    def _yaml_representer(dumper, data):

        return dumper.represent_scalar("tag:yaml.org,2002:float", str(float(data)))


class TimedArray:
    def __init__(
        self,
        *,
        frequency: float,
        start: float,
        data: np.ndarray | None = None,
        duration: float | None = None,
        aggregation: str = "sum",
    ) -> None:

        self.frequency = Frequency(frequency)
        self.start = start
        self.aggregation = aggregation
        exp_size = 0
        if duration is not None and duration < 0:
            raise ValueError(f"duration should be None or >=0, got {duration}")

        if data is None:
            if duration is None:
                raise ValueError("Missing data or duration")

            if not frequency:
                data = np.zeros((0,))
            else:
                exp_size = max(1, self.frequency.to_ind(duration))
                data = np.zeros((0, exp_size))
        self.data = data
        if frequency and duration is not None:
            exp_size = max(1, self.frequency.to_ind(duration))
            if not self.data.shape[-1]:
                msg = "Last dimension is empty but frequency is not null "
                msg += f"(shape={self.data.shape})"
                raise ValueError(msg)
            if abs(data.shape[-1] - exp_size) > 2:
                msg = f"Data has incorrect (last) dimension {data.shape} for duration "
                msg += f"{duration} and frequency {frequency} (expected {exp_size})"
                raise ValueError(msg)
        if frequency:
            self.duration = self.frequency.to_sec(data.shape[-1])
        elif duration is None:
            raise ValueError(f"duration must be provided if {frequency=}")
        else:
            self.duration = duration

        self._overlapping_data_count: None | np.ndarray = None
        if aggregation == "average":
            num = self.data.shape[-1] if self.frequency else 1
            self._overlapping_data_count = np.zeros(num, dtype=int)
        elif aggregation != "sum":
            raise ValueError(f"Unknown {aggregation=}")

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        fields = "frequency,start,duration,aggregation,data".split(",")
        string = ",".join(f"{f}={getattr(self, f)}" for f in fields)
        return f"{cls}({string})"

    def __iadd__(self, other: "TimedArray") -> "TimedArray":
        if other.frequency and self.frequency != other.frequency:
            diff = abs(self.frequency - other.frequency)
            if diff * max(self.duration, other.duration) >= 0.5:

                msg = f"Cannot add with different (non-0) frequencies ({other.frequency} and {self.frequency})"
                raise ValueError(msg)
        if not self.data.size:

            last = -1 if other.frequency else None
            shape = other.data.shape[:last]
            if self.frequency:
                shape += (self.data.shape[-1],)
            self.data = np.zeros(shape, dtype=other.data.dtype)
        if self.frequency:
            slices = [
                sa1._overlap_slice(sa2.start, sa2.duration)
                for sa1, sa2 in [(self, other), (other, self)]
            ]
            if slices[0] is None or slices[1] is None:
                return self

            self_slice = slices[0][-1]
            other_slice = slices[1][-1]
        else:
            self_slice = None
            other_slice = None
        if self._overlapping_data_count is None:

            self.data[..., self_slice] += other.data[..., other_slice]
        else:

            counts = self._overlapping_data_count[..., self_slice]
            upd = counts / (1.0 + counts)
            self.data[..., self_slice] *= upd
            self.data[..., self_slice] += (1 - upd) * other.data[..., other_slice]
            counts += 1
        return self

    def _overlap_slice(
        self, start: float, duration: float
    ) -> tuple[float, float, slice | None] | None:
        if duration < 0:
            raise ValueError(f"duration should be >=0, got {duration=}")
        overlap_start = max(start, self.start)
        overlap_stop = min(start + duration, self.start + self.duration)
        if overlap_stop < overlap_start:
            return None

        if overlap_stop == overlap_start and self.duration and duration:
            return None

        if not self.frequency:
            return overlap_start, overlap_stop - overlap_start, None
        start_ind = self.frequency.to_ind(overlap_start - self.start)
        duration_ind = self.frequency.to_ind(overlap_stop - overlap_start)

        if duration_ind <= 0:

            duration_ind = 1

        tps = self.data.shape[-1]
        if start_ind > tps - duration_ind:
            start_ind = tps - duration_ind
        if start_ind < 0:
            raise RuntimeError(f"Fail for {start=} {duration=} on {self}")
        start = self.frequency.to_sec(start_ind) + self.start
        duration = self.frequency.to_sec(duration_ind)

        out = start, duration, slice(start_ind, start_ind + duration_ind)
        return out

    def overlap(self, start: float, duration: float) -> tp.Optional["TimedArray"]:

        out = self._overlap_slice(start, duration)
        if out is None:
            return None
        ostart, oduration, sl = out
        return TimedArray(
            frequency=self.frequency,
            start=ostart,
            duration=oduration,
            data=self.data[..., sl],
        )


yaml.representer.SafeRepresenter.add_representer(Frequency, Frequency._yaml_representer)
