# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import functools
import inspect
import logging
import typing as tp
import urllib
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic

from .base import Frequency, StrCast
from .utils import ignore_all, warn_once

E = tp.TypeVar("E", bound="Event")
logger = logging.getLogger(__name__)


class Event(pydantic.BaseModel):

    start: float
    timeline: str
    duration: pydantic.NonNegativeFloat = 0.0
    extra: dict[str, tp.Any] = {}
    type: tp.ClassVar[str] = "Event"
    _CLASSES: tp.ClassVar[dict[str, tp.Type["Event"]]] = {}
    _index: int | None = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        cls.type = cls.__name__
        Event._CLASSES[cls.__name__] = cls

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if pd.isna(self.start):
            raise ValueError(f"Start time needs to be provided for {self!r}")

    @classmethod
    def from_dict(cls: tp.Type[E], row: tp.Any) -> E:

        index: int | None = None
        if hasattr(row, "_asdict"):
            index = getattr(row, "Index", None)
            row = row._asdict()

        cls_ = cls._CLASSES[row["type"]]
        if not issubclass(cls_, cls):
            raise TypeError(f"{cls_} is not a subclass of {cls}")
        fs = set(cls_.model_fields)

        kwargs: dict[str, tp.Any] = {}
        extra = {}
        for k, v in row.items():
            if pd.isna(v):

                continue
            if k in fs:
                kwargs[k] = v
            elif k != "type":
                if k.startswith("extra__"):

                    k = k[7:]
                extra[k] = v
        kwargs.setdefault("extra", {}).update(extra)

        try:
            out = cls_(**kwargs)
        except Exception as e:
            logger.warning(
                "Event.from_dict parsing failed for input %s\nmapped to %s\n with error: %s)",
                row.to_string() if hasattr(row, "to_string") else row,
                kwargs,
                e,
            )
            raise
        out._index = index
        return out

    def to_dict(self) -> dict[str, tp.Any]:

        out = dict(self.extra)
        out["type"] = self.type

        tag = "extra"
        fields = {x: str(y) if isinstance(y, Path) else y for x, y in self if x != tag}
        out.update(fields)
        return out

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def __str__(self) -> str:
        core_fields = {k: v for k, v in self if k != "extra"}
        return ", ".join([f"{k}={v}" for k, v in core_fields.items()])


Event._CLASSES["Event"] = Event


class EventTypesHelper:

    def __init__(self, event_types: str | tp.Type[Event] | tp.Sequence[str]) -> None:
        self.specified = event_types
        if inspect.isclass(event_types):
            self.classes: tp.Tuple[tp.Type[Event], ...] = (event_types,)
        else:
            if isinstance(event_types, str):
                event_types = (event_types,)
            try:
                self.classes = tuple(Event._CLASSES[x] for x in event_types)

            except KeyError as e:
                avail = list(Event._CLASSES)
                msg = f"{event_types} is an invalid event name, use one of {avail}"
                raise ValueError(msg) from e
        items = Event._CLASSES.items()
        self.names = [x for x, y in items if issubclass(y, self.classes)]


class BaseDataEvent(Event):

    filepath: Path | str = ""
    frequency: float = 0
    _read_method: tp.Any = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.filepath:
            raise ValueError("A filepath must be provided")

        self._set_read_method()
        fp = str(self.filepath)
        self.filepath = fp
        if ":" not in str(fp):

            if not Path(fp).exists():
                warn_once(f"file missing: {fp}")

    def _set_read_method(self) -> None:
        try:
            if getattr(self, "_read_method", None) is not None:
                return
        except TypeError:

            pass

        tag = "method:"
        fp = str(self.filepath)
        if not fp.startswith(tag):
            self._read_method = self._read
            return

        from .data import TIMELINES

        components = urllib.parse.urlparse(fp)
        assert components.netloc == ""
        assert components.params == ""
        assert components.fragment == ""

        inst = TIMELINES[self.timeline]
        kwargs = dict(urllib.parse.parse_qsl(components.query, strict_parsing=True))
        self._read_method = functools.partial(getattr(inst, components.path), **kwargs)

    def __hash__(self) -> int:
        return hash(self.to_dict())

    def __eq__(self, other: tp.Any) -> bool:
        if isinstance(other, self.__class__):
            return self.__hash__() == other.__hash__()
        return False

    def read(self) -> tp.Any:
        self._set_read_method()
        return self._read_method()

    @abstractmethod
    def _read(self) -> tp.Any:
        return

    def _missing_duration_or_frequency(self) -> bool:
        return any(not x or pd.isna(x) for x in [self.duration, self.frequency])


class BaseSplittableEvent(BaseDataEvent):

    offset: pydantic.NonNegativeFloat = 0.0

    def _split(
        self, timepoints: tp.List[float], min_duration: float | None = None
    ) -> tp.Sequence["BaseSplittableEvent"]:

        timepoints = [t for t in timepoints if 0 < t < self.duration]
        timepoints = sorted(set(timepoints))
        if min_duration:
            delta_before = np.diff(timepoints, prepend=0)
            delta_after = np.diff(timepoints, append=self.duration)
            timepoints = [
                t
                for t, db, da in zip(timepoints, delta_before, delta_after)
                if db >= min_duration and da >= min_duration
            ]
        timepoints.append(self.duration)

        start = 0.0
        data = dict(self)
        cls = self.__class__
        events = []
        for stop in list(timepoints):
            if start >= stop:
                raise ValueError(
                    f"Timepoints should be strictly increasing (got {start} and {stop})"
                )
            data.update(
                start=self.start + start,
                duration=stop - start,
                offset=self.offset + start,
            )
            events.append(cls(**data))
            start = stop
        return events


class Image(BaseDataEvent):

    caption: str = ""

    def _read(self) -> tp.Any:

        import PIL.Image

        return PIL.Image.open(self.filepath).convert("RGB")

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.duration <= 0:
            logger.info("Image event has null duration and will be ignored.")


class Sound(BaseSplittableEvent):


    def model_post_init(self, log__: tp.Any) -> None:

        if not Path(self.filepath).exists():
            raise ValueError(f"Sound filepath does not exist: {self.filepath}")
        if self._missing_duration_or_frequency():
            import soundfile

            info = soundfile.info(str(self.filepath))
            self.frequency = Frequency(info.samplerate)
            self.duration = info.duration
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import soundfile
        import torch

        sr = Frequency(self.frequency)
        offset = sr.to_ind(self.offset)
        num = sr.to_ind(self.duration)
        fp = str(self.filepath)
        wav = soundfile.read(fp, start=offset, frames=num)[0]
        out = torch.Tensor(wav)
        if out.ndim == 1:
            out = out[:, None]
        return out


class Video(BaseSplittableEvent):

    def model_post_init(self, log__: tp.Any) -> None:

        if not Path(self.filepath).exists():
            raise ValueError(f"Missing video file {self.filepath}")
        if self._missing_duration_or_frequency():
            from moviepy import VideoFileClip

            with ignore_all():
                video = VideoFileClip(str(self.filepath))
            self.frequency = Frequency(video.fps)
            self.duration = video.duration
            video.close()
        super().model_post_init(log__)

    def _read(self) -> None:
        from moviepy import VideoFileClip

        with ignore_all():
            clip = VideoFileClip(str(self.filepath))
            start, end = self.offset, self.offset + self.duration
            assert end <= clip.duration
            clip = clip.subclipped(start, end)
        return clip


class BaseText(Event):

    language: str = ""
    text: str = pydantic.Field("", min_length=1)
    context: str = ""


class Text(BaseText):
    pass


class Sentence(BaseText):
    pass


class Word(BaseText):
    sentence: str = ""

    sentence_char: int | None = None


class Phoneme(BaseText):
    pass


class Fmri(BaseDataEvent):
    subject: StrCast = ""

    def model_post_init(self, log__: tp.Any) -> None:
        self.subject = str(self.subject)

        if self._missing_duration_or_frequency():
            raise ValueError(
                "Duration and frequency must be provided for Fmri event: "
                "Don't rely on get_zooms as the header is sometimes unreliable.\n"
                f"Got: {self}"
            )
        if not self.subject:
            raise ValueError("Missing 'subject' field")
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import nibabel

        nii_img = nibabel.load(self.filepath, mmap=True)
        if nii_img.ndim not in (4, 2):

            msg = f"{self.filepath} should be 2D or 4D with time the last dim."
            raise ValueError(msg)
        return nii_img
