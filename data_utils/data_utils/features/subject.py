# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import typing as tp
import warnings

import pandas as pd
import pydantic
import torch

import data_utils as ns
from data_utils.base import Frequency as Frequency
from data_utils.base import TimedArray as TimedArray
from data_utils.events import Event, EventTypesHelper
from data_utils.segments import Segment

logger = logging.getLogger(__name__)


class SubjectEncoder(pydantic.BaseModel):
    frequency: float = 0.0

    frequency: float | tp.Literal["native"] = 0.0
    _effective_frequency: float | None = None
    _event_types_helper: EventTypesHelper
    _missing_default: torch.Tensor | None = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._event_types_helper = EventTypesHelper("Event")

    def _get_data(self, events: list[Event]) -> tp.Iterable[tp.Any]:
        for _ in events:
            yield None

    def _get_timed_arrays(
        self, events: list[Event], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        raise NotImplementedError

    def __call__(
        self,
        events: tp.Any,
        start: float,
        duration: float,
        trigger: float | dict[str, tp.Any] | None = None,
    ) -> torch.Tensor:
        _input_events = events

        from data_utils import helpers

        assert duration >= 0.0, f"{duration} must be >= 0."
        event_types = self._event_types_helper.classes
        name = self.__class__.__name__
        events = helpers.extract_events(events, types=self._event_types_helper)

        if not events and self._missing_default is not None:
            if self._effective_frequency is None:
                msg = f"_missing_default was set for {name} but _effective_frequency is missing"
                raise RuntimeError(msg)
            default = self._missing_default
            freq = Frequency(self._effective_frequency)
            if freq:
                n_times = max(1, freq.to_ind(duration))
                reps = [1 for _ in range(default.ndim)] + [n_times]
                default = default.unsqueeze(-1).repeat(reps)
            return default

        if not events:
            found_types = {type(e) for e in _input_events}
            msg = f"No {event_types} found in segment for feature {name} "
            msg += f"(types found: {found_types} in {_input_events}) "
            msg += "and feature shape not populated "
            msg += '(you may need to call "prepare" on the feature).'
            raise ValueError(msg)

        events = events[:1]
        tarrays = list(
            self._get_timed_arrays(events=events, start=start, duration=duration)
        )
        if self._effective_frequency is None:
            if self.frequency == "native":
                self._effective_frequency = tarrays[0].frequency
            else:
                self._effective_frequency = self.frequency

        time_info: dict[str, tp.Any] = {
            "start": start,
            "frequency": self._effective_frequency,
            "duration": duration,
        }
        aggreg = "sum"
        out = TimedArray(aggregation=aggreg, **time_info)
        for ta in tarrays:
            out += ta
        tensor = torch.from_numpy(out.data)
        if not tensor.ndim:
            tensor = tensor.unsqueeze(0)

        if self._missing_default is None:

            shape = tuple(tensor.shape[: -1 if self.frequency else None])
            self._missing_default = torch.zeros(*shape, dtype=tensor.dtype)
        return tensor

    def _events_from_dataframe(self, events: pd.DataFrame) -> list[tp.Any]:
        from data_utils import helpers

        warnings.warn(
            "_events_from_dataframe is deprecated, use ns.helpers.extract_events instead",
            DeprecationWarning,
        )
        events_ = helpers.extract_events(events, types=self._event_types_helper)
        return events_

    def _get_timed_arrays(
        self, events: list[Event], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        for event in events:
            embedding = self.get_static(event)
            ta = TimedArray(
                frequency=0,
                duration=event.duration,
                start=event.start,
                data=embedding.numpy(),
            )
            yield ta

    name: tp.Literal["SubjectEncoder"] = "SubjectEncoder"

    _label_to_ind: dict[str, int] = {}

    def _extract_event_field(self, event: ns.events.Event) -> str:
        if hasattr(event, "subject"):
            return getattr(event, "subject")
        else:
            return event.extra["subject"]

    def prepare(
        self, obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment]
    ) -> None:
        from data_utils import helpers

        events = helpers.extract_events(obj, types=self._event_types_helper)
        field = "subject"
        if not all(hasattr(e, field) or field in e.extra for e in events):
            msg = f"Field {field} not found in events for {self.__class__.__name__}"
            raise TypeError(msg)
        labels = set(self._extract_event_field(e) for e in events)
        if len(labels) < 2:
            logger.warning(
                f"SubjectEncoder has only found one label: {labels}. "
                "This was probably not intended."
            )
        self._label_to_ind = {label: i for i, label in enumerate(sorted(labels))}
        if events:
            self(events[0], events[0].start, duration=0.001, trigger=events[0].to_dict())

    def get_static(self, event: ns.events.Event) -> torch.Tensor:
        if not self._label_to_ind:
            msg = "Must call label_encoder.prepare(events) before using the feature."
            raise ValueError(msg)
        inds = [self._label_to_ind[self._extract_event_field(event)]]
        label = torch.tensor(inds, dtype=torch.long)
        return label
