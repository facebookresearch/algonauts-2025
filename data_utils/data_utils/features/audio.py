# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
import warnings

import numpy as np
import pandas as pd
import pydantic
import torch
from torch import nn
from torch.nn import functional as F

import data_utils as ns
from data_utils.base import Frequency, TimedArray
from data_utils.events import Event, EventTypesHelper
from data_utils.infra import MapInfra
from data_utils.segments import Segment

logger = logging.getLogger(__name__)


class AudioEncoder(pydantic.BaseModel):
    _effective_frequency: float | None = None
    _event_types_helper: EventTypesHelper
    _missing_default: torch.Tensor | None = None
    layers: list[float] = [0.5, 0.75, 1.0]
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        super().__init_subclass__()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._event_types_helper = EventTypesHelper("Sound")
        name = self.__class__.__name__
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare(
        self, obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment]
    ) -> None:
        from data_utils import helpers

        events = helpers.extract_events(obj, types=self._event_types_helper)

        self._get_data(events)
        if events:

            self(
                events[0],
                start=events[0].start,
                duration=0.001,
                trigger=events[0].to_dict(),
            )

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

        tarrays = list(
            self._get_timed_arrays(events=events, start=start, duration=duration)
        )
        if self._effective_frequency is None:
            self._effective_frequency = 2.0

        time_info: dict[str, tp.Any] = {
            "start": start,
            "frequency": self._effective_frequency,
            "duration": duration,
        }
        out = TimedArray(aggregation="sum", **time_info)
        for ta in tarrays:
            out += ta
        tensor = torch.from_numpy(out.data)
        if not tensor.ndim:
            tensor = tensor.unsqueeze(0)

        if self._missing_default is None:

            shape = tuple(tensor.shape[:-1])
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

    name: tp.Literal["AudioEncoder"] = "AudioEncoder"  # CHANGE NAME
    model_name: str = "facebook/wav2vec2-large-xlsr-53"  # SIMPLIFY
    device: tp.Literal["auto", "cpu", "cuda", "accelerate"] = "auto"
    layer_aggregation: tp.Literal["group_mean"] | None = "group_mean"

    frequency: tp.Literal["native"] | float = "native"
    _model: nn.Module
    _feature_extractor: nn.Module

    infra: MapInfra = MapInfra(
        timeout_min=25,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=4096,
        version="v5",
    )

    def _preprocess_wav(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torch.mean(wav, dim=1)

        wav = (wav - wav.mean()) / (1e-8 + wav.std())
        return wav

    def _resample_wav(
        self, wav: torch.Tensor, old_frequency: float, new_frequency: float
    ) -> torch.Tensor:
        for freq in (old_frequency, new_frequency):
            if not float(freq).is_integer():
                raise ValueError(f"Frequencies need to be integers, got {freq}")
        old_frequency, new_frequency = int(old_frequency), int(new_frequency)
        import julius

        wav = julius.resample.ResampleFrac(old_sr=old_frequency, new_sr=new_frequency)(
            wav.T
        ).T
        return wav

    @infra.apply(
        item_uid=lambda event: f"{event.filepath}_{event.offset:.2f}_{event.duration:.2f}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: list[ns.events.Event]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            from tqdm import tqdm

            events = tqdm(events, desc="Computing audio embeddings")

        for event in events:
            if isinstance(event, ns.events.Sound):
                wav = event.read()
                sfreq = event.frequency
            elif isinstance(event, ns.events.Video):
                audio = event.read().audio
                wav = torch.tensor(audio.to_soundarray(), dtype=torch.float32)
                sfreq = audio.fps
            else:
                raise ValueError(
                    f"Unsupported event type for Audio feature: {type(event)}"
                )
            wav = self._resample_wav(wav, sfreq, self._input_frequency)
            wav = self._preprocess_wav(wav)
            latents = self._process_wav(wav)

            timepoints = Frequency(2.0).to_ind(event.duration)

            if abs(timepoints - latents.shape[-1]) > 0:
                if len(latents.shape) == 2:

                    latents = F.interpolate(latents[None], timepoints)[0]
                else:

                    latents = F.interpolate(latents, timepoints)
            yield latents.numpy()

    def _aggregate_layers(self, latents: np.ndarray) -> np.ndarray:
        layer_indices = np.unique(
            [int(i * (latents.shape[0] - 1)) for i in self.layers]
        ).tolist()

        if len(layer_indices) == 1:
            if self.layer_aggregation is None:
                return latents[layer_indices[0]][None, :]
            else:
                return latents[layer_indices[0]]
        else:
            if self.layer_aggregation == "group_mean":
                groups = []
                layer_indices[-1] += 1
                for l1, l2 in zip(layer_indices[:-1], layer_indices[1:]):
                    groups.append(latents[l1:l2].mean(0))
                return np.stack(groups)
            elif self.layer_aggregation is None:
                return latents[layer_indices]
            else:
                raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")

    @property
    def _input_frequency(self) -> float:
        return getattr(self.feature_extractor, "sampling_rate", 16_000)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["device"]

    def _exclude_from_cache_uid(self) -> list[str]:
        return ["device"] + ["layers", "layer_aggregation"]

    @property
    def feature_extractor(self) -> nn.Module:
        if not hasattr(self, "_feature_extractor"):
            self._feature_extractor = self._get_feature_extractor(self.model_name)
        return self._feature_extractor

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            self._model = self._get_sound_model(self.model_name)
        return self._model

    def _get_feature_extractor(self, model_name: str) -> torch.nn.Module:
        from transformers import AutoFeatureExtractor

        return AutoFeatureExtractor.from_pretrained(model_name)

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        from transformers import AutoModel

        _model = AutoModel.from_pretrained(model_name)
        _model.to(self.device)
        _model.eval()
        return _model

    def _get_features(self, wav):
        out = self._feature_extractor(
            wav,
            return_tensors="pt",
            sampling_rate=self.feature_extractor.sampling_rate,
            do_normalize=True,
        )
        try:
            return out["input_features"]
        except KeyError:
            return out["input_values"]

    def _get_timed_arrays(
        self, events: list[ns.events.Event], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        if not events:
            raise RuntimeError("_get_timed_arrays should not be called with no event")
        freq = 2.0
        for latent, event in zip(self._get_data(events), events):
            if freq is None:

                freq = latent.shape[-1] / event.duration
                self._effective_frequency = freq

            tdata = TimedArray(data=latent, start=event.start, frequency=freq)
            sub = tdata.overlap(start=start, duration=duration)
            if sub is None:

                sub = tdata.overlap(start=tdata.start, duration=0)
            sub.data = self._aggregate_layers(sub.data)
            yield sub

    def _process_wav(self, wav: torch.Tensor) -> torch.Tensor:
        features = self._get_features(wav)
        with torch.no_grad():
            outputs = self.model(features.to(self.device), output_hidden_states=True)
        out: tp.Any = outputs.get("hidden_states")
        if isinstance(out, tuple):
            out = torch.stack(out)

        out = out.squeeze(1).detach().cpu().clone().transpose(-1, -2).numpy()

        return torch.Tensor(out)


class Wav2Vec(AudioEncoder):

    name: tp.Literal["Wav2Vec"] = "Wav2Vec"

    model_name: str = "facebook/wav2vec2-large-xlsr-53"


class Wav2VecBert(AudioEncoder):

    name: tp.Literal["Wav2VecBert"] = "Wav2VecBert"

    model_name: str = "facebook/w2v-bert-2.0"

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        from transformers import Wav2Vec2BertModel

        _model = Wav2Vec2BertModel.from_pretrained(model_name)
        _model.to(self.device)
        _model.eval()
        return _model


class Granite(AudioEncoder):

    name: tp.Literal["Granite"] = "Granite"

    model_name: str = "ibm-granite/granite-speech-3.3-2b"

    def _get_feature_extractor(self, model_name: str) -> torch.nn.Module:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(model_name)

    def _process_wav(self, wav: torch.Tensor) -> torch.Tensor:
        features = self._get_features(wav)
        with torch.no_grad():
            outputs = self.model(**features, output_hidden_states=True)
        out: tp.Any = outputs.get("hidden_states")
        if isinstance(out, tuple):
            out = torch.stack(out)

        out = out.squeeze(1).detach().cpu().clone().transpose(-1, -2).numpy()

        return torch.Tensor(out)

    def _get_features(self, wav):
        chat = [
            {
                "role": "system",
                "content": (
                    "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<|audio|>can you transcribe the speech into a written format?"
                ),
            },
        ]
        tokenizer = self._feature_extractor.tokenizer
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        out = self._feature_extractor(
            text=text,
            audio=wav,
            return_tensors="pt",
            device=self.device,
        ).to(self.device)
        return out

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        from transformers import AutoModelForSpeechSeq2Seq

        _model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        _model.to(self.device)
        _model.eval()
        return _model


AudioFeature = tp.Annotated[
    tp.Union[Wav2VecBert, Granite, Wav2Vec],
    pydantic.Field(discriminator="name"),
]
