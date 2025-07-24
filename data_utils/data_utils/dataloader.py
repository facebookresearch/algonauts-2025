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

import torch

import data_utils as ns

from .base import Frequency

logger = logging.getLogger(__name__)


class CollateSegments:

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise RuntimeError("CollateSegments is deprecated in favor of SegmentDataset")


@dataclasses.dataclass
class SegmentData:

    data: tp.Dict[str, torch.Tensor]
    segments: tp.List[ns.segments.Segment]

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict):
            raise TypeError(f"'features' need to be a dict, got: {self.features}")
        if not self.data:
            raise ValueError(f"No data in {self}")
        if not isinstance(self.segments, list):
            raise TypeError(f"'segments' needs to be a list, got {self.segments}")

        batch_size = next(iter(self.data.values())).shape[0]
        if len(self.segments) != batch_size:
            raise RuntimeError(
                f"Incoherent batch size {batch_size} for {len(self.segments)} segments in {self}"
            )

    def to(self, device: str) -> "SegmentData":

        out = {name: d.to(device) for name, d in self.data.items()}
        return SegmentData(data=out, segments=self.segments)

    def __getitem__(self, key: str) -> None:
        raise RuntimeError("New SegmentData batch is not a dict, use batch.data instead")


def validate_features(features: tp.Mapping[str, tp.Any]) -> tp.Mapping[str, tp.Any]:

    if not features:
        return {}

    if not isinstance(features, collections.abc.Mapping):
        raise ValueError(f"Only dict of features are supported, got {type(features)}")

    return features


def get_pad_lengths(
    feats: tp.Mapping[str, tp.Any],
    pad_duration: float | None,
) -> tp.Dict[str, int]:

    pad_lengths: tp.Dict[str, int] = {}
    if pad_duration is None:
        return pad_lengths
    for name, f in feats.items():
        if isinstance(
            f,
            ns.features.text.LLM
            | ns.features.audio.AudioFeature
            | ns.features.neuro.Fmri
            | ns.features.video.VideoFeature
            | ns.features.SubjectEncoder,
        ):
            freq = Frequency(f.frequency)
            pad_lengths[name] = freq.to_ind(pad_duration)
    return pad_lengths


def _pad_to(tensor: torch.Tensor, pad_len: int | None):

    if pad_len is None:
        return tensor
    if pad_len < tensor.shape[-1]:
        msg = "Pad duration is shorter than segment duration, cropping."
        warnings.warn(msg, UserWarning)
        return tensor[:, :pad_len]
    else:
        return torch.nn.functional.pad(tensor, (0, pad_len - tensor.shape[-1]))


def _apply_feature(segment: ns.segments.Segment, feature: tp.Any) -> torch.Tensor:

    return feature(
        segment.ns_events,
        start=segment.start,
        duration=segment.duration,
        trigger=segment._trigger,
    )


class SegmentDataset(torch.utils.data.Dataset[SegmentData]):

    def __init__(
        self,
        features: tp.Mapping[str, tp.Any],
        segments: tp.Sequence[ns.segments.Segment],
        pad_duration: float | None = None,
    ) -> None:
        self.features = validate_features(features)
        self.segments = segments
        self._pad_lengths = get_pad_lengths(self.features, pad_duration)

    def collate_fn(self, batches: tp.List[SegmentData]) -> SegmentData:

        if not batches:
            return SegmentData(data={}, segments=[])
        if len(batches) == 1:
            return batches[0]
        if not batches[0].data:
            raise ValueError(f"No feature in first batch: {batches[0]}")

        features = {}
        for name in batches[0].data:
            data = [b.data[name] for b in batches]
            try:
                features[name] = torch.cat(data, axis=0)

            except Exception:
                string = f"Failed to collate data with shapes {[d.shape for d in data]}\n"
                string += "Do you need specifying padding in SegmentDataset?"
                logger.warning(string)
                raise
        segments = [s for b in batches for s in b.segments]
        return SegmentData(data=features, segments=segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> SegmentData:
        seg = self.segments[idx]
        out: tp.Dict[str, torch.Tensor] = {}
        for name, feats in self.features.items():

            data = _apply_feature(seg, feats)

            data = _pad_to(data, self._pad_lengths.get(name, None))

            out[name] = data[None, ...]

        return SegmentData(data=out, segments=[seg])

    def build_dataloader(self, **kwargs: tp.Any) -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(self, collate_fn=self.collate_fn, **kwargs)

    def as_one_batch(self, num_workers: int = 0) -> SegmentData:

        num_workers = min(num_workers, len(self))
        batch_size = len(self)
        if num_workers > 1:
            batch_size = max(1, len(self) // (3 * num_workers))
        if num_workers == 1:
            num_workers = 0

        loader = self.build_dataloader(
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
        )
        return self.collate_fn(list(loader))
