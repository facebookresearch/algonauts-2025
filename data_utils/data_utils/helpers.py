# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import concurrent.futures
import logging
import typing as tp

import pandas as pd

from data_utils import events, segments

logger = logging.getLogger(__name__)
TypesParam = str | tp.Sequence[str] | tp.Type[events.Event] | events.EventTypesHelper


def extract_events(obj: tp.Any, types: TypesParam | None = None) -> list[events.Event]:

    helper: events.EventTypesHelper | None = None
    if isinstance(types, events.EventTypesHelper):
        helper = types
    elif types is not None:
        helper = events.EventTypesHelper(types)

    if isinstance(obj, (list, tuple)):
        if not obj:
            return []
        if isinstance(obj[0], events.Event):
            if helper is not None:
                obj = [e for e in obj if isinstance(e, helper.classes)]
            return obj
    if isinstance(obj, pd.DataFrame):
        if helper is not None:
            obj = obj.loc[obj.type.isin(helper.names), :]
        unknown = set(obj.type) - set(events.Event._CLASSES)
        if unknown:
            logger.warning("Ignoring unknown event types: %s", unknown)
            obj = obj.loc[~obj.type.isin(unknown), :]

        num = len(obj)
        iterable = (obj.iloc[k, :] for k in range(num)) if num <= 2 else obj.itertuples()
        out = [events.Event.from_dict(r) for r in iterable]
        for i, e in zip(obj.index, out):
            e._index = i

        return out
    if isinstance(obj, events.Event):
        obj = [obj]
    elif isinstance(obj, dict):
        obj = [events.Event.from_dict(obj)]
    if not isinstance(obj, (list, tuple)):
        raise NotImplementedError(f"Conversion of {type(obj)} is not supported")
    if not obj:
        return []
    if isinstance(obj[0], segments.Segment):
        event_dict = dict()
        for segment in obj:
            event_dict.update({id(e): e for e in segment.ns_events})
        obj = list(event_dict.values())
    if not isinstance(obj[0], events.Event):
        raise NotImplementedError(f"Unexpected list of {type(obj[0])} is not supported")
    return extract_events(obj, types=helper)


def prepare_features(
    features: list[tp.Any] | dict[str, tp.Any],
    events: pd.DataFrame | tp.Sequence[events.Event] | tp.Sequence[segments.Segment],
) -> None:

    events = extract_events(events)

    feature_list = list(features.values()) if isinstance(features, dict) else features
    features_using_slurm = [
        feature
        for feature in feature_list
        if hasattr(feature, "infra")
        and getattr(feature.infra, "cluster", None) == "slurm"
    ]
    other_features = [
        feature for feature in feature_list if feature not in features_using_slurm
    ]
    slurm_names = ", ".join(
        feature.__class__.__name__ for feature in features_using_slurm
    )
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for feature in features_using_slurm:
            futures.append(executor.submit(feature.prepare, events))

            futures[-1].__dict__["_name"] = feature.__class__.__name__
        if features_using_slurm:
            logger.info(
                f"Started parallel preparation of features {slurm_names} on slurm"
            )
        for feature in other_features:
            logger.info(f"Preparing feature: {feature.__class__.__name__}")
            feature.prepare(events)
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()

            except Exception as e:
                name = future.__dict__.get("_name", "UNKNOWN")
                logger.warning("Error occurred while preparing feature %s: %s", name, e)
                raise
