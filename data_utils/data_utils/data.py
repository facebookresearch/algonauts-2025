# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import logging
import shutil
import tempfile
import typing as tp
from collections import OrderedDict
from pathlib import Path

import exca
import pandas as pd
import pydantic

from .base import PathLike, StrCast
from .enhancers import Enhancer
from .infra import CacheDict, MapInfra
from .segments import validate_events
from .utils import compress_string

logger = logging.getLogger(__name__)


def _check_folder_path(path: PathLike, name: str) -> Path:

    path = Path(path)
    if not path.parent.exists():
        raise RuntimeError(f"Parent folder {path.parent} of {name} must exist first.")
    path.mkdir(exist_ok=True)
    return path


TIMELINES: tp.Dict[str, "BaseData"] = {}


class BaseData(pydantic.BaseModel):

    subject: StrCast
    path: PathLike
    timeline: str = ""

    version: tp.ClassVar[str] = "v2"
    study: tp.ClassVar[str]
    device: tp.ClassVar[str] = ""

    @tp.final
    @classmethod
    def iter_timelines(cls, path: PathLike) -> tp.Iterator["BaseData"]:
        path = _check_folder_path(path, name="path")
        study = "Algonauts2025"
        if path.name.lower() != study.lower():

            for name in (study, study.lower(), study.lower().replace("bold", "")):
                if (path / name).exists():
                    path = path / name
                    logger.debug("Updating study path to %s", path)
                    break
        found = False
        for tl in cls._iter_timelines(path):
            found = True
            yield tl
        if not found:
            raise RuntimeError(f"No timeline found for {study} in {path}")

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

        if not self.timeline:
            excludes = "path", "timeline"
            timeline = "Algonauts2025"
            for name, arg in type(self).model_fields.items():
                if name in excludes or arg.init_var is False:
                    continue
                value = getattr(self, name)
                timeline += f"_{name}-{str(value)}"
            self.timeline = compress_string(timeline)

        TIMELINES[self.timeline] = self

    @tp.final
    def load(self) -> pd.DataFrame:

        events = self._load_events()

        for col in ["subject", "timeline"]:
            if col in events:
                raise ValueError(f"Column {col} already exists in the events dataframe")
            events[col] = getattr(self, col)
        events["study"] = "Algonauts2025"

        events = validate_events(events)
        return events


class StudyLoader(pydantic.BaseModel):

    path: PathLike
    query: str | None = None

    enhancers: tp.List[Enhancer] | OrderedDict[str, Enhancer] = []
    infra: MapInfra = MapInfra(cluster="processpool")
    _build_infra: MapInfra = MapInfra()
    _timelines: tp.List[BaseData] | None = None

    def _exclude_from_cls_uid(self) -> tp.List[str]:
        return ["path"]

    def model_post_init(self, log__: tp.Any) -> None:
        if isinstance(self.enhancers, dict):
            version = exca.__version__
            if tuple(int(n) for n in version.split(".")) < (0, 4, 2):
                msg = f"study_loader.enhancers cannot be a dict with exca<0.4.2 ({version=})"
                raise RuntimeError(msg)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _ = CacheDict(folder=tmp, cache_type="ParquetPandasDataFrame")
        except ValueError as e:
            raise RuntimeError('Run "pip install pyarrow" to enable study cache') from e

        study = self.study()

        name = self.__class__.__name__ + ",{version}"
        i = self.infra

        if i.cluster is not None and "pool" in i.cluster:
            if "max_jobs" not in i.model_fields_set:
                i.max_jobs = None

        i.version = type(self).model_fields["infra"].default.version + f"-{study.version}"
        folder_name = f"{name},Algonauts2025"
        i._uid_string = folder_name + "/{method},{uid}"

        names = ["folder", "version", "_uid_string", "mode"]
        self._build_infra._update({x: getattr(i, x) for x in names})

        if self.infra.mode == "force" and self.infra.folder is not None:
            folder = Path(self.infra.folder) / folder_name
            if folder.exists():
                shutil.rmtree(folder)

    def study(self) -> tp.Type[BaseData]:

        return getattr(
            importlib.import_module("data_utils.studies.algonauts2025"), "Algonauts2025"
        )

    def iter_timelines(self) -> tp.Iterator[BaseData]:

        if self._timelines is None:
            self._timelines = list(self.study().iter_timelines(self.path))
        else:
            for tl in self._timelines:
                TIMELINES[tl.timeline] = tl

        return iter(self._timelines)

    def study_summary(self, apply_query: bool = True) -> pd.DataFrame:

        out = pd.DataFrame([dict(tl) for tl in self.iter_timelines()])
        out["subject"] = out.subject.apply(lambda x: f"Algonauts2025/{x}")
        if any(n in out.columns for n in ["subject_index", "timeline_index"]):
            msg = "Study dataframes are not allowed to have subject_index nor timeline_index"
            msg += f" in their column, found columns: {list(out.columns)}"
            raise RuntimeError(msg)
        groups = out.groupby("subject")
        out.loc[:, "subject_index"] = groups.ngroup()
        out.loc[:, "subject_timeline_index"] = groups.cumcount()
        out.loc[:, "timeline_index"] = out.index

        if apply_query and self.query is not None:
            out = out.query(self.query)
        return out

    def build(self) -> pd.DataFrame:

        for _ in self.iter_timelines():
            pass
        query = self.query
        out = list(self._build([query]))[0]

        return out

    @infra.apply(
        item_uid=lambda item: item.timeline,
        exclude_from_cache_uid=("enhancers", "query"),
    )
    def _load_timelines(
        self, timelines: tp.Iterable[BaseData]
    ) -> tp.Iterator[pd.DataFrame]:

        for tl in timelines:
            TIMELINES[tl.timeline] = tl

            out = tl.load()
            out.subject = f"Algonauts2025/{tl.subject}"
            yield out

    @_build_infra.apply(
        item_uid=str,
        exclude_from_cache_uid=("query",),
        cache_type="ParquetPandasDataFrame",
    )
    def _build(self, queries: tp.Iterable[str | None]) -> tp.Iterator[pd.DataFrame]:

        timelines = list(self.iter_timelines())
        summary: pd.DataFrame | None = None
        for query in queries:
            sub = timelines
            if query is not None:
                if summary is None:
                    summary = self.study_summary(apply_query=False)
                selected = summary.query(query)
                sub = [timelines[i] for i in selected.index]
            if not sub:
                msg = f"No timeline found for Algonauts2025 with {query=}"
                raise RuntimeError(msg)
            events = pd.concat(list(self._load_timelines(sub))).reset_index(drop=True)
            if isinstance(self.enhancers, dict):
                enhancers = list(self.enhancers.values())
            else:
                enhancers = list(self.enhancers)
            for enhancer in enhancers:
                events = enhancer(events)
            events = validate_events(events)
            yield events
