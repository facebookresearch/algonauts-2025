# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import dataclasses
import functools
import hashlib
import os
import re
import typing as tp
import warnings
from pathlib import Path

import numpy as np


def all_subclasses(cls):

    subs = set(cls.__subclasses__())
    return subs | {s for c in subs for s in all_subclasses(c)}


def match_list(A, B, on_replace="delete"):

    from Levenshtein import editops

    if not isinstance(A, str):
        unique = np.unique(np.r_[A, B])
        label_encoder = dict((k, v) for v, k in enumerate(unique))

        def int_to_unicode(array: np.ndarray) -> str:
            return "".join([str(chr(label_encoder[ii])) for ii in array])

        A = int_to_unicode(A)
        B = int_to_unicode(B)

    changes = editops(A, B)
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type_, val_a, val_b in changes:
        if type_ == "insert":
            B_sel[val_b] = np.nan
        elif type_ == "delete":
            A_sel[val_a] = np.nan
        elif on_replace == "delete":

            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == "keep":

            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


ISSUED_WARNINGS = set()


def warn_once(message: str) -> None:
    if message not in ISSUED_WARNINGS:
        warnings.warn(message)
        ISSUED_WARNINGS.add(message)


def compress_string(file_) -> str:
    def hash_(s: str) -> str:
        return hashlib.sha256(s.encode()).hexdigest()[:10]

    file_ = str(file_)
    fname = Path(file_).name

    pattern = r"[^a-zA-Z0-9.\-_]"
    valid = re.sub(pattern, "", fname)

    if len(fname) > 70:
        valid = "_".join([valid[:20], hash_(fname), valid[-20:]])

    folder = str(Path(file_).parent)
    if folder != "." or valid != fname:
        valid = f"{hash_(file_)}_{valid}"

    return valid


@contextlib.contextmanager
def ignore_all() -> tp.Iterator[None]:
    with open(os.devnull, "w", encoding="utf8") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield


@contextlib.contextmanager
def success_writer(
    fname: str | Path, suffix: str = "_success.txt", success_msg: str = "done"
):

    success_fname = Path(str(Path(fname).with_suffix("")) + suffix)
    file_exists = success_fname.exists()
    yield file_exists
    if not file_exists:
        with open(success_fname, "w") as f:
            f.write(success_msg)


class NoApproximateMatch(ValueError):

    def __init__(self, msg: str, matches: tp.Any) -> None:
        super().__init__(msg)
        self.matches = matches


@dataclasses.dataclass
class Tolerance:

    abs_tol: float
    rel_tol: float

    def __call__(self, value1: float, value2: float) -> bool:
        diff = abs(value1 - value2)
        tol = max(self.abs_tol, self.rel_tol * min(abs(value1), abs(value2)))
        return diff <= tol


@dataclasses.dataclass
class Sequence:

    sequence: tp.Sequence[float]

    current: int

    matches: tp.List[int]

    def valid_index(self, shift: int = 0) -> bool:
        return self.current + shift < len(self.sequence)

    def diff(self, shift: int = 0) -> float:
        return self.sequence[self.current + shift] - self.last_value

    @property
    def last_value(self) -> float:
        return self.sequence[self.matches[-1]]

    def diff_to(self, ind: int) -> np.ndarray:
        r = self.matches[-1]
        sub = self.sequence[r : r + ind] if ind > 0 else self.sequence[r + ind : r]
        return np.array(sub) - self.last_value


def get_spacy_model(*, model: str = "", language: str = "") -> tp.Any:

    if language and model:
        msg = f"Language and model cannot be specified at the same time, got {language=} and {model=}"
        raise ValueError(msg)
    if not language and not model:
        language = "english"

    if language:
        defaults = dict(
            english="en_core_web_lg",
            french="fr_core_news_lg",
            spanish="es_core_news_lg",
            chinese="zh_core_web_lg",
        )
        if language not in defaults:
            raise ValueError(f"Language {language!r} not available: {defaults}")
        model = defaults[language]
    return _get_model(model)


@functools.lru_cache(maxsize=3)
def _get_model(model: str) -> tp.Any:
    import spacy

    if not spacy.util.is_package(model):
        import spacy.cli

        spacy.cli.download(model)

    nlp = spacy.load(model)
    return nlp
