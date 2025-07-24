# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import bisect
import logging
import typing as tp
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
from exca.utils import DISCRIMINATOR_FIELD
from tqdm import tqdm

from . import events as ev
from . import splitting, utils
from .segments import find_enclosed
from .splitting import chunk_events

logger = logging.getLogger(__name__)


class BaseEnhancer(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: str

    _CLASSES: tp.ClassVar[dict[str, type["BaseEnhancer"]]] = {}
    _discriminating_type_adapter: tp.ClassVar[pydantic.TypeAdapter]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:

        super().__pydantic_init_subclass__(**kwargs)
        name = cls.__name__
        if "Base" not in name and not name.startswith("_"):
            if "name" not in cls.model_fields or cls.model_fields["name"].default != name:

                indication = f"name: tp.Literal[{name!r}] = {name!r}"
                raise NotImplementedError(
                    f"Enhancer {name} has incorret/missing name field, add:\n{indication}"
                )
            BaseEnhancer._CLASSES[cls.model_fields["name"].default] = cls

            BaseEnhancer._discriminating_type_adapter = pydantic.TypeAdapter(
                tp.Annotated[
                    tp.Union[tuple(cls._CLASSES.values())],
                    pydantic.Field(discriminator="name"),
                ]
            )

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _parse_into_subclass(
        cls, v: tp.Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> "BaseEnhancer":
        if cls is BaseEnhancer:
            out = BaseEnhancer._discriminating_type_adapter.validate_python(v)
        else:
            out = handler(v)

        out.__dict__[DISCRIMINATOR_FIELD] = "name"
        return out

    @pydantic.model_serializer
    def _dump(self, info: pydantic.SerializationInfo) -> dict[str, tp.Any]:

        out: dict[str, tp.Any] = {}
        for name, field in type(self).model_fields.items():
            val = getattr(self, name)
            if not info.exclude_defaults or val != field.default:
                out[name] = val
        return out

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


EnhancerConfig = BaseEnhancer
Enhancer = BaseEnhancer


class AddText(BaseEnhancer):
    name: tp.Literal["AddText"] = "AddText"

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        if "Text" in events.type.unique():
            logger.info("Text already present in events dataframe, skipping")
            return events
        language = events.loc[events.type == "Word", "language"].values[0]
        text_events = []
        for _, df in events.groupby("timeline"):
            words = df.loc[df.type == "Word"]
            words["stop"] = words["start"] + words["duration"]
            text = " ".join(words.text.values)
            doc = parse_text(text, language)
            sentences = [sent.text.capitalize().rstrip(".") for sent in doc.sents]
            punctuated_text = ". ".join(sentences)

            text_event = words.iloc[0].to_dict()
            text_event |= dict(
                type="Text",
                start=words.start.min(),
                duration=words.stop.max() - words.start.min(),
                timeline=df.timeline.values[0],
                text=punctuated_text,
            )
            text_events.append(text_event)
        events = pd.concat([events, pd.DataFrame(text_events)], ignore_index=True)
        return events


class AddTextToWords(AddText):
    name: tp.Literal["AddTextToWords"] = "AddTextToWords"


class AddSentenceToWords(BaseEnhancer):
    name: tp.Literal["AddSentenceToWords"] = "AddSentenceToWords"
    max_unmatched_ratio: float = 0.0

    override_sentences: bool = False
    _exclude_from_cls_uid: tp.Tuple[str, ...] = ("max_unmatched_ratio",)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.max_unmatched_ratio < 0 or self.max_unmatched_ratio >= 1:
            raise ValueError("max_unmatched_ratio must be >=0 and <1")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:

        if "Sentence" in events.type.unique():
            if not self.override_sentences:
                msg = "Sentence already present in events dataframe"
                logger.warning(msg)
                return events
            events = events[events.type != "Sentence"]
        if "timeline" in events.columns and len(events.timeline.unique()) > 1:
            timelines = []
            desc = "Add sentence to Word based on Text"

            for _, subevents in tqdm(events.groupby("timeline", sort=False), desc=desc):
                timelines.append(self(subevents))
            return pd.concat(timelines, ignore_index=True)
        contexts = events.loc[events.type == "Text"]
        events = events.copy(deep=True)

        wtypes = ev.EventTypesHelper("Word")
        words = events[events.type.isin(wtypes.names)]
        events.loc[:, "sentence_char"] = np.nan

        if events["sentence"].dtype != object:
            events["sentence"] = events["sentence"].astype(object)

        events["sentence"] = ""

        sentences = []
        for context in contexts.itertuples():

            encl = find_enclosed(events, start=context.start, duration=context.duration)

            sub = events.loc[encl]
            sel = sub[sub.type.isin(wtypes.names)].index
            if not len(sel):
                raise ValueError("No word overlapping with context")
            wordseq = words.loc[sel].text.tolist()
            lang = getattr(context, "language", None)
            if not isinstance(lang, str):
                raise ValueError(f"Need language for Text field {context}")
            info = pd.DataFrame(
                _match_text_words(context.text, wordseq, language=lang), index=sel
            )

            events.loc[sel, info.columns] = info

            context_sentences = [s.to_dict() for s in _extract_sentences(events)]
            subject = getattr(context, "subject", None)
            if subject is not None:
                for s in context_sentences:
                    s["subject"] = subject
            sentences.extend(context_sentences)
        sentence_df = pd.DataFrame(
            [s for s in sentences if s["text"] != MISSING_SENTENCE]
        )
        events = pd.concat([events, sentence_df], ignore_index=True)
        events = events.reset_index(drop=True)

        words = events[events.type.isin(wtypes.names)]
        if len(words) == 0:
            return events
        ratio = sum(not s or not isinstance(s, str) for s in words.sentence) / len(words)
        if ratio > self.max_unmatched_ratio:
            max_unmatched_ratio = self.max_unmatched_ratio
            cls = self.__class__.__name__
            msg = f"Ratio of unmatched words is {ratio:.4f} on {len(words)} words "
            msg += f"while {cls}.{max_unmatched_ratio=}"
            raise RuntimeError(msg)
        return events


MISSING_SENTENCE = "# MISSING SENTENCE #"


def _extract_sentences(events) -> tp.List[ev.Sentence]:

    wtypes = ev.EventTypesHelper("Word")
    words_df = events.loc[events.type.isin(wtypes.names), :]
    sentences = []
    words: tp.List[tp.Any] = []
    eps = 1e-6
    for k, word in enumerate(words_df.itertuples(index=False)):
        if words and words[-1].timeline == word.timeline:
            if word.start < words[-1].start:
                raise ValueError(
                    f"Words are not sorted within a timeline ({words!r} and then {word!r}"
                )
        sentence_end = False
        if k == len(words_df) - 1:

            sentence_end = True
            words.append(word)
        if words:
            sentence_end |= words[-1].timeline != word.timeline
            sentence_end |= word.sentence != words[-1].sentence
            sentence_end |= word.sentence_char <= words[-1].sentence_char
            if sentence_end:
                w0 = words[0]
                text = w0.sentence
                if not (isinstance(text, str) and text):
                    text = MISSING_SENTENCE
                sentences.append(
                    ev.Sentence(
                        start=w0.start - eps,
                        duration=words[-1].start
                        + words[-1].duration
                        - w0.start
                        + 2 * eps,
                        timeline=w0.timeline,
                        text=text,
                    )
                )
                words = []
        words.append(word)
    return sentences


class AssignSentenceSplit(BaseEnhancer):
    name: tp.Literal["AssignSentenceSplit"] = "AssignSentenceSplit"
    min_duration: float | None = None
    min_words: int | None = None
    ratios: tp.Tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0
    max_unmatched_ratio: float = 0.0

    _exclude_from_cls_uid: tp.Tuple[str, ...] = ("max_unmatched_ratio",)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not sum(self.ratios) == 1:
            raise ValueError("Split ratios must sum to 1")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        if "timeline" not in events.columns:
            events["timeline"] = "#foo#"
        wtypes = ev.EventTypesHelper("Word")
        words_df = events.loc[events.type.isin(wtypes.names), :]
        ratio = sum(not s or not isinstance(s, str) for s in words_df.sentence) / len(
            words_df
        )
        if ratio > self.max_unmatched_ratio:
            max_unmatched_ratio = self.max_unmatched_ratio
            cls = self.__class__.__name__
            raise RuntimeError(
                f"Ratio of words with no sentence match is {ratio:.2f} while {cls}.{max_unmatched_ratio=}"
            )
        sentences = _extract_sentences(events)
        merged = _merge_sentences(
            sentences, min_duration=self.min_duration, min_words=self.min_words
        )
        ratios = dict(train=self.ratios[0], val=self.ratios[1], test=self.ratios[2])
        ratios = {x: y for x, y in ratios.items() if y > 0}
        if len(ratios) == 1:

            events.loc[events.type.isin(wtypes.names), "split"] = list(ratios)[0]
            if tuple(events.timeline.unique()) == ("#foo#",):
                events = events.drop("timeline", axis=1)
            return events
        splitter = splitting.DeterministicSplitter(ratios, seed=self.seed)
        undef = "undefined"
        affectations: tp.Dict[str | float, tp.Tuple[str, ...] | str] = {
            MISSING_SENTENCE: undef
        }
        groups: tp.Dict[str, tp.Set[str]] = {}

        for part in merged:
            string = "".join(s.text for s in part)
            if string not in affectations:
                affectations[string] = splitter(string)
            split = affectations[string]
            for seq in part:
                groups.setdefault(seq.text, set()).add(string)
                if affectations.setdefault(seq.text, split) != split:
                    affectations[seq.text] = undef
                    conflicts = groups[seq.text]
                    logger.warning(
                        'Sequence split "%s" set to undefined because it belongs to conflicting groups: %s',
                        seq.text,
                        conflicts,
                    )
        valid = ~(np.logical_or(events.sentence.isnull(), events.sentence == ""))
        events.loc[valid, "split"] = (
            events.loc[valid].sentence.apply(str).apply(lambda x: affectations[x])
        )
        events.loc[np.logical_and(~valid, events.type.isin(wtypes.names)), "split"] = (
            undef
        )

        if tuple(events.timeline.unique()) == ("#foo#",):
            events = events.drop("timeline", axis=1)
        return events


class AddContextToWords(BaseEnhancer):

    name: tp.Literal["AddContextToWords"] = "AddContextToWords"
    sentence_only: bool = True

    max_context_len: int | None = None

    split_field: str = "split"

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        if hasattr(events, "context"):

            events.context = events.context.fillna("").astype(str)
        wtypes = ev.EventTypesHelper("Word")
        words = events.loc[events.type.isin(wtypes.names), :]
        past_sentences: tp.List[str] = []
        last_word: tp.Any = None
        contexts = []
        desc = "Add context to words"
        worditer: tp.Iterator[ev.Word] = words.itertuples(index=False)

        sfield = self.split_field
        if sfield and sfield not in words.columns:
            raise ValueError(f"split_field {sfield!r} is not part of dataframe columns")
        for word in tqdm(worditer, total=len(words), desc=desc, mininterval=10):

            sent = word.sentence
            if not (isinstance(sent, str) and sent):

                if sfield and last_word is not None:
                    if getattr(last_word, sfield, "") != getattr(word, sfield, ""):
                        past_sentences = []

                contexts.append("")
                last_word = None
                continue
            if last_word is not None:
                if word.sentence != last_word.sentence:
                    if word.sentence_char <= last_word.sentence_char:
                        if not self.sentence_only:
                            past_sentences.append(last_word.sentence)

                        if sfield:
                            if getattr(last_word, sfield, "") != getattr(
                                word, sfield, ""
                            ):
                                past_sentences = []

            if last_word is not None:
                if last_word.timeline != word.timeline:
                    past_sentences = []

                elif word.start < last_word.start:
                    msg = "Words are not in increasing order "
                    msg += f"({word} follows {last_word})"
                    raise ValueError(msg)
            if word.sentence_char is None or np.isnan(word.sentence_char):

                contexts.append("")
                continue
            last_word = word
            last_char = float(word.sentence_char) + len(word.text)
            context = "".join(past_sentences) + word.sentence[: int(last_char)]
            if self.max_context_len is not None:
                context = " ".join(context.split(" ")[-self.max_context_len - 1 :])
            contexts.append(context)
        events.loc[events.type.isin(wtypes.names), "context"] = contexts
        return events


class RemoveMissing(BaseEnhancer):
    name: tp.Literal["RemoveMissing"] = "RemoveMissing"
    event_types: str | tp.Sequence[str] = "Word"
    field: str = "context"

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.field not in events.columns:
            msg = f"Field {self.field} not found in events dataframe, skipping RemoveMissing"
            logger.warning(msg)
            return events
        names = ev.EventTypesHelper(self.event_types).names
        data = events.loc[:, self.field]
        missing = np.logical_or(data.isnull(), data == "")
        return events.loc[np.logical_or(~events.type.isin(names), ~missing)]


class ChunkEvents(BaseEnhancer):
    name: tp.Literal["ChunkEvents"] = "ChunkEvents"
    event_type_to_chunk: tp.Literal["Sound", "Video"]
    event_type_to_use: str | None = None
    min_duration: float | None = None
    max_duration: float = np.inf

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        return chunk_events(
            events,
            self.event_type_to_chunk,
            self.event_type_to_use,
            self.min_duration,
            self.max_duration,
        )


class ExtractAudioFromVideo(BaseEnhancer):
    name: tp.Literal["ExtractAudioFromVideo"] = "ExtractAudioFromVideo"

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        video_events = events.loc[events.type == "Video"]
        if len(video_events) == 0:
            return events
        events_to_add = []
        for video_event in tqdm(
            video_events.itertuples(),
            total=len(video_events),
            desc="Extract audio from video events",
        ):
            audio_filepath = Path(video_event.filepath).with_suffix(".wav")

            video_ns_event = ev.Video.from_dict(video_event)
            audio = video_ns_event.read().audio
            if not audio:
                continue
            if not audio_filepath.exists():
                audio.write_audiofile(audio_filepath)
            audio.close()
            audio_event = video_event._replace(
                type="Sound", filepath=str(audio_filepath), frequency=pd.NA
            )

            events_to_add.append(audio_event)
        events = pd.concat([events, pd.DataFrame(events_to_add)], ignore_index=True)
        events = events.reset_index(drop=True)
        return events


@lru_cache
def parse_text(text: str, language: str = "") -> tp.Any:
    nlp = utils.get_spacy_model(language=language)
    return nlp(text)


def _merge_sentences(
    sentences: tp.List[ev.Sentence],
    min_duration: float | None = None,
    min_words: int | None = None,
) -> tp.List[tp.List[ev.Sentence]]:

    out: tp.List[tp.List[ev.Sentence]] = []
    for s in sentences:
        new = True
        if out:
            if min_duration is not None:

                new &= s.start - out[-1][0].start >= min_duration
            if min_words is not None:
                new &= sum(len(s.text.split()) for s in out[-1]) >= min_words
        if not new:

            new |= out[-1][-1].timeline != s.timeline

        if new:
            out.append([s])
        else:
            out[-1].append(s)
    return out


def _word_preproc(word: str) -> str:

    return word.lower().strip('",. ()?!\n\t')


def _match_text_words(
    text: str, words: tp.Sequence[str], language: str = ""
) -> tp.List[tp.Dict[str, tp.Any]]:

    doc = parse_text(text, language=language)
    text_words = [word for sentence in doc.sents for word in sentence]
    text_words_str = [_word_preproc(w.text) for w in text_words]
    text_match, words_match = utils.match_list(
        text_words_str, [_word_preproc(w) for w in words]
    )
    info: tp.List[tp.Dict[str, tp.Any]] = [{"word": word} for word in words]
    mkey = "text_match"
    for tm, wm in zip(text_match, words_match):
        info[wm][mkey] = tm

    todebug = []
    first: tp.Any = None
    last: tp.Any = None
    for k, i in enumerate(info):
        if mkey not in i:
            todebug.append(i)
            if k != len(info) - 1:
                continue
        if mkey in i:
            last = i
        if todebug:
            start = 0
            if first is not None:
                w = text_words[first[mkey]]
                start = w.idx + len(w)
            end = len(text)
            if last is not None:
                w = text_words[last[mkey]]
                end = w.idx
            subtext = text[start:end].lower()
            concat_words = " ".join(_word_preproc(j["word"]) for j in todebug)
            text_match, words_match = utils.match_list(subtext, concat_words)
            word_idx_charnum = [
                (k, c) for k, i in enumerate(todebug) for c in range(len(i["word"]) + 1)
            ]
            for mtext, mwordseq in zip(text_match, words_match):
                idx, charnum = word_idx_charnum[mwordseq]
                todebug[idx].setdefault("votes", []).append(start + mtext - charnum)
            for i in todebug:
                if "votes" not in i:
                    continue

                votes: tp.List[int] = i.pop("votes")
                best_bet = max(votes, key=votes.count)
                count = votes.count(best_bet)
                if count / len(i["word"]) <= 0.5:
                    logger.warning(
                        "Ignoring unreliable matching for '%s' in '%s'",
                        i["word"],
                        subtext,
                    )
                    continue

                word = i["word"]
                found = text[best_bet : best_bet + len(word)]
                if _word_preproc(word) != _word_preproc(found):
                    logger.warning(
                        "Approximately matched annotated %r with %r in text", word, found
                    )

                bounds = [j[mkey] if j is not None else None for j in [first, last]]
                sub = text_words[bounds[0] : bounds[1]]
                ind = bisect.bisect_left(sub, best_bet, key=lambda w: w.idx + len(w))
                i["sentence"] = sub[ind].sent.text_with_ws
                i["sentence_char"] = best_bet - sub[ind].sent[0].idx
            todebug = []

        if last is not None:
            first = last
            last = None

    for i in info:
        i.pop("word")
        if mkey in i:
            tword = text_words[i.pop(mkey)]
            i["sentence_char"] = tword.idx - tword.sent[0].idx
            i["sentence"] = tword.sent.text_with_ws

    prev: str | None = None
    missing = []
    for i in info:
        sent = i.get("sentence", None)
        if sent is None:
            missing.append(i)
            continue
        if prev == sent:
            for i2 in missing:
                i2["sentence"] = sent
        missing = []
        prev = sent
    return info
