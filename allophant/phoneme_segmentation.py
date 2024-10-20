from collections.abc import Iterable, Iterator, Sequence
import itertools
import logging
from logging import FileHandler, Filter, LogRecord, Logger
from typing import List, TypeVar
import unicodedata
import re

from regex import regex

from allophant.phonemes import IpaSegmenter
from allophant.utils import AnyPath


TONES = [
    "꜀",
    "꜁",
    "꜂",
    "꜃",
    "꜄",
    "꜅",
    "꜆",
    "꜇",
    "꜈",
    "꜉",
    "꜊",
    "꜋",
    "꜌",
    "꜍",
    "꜎",
    "꜏",
    "꜐",
    "꜑",
    "꜒",
    "꜓",
    "꜔",
    "꜕",
    "꜖",
    "ꜗ",
    "ꜘ",
    "ꜙ",
    "ꜚ",
    "ꜛ",
    "ꜜ",
    "ꜝ",
    "ꜞ",
    "ꜟ",
]


TONE_PATTERN = f"[{''.join(TONES)}]"


def filter_tones(inventory: Iterable[str]) -> Iterator[str]:
    for phoneme in inventory:
        if not re.search(phoneme, TONE_PATTERN):
            yield phoneme


VOWELS = {
    "a",
    "e",
    "i",
    "o",
    "u",
    "y",
    "æ",
    "ø",
    "œ",
    "ɐ",
    "ɑ",
    "ɒ",
    "ɔ",
    "ɘ",
    "ə",
    "ɚ",
    "ɛ",
    "ɜ",
    "ɝ",
    "ɞ",
    "ɤ",
    "ɨ",
    "ɪ",
    "ɯ",
    "ɵ",
    "ɶ",
    "ʉ",
    "ʊ",
    "ʌ",
    "ʏ",
}


class _MissingPhonemeFilter(Filter):
    _DEFAULT_LANGUAGE = "unknown"

    def __init__(self):
        self._seen = set()
        self.language = self._DEFAULT_LANGUAGE

    def filter(self, record: LogRecord) -> bool:
        phoneme = record.msg
        entry = (self.language, phoneme)
        if entry in self._seen:
            return False
        self._seen.add(entry)
        record.msg = f"Missing phoneme segment: ({self.language}) {phoneme!r}"
        return True

    def reset(self) -> None:
        self._seen = set()
        self.language = self._DEFAULT_LANGUAGE


class SegmentationLogger:
    def __init__(self) -> None:
        self._reset_steps = 0
        self._segmentation_logger = logging.getLogger("allophant.phoneme_segmentation")
        self._segmentation_logger.setLevel(logging.WARNING)
        self._filter = _MissingPhonemeFilter()
        self._segmentation_logger.addFilter(self._filter)

    def add_file_handler(self, log_path: AnyPath) -> None:
        self._segmentation_logger.addHandler(FileHandler(log_path, mode="w", encoding="utf-8"))

    def context_language(self, language: str) -> None:
        self._filter.language = language

    @property
    def log(self) -> Logger:
        return self._segmentation_logger

    def reset(self) -> None:
        if self._reset_steps > 0:
            self._reset_steps -= 1
        else:
            self._filter.reset()

    def suppress_resets(self, count: int = 1) -> None:
        self._reset_steps += count


SEGMENTATION_LOGGER = SegmentationLogger()


I = TypeVar("I", bound=int | Sequence[int])


class SegmentationProcessor:
    def pre_process(self, phoneme: str, _phoneme_iterator: Iterator[str]) -> str:
        return phoneme

    def post_process(self, sub_segments: List[str], _phoneme_iterator: Iterator[str]) -> Sequence[str]:
        return sub_segments


class IpaSentenceSegmenter:
    def __init__(self, dictionary: List[str], processor: SegmentationProcessor | None = None) -> None:
        self._segmenter = IpaSegmenter(dictionary)
        self._processor = SegmentationProcessor() if processor is None else processor

    @property
    def word_segmenter(self) -> IpaSegmenter:
        return self._segmenter

    def __call__(self, phonetic_sentences: Iterable[List[str]]) -> Iterator[List[str]]:
        return (self._segmenter.segment_words_checked(sentence) for sentence in phonetic_sentences)

    def lossy_segment(self, phonetic_sentences: Iterable[List[str]]) -> Iterator[List[str]]:
        for sentence in phonetic_sentences:
            sentence_phonemes = []
            phoneme_iterator = iter(sentence)
            for phoneme in phoneme_iterator:
                pre_processed = self._processor.pre_process(phoneme, phoneme_iterator)
                sub_segments = self._processor.post_process(
                    self._segmenter.segment(pre_processed),
                    phoneme_iterator,
                )
                sentence_phonemes.extend(sub_segments)
                if len(sub_segments) != 1:
                    if "".join(sub_segments) != pre_processed:
                        SEGMENTATION_LOGGER.log.warning(pre_processed + " (Missing sub-segment when split)")
                    else:
                        SEGMENTATION_LOGGER.log.warning(pre_processed)

            yield sentence_phonemes


def _is_mark(character: str) -> bool:
    category = unicodedata.category(character)
    return category.endswith("m") or category == "Sk" or category.startswith("M")


def is_vowel_only_segment(segment: str) -> bool:
    return all(character in VOWELS for character in segment if not _is_mark(character))


def base_phonemes(segment: str) -> Iterator[str]:
    # Any symbols that are not marks or diacritics.
    # Should leave only phoneme base characters
    return (character for character in segment if not _is_mark(character))


def complex_with_vowel(base_phonemes: Iterable[str]) -> bool:
    count = 0
    has_vowel = False
    for character in base_phonemes:
        if character in VOWELS:
            has_vowel = True
        count += 1

    # Any segment with more than one base phoneme and a vowel
    return has_vowel and count > 1


def is_multi_vowel(base_phonemes: Iterable[str]) -> bool:
    count = 0
    for character in base_phonemes:
        # Not a multi vowel segment if there's a symbol that is not a
        # diacritic, mark or a vowel
        if character not in VOWELS:
            return False
        count += 1

    # Only multi-vowel if there is more than one
    return count > 1


def split_complex_segment(segment: str) -> List[str]:
    vowels = []
    prefix = ""
    # Get all grapheme clusters - including vowels with attached combining characters
    for grapheme in regex.finditer(r"\X", segment):
        # Skip IPA marks that count as separate
        grapheme = grapheme.group()
        if len(grapheme) == 1 and _is_mark(grapheme):
            if not vowels:
                prefix += grapheme
            else:
                vowels[len(vowels) - 1] += grapheme
        else:
            vowels.append(prefix + grapheme)
            prefix = ""

    # Handle combining mark only segments, this mostly affects tone contours, keeping them together
    if prefix:
        vowels.append(prefix)

    return vowels


def split_all_complex_segments(segments: Iterable[str]) -> Iterator[str]:
    return itertools.chain.from_iterable(map(split_complex_segment, segments))


def split_phoneme_segment(segment: str) -> List[List[str]]:
    return [split_complex_segment(subsegment) for subsegment in segment.split("|")]
