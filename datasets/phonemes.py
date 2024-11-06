from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from typing import Dict, Generic, List, Literal, Optional, Protocol, Tuple, Type, TypeVar, Union
import unicodedata
import itertools
import logging
import re

import regex
from stanza import Pipeline
from stanza.resources import common
from epitran.backoff import Backoff, PuncNorm, StripDiacritics, XSampa
from epitran.backoff import panphon
from epitran import Epitran, meta
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from mashumaro.mixins.msgpack import DataClassMessagePackMixin

from allophant.language_codes import LanguageCode, LanguageCodeAny, LanguageCodeMap
from allophant import language_codes
from allophant.package_data import PHONEME_REPLACEMENTS_PATH
from allophant.phoneme_segmentation import SegmentationProcessor
from allophant.phonetic_features import TIE, IpaSentenceSegmenter
from allophant.utils import classproperty


@dataclass
class TaggedTranscription(DataClassMessagePackMixin):
    transcription: List[str]
    language: Optional[str] = None


SentenceBatch = Sequence[List[str]]
TaggedTranscriptionBatch = Sequence[List[TaggedTranscription]]


S = TypeVar("S")


@dataclass
class PhonemeTranscription(DataClassMessagePackMixin):
    words: Optional[List[str]]
    phonemes: List[List[TaggedTranscription]]

    def only_primary_script(self) -> bool:
        """
        Checks if all transcriptions for an utterance only contain phonemes
        transcribed from the primary script of the language

        :param index: The index of the utterance in the corpus

        :return: `True` if each transcription consists of only one segment in the primary script of the language
        """
        return all(len(transcription) == 1 and transcription[0].language is None for transcription in self.phonemes)

    def flattened_transcriptions(self) -> Iterator[List[str]]:
        """
        Flattens the subsegements of each transcription into single sequences of phonemes

        :return: An iterator over flattened phoneme sequences
        """
        for transcription in self.phonemes:
            yield [phoneme for subsequence in transcription for phoneme in subsequence.transcription]

    def flattened_primary_transcriptions(self) -> Iterator[List[str]]:
        """
        Keeps only transcript segments that correspond to the primary script of
        the language and flattens them into single sequences of phonemes

        :return: An iterator over flattened phoneme sequences containing only
            segments transcribed from the primary script of the languages.
            If a transcription contains no such segment, an empty list is yielded for it.
        """
        for transcription in self.phonemes:
            yield [
                phoneme
                for subsequence in transcription
                for phoneme in subsequence.transcription
                if subsequence.language is None
            ]


@dataclass
class PhonemeTranscriptionBatch:
    words: SentenceBatch
    phonemes: List[TaggedTranscriptionBatch]

    def __iter__(self) -> Iterator["PhonemeTranscription"]:
        for transcribed in itertools.zip_longest(self.words, *self.phonemes):
            yield PhonemeTranscription(transcribed[0], list(transcribed[1:]))


class HasSentence(Protocol):
    sentence: str


E = TypeVar("E", bound=HasSentence)


class PhonemeExtractor:
    _TOKENIZER_LANGUAGES_STORAGE = None

    @classproperty
    def _TOKENIZER_LANGUAGES(cls) -> Dict[str, str]:
        # Lazily initialize to avoid downloading the stanza model list on
        # import since it may not actually be used
        if cls._TOKENIZER_LANGUAGES_STORAGE is None:
            cls._TOKENIZER_LANGUAGES_STORAGE = {
                LanguageCode.from_str(code).language: code
                for code in common.list_available_languages()
                if code != "multilingual"
            }
        return cls._TOKENIZER_LANGUAGES_STORAGE

    def __init__(
        self, language_code: LanguageCodeAny, g2p_models: List["GraphemeToPhonemeModel"], token_batch_size: int = 512
    ) -> None:
        self._language_code = language_codes.to_language_code(language_code)
        # Creates a stanza tokenization pipeline if necessary
        if any(model.REQUIRES_TOKENIZER for model in g2p_models):
            # Falls back to multilingual tokenizer model if there's no tokenizer model for the given language
            self._pipeline = Pipeline(
                self._TOKENIZER_LANGUAGES[self._language_code.language],
                processors=["tokenize"],
                tokenize_no_ssplit=True,
                token_batch_size=token_batch_size,
                logging_level="ERROR",
            )
        else:
            self._pipeline = None

        self._g2p_models = g2p_models

    def sentences_to_phoneme(self, sentences: List[str]) -> PhonemeTranscriptionBatch:
        # Tokenize and filter punctuation if a tokenization pipeline is required
        words = (
            []
            if self._pipeline is None
            else [
                [
                    word.text
                    for word in tokenized.words
                    if not all(unicodedata.category(character)[0] == "P" for character in word.text)
                ]
                for tokenized in self._pipeline(sentences).sentences  # type: ignore
            ]
        )
        return PhonemeTranscriptionBatch(
            words, [model(words) if model.REQUIRES_TOKENIZER else model(sentences) for model in self._g2p_models]
        )

    def auto_batch_g2p(self, sentences: Iterable[str], batch_size: int) -> Iterator[PhonemeTranscriptionBatch]:
        current_batch = []
        for sentence in sentences:
            current_batch.append(sentence)
            if len(current_batch) == batch_size:
                yield self.sentences_to_phoneme(current_batch)
                current_batch = []

        # Handle final batch
        if current_batch:
            yield self.sentences_to_phoneme(current_batch)

    def auto_batch_g2p_transcribe(
        self, entries: Iterable[E], batch_size: int
    ) -> Iterator[Tuple[Sequence[E], PhonemeTranscriptionBatch]]:
        current_batch = []
        for entry in entries:
            current_batch.append(entry)
            if len(current_batch) == batch_size:
                yield (
                    current_batch,
                    self.sentences_to_phoneme([batch_entry.sentence for batch_entry in current_batch]),
                )
                current_batch = []

        # Handle final batch
        if current_batch:
            yield (current_batch, self.sentences_to_phoneme([batch_entry.sentence for batch_entry in current_batch]))


class PhonemeSource(Enum):
    MANUAL = "manual"
    EPITRAN = "epitran"
    ESPEAK_NG = "espeak-ng"


class GraphemeToPhonemeModel(metaclass=ABCMeta):
    REQUIRES_TOKENIZER: bool
    _PHONEME_SOURCE: PhonemeSource

    def __init__(self, language_code: LanguageCodeAny, segment_inventory: List[str]) -> None:
        self._language_code = language_codes.to_language_code(language_code)
        self._segment_inventory = segment_inventory

    @abstractmethod
    def _process_batch(self, _: Union[SentenceBatch, List[str]]) -> TaggedTranscriptionBatch:
        pass

    def __call__(self, words: Union[SentenceBatch, List[str]]) -> TaggedTranscriptionBatch:
        return self._process_batch(words)

    @classmethod
    def _initialize(cls):
        return None

    @staticmethod
    @abstractmethod
    def supports(_language_code: LanguageCodeAny) -> bool:
        return False


T = TypeVar("T", bound="GraphemeToPhonemeModel")


class PhonemeTranscriber(metaclass=ABCMeta):
    _phoneme_sources: List[PhonemeSource]
    _requires_tokenization: bool

    def supports_tokenization(self, language_code: LanguageCodeAny) -> bool:
        return (
            not self._requires_tokenization
            or language_codes.to_language_code(language_code).language in PhonemeExtractor._TOKENIZER_LANGUAGES
        )

    @abstractmethod
    def extractor(self, _language_code: LanguageCodeAny) -> PhonemeExtractor:
        pass

    @abstractmethod
    def supports(self, _language_code: LanguageCodeAny) -> bool:
        return False

    @property
    def phoneme_sources(self) -> List[PhonemeSource]:
        return self._phoneme_sources

    @property
    def num_engines(self) -> int:
        return 0


class GraphemeToPhonemeEngine(Generic[T], PhonemeTranscriber):
    def __init__(self, g2p_model: Type[T], segment_inventory: List[str], token_batch_size: int = 512) -> None:
        self._g2p_type = g2p_model
        self._phoneme_sources = [g2p_model._PHONEME_SOURCE]
        # Pre-initialization for multilingual models to avoid loading the model more than once
        self._g2p_model = g2p_model._initialize()
        self._token_batch_size = token_batch_size
        self._segment_inventory = segment_inventory
        self._requires_tokenization = g2p_model.REQUIRES_TOKENIZER

    def extractor(self, language_code: LanguageCodeAny) -> PhonemeExtractor:
        return PhonemeExtractor(
            language_code,
            [self._g2p_type(language_code, self._segment_inventory) if self._g2p_model is None else self._g2p_model],
            self._token_batch_size,
        )

    def supports(self, language_code: LanguageCodeAny) -> bool:
        return self._g2p_type.supports(language_code)

    @property
    def num_engines(self) -> Literal[1]:
        return 1


class GraphemeToPhonemeEnsemble(PhonemeTranscriber):
    def __init__(
        self,
        g2p_models: Sequence[Type[GraphemeToPhonemeModel]],
        segment_inventory: List[str],
        token_batch_size: int = 512,
    ) -> None:
        self._g2p_types = g2p_models
        self._phoneme_sources = [model._PHONEME_SOURCE for model in g2p_models]
        # Pre-initialization for multilingual models to avoid loading the model more than once
        self._g2p_models = [g2p_model._initialize() for g2p_model in g2p_models]
        self._token_batch_size = token_batch_size
        self._segment_inventory = segment_inventory
        self._requires_tokenization = any(g2p_model.REQUIRES_TOKENIZER for g2p_model in g2p_models)

    def extractor(self, language_code: LanguageCodeAny) -> PhonemeExtractor:
        return PhonemeExtractor(
            language_code,
            [
                g2p_type(language_code, self._segment_inventory) if g2p_model is None else g2p_model
                for g2p_model, g2p_type in zip(self._g2p_models, self._g2p_types)
            ],
            self._token_batch_size,
        )

    def supports(self, language_code: LanguageCodeAny) -> bool:
        return all(g2p_type.supports(language_code) for g2p_type in self._g2p_types)

    @property
    def num_engines(self) -> int:
        return len(self._g2p_types)


class ToneBackoff(Backoff):
    """Extends :py:class:`epitran.Backoff` with tone support"""

    # Derived from the original __init__ from epitran.Backoff licensed under the following terms:
    #
    # The MIT License (MIT)
    #
    # Copyright (c) 2016 David Mortensen
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    def __init__(self, lang_script_codes: Sequence[str], cedict_file: Optional[str] = None, tones: bool = True):
        """
        Creates a `ToneBackoff` instance

        :param lang_script_codes: Codes for languages to try, starting with the highest priority languages
        :param cedict_file: Path to the CC-CEdict dictionary file (necessary only when cmn-Hans or cmn-Hant are used)
        :param tones: Enables tones in transcriptions for languages that support them
        """
        self.langs = [Epitran(c, cedict_file=cedict_file, tones=tones) for c in lang_script_codes]  # type: ignore
        self.num_re = regex.compile(r"\p{Number}+")
        self.ft = panphon.featuretable.FeatureTable()
        self.xsampa = XSampa()
        self.puncnorm = PuncNorm()
        self.dias = [StripDiacritics(c) for c in lang_script_codes]


class EpitranG2P(GraphemeToPhonemeModel):
    REQUIRES_TOKENIZER = True
    _PHONEME_SOURCE = PhonemeSource.EPITRAN
    _ENGLISH = "eng-Latn"

    def __init__(self, language_code: LanguageCodeAny, segment_inventory: List[str], tones: bool = False) -> None:
        super().__init__(language_code, segment_inventory)
        code = self._language_code.alpha3
        # Workaround since `get_default_mode` doesn't contain English
        if code == "eng":
            mode = self._ENGLISH
        else:
            mode = meta.get_default_mode(self._language_code.alpha3)
            if mode is None:
                raise ValueError(f"Language {self._language_code.alpha3!r} not supported")
        # Loads model for the first script specified in the supported list
        if not tones or mode.split("-")[1] == "Latn":
            self._epitran = Epitran(mode, tones=tones)
        else:
            # Backs off to English G2P to handle words in the Latin script for non-latin scripts
            self._epitran = ToneBackoff([mode, self._ENGLISH], tones=tones)
        self._segmenter = IpaSentenceSegmenter(self._segment_inventory)

    def _process_batch(self, words: SentenceBatch) -> TaggedTranscriptionBatch:
        return [
            [TaggedTranscription(transcription)]
            for transcription in self._segmenter.lossy_segment(
                [phoneme for word in sentence for phoneme in self._epitran.trans_list(word) if word]
                for sentence in words
            )
        ]

    @staticmethod
    def supports(language_code: LanguageCodeAny) -> bool:
        code = language_codes.to_language_code(language_code).alpha3
        # NOTE: Workaround since supported_lang doesn't include English even with the external library installed
        if code == "eng":
            return True
        return meta.supported_lang(code)


class _EspeakProcessor(SegmentationProcessor):
    _REPLACEMENT_STORAGE = None

    @classproperty
    def _REPLACEMENTS(cls) -> Dict[str, str]:
        """Mappings for unsupported IPA and noise from Espeak NG"""
        if cls._REPLACEMENT_STORAGE is not None:
            return cls._REPLACEMENT_STORAGE
        with PHONEME_REPLACEMENTS_PATH.open("r", encoding="utf-8") as file:
            cls._REPLACEMENT_STORAGE = json.load(file)
            return cls._REPLACEMENT_STORAGE

    def pre_process(self, phoneme: str, phoneme_iterator: Iterator[str]) -> str:
        # Processing for palatalized vowel segments, which are incorrectly split by phonemizer
        if phoneme == "ʲ":
            return phoneme + next(phoneme_iterator)
        # Replace unsupported IPA characters and noise
        phoneme = self._REPLACEMENTS.get(phoneme, phoneme)
        # Some languages contain ties even with ties disabled in phonemizer
        # Remove any ties for maximal compatibility - selected espeak settings shouldn't include them anyway
        return phoneme.replace(TIE, "")


class EspeakNg(GraphemeToPhonemeModel):
    REQUIRES_TOKENIZER = False
    _PHONEME_SOURCE = PhonemeSource.ESPEAK_NG
    _SUPPORTED_STORAGE = None

    @classproperty
    def _SUPPORTED(cls) -> LanguageCodeMap:
        # Lazily initialize since espeak-ng has to be manually installed on the
        # system first and is otherwise not needed
        if cls._SUPPORTED_STORAGE is None:
            # NOTE: Fixes incorrect script tags in Espeak-ng version 1.51
            fixed_tags = {"chr-US-Qaaa-x-west": "chr-Qaaa-US-x-west", "en-us-nyc": "en-us-x-nyc"}
            cls._SUPPORTED_STORAGE = LanguageCodeMap(
                [fixed_tags.get(code, code) for code in EspeakBackend.supported_languages().keys()],
                {"fr": "fr-fr", "en": "en-us"},
            )
        return cls._SUPPORTED_STORAGE

    def __init__(
        self,
        language_code: LanguageCodeAny,
        segment_inventory: List[str],
        with_stress: bool = False,
    ) -> None:
        super().__init__(language_code, segment_inventory)
        language = self._SUPPORTED[language_codes.to_language_code(language_code).language]
        # This is a workaround since word mismatches are otherwise still printed even though they're set to "ignore"
        null_logger = logging.getLogger("espeak-null")
        null_logger.addHandler(logging.NullHandler())
        self._espeak_main_language = language
        self._backend = EspeakBackend(
            language, language_switch="keep-flags", with_stress=with_stress, logger=null_logger
        )
        self._segmenter = IpaSentenceSegmenter(self._segment_inventory, _EspeakProcessor())

    def _process_phonemes(self, sentences: List[str]) -> Iterator[List[TaggedTranscription]]:
        for phonemes in self._backend.phonemize(
            sentences,
            Separator(word="", phone=" "),  # Only whitespace between phones
        ):
            subsequences = []
            flag = None
            # Split by language flags
            for subsequence in re.split(r"\s*(\(\w+)\)\s*", phonemes):
                if subsequence.startswith("("):
                    new_flag = subsequence[1:]
                    flag = new_flag if new_flag != self._espeak_main_language else None
                # Ignores empty subsequences that can happen if a language flag occurs at the start
                elif subsequence:
                    subsequences.append(
                        TaggedTranscription(list(self._segmenter.lossy_segment([subsequence.split()]))[0], flag)
                    )

            yield subsequences

    def _process_batch(self, sentences: List[str]) -> TaggedTranscriptionBatch:
        return list(self._process_phonemes(sentences))

    @classmethod
    def supports(cls, language_code: LanguageCodeAny) -> bool:
        return language_code in cls._SUPPORTED


class G2PEngineType(Enum):
    EPITRAN = "epitran"
    ESPEAK_NG = "espeak-ng"

    def model(self) -> Type[GraphemeToPhonemeModel]:
        match self:
            case self.EPITRAN:
                return EpitranG2P
            case self.ESPEAK_NG:
                return EspeakNg
            case _:
                raise NotImplementedError(f"Unsupported Engine Type: {self}")
