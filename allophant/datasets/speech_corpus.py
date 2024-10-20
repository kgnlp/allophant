from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import itertools
from os import path
from typing import Any, ClassVar, Dict, Generic, Iterator, List, Literal, Protocol, Tuple, TypeVar, runtime_checkable
from pandas.io.formats.format import math

from torch import Tensor
import torchaudio
from mutagen import File
from torchaudio.transforms import Resample

from allophant.datasets.phonemes import PhonemeTranscription, PhonemeSource
from allophant.phonetic_features import LanguageInventories
from allophant import utils
from allophant.utils import AnyPath


@dataclass
class LanguageInfo:
    """
    ISO 6393 language code and phoneme inventory for a language

    :param code: ISO 6393 language code
    :param phoneme_inventory: A list of IPA segments that form that languages phoneme inventory
    :param phoneme_mappings: Inventory mappings from Grapheme to Phoneme engines used to produce transcriptions
    """

    code: str

    phoneme_inventory: List[str]
    phoneme_mappings: List[Dict[str, List[str]]]


class TranscribedUtterance(Protocol):
    """A transcribed utterance with a unique ID"""

    utterance_id: str


@runtime_checkable
class PhoneticallyTranscribedUtterance(TranscribedUtterance, Protocol):
    """An utterance with a phonemic transcription words and the original orthographic forms for each word"""

    phonemes: PhonemeTranscription


@runtime_checkable
class PhoneticallySegmentedUtterance(TranscribedUtterance, Protocol):
    "An utterance with a phonemic transcription"
    phonemes: List[str]


T = TypeVar("T", bound=TranscribedUtterance)
I = TypeVar("I", bound=LanguageInfo)


@dataclass
class LanguageData(Generic[I, T]):
    """
    Metadata and transcribed utterances for a monolingual corpus

    :param info: Language-specific metadata
    :param transcribed_samples: A list of transcribed utterances for a language
    """

    info: I
    transcribed_samples: List[T]

    def __getitem__(self, index: int) -> T:
        return self.transcribed_samples[index]

    def __len__(self) -> int:
        return len(self.transcribed_samples)


@dataclass
class _LanguageEntry(Generic[I]):
    """
    Monolingual subset range in a `MultilingualCorpus`

    :param info: Language-specific metadata
    :param offset: Start offset of the monolingual subset in the multilingual utterance list
    :param num_utterances: The number of utterances available for monolingual subset
    """

    info: I
    offset: int
    num_utterances: int


@dataclass
class IndexedEntry(Generic[T]):
    """
    Represents an entry in a `MultilingualCorpus` with a language ID

    :param language_id: Language ID within the `MultilingualCorpus` that the entry belongs to
    :param entry: An entry in the `MultilingualCorpus` - usually an utterance
    """

    language_id: int
    entry: T


@dataclass
class AudioInfo:
    """
    Container for audio metadata

    :param sample_rate: The sample rate
    :param bits_per_sample: The bit depth
    :param num_channels: The number of audio channels
    """

    sample_rate: int = 16_000
    bits_per_sample: int = 16
    num_channels: int = 1

    @classmethod
    def none(cls):
        """Creates a fallback instance with every parameter set to `0`"""
        return cls(0, 0, 0)


class MultilingualCorpus(Generic[I, T]):
    """
    Generic interface for a multilingual corpus that allows both access to monolingual subsets and random access to utterances across languages
    """

    UTTERANCE_TYPE = TranscribedUtterance
    _SAMPLE_RATES = [44100]

    def __init__(
        self,
        base_directory: AnyPath,
        language_data: Iterable[LanguageData[I, T]],
        audio_subdirectory: str,
        audio_extension: str,
        limits: Dict[str, int] | int | None = None,
        resample: int | None = None,
        phoneme_sources: Sequence[PhonemeSource] | None = None,
        meta_data: Dict[str, Any] | None = None,
    ) -> None:
        """
        Creates a new multilingual corpus that adheres to the format of a base
        directory with a single subdirectory containing all audio files with
        the same extension. For construction, an iterable has to be provided which
        yields `LanguageData` instances for all or a subset of the languages
        contained in the corpus.

        :param base_directory: Base directory of the multilingual corpus
        :param language_data: An iterable yielding `LanguageData` instances which
            contain metadata and a list of utterances for monolingual subsets from the corpus.
            If multiple `LanguageData` instances are provided, a `ValueError` will be raised.
        :param audio_subdirectory: Base directory of the multilingual corpus
        :param audio_extension: File extension of the audio files contained in the dataset
        :param limits: Either a map of language to integer to set an utterance
            limit per language or a single integer to set an utterance limit globally
        :param resample: If given, all utterances will be resampled to the given sample rate when they are loaded
        :param phoneme_sources: List of sources for phoneme transcriptions available in this corpus. By default,
            it is assumed that there is one phoneme source which was manual transcribed.
            This corresponds to :py:attr:`~allophant.dataset.phonemes.PhonemeTranscriptionType.MANUAL`
        :param meta_data: An optional dictionary of key value pairs
        """
        self._meta_data = meta_data
        # Absolutize data path for later re-use from another location
        self._base_directory = path.abspath(path.expanduser(base_directory))
        self._audio_extension = audio_extension
        self._audio_subdirectory = audio_subdirectory
        self._phoneme_sources = [PhonemeSource.MANUAL] if phoneme_sources is None else phoneme_sources
        self._languages: Dict[str, _LanguageEntry] = {}
        self._language_ids: Dict[str, int] = {}
        self._language_list: List[str] = []
        self._utterances: List[IndexedEntry[T]] = []
        self._num_utterances = 0

        if resample is not None:
            self._resamplers = {sample_rate: Resample(sample_rate, resample) for sample_rate in self._SAMPLE_RATES}
        else:
            self._resamplers = None

        for i, language in enumerate(language_data):
            language_code = language.info.code
            limit = utils.global_or_local_limit(limits, language_code)
            num_utterances = len(language.transcribed_samples)
            if limit is not None:
                num_utterances = min(num_utterances, limit)

            if language_code in self._language_ids:
                raise ValueError(f"Duplicate language data for code: {language_code}")
            self._languages[language_code] = _LanguageEntry(language.info, self._num_utterances, num_utterances)
            self._language_ids[language_code] = i
            self._language_list.append(language_code)

            # Add utterances for the current language, optionally limited
            self._utterances.extend(
                IndexedEntry(i, transcription)
                for transcription, _ in zip(language.transcribed_samples, utils.limit_indices(limit))
            )
            self._num_utterances += num_utterances

        # Assumes audio sampling rate (if no resampling is used), channels and bit depth are same for every audio file in the corpus
        self._audio_info = AudioInfo.none()
        if resample is not None:
            self._audio_info.sample_rate = resample

    @property
    def meta_data(self) -> Dict[str, Any] | None:
        return self._meta_data

    @property
    def audio_info(self) -> AudioInfo:
        """
        Returns audio information about the `MultilingualCorpus`.
        Currently, only the sample rate is returned and only if resampling is enabled,
        since the original sample rates may differ between audio files
        """
        return self._audio_info

    @property
    def phoneme_sources(self) -> Sequence[PhonemeSource]:
        return self._phoneme_sources

    @staticmethod
    def empty() -> "MultilingualCorpus":
        """Creates a dummy empty corpus"""
        return MultilingualCorpus("", [], "", "")

    def path_from_utterance(self, language: str, utterance_id: str) -> str:
        """
        Retrieves the path to an utterance with the given language and ID from the corpus

        :param language: language code of the utterance in the format used by the corpus
        :param utterance_id: The utterance_id in the format used by the corpus

        :return: The path to the requested utterance
        """
        return path.join(
            self._base_directory, language, self._audio_subdirectory, f"{utterance_id}.{self._audio_extension}"
        )

    def path_for(self, indexed_transcription: IndexedEntry[T]) -> str:
        """
        Retrieves the path to an utterance for the given indexed transcription from this corpus

        :param indexed_transcription: An `IndexedEntry` from the corpus

        :return: The path to the requested utterance
        """
        return self.path_from_utterance(
            self._language_list[indexed_transcription.language_id], indexed_transcription.entry.utterance_id
        )

    def path(self, index: int) -> str:
        """
        Retrieves the path to an utterance for the given utterance index

        :param index: Index of an utterance in the multilingual corpus

        :return: The path to the requested utterance
        """
        return self.path_for(self._utterances[index])

    def audio_from_utterance(self, language: str, utterance_id: str) -> Tuple[Tensor, int]:
        """
        Retrieves the audio and original sample rate of an utterance with the given language and ID from the corpus

        :param language: language code of the utterance in the format used by the corpus
        :param utterance_id: The utterance_id in the format used by the corpus

        :return: A tuple of (possibly resampled) audio samples as a tensor and the original sample rate before resampling
        """
        samples, sample_rate = torchaudio.load(self.path_from_utterance(language, utterance_id))  # pyright: ignore
        return (samples if self._resamplers is None else self._resamplers[sample_rate](samples), sample_rate)

    def audio_for(self, indexed_transcription: IndexedEntry[T]) -> Tuple[Tensor, int]:
        """
        Retrieves the audio and original sample rate of an utterance for the given indexed transcription from this corpus

        :param indexed_transcription: An `IndexedEntry` from the corpus

        :return: A tuple of (possibly resampled) audio samples as a tensor and the original sample rate before resampling
        """
        return self.audio_from_utterance(
            self._language_list[indexed_transcription.language_id], indexed_transcription.entry.utterance_id
        )

    def audio(self, index: int) -> Tuple[Tensor, int]:
        """
        Retrieves the audio and original sample rate of an utterance for the given utterance index

        :param index: Index of an utterance in the multilingual corpus

        :return: A tuple of (possibly resampled) audio samples as a tensor and the original sample rate before resampling
        """
        return self.audio_for(self._utterances[index])

    @property
    def languages(self) -> List[str]:
        return self._language_list

    def utterance_languages(self) -> Iterator[str]:
        """
        Generates language codes for each utterance in the corpus

        :return: Language codes for each utterance in the corpus
        """
        # Repeat language codes for all languages in order
        # - only works since dict is guaranteed to be ordered in newer Python versions
        for code, language_entry in self._languages.items():
            for _ in range(language_entry.num_utterances):
                yield code

    def monolingual_index_range(self, language_code: str) -> range:
        """
        Retrieves the range from the start to the stop index of utterance in
        the given language

        :param language_code: An ISO 6393 language code

        :return: The utterance index range for `language_code`
        """
        language_entry = self._languages[language_code]
        offset = language_entry.offset
        return range(offset, offset + language_entry.num_utterances)

    def _monolingual_islice(self, language_entry: _LanguageEntry) -> Iterator[IndexedEntry[T]]:
        offset = language_entry.offset
        return itertools.islice(self._utterances, offset, offset + language_entry.num_utterances)

    def monolingual_subset(self, language_code: str) -> LanguageData[I, T]:
        """
        Retrieves the monolingual subcorpus with the given language code

        :param language_code: An ISO 6393 language code

        :return: The monolingual subcorpus for `language_code`
        """
        language_entry = self._languages[language_code]
        return LanguageData(
            language_entry.info, [utterance.entry for utterance in self._monolingual_islice(language_entry)]
        )

    def subset(self, language_codes: Iterable[str]) -> "MultilingualCorpus[I, T]":
        """
        Creates a multilingual subcorpus from the given language codes

        :param language_codes: An iterable of ISO 6393 language codes

        :return: A subcorpus containing only the languages included in `language_codes`
        """
        return MultilingualCorpus(
            self._base_directory,
            (self.monolingual_subset(language_code) for language_code in language_codes),
            self._audio_subdirectory,
            self._audio_extension,
        )

    def shared_inventory(self) -> List[str]:
        """
        Collects the shared phoneme inventory from all languages in the corpus.
        The resulting IPA segments are returned in sorted order.

        :return: The shared phoneme inventory of the corpus in sorted order
        """
        # Sorted for more determinism
        return sorted(
            {
                phoneme
                for language_entry in self._languages.values()
                for phoneme in language_entry.info.phoneme_inventory
            }
        )

    def language_id_inventories(self) -> LanguageInventories:
        """
        Creates a mapping between corpus-specific language IDs and phoneme
        inventories, including the language codes for each ID

        :return: A mapping from language IDs to phoneme inventories and a list of language codes corresponding to the IDs
        """
        return LanguageInventories(
            {
                self.language_id(language_entry.info.code): language_entry.info.phoneme_inventory
                for language_entry in self._languages.values()
            },
            self.languages,
        )

    def language_id(self, language: str) -> int:
        """
        Retrieves the corpus-specific language ID for a language

        :param language: An ISO 6393 language code

        :return: The ID of the given language in the corpus
        """
        return self._language_ids[language]

    def language(self, language_id: int) -> str:
        """
        Retrieves the ISO 6393 language code for a corpus-specific language ID

        :param language_id: A language ID from this corpus

        :return: The ISO6393 language code corresponding to the `language_id` in this corpus
        """
        return self._language_list[language_id]

    def inventory(self, language: str) -> List[str]:
        """
        Retrieves the phoneme inventory for a language

        :param language: An ISO 6393 language code

        :return: The phoneme inventory for the given language
        """
        return self._languages[language].info.phoneme_inventory

    def _language_lengths(self, language_entry: _LanguageEntry) -> Iterator[int]:
        """
        Calculates the number of samples after resampling for every utterance in the `language_entry`

        :param language_entry: A monolingual subset for which to compute sample counts

        :return: A generator of sample counts for each utterance in `language_entry`
        """
        sample_rate = self._audio_info.sample_rate
        offset = language_entry.offset
        for utterance_path in self._monolingual_islice(language_entry):
            audio_path = self.path_for(utterance_path)
            audio_file = File(audio_path)
            if audio_file is None:
                raise ValueError(f"Could not guess audio file type for {audio_path!r}")
            file_info = audio_file.info
            # Get original sample count
            original_sample_count = file_info.length * file_info.sample_rate
            # Get target samples after resampling - adapted from _apply_sinc_resample_kernel
            # in https://github.com/pytorch/audio/blob/7ac3e2e237e443baf91dfbf9893fca114c1c6001/torchaudio/functional/functional.py
            yield int(math.ceil((sample_rate * original_sample_count) / file_info.sample_rate))

    def read_lengths(self) -> Iterator[Tuple[str, Iterator[int]]]:
        """
        Calculates the number of samples after resampling for every utterance in the corpus one language at a time

        :return: A generator of ISO 6393 language code and sample count generator tuples
        """
        return ((code, self._language_lengths(language_entry)) for code, language_entry in self._languages.items())

    def __getitem__(self, index: int) -> IndexedEntry[T]:
        return self._utterances[index]

    def __iter__(self) -> Iterator[LanguageData[I, T]]:
        for language in self._language_list:
            yield self.monolingual_subset(language)

    def __len__(self) -> int:
        return self._num_utterances

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._base_directory!r}, {self._languages!r})"


@dataclass
class SplitMetaData:
    train: Dict[str, Any] | None
    dev: Dict[str, Any] | None
    test: Dict[str, Any] | None


C = TypeVar("C", bound=MultilingualCorpus)


@dataclass
class MultilingualSplits(Generic[C]):
    """
    Represents train, dev and test splits of a corpus

    :param train: A subset used for training
    :param dev: A subset used for validation
    :param test: A held-out test set
    """

    SPLIT_NAMES: ClassVar[Tuple[Literal["train"], Literal["dev"], Literal["test"]]] = ("train", "dev", "test")

    train: C
    dev: C
    test: C

    def audio_info(self) -> AudioInfo:
        """
        Retrieves the `AudioInfo` for the first non-empty split or :func:`AudioInfo.none` if all splits are empty.

        :return: The `AudioInfo` for the first non-empty split
        """
        no_info = AudioInfo.none()
        return next(
            (split.audio_info for split in (self.train, self.dev, self.test) if split.audio_info != no_info), no_info
        )

    def meta_data(self) -> SplitMetaData:
        return SplitMetaData(self.train.meta_data, self.dev.meta_data, self.test.meta_data)

    @classmethod
    def single(cls, corpus: C, split: Literal["train"] | Literal["dev"] | Literal["test"] = "test"):
        """
        Creates `MultilingualSplits` with only a single split

        :param corpus: The `MultilingualCorpus` to be used as a split
        :param split: The type of split the `corpus` should be assigned to

        :return: `MultilingualSplits` with a single split where `corpus` is assigned to `split`
        """
        splits = {split: corpus}
        corpus_type = corpus.__class__
        for data_split in ("train", "dev", "test"):
            if data_split not in splits:
                splits[data_split] = corpus_type.empty()  # type: ignore

        return cls(**splits)

    def __iter__(self) -> Iterator[C]:
        # Yields splits sequentially
        yield self.train
        yield self.dev
        yield self.test
