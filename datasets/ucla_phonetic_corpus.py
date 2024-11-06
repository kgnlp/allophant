from collections.abc import Sequence
from dataclasses import dataclass
import os
from os import path
from typing import Dict, Iterable, Iterator, List

from tqdm import tqdm

from allophant.datasets import MultilingualCorpus
from allophant.datasets.phonemes import LanguageCodeAny
from allophant import language_codes
from allophant.datasets.speech_corpus import LanguageData, LanguageInfo, PhoneticallySegmentedUtterance
from allophant.utils import AnyPath


@dataclass
class Transcription:
    """
    A transcription from the :py:class:`UCLAPhoneticCorpus`

    :param utterance_id: The ID of the utterance
    :param raw: The narrow phone annotations of the utterance
    :param phonemes: The segmented and normalized transcription from the utterance
    """

    utterance_id: str
    raw: str
    phonemes: List[str]


@dataclass
class UCLALanguageData(LanguageData[LanguageInfo, Transcription]):
    """
    Metadata and transcribed utterances for a monolingual subset of the :py:class:`UCLAPhoneticCorpus`

    :param info: Language-specific metadata
    :param transcribed_samples: A list of transcribed utterances for a language
    :param id_map: A map from utterance IDs to generated utterance indices to allow access by ID
    """

    id_map: Dict[str, int]


_INVENTORY_FILE = "inventory"
_RAW_FILE = "raw"
_TEXT_FILE = "text"


def _load_languages(
    directory: AnyPath, languages: Sequence[LanguageCodeAny] | None = None, progress_bar: bool = False
) -> Iterator[UCLALanguageData]:
    """
    Loads transcriptions and inventories from the full
    :py:class:`UCLAPhoneticCorpus` or a subset with fewer languages

    :param directory: Base directory of the corpus
    :param languages: A language subset or `None` to load the full corpus
    :param progress_bar: If true, loading progress is displayed as a progress bar

    :return: A generator over monolingual subsets of the :py:class:`UCLAPhoneticCorpus`
    """
    if languages is None:
        language_generator = os.listdir(directory)
        total = len(language_generator)
    else:
        language_generator = (language_codes.to_language_code(code).alpha3 for code in languages)
        total = len(languages)

    for language in (
        tqdm(language_generator, total=total, unit=" languages", desc="Loading transcriptions")
        if progress_bar
        else language_generator
    ):
        language_directory = path.join(directory, language)

        with open(path.join(language_directory, _INVENTORY_FILE), "r", encoding="utf-8") as file:
            # Sort inventories for consistency with other processing functions
            info = LanguageInfo(language, sorted(phoneme for phoneme, _ in map(str.split, file)), [])

        with (
            open(path.join(language_directory, _RAW_FILE), "r", encoding="utf-8") as raw_file,
            open(path.join(language_directory, _TEXT_FILE), "r", encoding="utf-8") as text_file,
        ):
            transcriptions = []
            id_map = {}
            for i, (raw_line, text_line) in enumerate(zip(raw_file, text_file)):
                raw_id, raw = raw_line.split()
                text_columns = text_line.split()
                assert raw_id == text_columns[0], "Mismatch between raw and text file IDs"
                transcriptions.append(Transcription(raw_id, raw, text_columns[1:]))
                id_map[raw_id] = i

        yield UCLALanguageData(info, transcriptions, id_map)


class UCLAPhoneticCorpus(MultilingualCorpus):
    """
    An interface to the utterances transcriptions and inventories contained in the UCLA Phonetic Corpus introduced by Li et al. (2021)

    .. references::
        Li, X., Mortensen, D. R., Metze, F., & Black, A. W. (2021, June).
        Multilingual phonetic dataset for low resource speech recognition.
        In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and
        Signal Processing (ICASSP) (pp. 6958-6962). IEEE.
    """

    UTTERANCE_TYPE = PhoneticallySegmentedUtterance

    _SAMPLE_RATES = [44100, 48000]
    _AUDIO_DIRECTORY = "audio"

    def __init__(self, base_directory: AnyPath, languages: Iterable[LanguageData], resample: int | None = None) -> None:
        """
        Creates a new UCLA Phonetic Corpus from the given data and base directory

        :param base_directory: Base directory of the corpus
        :param language_data: An iterable yielding `LanguageData` instances which
            contain metadata and a list of utterances for monolingual subsets from the corpus.
            If multiple `LanguageData` instances are provided, a `ValueError` will be raised.
        :param resample: If given, all utterances will be resampled to the given sample rate when they are loaded
        """
        super().__init__(base_directory, languages, self._AUDIO_DIRECTORY, "wav", resample=resample)

    @classmethod
    def load(
        cls,
        directory: AnyPath,
        resample: int | None = None,
        languages: Sequence[LanguageCodeAny] | None = None,
        progress_bar: bool = False,
    ):
        """
        Loads (a subset of) the UCLA Phonetic Corpus from the given directory

        :param directory: Base directory of the corpus
        :param resample: If given, all utterances will be resampled to the given sample rate when they are loaded
        :param languages: A language subset or `None` to load the full corpus
        :param progress_bar: If true, loading progress is displayed as a progress bar
        """
        return cls(directory, _load_languages(directory, languages, progress_bar), resample)

    def __str__(self) -> str:
        return (
            f"UCLA Phonetic Corpus containing {len(self._languages)} languages with {self._num_utterances} utterances"
        )
