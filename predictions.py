from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib.metadata import version
from io import TextIOWrapper
from os import path
import typing
from typing import Any, BinaryIO, ClassVar, Dict, Generic, List, Tuple, Type, TypeVar
import gzip

import torch
from torch import Tensor
from torchaudio.models import decoder
from torchaudio.models.decoder import CTCDecoder, CTCHypothesis

from marshmallow import Schema
from mashumaro.types import SerializationStrategy
from mashumaro.mixins.json import DataClassJSONMixin
import marshmallow_dataclass

import allophant
from allophant import phonemes
from allophant.phonemes import Action
from allophant import utils
from allophant.config import FeatureSet
from allophant.phonetic_features import ArticulatoryAttributes, PhonemeIndexer, PhoneticIndexerState
from allophant.utils import MarshmallowDataclassLoadMixin, PathOrFile, PathOrFileBinary


# Major, Minor, Patch
CURRENT_FORMAT_VERSION = (1, 1, 0)
SUPPORTED_VERSIONS = [CURRENT_FORMAT_VERSION]


@marshmallow_dataclass.add_schema
@dataclass
class PredictionMetaData(MarshmallowDataclassLoadMixin):
    Schema: ClassVar[Type[Schema]]

    prediction_arguments: str
    corpus_type: str
    languages: List[str]
    feature_set: FeatureSet
    indexer_state: PhoneticIndexerState
    classifiers: List[str]
    label_inventories: Dict[str, List[str]] | None = None
    package_version: str = version(allophant.__package__)
    format_version: Tuple[int, int, int] = CURRENT_FORMAT_VERSION


@dataclass
class UtterancePrediction(DataClassJSONMixin):
    language: str
    utterance_id: str
    predictions: Dict[str, List[List[str]]]
    labels: List[List[str]] | None = None


def levensthein_substitutions(expected: List[str], actual: List[str]) -> List[Tuple[Action, str, str]]:
    return phonemes.to_substitutions(expected, actual, phonemes.levensthein_operations(expected, actual)[0])


class ActionSerializationStrategy(SerializationStrategy, use_annotations=True):
    def serialize(self, value: Action) -> int:
        return int(value)

    def deserialize(self, value: int) -> Action:
        return Action.from_int(value)


@dataclass
class UtteranceEdits(DataClassJSONMixin):
    language: str
    utterance_id: str
    expected: Dict[str, List[str]]
    edit_operations: Dict[str, List[Tuple[Action, str, str]]]

    class Config:
        serialization_strategy = {
            Action: ActionSerializationStrategy(),
        }


def _infer_gzip(filepath: PathOrFile) -> bool:
    """
    Infers gzip format from a path or named file object if it points to a file with the `.gz` extension

    :param filepath: A path or named file object to infer the format from

    :return: `True` if the path or file object path points to a file with the `.gz` extension
    """
    return path.splitext(utils.get_filepath(filepath))[1] == ".gz"


T = TypeVar("T")


class JsonlReader(Generic[T]):
    def __init__(self, file: PathOrFileBinary, gzip: bool | None = None) -> None:
        """
        Creates a prediction reader for a path or file object to read predictions from

        :param file: A file or path object of the predictions either
            uncompressed or gzip compressed.
        :param gzip: Whether to treat the input file as gzip compressed.
            If the `gzip` argument was not given or `None`, compression is inferred if the path ends in ".gz"
        """
        self._wrapped_file = file
        self._gzip = _infer_gzip(file) if gzip is None else gzip

    def read_meta(self) -> Any:
        return None

    def process_line(self, line) -> T:
        return line

    def __iter__(self) -> Iterator[T]:
        for line in self._file:
            yield self.process_line(line)

    def __enter__(self):
        self._file = TextIOWrapper(
            typing.cast(
                BinaryIO,
                gzip.open(self._wrapped_file, "r") if self._gzip else utils.file_from(self._wrapped_file, "rb"),
            ),
            encoding="utf-8",
        )
        self._metadata = self.read_meta()
        return self

    def __exit__(self, *_) -> None:
        self._file.close()


class PredictionReader(JsonlReader[UtterancePrediction]):
    def read_meta(self) -> PredictionMetaData:
        return PredictionMetaData.loads(self._file.readline())

    @property
    def metadata(self) -> PredictionMetaData:
        return self._metadata

    def process_line(self, line) -> UtterancePrediction:
        return UtterancePrediction.from_json(line)


class StatisticsReader(JsonlReader[UtteranceEdits]):
    def read_meta(self) -> PredictionMetaData:
        return PredictionMetaData.loads(self._file.readline())

    @property
    def metadata(self) -> PredictionMetaData:
        return self._metadata

    def process_line(self, line) -> UtteranceEdits:
        return UtteranceEdits.from_json(line)


class JsonlWriter:
    def __init__(self, file: PathOrFileBinary, metadata: PredictionMetaData, gzip: bool = False) -> None:
        """
        Creates a prediction writer for a path or file object to save predictions to

        :param file: A file or path object to write metadata and predictions to
        :param gzip: Whether to compress outputs with gzip
        """
        self._wrapped_file = file
        self._gzip = _infer_gzip(file) if gzip is None else gzip
        self._meta_data = metadata

    def __enter__(self):
        self._file = TextIOWrapper(
            typing.cast(
                BinaryIO,
                gzip.open(self._wrapped_file, "x") if self._gzip else utils.file_from(self._wrapped_file, "xb"),
            ),
            encoding="utf-8",
        )
        self._file.write(self._meta_data.dumps() + "\n")
        return self

    def __exit__(self, *_) -> None:
        self._file.close()

    def write(self, serialized: DataClassJSONMixin) -> None:
        self._file.write(str(serialized.to_json()) + "\n")


class GreedyCTCDecoder:
    def __init__(self, blank_index: int = 0):
        super().__init__()
        self._blank_index = blank_index

    def __call__(self, log_emissions: Tensor, lengths: Tensor) -> List[List[CTCHypothesis]]:
        batch_max = torch.max(log_emissions, dim=-1)
        outputs = []
        for i, indices in enumerate(batch_max.indices):
            length = lengths[i]
            indices = indices[:length]
            decoded, sizes = torch.unique_consecutive(indices, return_counts=True)
            non_blanks = decoded != self._blank_index
            # Compute flashlight CTCDecoder compatible 1-based timesteps
            timesteps = (sizes.cumsum(0) - sizes + 1)[non_blanks]

            # Compute score as the sum of the highest probabilities as in the flashlight CTCDecoder
            outputs.append([CTCHypothesis(decoded[non_blanks], [], batch_max.values[i, :length].sum(), timesteps)])
        return outputs


class BeamCTCDecoder:
    def __init__(self, tokens: List[str], beam_width: int, n_best: int = 1, blank_index: int = 0) -> None:
        blank_token = tokens[blank_index]
        self._decoder = decoder.ctc_decoder(
            lexicon=None,
            tokens=tokens,
            blank_token=blank_token,
            # Ignore silence
            sil_token=blank_token,
            beam_size=beam_width,
            nbest=n_best,
            log_add=True,
        )

    def __call__(self, log_emissions: Tensor, lengths: Tensor | None = None) -> Any:
        # Compatibility with CTCDecoder, which requires regular probabilities
        return self._decoder(log_emissions.exp(), lengths)


def _ctc_decoder(categories: Iterable[str], beam_width: int = 1, n_best: int = 1) -> CTCDecoder:
    assert n_best <= beam_width, "N-best can not exceed beam width"
    # Optimized decoder for greedy decoding with log probabilities
    if beam_width == 1:
        return GreedyCTCDecoder()

    return BeamCTCDecoder(["<blank>", *categories], beam_width, n_best)


class FeatureDecoder:
    def __init__(self, feature_attributes: ArticulatoryAttributes, beam_width: int = 1, n_best: int = 1):
        self.attributes = feature_attributes
        self.feature_matrix = feature_attributes.dense_feature_table.long()
        self.decoder = _ctc_decoder(feature_attributes.phonemes, beam_width, n_best)


def feature_decoders(
    indexer: PhonemeIndexer,
    beam_width: int = 1,
    feature_names: Iterable[str] | None = None,
    n_best: int = 1,
) -> Dict[str, CTCDecoder]:
    return {
        name: _ctc_decoder(indexer.feature_categories(name), beam_width, n_best)
        for name in (indexer.feature_names if feature_names is None else feature_names)
    }
