from dataclasses import dataclass
from enum import Enum
import json
import os
import sys
from argparse import Action, ArgumentParser, ArgumentTypeError, FileType, Namespace
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar
from marshmallow import fields
import logging
import marshmallow_dataclass

import toml
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import zarr
from zarr import Blosc
from zarr.errors import ContainsArrayError
import numpy as np

from allophant import utils
from allophant.config import Config, FeatureSet
from allophant.datasets.speech_corpus import (
    LanguageData,
    LanguageInfo,
    MultilingualCorpus,
    MultilingualSplits,
    TranscribedUtterance,
)
from allophant.package_data import DEFAULT_CONFIG_PATH
from allophant.phoneme_segmentation import SEGMENTATION_LOGGER
from allophant.phonetic_features import PhoneticAttributeIndexer
from allophant.preprocessing import FeatureFunction
from allophant.utils import EnumAction, MarshmallowDataclassLoadMixin
from allophant.datasets import corpus_loading, mozilla_common_voice
from allophant.datasets.corpus_loading import PreprocessedSplitData
from allophant.datasets.mozilla_common_voice import CommonVoiceCorpus, ReleaseMeta
from allophant.datasets.phonemes import G2PEngineType, GraphemeToPhonemeEnsemble
from allophant.language_codes import LanguageCodeMap


COMMON_VOICE_LOGGER = logging.getLogger("allophant.datasets")
COMMON_VOICE_LOGGER.setLevel(logging.INFO)


def generate_phoneme_transcriptions(parser: ArgumentParser, arguments: Namespace) -> None:
    if arguments.engine is None:
        g2p_engine = None

        if arguments.feature_set or arguments.attribute_path:
            parser.error(
                "Attribute path (-a/--attribute-path) and/or feature sets (-f/--feature-set) are only valid when a grapheme to phoneme engine is specified"
            )
    else:
        if arguments.feature_set:
            phoneme_inventory = PhoneticAttributeIndexer(
                arguments.feature_set, arguments.attribute_path
            ).phonemes.tolist()
        elif arguments.attribute_path:
            with arguments.attribute_path as file:
                phoneme_inventory = [segment.strip() for segment in file]
        else:
            parser.error(
                "At least one of -f/--feature-set and -a/--attribute-path has to be specified if grapheme to phoneme engines are given"
            )

        existing_engines = set()
        engines = []
        for engine_type in arguments.engine:
            # Sanity check since arguments might be passed from another source than __main__
            if engine_type in existing_engines:
                parser.error(f"G2P engine {engine_type!r} specified more than once")

            engines.append(engine_type.model())
            existing_engines.add(engine_type)

        g2p_engine = GraphemeToPhonemeEnsemble(engines, phoneme_inventory, arguments.batch_size)
        if arguments.log is not None:
            SEGMENTATION_LOGGER.add_file_handler(arguments.log)

    has_language_codes = arguments.language_codes is not None
    has_language_limits = isinstance(arguments.training_limits, dict)
    if has_language_codes or has_language_limits:
        language_map = LanguageCodeMap(
            ReleaseMeta.load(os.path.join(arguments.common_voice_path, CommonVoiceCorpus.META_FILE)).language_codes()
        )
        if has_language_codes:
            arguments.language_codes = [language_map[code] for code in arguments.language_codes]
        if has_language_limits:
            arguments.training_limits = {language_map[code]: limit for code, limit in arguments.training_limits.items()}

    mozilla_common_voice.load_common_voice(
        arguments.common_voice_path,
        g2p_engine,
        arguments.feature_set or arguments.attribute_path,
        arguments.include_single_upvote,
        arguments.batch_size,
        arguments.language_codes,
        arguments.map_to_allophoible,
        arguments.training_limits,
        arguments.progress,
        arguments.mapping_threshold,
    ).save(arguments.common_voice_path, arguments.output_directory)


def download_meta(_: ArgumentParser, arguments: Namespace) -> None:
    with arguments.output as output:
        json.dump(mozilla_common_voice.download_release_meta(arguments.version), output)


def parse_limits(limits: str) -> int | Dict[str, int]:
    try:
        return int(limits)
    except ValueError:
        return fields.Dict(fields.String(), fields.Integer()).deserialize(json.loads(limits))  # type: ignore


I = TypeVar("I", bound=LanguageInfo)
T = TypeVar("T", bound=TranscribedUtterance)


class FeaturePreprocessingDataset(Dataset, Generic[I, T]):
    def __init__(
        self, corpus: MultilingualCorpus[I, T], language_data: LanguageData[I, T], feature_function: FeatureFunction
    ) -> None:
        self._corpus = corpus
        self._samples = language_data.transcribed_samples
        self._language = language_data.info.code
        self._feature_function = feature_function

    def __getitem__(self, index: int) -> Tensor:
        # Get left channel only (or single channel for mono audio)
        return self._feature_function(
            self._corpus.audio_from_utterance(self._language, self._samples[index].utterance_id)[0][0]
        )

    def __len__(self) -> int:
        return len(self._samples)


def _create_audio_batch(batch: List[Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Workaround for when subarrays happen to have the same size,
    which would yield a regular array instead of an object array
    """
    batch_size = len(batch)
    object_array = np.empty(batch_size, dtype="object")
    lengths = np.empty(batch_size)
    for i, samples in enumerate(batch):
        lengths[i] = samples.shape[0]
        object_array[i] = samples.flatten().numpy()

    return object_array, lengths


_FEATURE_CHUNK_SIZE = 128


def preprocess_filters(_: ArgumentParser, arguments: Namespace) -> None:
    config = Config.load(arguments.config)
    corpus = corpus_loading.load_corpus(
        arguments.dataset_path,
        arguments.dataset_type,
        config.preprocessing.resample,
        only_primary_script=config.data.only_primary_script,
        progress_bar=True,
    )
    feature_function = FeatureFunction.from_config(config, corpus.audio_info().sample_rate)
    data_workers = utils.get_worker_count(arguments.data_workers)

    store = zarr.DirectoryStore(arguments.output_directory)
    data_root = zarr.group(store)
    # Store metadata for unraveling tensors later
    data_root.attrs["feature_size"] = feature_function.feature_size
    split_groups = data_root.require_groups(*corpus.SPLIT_NAMES)
    with tqdm(total=sum(map(len, corpus)), unit=" files") as progress:
        for corpus_split, split_group in zip(corpus, split_groups):
            for language_data in corpus_split:
                language_group = split_group.require_group(language_data.info.code)
                # Saves preprocessed features in flattened variable length arrays
                feature_data = language_group.zeros(
                    "features",
                    shape=len(language_data),
                    chunks=_FEATURE_CHUNK_SIZE,
                    compressor=Blosc(cname="lz4", shuffle=Blosc.BITSHUFFLE),
                    dtype="array:float32",
                )
                try:
                    length_data = language_group.zeros(
                        "lengths", shape=len(language_data), chunks=_FEATURE_CHUNK_SIZE, dtype=np.int64
                    )
                # Accept lengths already being generated by save-lengths
                except ContainsArrayError:
                    length_data = None

                offset = 0
                for features, lengths in DataLoader(
                    FeaturePreprocessingDataset(corpus_split, language_data, feature_function),
                    batch_size=_FEATURE_CHUNK_SIZE,
                    collate_fn=_create_audio_batch,
                    shuffle=False,
                    num_workers=data_workers,
                    persistent_workers=True,
                ):
                    length = len(features)
                    end_offset = offset + length
                    feature_data[offset:end_offset] = features
                    if length_data is not None:
                        length_data[offset:end_offset] = lengths
                    offset += length
                    progress.update(length)


def save_lengths(_: ArgumentParser, arguments: Namespace) -> None:
    config = Config.load(arguments.config)
    corpus = corpus_loading.load_corpus(
        arguments.dataset_path,
        arguments.dataset_type,
        config.preprocessing.resample,
        only_primary_script=config.data.only_primary_script,
        progress_bar=True,
    )
    feature_function = FeatureFunction.from_config(config, corpus.audio_info().sample_rate)

    store = zarr.DirectoryStore(arguments.output_directory)
    data_root = zarr.group(store)
    split_groups = data_root.create_groups(*corpus.SPLIT_NAMES)

    with tqdm(total=sum(map(len, corpus)), unit=" files") as progress:

        def _frame_counts(lengths: Iterable[int]) -> Iterator[int]:
            for length in lengths:
                yield feature_function.frame_count(length)
                progress.update(1)

        for split, split_group in zip(corpus, split_groups):
            for language_code, lengths in split.read_lengths():
                split_group.create_group(language_code).array(
                    "lengths", list(_frame_counts(lengths)), chunks=_FEATURE_CHUNK_SIZE, dtype=np.int64
                )


@dataclass
class UtteranceDurations:
    total: int
    average: float
    sample_rate: Optional[int] = None

    def __str__(self) -> str:
        if self.sample_rate is None:
            seconds = hours = 1
        else:
            seconds = self.sample_rate
            hours = self.sample_rate * 60 * 60
        return f"{self.total} frames at {self.sample_rate}: {self.total / hours:.2f}h (Average: {self.average / seconds:.4f}s)"

    @classmethod
    def compute_durations(cls, lengths: Tensor, sample_rate: int | None = None):
        return cls(
            int(lengths.sum().item()),
            lengths.mean(dtype=torch.float64).item(),
            sample_rate,
        )


@dataclass
class SplitStatistics:
    languages: List[str]
    utterance_counts: Dict[str, int]
    phoneme_statistics: Dict[str, Dict[str, int]]
    durations: Optional[Dict[str, UtteranceDurations]] = None


@marshmallow_dataclass.add_schema
@dataclass
class CorpusStatistics(MarshmallowDataclassLoadMixin):
    splits: Dict[str, SplitStatistics]


def _range_to_slice(input_range: range) -> slice:
    return slice(input_range.start, input_range.stop)


def _length_statistics(
    split: MultilingualCorpus, data: PreprocessedSplitData, sample_rate: int | None = None
) -> Dict[str, UtteranceDurations]:
    lengths = data.lengths
    return {
        language: UtteranceDurations.compute_durations(
            lengths[_range_to_slice(split.monolingual_index_range(language))], sample_rate
        )
        for language in split.languages
    }


def corpus_statistics(_: ArgumentParser, arguments: Namespace) -> None:
    corpus = corpus_loading.load_corpus(
        arguments.dataset_path,
        arguments.dataset_type,
        only_primary_script=arguments.only_primary_script,
        progress_bar=True,
    )

    statistics = CorpusStatistics({})
    for split_name in arguments.splits:
        split = getattr(corpus, split_name)
        statistics.splits[split_name] = SplitStatistics(split.languages.copy(), {}, {})

    if arguments.lengths is not None:
        data = corpus_loading.preprocessed_features_or_lengths(
            arguments.lengths, {split: getattr(corpus, split).languages for split in arguments.splits}
        )

        for split_name, split_data in data.items():
            statistics.splits[split_name].durations = _length_statistics(
                getattr(corpus, split_name), split_data, arguments.sample_rate
            )

    if arguments.json:
        print(statistics.dumps())
    else:
        print(statistics)


class EnumUniqueAppendAction(Action):
    """
    An action for handling sets of multiple Enum instances
    """

    def __init__(self, type: Type[Enum], **kwargs) -> None:
        self._enum_action = EnumAction(type, **kwargs)
        self._seen = set()
        super().__init__(**kwargs)

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        # Get current values
        current_values = getattr(namespace, self.dest) or []
        # Retrieve destination value from the enum action
        self._enum_action(parser, namespace, values, option_string)
        new_value = getattr(namespace, self.dest)
        # Ignore duplicates and append new elements
        if new_value not in self._seen:
            current_values.append(new_value)
            self._seen.add(new_value)
        setattr(namespace, self.dest, current_values)


def main(args: Sequence[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    parser = ArgumentParser(description="Tools for preprocessing the Mozilla Common Voice corpus")
    subparsers = parser.add_subparsers(dest="mode")
    subparsers.required = True

    transcription_parser = subparsers.add_parser(
        "transcribe",
        help="Generates phonemic transcriptions for (a subset of) the Mozilla Common Voice corpus and extracts phoneme inventories",
        description="Generates phonemic transcriptions for (a subset of) the Mozilla Common Voice corpus and extracts phoneme inventories",
    )
    transcription_parser.add_argument("common_voice_path", help="Path to a version of the Mozilla Common Voice dataset")
    transcription_parser.add_argument(
        "-e",
        "--engine",
        action=EnumUniqueAppendAction,
        type=G2PEngineType,
        help="G2P engine(s) for extracting phonemic transcriptions for each utterance\nMay be none or multiple",
    )
    transcription_parser.add_argument(
        "-u",
        "--include-single-upvote",
        action="store_true",
        help="If given, unvalidated utterances with at least one more upvotes than downvotes will also be included in the training set",
    )
    transcription_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for tokenization and G2P engines (if provided) in maximum number of utterances per batch",
    )
    transcription_parser.add_argument(
        "-l",
        "--language-codes",
        type=lambda codes: codes.split(","),
        help="ISO639 language codes for each language to be transcribed",
    )
    transcription_parser.add_argument(
        "-t",
        "--training-limits",
        type=utils.argparse_type_wrapper(parse_limits),
        help="Limit for the number of utterances per language, may be a single number or a json dictionary from IOS639 language codes to language specific limits",
    )
    transcription_parser.add_argument(
        "-a",
        "--attribute-path",
        type=FileType("r", encoding="utf-8"),
        help="Path to a phoneme inventory with one phoneme per line or a supported phoneme feature file",
    )
    transcription_parser.add_argument(
        "-f",
        "--feature-set",
        action=EnumAction,
        type=FeatureSet,
        help="Type of feature set to load - loads Allophoible for the phoible feature set unless --attribute-path is given",
    )
    transcription_parser.add_argument(
        "-p", "--progress", action="store_true", help="Shows progress bars while utterances are being transcribed"
    )
    transcription_parser.add_argument(
        "-o",
        "--output-directory",
        help=(
            "A different directory from the corpus directory for storing transcriptions."
            "Note that moving the original corpus directory makes audio files inaccessible from this folder."
        ),
    )
    transcription_parser.add_argument(
        "-m",
        "--map-to-allophoible",
        action="store_true",
        help=(
            "Maps the inventories collected from the G2P transcriptions to language specific allophoible inventories, "
            "including in the transcriptions themselves"
        ),
    )
    transcription_parser.add_argument(
        "--log", help="Path to the log file for phonemes missing from the chosen feature set"
    )
    transcription_parser.add_argument(
        "--mapping-threshold",
        type=int,
        help="Distance threshold for allophoible mapping above which mapping should not take place",
    )

    download_meta_parser = subparsers.add_parser(
        "download-meta",
        help="Downloads metadata for a given version of the Mozilla Common Voice corpus",
        description="Downloads metadata for a given version of the Mozilla Common Voice corpus",
    )
    download_meta_parser.add_argument(
        "output",
        type=FileType("w", encoding="utf-8"),
        default=sys.stdout,
        help="Output file which the metadata will be saved to",
    )
    download_meta_parser.add_argument(
        "-v", "--version", default="10.0-2022-07-04", help="Version of the Common Voice corpus"
    )

    data_config_parser = ArgumentParser(add_help=False)
    data_config_parser.add_argument(
        "-t", "--dataset-type", choices={"common-voice"}, default="common-voice", help="Type of the Dataset"
    )
    data_config_parser.add_argument(
        "-c",
        "--config",
        type=utils.argparse_type_wrapper(toml.load),
        default=toml.load(DEFAULT_CONFIG_PATH),
        help=(
            "Path to a configuration file in toml format, "
            "usually modified from a default config generated using generate-config"
        ),
    )
    data_config_parser.add_argument(
        "-j",
        "--config-json-data",
        default=None,
        help="Parses configuration files passed to `-c/--config` using JSON instead of TOML",
    )

    data_processing_parser = ArgumentParser(add_help=False)
    data_processing_parser.add_argument(
        "dataset_path", help="Path to a corpus containing phonetically transcribed utterances"
    )
    data_processing_parser.add_argument(
        "output_directory", help="Path to a directory which the processed data will be stored in"
    )

    subparsers.add_parser(
        "save-lengths",
        parents=[data_config_parser, data_processing_parser],
        help="Generates and saves frame lengths for the given dataset based on the feature function in the given configuration",
        description="Generates and saves frame lengths for the given dataset based on the feature function in the given configuration",
    )

    preprocessing_parser = subparsers.add_parser(
        "preprocess",
        parents=[data_config_parser, data_processing_parser],
        help="Preprocesses acoustic features for the given dataset based on the given configuration",
        description="Preprocesses acoustic features for the given dataset based on the given configuration",
    )
    preprocessing_parser.add_argument(
        "-w",
        "--data-workers",
        type=int,
        help=(
            "Number of workers - 0 disables workers and runs data processing synchronously on the main thread, "
            "by default the number of workers is auto-detected based on CPU thread count"
        ),
    )

    statistics_parser = subparsers.add_parser("stats")
    statistics_parser.add_argument(
        "dataset_path", help="Path to a corpus containing phonetically transcribed utterances"
    )
    statistics_parser.add_argument(
        "-t", "--dataset-type", choices={"common-voice"}, default="common-voice", help="Type of the Dataset"
    )
    statistics_parser.add_argument(
        "-l", "--lengths", help="Path to previously saved lengths for utterances in the corpus"
    )
    statistics_parser.add_argument(
        "-p",
        "--only-primary-script",
        action="store_true",
        help="Keeps only transcriptions that consist of a single script that corresponds to the primary script of each language",
    )
    statistics_parser.add_argument(
        "-s",
        "--splits",
        type=lambda string: string.split(","),
        default=MultilingualSplits.SPLIT_NAMES,
        help="Splits to calculate statistics for separated by commas. E.g.: train,dev,test",
    )
    statistics_parser.add_argument("-j", "--json", action="store_true", help="Outputs statistics in JSON format")
    statistics_parser.add_argument(
        "--sample-rate",
        type=int,
        help="Interprets lengths as frame counts at the given sample rate, if provided, and displays durations also in hours",
    )

    arguments = parser.parse_args(args)

    # Handles cases in which the configuration is provided in json format
    if hasattr(arguments, "config_json_data") and arguments.config_json_data is not None:
        arguments.config = json.loads(arguments.config_json_data)

    match arguments.mode:
        case "transcribe":
            generate_phoneme_transcriptions(parser, arguments)
        case "download-meta":
            download_meta(parser, arguments)
        case "save-lengths":
            save_lengths(parser, arguments)
        case "preprocess":
            preprocess_filters(parser, arguments)
        case "stats":
            corpus_statistics(parser, arguments)
        case mode:
            raise ValueError(f"Unsupported action: {mode}")


if __name__ == "__main__":
    main(sys.argv[1:])
