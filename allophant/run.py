from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence, Set
import contextlib
from enum import Enum
import math
import os
from typing import List, NamedTuple, Tuple, Dict
from argparse import ArgumentParser, FileType, Namespace
import warnings
import toml
import shutil
import json
import sys
import random
from os import path
from dataclasses import dataclass
import resource
from importlib.metadata import version

import torch
from torch.backends import cudnn
from torch.utils.tensorboard.writer import SummaryWriter
from torch import autograd
from torch import Tensor
from tqdm import tqdm
import numpy as np
from allophant import MAIN_LOGGER, language_codes, phoneme_segmentation, phonemes

from allophant.config import BatchingMode, Config, PhonemeLayerType, ProjectionEntryConfig
from allophant.dataset_processing import (
    BatchType,
    PhoneticallySegmentedDataset,
    RawLabeledBatch,
    SamplesProcessor,
    TranscribedDataset,
)
from allophant.datasets import corpus_loading
from allophant.datasets.speech_corpus import (
    MultilingualCorpus,
    MultilingualSplits,
    PhoneticallySegmentedUtterance,
    PhoneticallyTranscribedUtterance,
)
from allophant.package_data import DEFAULT_CONFIG_PATH
from allophant import estimator, utils, predictions
from allophant.utils import EnumAction
from allophant.estimator import EpochPosition, Estimator, DatasetManager, Checkpoint, TrainDevFeatures, TrainDevLengths
from allophant.batching import Batcher
from allophant.phonemes import EditStatistics
from allophant.evaluation import EvaluationResults, MultilingualEvaluationResults
from allophant.phonetic_features import PhoneticAttributeIndexer
from allophant.preprocessing import FeatureFunction
from allophant.predictions import (
    FeatureDecoder,
    PredictionMetaData,
    PredictionReader,
    JsonlWriter,
    UtteranceEdits,
    UtterancePrediction,
)


# Workaround for too many file descriptors in multiprocessing dataloader
FILE_LIMIT = 8192
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
if soft_limit < FILE_LIMIT:
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(FILE_LIMIT, hard_limit), hard_limit))


def generate_config(arguments: Namespace) -> None:
    if arguments.config_path is None:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as toml_file:
            print(toml_file.read(), end="")
    else:
        shutil.copyfile(DEFAULT_CONFIG_PATH, arguments.config_path)


BEST_CHECKPOINT = "best.pt"


def _checkpoint_name(epoch_position: EpochPosition):
    if epoch_position.step is None:
        return f"checkpoint-{epoch_position.epoch}.pt"
    return f"checkpoint-{epoch_position.epoch}-{epoch_position.step}.pt"


def _set_random_seed(seed: int) -> None:
    """
    Seeds standard library random, torch and cuda. Cuda benchmarking is also
    disabled and the cuda deterministic mode set for better reproducability of experiments.

    :param seed: A seed to use for the standard library, torch and cuda random number generators
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def _try_restoring_from_checkpoint(arguments: Namespace) -> Tuple[Config, Checkpoint | None]:
    model_file = (
        path.join(arguments.save_path, BEST_CHECKPOINT)
        if arguments.save_path is not None and path.isdir(arguments.save_path)
        else arguments.save_path
    )
    model_path_exists = arguments.save_path is not None and path.exists(arguments.save_path)
    model_file_exists = model_file is not None and path.exists(model_file)
    if arguments.restore:
        if not model_path_exists:
            raise FileNotFoundError(
                f"Can't restored checkpoint from {arguments.save_path!r}, no such file or directory"
            )
        elif not model_file_exists:
            raise FileNotFoundError(f"Couldn't find a best checkpoint to continue from at {model_file!r}")
        restore_model = True
    else:
        restore_model = False

    if restore_model:
        checkpoint = Checkpoint.restore(model_file)
        return checkpoint.config, checkpoint
    else:
        if model_path_exists:
            raise ValueError(
                f"Checkpoint file already exists at {arguments.save_path} - either use --restore to continue training or remove the existing model file or directory"
            )
        return Config.load(arguments.config), None


def _log_training_data_statistics(
    dataset_type: str, dataset_path: str, corpus: MultilingualSplits, validation_enabled: bool = True
):
    MAIN_LOGGER.info(f"using the {dataset_type} corpus from {dataset_path!r}")

    num_train_languages = len(corpus.train.languages)
    init_message = f"Training with {len(corpus.train)} utterances from {num_train_languages} language{'s' if num_train_languages > 1 else ''}"
    if validation_enabled:
        num_dev_languages = len(corpus.dev.languages)
        init_message += f" and validating with {len(corpus.dev)} utterances from {num_dev_languages} language{'s' if num_dev_languages > 1 else ''}"

    MAIN_LOGGER.info(init_message)


def train_network(arguments: Namespace) -> None:
    if arguments.detect_anomaly:
        autograd.set_detect_anomaly(True)

    config, checkpoint = _try_restoring_from_checkpoint(arguments)
    # Set a seed for more reproducible results
    seed = config.nn.seed
    if seed is not None:
        _set_random_seed(seed)

    corpus = corpus_loading.load_corpus(
        arguments.dataset_path,
        arguments.dataset_type,
        config.preprocessing.resample,
        config.data.languages,
        config.data.validation_limits,
        config.data.only_primary_script,
        not arguments.no_progress,
    )

    validate = not arguments.no_validation
    _log_training_data_statistics(arguments.dataset_type, arguments.dataset_path, corpus, validate)

    if arguments.features is None:
        features = None
        if arguments.lengths is None:
            training_lengths = None
        else:
            languages = {name: getattr(corpus, name).languages for name in MultilingualSplits.SPLIT_NAMES[:2]}
            processed_data = corpus_loading.preprocessed_features_or_lengths(arguments.lengths, languages)
            training_lengths = TrainDevLengths(
                processed_data["train"].lengths,
                processed_data["dev"].lengths,
            )
    else:
        languages = {name: getattr(corpus, name).languages for name in MultilingualSplits.SPLIT_NAMES[:2]}
        processed_data = corpus_loading.preprocessed_features_or_lengths(
            arguments.features, languages, lengths_only=False
        )
        training_data = processed_data["train"]
        validation_data = processed_data["dev"]
        if training_data.features is None or validation_data.features is None:
            raise ValueError("Missing feature data in preprocessed data")
        features = TrainDevFeatures(training_data.features, validation_data.features)
        training_lengths = TrainDevLengths(training_data.lengths, validation_data.lengths)

    if checkpoint is not None:
        MAIN_LOGGER.info(f"Restoring from {arguments.save_path}")
        estimator, attribute_indexer = Estimator.restore(checkpoint, "cpu" if arguments.cpu else "cuda")
        optimization_states = checkpoint.optimization_states
        dataset_manager = DatasetManager.from_config(
            config,
            corpus,
            attribute_indexer,
            training_lengths,
            features,
            arguments.data_workers,
            validate,
        )
    else:
        MAIN_LOGGER.info(f"Generating model from config")
        # Create directory for storing all checkpoints
        if arguments.save_all and arguments.save_path is not None:
            os.mkdir(arguments.save_path)

        training_split: MultilingualCorpus = corpus.train
        attribute_indexer = PhoneticAttributeIndexer.from_config(
            config,
            arguments.attribute_path,
            training_split.language_id_inventories(),
        )

        dataset_manager = DatasetManager.from_config(
            config,
            corpus,
            attribute_indexer,
            training_lengths,
            features,
            arguments.data_workers,
            validate,
        )

        estimator = Estimator.from_config(
            config,
            dataset_manager.feature_size,
            dataset_manager.sample_rate,
            dataset_manager.attribute_graph(config),
            attribute_indexer,
            use_cuda,
        )
        optimization_states = None

    optimizer = estimator.create_optimizer()

    tensorboard_directory = arguments.tensorboard_directory
    with (
        contextlib.nullcontext()
        if tensorboard_directory is None
        else SummaryWriter(tensorboard_directory if tensorboard_directory else None)
    ) as writer:
        for status, statistics in estimator.train(
            dataset_manager,
            optimizer,
            max_iterations=config.nn.maximum_iterations,
            validate=validate,
            progress=not arguments.no_progress,
            optimization_states=optimization_states,
            summary_writer=writer,
        ):
            MAIN_LOGGER.info(statistics)
            # Save model to the target path if it was provided every time the network improves
            if arguments.save_path is not None:
                epoch_checkpoint_path = path.join(arguments.save_path, _checkpoint_name(statistics.epoch))
                if arguments.save_all:
                    # Save a checkpoint every epoch with its epoch data preserved if save_all is enabled
                    estimator.save(epoch_checkpoint_path, optimizer.state_dict(), dataset_manager.indexer_state())
                if status.improvement:
                    if arguments.save_all:
                        # Copy the best epoch checkpoint to `best.pt` for easier access
                        shutil.copy2(epoch_checkpoint_path, path.join(arguments.save_path, BEST_CHECKPOINT))
                    else:
                        # Save only the best checkpoint if save_all was false and save_path is a file
                        estimator.save(arguments.save_path, optimizer.state_dict(), dataset_manager.indexer_state())

            if status.stop:
                if not arguments.no_progress:
                    MAIN_LOGGER.info("Training stopped by scheduler")
                return


def _fix_inventories(
    target_inventories: Dict[str, List[str]], missing_mappings: Dict[str, str]
) -> Dict[str, List[str]]:
    # Fix missing phonemes when necessary or keep the current phonemes if they aren't in the missing mapping
    # Afterwards, sort the fixed inventory for determinism to ensure consistent evaluation
    return {
        language: sorted({missing_mappings.get(phoneme, phoneme) for phoneme in raw_inventory})
        for language, raw_inventory in target_inventories.items()
    }


def _make_source_maps(
    attribute_indexer: PhoneticAttributeIndexer,
    target_inventories: Dict[str, List[str]],
) -> Dict[str, Dict[str, str]]:
    # Remap the indexer inventory to the fixed target inventories
    return {
        language: attribute_indexer.map_target_inventory(inventory)
        for language, inventory in target_inventories.items()
    }


def _missing_mappings(
    attribute_indexer: PhoneticAttributeIndexer, target_inventories: Dict[str, List[str]]
) -> Dict[str, str]:
    # Construct mappings from missing phonemes for the union of all target inventories which are sorted first for more determinism
    return attribute_indexer.full_attributes.missing_inventory_mappings(
        sorted({phoneme for inventory in target_inventories.values() for phoneme in inventory})
    )


@dataclass
class _EvaluationMappings:
    source_maps: Dict[str, Dict[str, str]]
    missing_mappings: Dict[str, str] | None = None


def _evaluation_mappings(
    attribute_indexer: PhoneticAttributeIndexer,
    target_inventories: Dict[str, List[str]] | None = None,
    remap: bool = False,
    fix_unicode: bool = False,
) -> _EvaluationMappings:
    if target_inventories is None:
        raise ValueError("Target inventories are required for phoneme remapping but none were given")

    missing_mappings = _missing_mappings(attribute_indexer, target_inventories) if fix_unicode else None

    return _EvaluationMappings(
        {} if not remap else _make_source_maps(attribute_indexer, target_inventories), missing_mappings
    )


_IPA_LAYER = {ProjectionEntryConfig.PHONEME_LAYER, ProjectionEntryConfig.PHONE}


class MissingFeatureWarning(UserWarning):
    """Warns about a feature for which outputs exist in predictions but without labels being encountered in evaluation"""


warnings.simplefilter("once", MissingFeatureWarning)


class EvaluationProcessor:
    def __init__(
        self,
        predictions_meta: PredictionMetaData,
        map_phonemes: bool = False,
        fix_unicode: bool = False,
        split_complex: bool = False,
    ) -> None:
        self.attribute_indexer = PhoneticAttributeIndexer.from_state(
            predictions_meta.feature_set,
            predictions_meta.indexer_state,
        )
        self.full_attributes = self.attribute_indexer.full_attributes
        self.evaluation_mappings = _evaluation_mappings(
            self.attribute_indexer, predictions_meta.label_inventories, map_phonemes, fix_unicode
        )

        self._map_phonemes = map_phonemes
        self._split_complex = split_complex

    @property
    def split_complex(self) -> bool:
        return self._split_complex

    def language_mapper(self, language: str) -> Dict[str, str] | None:
        return self.evaluation_mappings.source_maps[language] if self._map_phonemes else None

    def attribute_indices(self, phonemes: List[str]) -> Dict[str, Tensor]:
        missing_mappings = self.evaluation_mappings.missing_mappings
        return self.full_attributes.get_named(
            phonemes if missing_mappings is None else [missing_mappings.get(phoneme, phoneme) for phoneme in phonemes]
        )


def _labeled_predictions(
    arguments: Namespace, reader: PredictionReader
) -> Iterator[Tuple[int, UtterancePrediction, List[List[str]]]]:
    predictions_meta = reader.metadata

    format_version = predictions_meta.format_version
    if format_version != predictions.CURRENT_FORMAT_VERSION:
        MAIN_LOGGER.warn(
            f"Predictions file uses older format {'.'.join(map(str, format_version))}"
            f" while the current version is {'.'.join(map(str, predictions.CURRENT_FORMAT_VERSION))}"
        )

    for line, prediction in enumerate(tqdm(reader, unit=" utterances", disable=arguments.no_progress), 1):
        references = prediction.labels
        if references is None:
            raise ValueError(f"Missing label for evaluation in line {line}")

        yield line, prediction, references


def _process_prediction(
    prediction: UtterancePrediction, references: List[List[str]], evaluation_processor: EvaluationProcessor
) -> Iterator[Tuple[str, List[str], List[List[str]]]]:
    [reference] = references
    reference_feature_indices = evaluation_processor.attribute_indices(reference)

    for name, candidates in prediction.predictions.items():
        # Special case for phone and phoneme labels
        is_ipa_output = name in _IPA_LAYER
        if is_ipa_output:
            expected = reference
        else:
            try:
                expected = evaluation_processor.full_attributes.feature_values(name, reference_feature_indices[name])
            except KeyError:
                warnings.warn(f'Missing feature in attributes: "{name}" - skipping', MissingFeatureWarning)
                continue

        if is_ipa_output and evaluation_processor.split_complex:
            expected = list(phoneme_segmentation.split_all_complex_segments(expected))

        yield name, expected, candidates


def _process_candidates(
    candidates, evaluation_processor: EvaluationProcessor, language: str, is_ipa_output: bool = False
) -> Iterator[List[str]]:
    source_map = evaluation_processor.language_mapper(language)

    for candidate in candidates:
        if not is_ipa_output:
            yield candidate
            continue

        actual = candidate if source_map is None else [source_map[phoneme] for phoneme in candidate]
        # Split all complex segments if requested before comparison
        if evaluation_processor.split_complex:
            actual = list(phoneme_segmentation.split_all_complex_segments(actual))

        yield actual


def _compute_edit_statistics(arguments: Namespace, reader: PredictionReader) -> Dict[str, Dict[str, EditStatistics]]:
    predictions_meta = reader.metadata
    classifiers = predictions_meta.classifiers
    languages = predictions_meta.languages
    map_phonemes = not arguments.no_remap

    split_complex = arguments.split_complex
    processor = EvaluationProcessor(predictions_meta, map_phonemes, arguments.fix_unicode, split_complex)
    edit_statistics = {language: {name: EditStatistics.zeros() for name in classifiers} for language in languages}

    for line, prediction, references in _labeled_predictions(arguments, reader):
        language = prediction.language

        for name, expected, candidates in _process_prediction(prediction, references, processor):
            lowest_error_rate = math.inf
            best_statistics = None

            for actual in _process_candidates(candidates, processor, language, name in _IPA_LAYER):
                statistics = phonemes.levensthein_statistics(expected, actual)
                error_rate = statistics.word_error_rate()
                if error_rate < lowest_error_rate:
                    lowest_error_rate = error_rate
                    best_statistics = statistics

            if best_statistics is None:
                warnings.warn(
                    f"Each category needs at least one candidate output, got no candidates for {name!r} in line {line}"
                    f" with utterance ID {prediction.utterance_id}"
                )
                continue
            edit_statistics[language][name] += best_statistics

    return edit_statistics


def evaluate(arguments: Namespace) -> None:
    with PredictionReader(arguments.prediction_path, gzip=arguments.decompress) as reader:
        edit_statistics = _compute_edit_statistics(arguments, reader)
        metadata = reader.metadata

    classifiers = metadata.classifiers

    total_statistics = defaultdict(EditStatistics.zeros)
    stats = {}
    for language, language_statistics in edit_statistics.items():
        language_error_rates = {}
        for name, statistics in language_statistics.items():
            total_statistics[name] += statistics
            language_error_rates[name] = statistics.word_error_rate()
        stats[language] = EvaluationResults(classifiers, language_error_rates, language_statistics)

    total_error_rates = {}
    for name, statistics in total_statistics.items():
        total_error_rates[name] = statistics.word_error_rate()

    stats["total"] = EvaluationResults(classifiers, total_error_rates, dict(total_statistics))
    results = MultilingualEvaluationResults(
        str(arguments), {language: language_stats for language, language_stats in stats.items()}
    )

    with arguments.output or sys.stdout as file:
        if arguments.json:
            results.dump(file)
        else:
            file.write(str(results))
            file.write("\n")


def _compute_edits(arguments: Namespace, reader: PredictionReader) -> Iterator[UtteranceEdits]:
    map_phonemes = not arguments.no_remap

    split_complex = arguments.split_complex
    processor = EvaluationProcessor(reader.metadata, map_phonemes, arguments.fix_unicode, split_complex)

    for _, prediction, references in _labeled_predictions(arguments, reader):
        language = prediction.language

        edits = {}
        expected_sequences = {}
        for name, expected, candidates in _process_prediction(prediction, references, processor):
            # Get only the highest scoring beam
            actual = next(_process_candidates(candidates, processor, language, name in _IPA_LAYER))
            edits[name] = predictions.levensthein_substitutions(expected, actual)
            expected_sequences[name] = expected

        yield UtteranceEdits(language, prediction.utterance_id, expected_sequences, edits)


def edits(arguments: Namespace) -> None:
    with (
        PredictionReader(arguments.prediction_path, gzip=arguments.decompress) as reader,
        JsonlWriter(arguments.output, reader.metadata, gzip=arguments.compress) as writer,
    ):
        for edits in _compute_edits(arguments, reader):
            writer.write(edits)


def _dataset_from_data(data: MultilingualCorpus, config: Config, indexer: PhoneticAttributeIndexer):
    sample_processor = SamplesProcessor(
        FeatureFunction.from_config(config, data.audio_info.sample_rate),
        indexer,
    )

    if data.UTTERANCE_TYPE == PhoneticallyTranscribedUtterance:
        return TranscribedDataset(
            BatchType.RAW,
            data,
            sample_processor,
        )
    elif data.UTTERANCE_TYPE == PhoneticallySegmentedUtterance:
        return PhoneticallySegmentedDataset(
            BatchType.RAW,
            data,
            sample_processor,
        )
    else:
        raise ValueError(f"Unknown utterance type {data.UTTERANCE_TYPE}")


class GeneratedBatch(NamedTuple):
    batch: RawLabeledBatch
    language_batch: List[str]


def _filter_split_raw_batches_by_language(
    batch_generator: Iterable[RawLabeledBatch], data: MultilingualCorpus, excluded_languages: Set[str]
) -> Iterator[GeneratedBatch]:
    for original_batch in batch_generator:
        for language_id, batch in original_batch.split_by_language():
            language_code = data.language(language_id)
            # Skip filtered languages
            if language_code in excluded_languages:
                continue
            # Add language batch by just repeating the code since the batches only contain utterances in a single language
            yield GeneratedBatch(batch, [language_code] * len(batch))


def predict(arguments: Namespace) -> None:
    n_candidates = arguments.n_best
    if n_candidates > arguments.ctc_beam:
        raise ValueError(f"n_best {n_candidates} larger than the beam size {arguments.ctc_beam}")

    inference_estimator, attribute_indexer = Estimator.restore(arguments.model_path, "cpu" if arguments.cpu else "cuda")
    config = inference_estimator.config
    # Set Model entirely to evaluation mode for prediction
    inference_estimator.model.eval()
    evaluation_device = "cpu" if arguments.cpu else "cuda"

    is_allophone_model = config.nn.projection.phoneme_layer == PhonemeLayerType.ALLOPHONES
    if arguments.language_phonemes and not is_allophone_model:
        raise ValueError(
            "--language-phonemes can only be used with models that use an allophone layer."
            f" The restored checkpoint layer type is {config.nn.projection.phoneme_layer}"
        )

    feature_names = attribute_indexer.feature_names
    is_composition_model = config.nn.projection.embedding_composition is not None

    # Sets phoneme indexer to phones instead if universal phone outputs are used from allophone layer models
    # and embedding composition isn't used
    map_allophones = False
    composition_output_name = ProjectionEntryConfig.PHONEME_LAYER
    if is_allophone_model:
        if attribute_indexer.allophone_data is None:
            raise ValueError("Allophone data is missing from the attribute indexer")
        phone_indexer = attribute_indexer.allophone_data.shared_phone_indexer

        feature_names.append(ProjectionEntryConfig.PHONE)
        composition_output_name = ProjectionEntryConfig.PHONE
        if arguments.language_phonemes:
            map_allophones = True
        else:
            # Don't try predicting phonemes with language specific matrices if --language-phonemes isn't set
            feature_names.remove(ProjectionEntryConfig.PHONEME_LAYER)
    else:
        phone_indexer = None

    if arguments.feature_subset is not None:
        if not set(arguments.feature_subset).issubset(set(feature_names)):
            raise ValueError(
                f"The provided feature subset {sorted(arguments.feature_subset)} "
                f"is not a subset of {sorted(feature_names)}"
            )
        feature_names = arguments.feature_subset

    MAIN_LOGGER.info(f"Predicting with attributes: {feature_names}")

    num_workers = utils.get_worker_count(arguments.data_workers)
    batching_mode = config.nn.batching_mode if arguments.batch_mode is None else arguments.batch_mode
    batcher = Batcher(
        (
            estimator.split_batch_size(config.nn.batch_size, config.nn.accumulation_factor)
            if arguments.batch_size is None
            else arguments.batch_size
        ),
        batching_mode,
        data_workers=num_workers,
    )

    training_language_mode: TrainingLanguageMode = arguments.training_languages
    match training_language_mode:
        case TrainingLanguageMode.ONLY:
            corpus_languages = config.data.languages
            exclude_known = False
        case TrainingLanguageMode.EXCLUDE:
            corpus_languages = None
            exclude_known = True
        case TrainingLanguageMode.INCLUDE:
            corpus_languages = None
            exclude_known = False

    test_data: MultilingualCorpus = corpus_loading.load_corpus(
        arguments.dataset_path,
        arguments.dataset_type,
        config.preprocessing.resample,
        corpus_languages,
        only_primary_script=config.data.only_primary_script,
        progress_bar=not arguments.no_progress,
    ).test

    test_languages = test_data.languages
    excluded_languages = set()

    if exclude_known:
        language_set = set(map(language_codes.standardize_to_iso6393, config.data.languages))
        kept_languages = []
        for language in test_languages:
            if language in language_set:
                excluded_languages.add(language)
            else:
                kept_languages.append(language)
        test_languages = kept_languages
        MAIN_LOGGER.info(f"Excluding languages: {excluded_languages}")

    if batching_mode != BatchingMode.FRAMES and arguments.lengths is None:
        test_data_lengths = None
    else:
        test_data_lengths = corpus_loading.preprocessed_features_or_lengths(
            arguments.lengths, {"test": test_data.languages}
        )["test"].lengths

    dataset = _dataset_from_data(test_data, config, attribute_indexer)
    model_output_start_offset = config.nn.loss.BLANK_OFFSET

    ctc_decoders = predictions.feature_decoders(attribute_indexer, arguments.ctc_beam, feature_names, n_candidates)

    # Handle embedding inventory subsets for models using embedding composition
    if not is_composition_model:
        per_language_decoders = None
    else:
        attributes = attribute_indexer.composition_features
        # Merge training and test inventories for fairness when using common voice
        if arguments.dataset_type == "common-voice":
            training_inventories = test_data.load_inventories_for("train")
            language_inventories = {
                language: sorted(set(test_data.inventory(language)) | set(training_inventories[language]))
                for language in test_languages
            }
        else:
            language_inventories = {language: test_data.inventory(language) for language in test_languages}

        if arguments.fix_unicode:
            language_inventories = _fix_inventories(
                language_inventories, _missing_mappings(attribute_indexer, language_inventories)
            )

        # Create CTC decoders for each language based on their phoneme inventories
        per_language_decoders = {
            language: FeatureDecoder(
                attribute_indexer.full_attributes.subset(inventory, attributes),
                arguments.ctc_beam,
                n_candidates,
            )
            for language, inventory in language_inventories.items()
        }

    batch_generator = batcher.batches(dataset, test_data_lengths)
    # Splits batches by language for models using composition layers or where languages are filtered
    if is_composition_model or excluded_languages:
        batch_generator = _filter_split_raw_batches_by_language(batch_generator, test_data, excluded_languages)
    else:
        # Add language batch only
        batch_generator = (
            GeneratedBatch(batch, [test_data.language(language_id) for language_id in map(int, batch.language_ids)])
            for batch in batch_generator
        )

    with (
        JsonlWriter(
            arguments.output,
            PredictionMetaData(
                str(arguments),
                arguments.dataset_type,
                test_languages,
                config.nn.projection.feature_set,
                attribute_indexer.state(),
                feature_names,
                {language: test_data.inventory(language) for language in test_languages},
            ),
            gzip=arguments.compress,
        ) as writer,
        tqdm(
            total=len(dataset)
            - sum(len(test_data.monolingual_index_range(language)) for language in excluded_languages),
            unit=" utterances",
            disable=arguments.no_progress,
        ) as progress_bar,
    ):
        for batch, language_batch in batch_generator:
            batch = batch.to(evaluation_device, non_blocking=True)

            if per_language_decoders is not None:
                language_decoder = per_language_decoders[language_batch[0]]
                ctc_decoders[composition_output_name] = language_decoder.decoder
                model_outputs = inference_estimator.predict(
                    batch, language_decoder.feature_matrix.to(evaluation_device)
                )
            else:
                language_decoder = None
                model_outputs = inference_estimator.predict(batch)

            if map_allophones:
                # Predict in inference mode
                with torch.inference_mode():
                    model_outputs.outputs[ProjectionEntryConfig.PHONEME_LAYER] = inference_estimator.map_allophones(
                        model_outputs.outputs[ProjectionEntryConfig.PHONE], batch.language_ids
                    )

            label_batches = batch.raw_labels
            transposed_label_batches = []
            prediction_batches = {}
            output_lengths = model_outputs.lengths

            for name, decoder in ctc_decoders.items():
                outputs = model_outputs.outputs[name]
                # Decode using emission probabilities from model log probabilities
                # Note: flashlight CTCDecoder requires emissions to be contiguous
                beam_results = decoder(
                    outputs.transpose(1, 0).contiguous(),
                    output_lengths,
                )
                prediction_batches: Dict[str, List[List[List[str]]]]
                prediction_batches[name] = prediction_batch = []

                # Check if it's either a phone or phoneme layer that outputs IPA characters
                is_ipa_layer = name in _IPA_LAYER
                for result in range(len(beam_results)):
                    candidates = []
                    for beam_index in range(n_candidates):
                        # Get most probable beam labels
                        outputs = beam_results[result][beam_index].tokens
                        # Map indices to phonemes
                        if is_ipa_layer:
                            actual_indices: np.ndarray = outputs.long().numpy()
                            # Phone logits are offset by one for the blank label
                            if language_decoder is None:
                                if name == ProjectionEntryConfig.PHONE:
                                    if phone_indexer is None:
                                        raise ValueError("Missing phone indexer for allophone layer")
                                    actual = phone_indexer.phoneme(actual_indices - 1).tolist()
                                else:
                                    actual = attribute_indexer.phoneme(actual_indices - 1).tolist()
                            else:
                                actual = language_decoder.attributes.phoneme(actual_indices - 1).tolist()

                            expected = [batch[result] for batch in label_batches]
                            transposed_label_batches.append(expected)
                        else:
                            # Subtracts blank start index from outputs and gets features as strings for future compatibility
                            actual = attribute_indexer.feature_values(name, outputs - model_output_start_offset)

                        candidates.append(actual)

                    prediction_batch.append(candidates)

                # Write utterance level predictions from the batch
                for index, (utterance_id, label_batch, language) in enumerate(
                    zip(batch.utterance_ids, transposed_label_batches, language_batch)
                ):
                    writer.write(
                        UtterancePrediction(
                            language,
                            utterance_id,
                            {name: batch[index] for name, batch in prediction_batches.items()},
                            label_batch,
                        )
                    )

            progress_bar.update(len(batch))


class TrainingLanguageMode(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    ONLY = "only"


def make_parser() -> ArgumentParser:
    needs_config_parser = ArgumentParser(add_help=False)
    needs_config_parser.add_argument(
        "-c",
        "--config",
        type=utils.argparse_type_wrapper(toml.load),
        default=toml.load(DEFAULT_CONFIG_PATH),
        help=(
            "Path to a configuration file in toml format, "
            "usually modified from a default config generated using generate-config"
        ),
    )
    needs_config_parser.add_argument(
        "-j",
        "--config-json-data",
        default=None,
        help="Parses configuration files passed to `-c/--config` using JSON instead of TOML",
    )

    progress_parser = ArgumentParser(add_help=False)
    progress_parser.add_argument(
        "--no-progress", action="store_true", help="Disables step level progress printing and progress bars"
    )

    fix_inventory_parser = ArgumentParser(add_help=False)
    fix_inventory_parser.add_argument(
        "--fix-unicode",
        action="store_true",
        help="Attempts resolving phonemes by performing unicode normalization",
    )

    dataset_processing_parser = ArgumentParser(add_help=False)
    dataset_processing_parser.add_argument(
        "dataset_path", help="Path to a corpus containing phonetically transcribed utterance"
    )
    dataset_processing_parser.add_argument(
        "-w",
        "--data-workers",
        type=int,
        help=(
            "Number of workers - 0 disables workers and runs data processing synchronously on the main thread, "
            "by default the number of workers is auto-detected based on CPU thread count"
        ),
    )
    dataset_processing_parser.add_argument("--cpu", action="store_true", help="Uses the CPU instead of CUDA GPUs")

    parser = ArgumentParser(description="Trains and evaluates universal phoneme recognizer models")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {version(__package__)}")

    subparsers = parser.add_subparsers(dest="mode")
    subparsers.required = True

    train_parser = subparsers.add_parser(
        "train",
        parents=[needs_config_parser, dataset_processing_parser, progress_parser],
        help="Trains a universal phoneme recognizer model",
        description="Trains a universal phoneme recognizer model",
    )
    train_parser.add_argument(
        "-a", "--attribute-path", help="Path to a (processed) feature table", type=FileType("r", encoding="utf-8")
    )
    train_parser.add_argument(
        "-t", "--dataset-type", choices={"common-voice"}, default="common-voice", help="Type of the Dataset"
    )
    train_parser.add_argument(
        "-s",
        "--save-path",
        help=(
            "Path the best model will be saved to during training (usually ending in .pt) "
            "or the directory that checkpoints will be saved in if the `-a/--save-all` flag is set"
        ),
    )
    train_parser.add_argument(
        "-r",
        "--restore",
        action="store_true",
        help="Restores the model from --save-path instead of initializing a new model from scratch",
    )
    train_parser.add_argument(
        "-n",
        "--no-validation",
        action="store_true",
        help="Disables validation during training and stops based on training error instead",
    )
    train_parser.add_argument(
        "-d",
        "--save-all",
        action="store_true",
        help=(
            "Saves all checkpoints in a directory instead of only saving the best one. "
            'Checkpoints are numbered by step and epoch and the best checkpoint is copied to "best.pt"'
        ),
    )
    train_parser.add_argument(
        "-b",
        "--tensorboard",
        nargs="?",
        const="",
        action="store",
        dest="tensorboard_directory",
        help=(
            "Enables tensorboard summary writing. If the optional argument is provided, "
            "it will be used as the log dir over the default ./runs/(run_name) directory naming"
        ),
    )
    train_parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Exits with a stack trace if an anomaly is detected during backpropagation such as NaN gradients",
    )

    preprocessed_group = train_parser.add_mutually_exclusive_group()
    preprocessed_group.add_argument(
        "-f",
        "--features",
        help="Path to the precomputed features and feature lengths generated via the datasets 'preprocess' command",
    )
    preprocessed_group.add_argument(
        "-l",
        "--lengths",
        help='Feature lengths of each utterance - only required if batching_mode is set to "utterances"',
    )

    config_parser = subparsers.add_parser(
        "generate-config",
        help="Generates a configuration file from a default template",
        description="Generates a configuration file from a default template",
    )
    config_parser.add_argument(
        "config_path",
        nargs="?",
        default=None,
        help="Path to the new config file\nThe configuration file will be written to stdout if no provided",
    )

    error_analysis_parser = ArgumentParser(add_help=False)
    error_analysis_parser.add_argument(
        "prediction_path",
        type=FileType("rb"),
        default=sys.stdin.buffer,
        help="Path to a file containing labeled predictions from `predict`",
    )
    error_analysis_parser.add_argument(
        "-d",
        "--decompress",
        default=None,
        action="store_true",
        help="Forces the inputs to be decompressed with gzip. This can be used when gzip compression cannot be inferred from the input file e.g. when using pipes and stdin",
    )
    error_analysis_parser.add_argument(
        "--no-remap",
        action="store_true",
        help="Prevents phonemes from being remapped with a training set to target scheme before computing edit statistics",
    )
    error_analysis_parser.add_argument(
        "-s",
        "--split-complex",
        action="store_true",
        help="Splits all complex segments in the predictions and reference before comparison. If `--no-remap` isn't specified, splitting is performed after remapping",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluates a trained model on a supported dataset",
        description="Evaluates a trained model on a supported dataset",
        parents=[progress_parser, fix_inventory_parser, error_analysis_parser],
    )
    evaluate_parser.add_argument("-j", "--json", action="store_true", help="Outputs evaluation results in json format")
    evaluate_parser.add_argument(
        "-o", "--output", type=FileType("x", encoding="utf-8"), help="Writes evaluation outputs to the given file"
    )

    compressed_output_parser = ArgumentParser(add_help=False)
    compressed_output_parser.add_argument(
        "-o", "--output", type=FileType("xb"), default=sys.stdout.buffer, help="Writes output to the given file"
    )
    compressed_output_parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        default=None,
        help="Uses gzip compression for outputs if it can't be inferred from a *.gz extension",
    )

    # Parser for computing and storing Levensthein edit operations
    subparsers.add_parser(
        "edits",
        help="Computes the lowest cost Levensthein edit operations for detailed analysis",
        description="Computes the lowest cost Levensthein edit operations for detailed analysis",
        parents=[progress_parser, fix_inventory_parser, error_analysis_parser, compressed_output_parser],
    )

    predict_parser = subparsers.add_parser(
        "predict", parents=[dataset_processing_parser, progress_parser, fix_inventory_parser, compressed_output_parser]
    )
    predict_parser.add_argument(
        "--training-languages",
        action=EnumAction,
        type=TrainingLanguageMode,
        default=TrainingLanguageMode.INCLUDE,
        help="Whether to only evaluate on languages the model was trained on, include or exclude them from the training set",
    )
    predict_parser.add_argument("model_path", help="Huggingface model ID or path to the model checkpoint for transcribing the data")
    predict_parser.add_argument(
        "-t",
        "--dataset-type",
        choices={"common-voice", "ucla-phonetic"},
        default="ucla-phonetic",
        help="Type of the evaluation dataset",
    )
    predict_parser.add_argument(
        "-l",
        "--lengths",
        help='Feature lengths of each utterance - only required if batching_mode is set to "utterances"',
    )
    predict_parser.add_argument(
        "-s",
        "--batch-size",
        type=int,
        default=None,
        help="Batch size used for evaluation\nIf not given, the same batch size as during training will be used",
    )
    predict_parser.add_argument(
        "-m", "--batch-mode", action=EnumAction, type=BatchingMode, default=BatchingMode.FRAMES, help="Type of batching"
    )
    predict_parser.add_argument(
        "-f",
        "--feature-subset",
        type=lambda codes: codes.split(","),
        help="Comma separated subet of features (including phoneme) to evaluate on",
    )
    predict_parser.add_argument("-b", "--ctc-beam", type=int, default=1, help="Beam size used during beam decoding")
    predict_parser.add_argument(
        "-n",
        "--n-best",
        type=int,
        default=1,
        help="Uses the best result from the `n` highest scoring beams\nCan be at most as large as the given `--ctc-beam/-b`",
    )
    predict_parser.add_argument(
        "--language-phonemes",
        action="store_true",
        default=None,
        help=(
            "For allophone models, the outputs of the language specific phoneme classifiers are used instead of the universal phone classifier."
            " Raises an error if utterances are encountered that are in a language for which no language specific phoneme layer exists in the model"
        ),
    )

    return parser


def main(args: Sequence[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    parser = make_parser()
    arguments = parser.parse_args(args)
    # Handles cases in which the configuration is provided in json format
    if hasattr(arguments, "config_json_data") and arguments.config_json_data is not None:
        arguments.config = json.loads(arguments.config_json_data)

    match arguments.mode:
        case "generate-config":
            generate_config(arguments)
        case "train":
            train_network(arguments)
        case "evaluate":
            evaluate(arguments)
        case "predict":
            predict(arguments)
        case "edits":
            edits(arguments)
        case mode:
            raise ValueError(f"Unsupported action: {mode}")
