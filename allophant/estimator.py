from collections.abc import Set
from importlib.metadata import version
from typing import (
    Callable,
    Iterable,
    Iterator,
    Optional,
    Dict,
    Any,
    Tuple,
    Union,
    TypeVar,
    Type,
    ClassVar,
    List,
    Protocol,
)
from dataclasses import dataclass, field
from contextlib import contextmanager
import itertools
import math
from datetime import timedelta
import timeit
import contextlib
import typing
import gc
import os

import marshmallow_dataclass
import torch
from torch import profiler
from torch.profiler import ProfilerActivity
from torch import Tensor
from torch import nn
from torch import cuda
from torch.cuda.amp import autocast_mode
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
from marshmallow import Schema
from marshmallow.fields import Raw
from tqdm import tqdm
import numpy as np
import transformers

from allophant import utils
from allophant.attribute_graph import AttributeGraph, AttributeGraphField, AttributeNode
from allophant.dataset_processing import BatchType, TranscribedDataset, SamplesProcessor
from allophant.datasets.speech_corpus import AudioInfo, MultilingualCorpus, MultilingualSplits, SplitMetaData
from allophant.phonetic_features import PhoneticAttributeIndexer, PhoneticIndexerState
from allophant.utils import OnlineMean, PathOrFile, PathOrFileBinary
from allophant.network.acoustic_model import Allophant, Predictions, UnfreezeSchedule
from allophant.config import (
    Config,
    OptimizerWrapper,
    ProfilingConfig,
    ProjectionEntryConfig,
    WarmupInfo,
    Wav2Vec2PretrainedConfig,
)
from allophant.batching import Batch, Batcher, LabeledBatch
from allophant.loss_functions import LossWrapper


@dataclass
class TrainingStatus:
    """
    Named tuple representing the training status of an `Estimator` indicating
    whether training should stop and whether there was an improvement over the
    previous best epoch
    """

    stop: bool
    improvement: bool


class StoppingCriterion(Protocol):
    """Protocol for a stopping criterion for `Estimator` training"""

    def status(self, validation_losses: float) -> TrainingStatus:
        """
        Checks whether training should stop and whether there was an
        improvement over the previous best epoch based on validation losses

        :param validation_losses: Losses on the validation set on the current epoch

        :return: A `TrainingStatus` for the current epoch
        """
        raise NotImplementedError


class ImprovementTrackingCriterion(StoppingCriterion):
    """
    Basic :py:class:`StoppingCriterion` that never stops but reports validation loss improvements
    """

    def __init__(self):
        """Initializes the criterion"""
        self._minimum_losses = math.inf

    def status(self, validation_losses: float) -> TrainingStatus:
        if validation_losses < self._minimum_losses:
            self._minimum_losses = validation_losses
            return TrainingStatus(False, True)

        return TrainingStatus(False, False)


class EarlyStopping(StoppingCriterion):
    """
    Stops training after a set number of epochs without improvement on validation set loss
    """

    def __init__(self, patience: int):
        """
        Initializes early stopping with the given patience.

        :param patience: Number of epochs without improvement on validation set loss before stopping training
        """
        self.patience = patience
        self._minimum_losses = math.inf
        self._epochs_without_improvement = 0

    def status(self, validation_losses: float) -> TrainingStatus:
        if validation_losses < self._minimum_losses:
            self._minimum_losses = validation_losses
            self._epochs_without_improvement = 0
            # Continue training after improvement
            return TrainingStatus(False, True)

        self._epochs_without_improvement += 1
        # Decide whether to continue after no improvement
        return TrainingStatus(self._epochs_without_improvement == self.patience, False)


M = TypeVar("M", bound=nn.Module)


@contextmanager
def evaluation(model: M) -> Iterator[M]:
    """
    Context manager which temporarily sets a model into evaluation mode

    :param model: The target model

    :return: The `model` in evaluation mode
    """
    original_mode = model.training
    model.eval()
    yield model
    model.train(original_mode)


@dataclass
class EpochPosition:
    epoch: int = 0
    global_step: int = 0
    step: int = 0

    def next_step(self) -> None:
        self.global_step += 1
        self.step += 1

    def __str__(self) -> str:
        return f"Epoch {self.epoch}" + (
            "" if self.step is None else f", Step {self.step}" + f" | Global Step: {self.global_step}"
        )


@dataclass
class EpochStatistics:
    """Stores statistics about loss and accuracy for an epoch"""

    epoch: EpochPosition
    training_loss: float
    validation_loss: float
    training_seconds: float = 0
    validation_seconds: float = 0

    def __str__(self) -> str:
        epoch_delta = timedelta(seconds=self.training_seconds + self.validation_seconds)
        base_info = (
            f"{self.epoch}"
            f" | Training loss: {self.training_loss:.4f}"
            f" | Validation loss: {self.validation_loss:.4f}"
            f" | Δt: {epoch_delta}"
        )
        return base_info


@dataclass
class OptimizationStates:
    optimizer: Dict[str, Tensor] = utils.schema_field(Raw(default=None, missing=None))
    grad_scaler: Optional[Dict[str, Any]] = None


CheckpointCls = TypeVar("CheckpointCls", bound="Checkpoint")


@marshmallow_dataclass.add_schema
@dataclass
class Checkpoint:
    """
    Holds data for an `Estimator` checkpoint and functions for serialization and deserialization
    """

    Schema: ClassVar[Type[Schema]]

    config: Config
    allophant_version: str
    feature_size: int
    sample_rate: int
    attribute_graph: AttributeGraph = field(metadata={"marshmallow_field": AttributeGraphField()})
    epoch: EpochPosition
    phonetic_indexer_state: PhoneticIndexerState
    dataset_meta_data: List[SplitMetaData]
    model_state: Dict[str, Tensor] = utils.schema_field(Raw())
    additional: Optional[Dict[str, Any]] = utils.schema_field(Raw(default=None, missing=None))
    history: List[Tuple[TrainingStatus, EpochStatistics]] = field(default_factory=list)
    optimization_states: Optional[OptimizationStates] = None

    def save(self, file: PathOrFileBinary) -> None:
        """
        Saves the checkpoint to a path or file object

        :param file: path or file object the checkpoint is saved to
        """
        torch.save(self.Schema().dump(self), file)

    @classmethod
    def restore(
        cls: Type[CheckpointCls], file: PathOrFile, device: torch.device | str | None = None, **kwargs
    ) -> CheckpointCls:
        """
        Restores a checkpoint from a huggingface *model id*, a *path* or a
        *file object*. Additional keyword arguments are passed to :py:func:`transformers.utils.cached_file`.

        :param file: huggingface *model id*, *path* or *file object* the checkpoint is read from
        :param device: Target device for model weights, specify "cuda" for loading weights on the GPU

        :return: A `Checkpoint` which was read from the `file`
        """
        # Try loading a cached file from huggingface if a string path is given but does not exist locally
        if isinstance(file, str) and not os.path.isfile(file):
            resolved = transformers.utils.cached_file(file, "allophant.pt", **kwargs)
            if resolved is None:
                raise FileNotFoundError(f"No checkpoint found at {file!r}")
            file = resolved

        return cls.Schema().load(torch.load(file, map_location=device, weights_only=True))  # type: ignore


def split_batch_size(batch_size: int, accumulation_factor: int) -> int:
    batch_size, remainder = divmod(batch_size, accumulation_factor)
    if remainder > 0:
        raise ValueError(f"Batch size {batch_size} is not divisble by the accumulation factor {accumulation_factor}")
    return batch_size


@dataclass
class TrainDevLengths:
    train: Tensor
    development: Tensor


@dataclass
class TrainDevFeatures:
    train: List[np.ndarray]
    dev: List[np.ndarray]


def _generate_attribute_graph_data(
    phonetic_indexer: PhoneticAttributeIndexer,
    config: Config,
) -> Iterator[AttributeNode]:
    for phoneme_attribute in config.nn.projection.classes:
        yield AttributeNode(
            phoneme_attribute.name,
            phonetic_indexer.size(phoneme_attribute.name),
            phoneme_attribute.time_layer,
            phoneme_attribute.dependencies,
        )


DatasetManagerCls = TypeVar("DatasetManagerCls", bound="DatasetManager")


class DatasetManager:
    def __init__(
        self,
        config: Config,
        data_splits: MultilingualSplits[MultilingualCorpus],
        samples_processor: SamplesProcessor,
        data_lengths: TrainDevLengths | None = None,
        data_features: TrainDevFeatures | None = None,
        data_workers: int | None = 0,
        perform_validation: bool = True,
    ):
        nn_config = config.nn
        self._index_start_offset = nn_config.loss.BLANK_OFFSET
        batch_size = split_batch_size(nn_config.batch_size, nn_config.accumulation_factor)
        # Only the custom Transformer Acoustic Model requires packed batches while Wav2Vec2 uses custom processing
        self.training_batcher = Batcher(
            batch_size, nn_config.batching_mode, nn_config.language_oversampling_factor, data_workers
        )
        self.validation_batcher = Batcher(batch_size, nn_config.batching_mode, data_workers=data_workers)
        # Reuse the global seed specifically for the batcher to ensure consistent data sampling for model comparisons
        self._batching_seed = config.nn.seed

        self._data_splits = data_splits
        self._data_workers = data_workers
        self._perform_validation = perform_validation
        self._sample_processor = samples_processor
        if data_lengths is None:
            self._training_lengths = self._development_lengths = None
        else:
            self._training_lengths = data_lengths.train
            self._development_lengths = data_lengths.development

        self._training_data = self._process_dataset(
            data_splits.train, None if data_features is None else data_features.train
        )
        self._validation_data = self._process_dataset(
            data_splits.dev,
            None if data_features is None else data_features.dev,
            # Map phonemes that might not appear in the training data
            {
                language_id: set(inventory)
                for language_id, inventory in data_splits.train.language_id_inventories().inventories.items()
            },
        )

        self._resampled = config.preprocessing.resample
        self._training_set_size = len(self._data_splits.train)
        self._validation_set_size = len(self._data_splits.dev)
        self._audio_info = self._data_splits.audio_info()
        self._sample_rate = self._audio_info.sample_rate

    def _process_dataset(
        self,
        data_split: MultilingualCorpus,
        features: List[np.ndarray] | None = None,
        phoneme_mapper: Dict[int, Set[str]] | None = None,
    ) -> TranscribedDataset:
        return TranscribedDataset(
            BatchType.INDEXED, data_split, self._sample_processor, self._index_start_offset, features, phoneme_mapper
        )

    def training_batches(self, shuffle: bool = True) -> Iterator[LabeledBatch]:
        return self.training_batcher.batches(self._training_data, self._training_lengths, shuffle, self._batching_seed)

    def development_batches(self, shuffle: bool = False) -> Iterator[LabeledBatch]:
        return self.validation_batcher.batches(
            self._validation_data, self._development_lengths, shuffle, self._batching_seed
        )

    @property
    def splits(self) -> MultilingualSplits:
        return self._data_splits

    @property
    def training_set_size(self) -> int:
        return self._training_set_size

    @property
    def validation_set_size(self) -> int:
        return self._validation_set_size

    @property
    def feature_size(self) -> int:
        return self._sample_processor.feature_size

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def audio_info(self) -> AudioInfo:
        return self._audio_info

    def indexer_state(self) -> PhoneticIndexerState:
        return self._sample_processor.indexer_state()

    def attribute_graph(self, config: Config) -> AttributeGraph:
        return AttributeGraph(_generate_attribute_graph_data(self._sample_processor.attribute_indexer, config))

    @classmethod
    def from_config(
        cls: Type[DatasetManagerCls],
        config: Config,
        data_splits: MultilingualSplits,
        attribute_indexer: PhoneticAttributeIndexer,
        data_lengths: Optional[TrainDevLengths] = None,
        data_features: Optional[TrainDevFeatures] = None,
        data_workers: Optional[int] = 0,
        perform_validation: bool = True,
    ) -> DatasetManagerCls:
        sample_processor = SamplesProcessor.from_config(
            config,
            data_splits.audio_info().sample_rate,
            attribute_indexer,
        )
        return cls(config, data_splits, sample_processor, data_lengths, data_features, data_workers, perform_validation)


def profiler_trace_handler(profiling_config: ProfilingConfig) -> Callable[[Any], None]:
    if profiling_config.tensorboard_dir is not None:
        return profiler.tensorboard_trace_handler(profiling_config.tensorboard_dir)

    flame_graph_path_cpu = profiling_config.flame_graph_path_cpu
    flame_graph_path_gpu = profiling_config.flame_graph_path_gpu

    def trace_handler(profiling_scheduler) -> None:
        if flame_graph_path_cpu is not None:
            profiling_scheduler.export_stacks(flame_graph_path_cpu, "self_cpu_time_total")
        if flame_graph_path_gpu is not None:
            profiling_scheduler.export_stacks(flame_graph_path_gpu, "self_cuda_time_total")

    return trace_handler


EstimatorCls = TypeVar("EstimatorCls", bound="Estimator")


class TrainingParameters:
    def __init__(
        self,
        stopping_criterion: Optional[StoppingCriterion] = None,
        max_iterations: Optional[int] = None,
        validate: bool = True,
        progress: bool = True,
        summary_writer: Optional[SummaryWriter] = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.validate = validate
        self.progress = progress
        self.stopping_criterion = stopping_criterion
        self.summary_writer = summary_writer


@dataclass
class TrainingProgressStatistics:
    batch_mean: OnlineMean = field(default_factory=OnlineMean)
    batch_sizes: List[int] = field(default_factory=list)
    running_mean_training_losses: float = 0
    training_losses: float = 0
    training_lengths: float = 0

    def update_mean(self, batch_training_lengths: int) -> float:
        self.training_lengths += batch_training_lengths
        self.running_mean_training_losses = self.training_losses / self.training_lengths
        return self.running_mean_training_losses


class CategoryBatchStatistics:
    def __init__(self, classes: List[str]):
        self.losses = {category: 0.0 for category in classes}
        self.lengths = typing.cast(Dict[str, int], self.losses.copy())

    def add(self, name: str, losses: float, length: int) -> None:
        self.losses[name] += losses
        self.lengths[name] += length

    def sum_lengths(self) -> int:
        return sum(self.lengths.values())

    def mean_losses(self) -> Dict[str, float]:
        return {name: losses / self.lengths[name] for name, losses in self.losses.items()}


def _create_training_progress_bar(
    training_set_size: int, step_size: Optional[int] = None, progress: bool = True
) -> tqdm:
    return tqdm(
        total=training_set_size if step_size is None else step_size,
        unit="utterances" if step_size is None else "update",
        desc="Training",
        disable=not progress,
    )


def _clear_cache() -> None:
    gc.collect()
    cuda.empty_cache()


class TrainingRun:
    def __init__(
        self,
        estimator: "Estimator",
        dataset_manager: DatasetManager,
        optimizer: OptimizerWrapper,
        training_parameters: TrainingParameters | None = None,
    ) -> None:
        self._estimator = estimator
        self._config = estimator.config
        self._model = estimator.model
        self._loss_functions = estimator.loss_functions

        self._dataset_manager = dataset_manager
        self._optimizer = optimizer
        self._parameters = TrainingParameters() if training_parameters is None else training_parameters

        if self._parameters.stopping_criterion is None:
            patience = self._config.nn.early_stopping_patience
            self._parameters.stopping_criterion = (
                ImprovementTrackingCriterion() if patience is None else EarlyStopping(patience)
            )

        self._stopping_criterion = self._parameters.stopping_criterion
        self._available_memory = cuda.get_device_properties(0).total_memory
        self._max_batch_elements = 0
        acoustic_model_config = self._config.nn.acoustic_model
        if (
            isinstance(acoustic_model_config, Wav2Vec2PretrainedConfig)
            and acoustic_model_config.unfreeze_schedule is not None
        ):
            self._unfreeze_schedule = UnfreezeSchedule.from_config(acoustic_model_config.unfreeze_schedule)
        else:
            self._unfreeze_schedule = None

    def _log_training_step(
        self,
        summary_writer: SummaryWriter,
        global_step: int,
        batch_size: int,
        progress_statistics: TrainingProgressStatistics,
        category_statistics: CategoryBatchStatistics,
    ):
        summary_writer.add_scalars("Training/Loss/Classifiers", category_statistics.mean_losses(), global_step)
        summary_writer.add_scalar("Training/Loss/Mean", progress_statistics.running_mean_training_losses, global_step)
        # Compute gradient norms to log
        total_norm = 0
        for parameter in self._model.parameters():
            gradient = parameter.grad
            if gradient is None or not parameter.requires_grad:
                continue
            param_norm = gradient.detach().data.norm()
            total_norm += param_norm.item() ** 2

        summary_writer.add_scalar("Training/GradientNorm", math.sqrt(total_norm), global_step)
        summary_writer.add_scalar("Training/LearningRate", self._optimizer.current_learning_rate(), global_step)
        summary_writer.add_scalars(
            "Training/GPUMemory",
            {
                "cached": cuda.memory_reserved() / self._available_memory * 100,
                "actual": cuda.memory_allocated() / self._available_memory * 100,
                "max_cached": cuda.max_memory_reserved() / self._available_memory * 100,
                "max_actual": cuda.max_memory_allocated() / self._available_memory * 100,
            },
            global_step,
        )
        summary_writer.add_scalar("Training/Batch/Size", batch_size, global_step)
        summary_writer.add_scalar("Training/Batch/MaximumElementsWithPadding", self._max_batch_elements, global_step)
        summary_writer.flush()

    def _log_validation_step(
        self,
        summary_writer: SummaryWriter,
        global_step: int,
        validation_losses: float,
        category_statistics: CategoryBatchStatistics,
    ) -> None:
        summary_writer.add_scalars("Validation/Loss/Classifiers", category_statistics.mean_losses(), global_step)
        summary_writer.add_scalar("Validation/Loss/Mean", validation_losses, global_step)
        summary_writer.flush()

    def _finish_validation(
        self,
        epoch: EpochPosition,
        training_delta: float,
        validation_start_time: float,
        average_training_losses: float,
        average_validation_losses: float,
    ) -> Tuple[TrainingStatus, EpochStatistics]:
        validation_delta = (timeit.default_timer() - validation_start_time) if self._parameters.validate else 0
        epoch_statistics = EpochStatistics(
            epoch,
            average_training_losses,
            average_validation_losses,
            training_delta,
            validation_delta,
        )

        # Get and record state and statistics of the training step or epoch
        training_state = (self._stopping_criterion.status(epoch_statistics.validation_loss), epoch_statistics)
        self._estimator.history.append(training_state)
        return training_state

    @torch.inference_mode()
    def _end_step(
        self,
        epoch: EpochPosition,
        training_device: torch.device,
        training_start_time: float,
        average_training_losses: float,
        use_mixed_precision: bool = False,
    ) -> Tuple[TrainingStatus, EpochStatistics]:
        training_delta = timeit.default_timer() - training_start_time
        total_validation_loss = 0
        sum_validation_lengths = 0

        validation_start_time = timeit.default_timer()
        if not self._parameters.validate:
            return self._finish_validation(
                epoch,
                training_delta,
                validation_start_time,
                average_training_losses,
                # Reports training losses as validation losses if validation is disabled
                # for compatibility with stopping schedules
                average_training_losses,
            )

        with (
            evaluation(self._model),
            tqdm(
                total=self._dataset_manager.validation_set_size,
                unit="utterances",
                desc="Validation",
                leave=False,
                disable=not self._parameters.progress,
                position=0,
            ) as progress_bar,
        ):
            category_statistics = CategoryBatchStatistics(self._model.classes)
            validation_mean = 0
            batch_mean = OnlineMean()

            for batch in self._dataset_manager.development_batches():
                batch_size = len(batch)
                batch_mean += batch_size
                with autocast_mode.autocast() if use_mixed_precision else contextlib.nullcontext():
                    batch = batch.to(training_device)
                    predictions = self._model(batch)
                    validation_loss = 0
                    [batch_labels] = batch.attribute_indices
                    [label_lengths] = batch.label_lengths
                    # Silently remove phone node since it is never explicitly trained
                    predictions.outputs.pop(ProjectionEntryConfig.PHONE, None)

                    for name, output in predictions.outputs.items():
                        category_lengths = label_lengths[batch.label_length_indices[name]]
                        labels = batch_labels[name]

                        category_loss = self._loss_functions[name](
                            output,
                            labels,
                            predictions.lengths,
                            category_lengths,
                        ).item()

                        total_attribute_label_lengths = int(category_lengths.sum().item())
                        category_statistics.add(name, category_loss, total_attribute_label_lengths)

                        validation_loss += category_loss
                        sum_validation_lengths += total_attribute_label_lengths

                total_validation_loss += float(validation_loss)
                validation_mean = total_validation_loss / sum_validation_lengths

                progress_bar.update(len(batch))
                progress_bar.set_postfix({"average batch size": float(batch_mean), "loss": validation_mean})

        summary_writer = self._parameters.summary_writer
        if summary_writer is not None:
            self._log_validation_step(summary_writer, epoch.global_step, validation_mean, category_statistics)

        return self._finish_validation(
            epoch,
            training_delta,
            validation_start_time,
            average_training_losses,
            total_validation_loss / sum_validation_lengths,
        )

    def _create_epoch_counter(self, start: int = 0) -> Iterable[int]:
        if self._parameters.max_iterations is None:
            return itertools.count(start)
        return range(start, self._parameters.max_iterations + 1)

    def _scaled_backward(self, losses: Tensor) -> None:
        if self._scaler is not None:
            typing.cast(Tensor, self._scaler.scale(losses)).backward()
        else:
            losses.backward()

    def _training_batch_accumulation(
        self,
        batches: List[LabeledBatch],
        progress_statistics: TrainingProgressStatistics,
        progress_bar: tqdm,
        use_mixed_precision: bool,
        training_device: torch.device,
    ) -> CategoryBatchStatistics:
        allophone_l2_alpha = self._config.nn.projection.allophone_l2_alpha
        step_size = self._config.nn.step_size

        category_statistics = CategoryBatchStatistics(self._model.classes)
        # Number of accumulated batches before draining them gradually from the list
        batch_count = len(batches)

        while len(batches) > 0:
            batch = batches.pop().to(training_device)
            batch_size = len(batch)
            label_length_indices = batch.label_length_indices

            with autocast_mode.autocast() if use_mixed_precision else contextlib.nullcontext():
                predictions = self._model(batch)
                output_lengths = predictions.lengths
                mini_batch_training_loss = torch.tensor(0, dtype=torch.float32, device=output_lengths.device)

                [batch_labels] = batch.attribute_indices
                [label_lengths] = batch.label_lengths

                # Remove batch early
                del batch

                # Silently remove phone node since it is never explicitly trained
                predictions.outputs.pop(ProjectionEntryConfig.PHONE, None)

                for name, output in predictions.outputs.items():
                    category_lengths = label_lengths[label_length_indices[name]]
                    labels = batch_labels[name]

                    category_loss = self._loss_functions[name](
                        output,
                        labels,
                        output_lengths,
                        category_lengths,
                    )

                    # Record losses for each category and on average
                    category_statistics.add(name, category_loss.item(), int(category_lengths.sum().item()))
                    mini_batch_training_loss += category_loss

            # Accumulate gradients for each mini batch after computing the mean over losses for each frame
            mini_batch_mean_training_losses = mini_batch_training_loss / category_statistics.sum_lengths()
            self._scaled_backward(mini_batch_mean_training_losses)
            # Running loss for epoch statistics
            progress_statistics.training_losses += mini_batch_training_loss.item()

            progress_bar.update(batch_size if step_size is None else 1 / batch_count)
            progress_statistics.batch_mean += batch_size
            progress_statistics.batch_sizes.append(batch_size)

            progress_bar.set_postfix(
                {
                    "average batch size": float(progress_statistics.batch_mean),
                    "loss": float(progress_statistics.running_mean_training_losses),
                    "lr": self._optimizer.current_learning_rate(),
                }
            )

            # Adds l2 penalty after accumulating losses over the whole batch
            with autocast_mode.autocast() if use_mixed_precision else contextlib.nullcontext():
                l2_penalty = self._model.l2_penalty()
            if l2_penalty is not None:
                self._scaled_backward(allophone_l2_alpha * l2_penalty)

        return category_statistics

    def _training_step(
        self,
        batches: List[LabeledBatch],
        progress_statistics: TrainingProgressStatistics,
        progress_bar: tqdm,
        use_mixed_precision: bool,
        training_device: torch.device,
    ) -> CategoryBatchStatistics:
        category_statistics = self._training_batch_accumulation(
            batches,
            progress_statistics,
            progress_bar,
            use_mixed_precision,
            training_device,
        )

        clip_norm = self._config.nn.clip_norm

        if clip_norm is not None:
            # Unscales gradients before gradient clipping as in https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            if self._scaler is not None:
                self._scaler.unscale_(self._optimizer.optimizer)
            nn.utils.clip_grad.clip_grad_norm_(self._model.parameters(), clip_norm)

        # Perform optimization step and scaler update with mixed precision after accumulating backward passes
        if self._scaler is not None:
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()

        if self._unfreeze_schedule is not None:
            self._unfreeze_schedule.step(self._model.acoustic_model)

        progress_statistics.update_mean(category_statistics.sum_lengths())

        return category_statistics

    def __iter__(self) -> Iterator[Tuple[TrainingStatus, EpochStatistics]]:
        training_device = next(self._model.parameters()).device

        # Enables mixed precision if it's both enabled and training is performed on a GPU
        use_cuda = training_device.type == "cuda"
        use_mixed_precision = self._config.nn.mixed_precision and use_cuda
        self._scaler = GradScaler() if use_mixed_precision else None

        step_size = self._config.nn.step_size
        accumulation_factor = self._config.nn.accumulation_factor

        # Epoch starts at 1 with epoch 0 indicating no training
        epoch_position = EpochPosition()

        profiling_config = self._config.profiling
        trace_handler = None if profiling_config is None else profiler_trace_handler(profiling_config)

        progress_statistics = TrainingProgressStatistics()
        step_update = 0
        training_start_time = timeit.default_timer()

        progress_bar = _create_training_progress_bar(
            self._dataset_manager.training_set_size, step_size, self._parameters.progress
        )
        for epoch in self._create_epoch_counter(epoch_position.epoch):
            # Set epoch based on the counter
            epoch_position.epoch = epoch
            self._estimator.epoch = epoch_position

            with (
                profiler.profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if use_cuda else [ProfilerActivity.CPU],
                    schedule=profiler.schedule(wait=1, warmup=2, active=profiling_config.active_steps),
                    record_shapes=profiling_config.record_shapes,
                    profile_memory=profiling_config.profile_memory,
                    # Required by flamegraph
                    with_stack=profiling_config.flame_graph_path_cpu is not None
                    or profiling_config.flame_graph_path_gpu is not None,
                    on_trace_ready=trace_handler,
                )
                if profiling_config is not None
                else contextlib.nullcontext()
            ) as profiling_scheduler:
                if step_size is not None:
                    # Reset at the start of an epoch only if epoch based training is used
                    progress_statistics = TrainingProgressStatistics()

                # NOTE: Requires higher ulimit or copy.deepcopy in some cases due to file descriptor limit
                batch_generator = self._dataset_manager.training_batches(shuffle=True)
                while batches := list(itertools.islice(batch_generator, accumulation_factor)):
                    epoch_position.next_step()
                    step_update += 1

                    batch_size = 0
                    for batch in batches:
                        batch_size += batch.audio_features.shape[0]
                        batch_elements_count = batch.audio_features.numel()
                        if batch_elements_count > self._max_batch_elements:
                            self._max_batch_elements = batch_elements_count

                    category_statistics = self._training_step(
                        batches, progress_statistics, progress_bar, use_mixed_precision, training_device
                    )

                    # Clear cache every 100 steps
                    if not epoch_position.global_step % 100:
                        _clear_cache()

                    summary_writer = self._parameters.summary_writer
                    if summary_writer is not None:
                        self._log_training_step(
                            summary_writer,
                            epoch_position.global_step,
                            batch_size,
                            progress_statistics,
                            category_statistics,
                        )

                    # Gradient zeroing is performed after potential logging in tensorboard
                    self._optimizer.optimizer.zero_grad(set_to_none=True)

                    if profiling_scheduler is not None:
                        profiling_scheduler.step()

                    if step_size is not None and step_size == step_update:
                        # Reset step
                        step_update = 0
                        # Clear output/move line upwards
                        progress_bar.reset()
                        status, epoch_statistics = self._end_step(
                            epoch_position,
                            training_device,
                            training_start_time,
                            progress_statistics.running_mean_training_losses,
                            use_mixed_precision,
                        )
                        # Reset progress statistics
                        progress_statistics = TrainingProgressStatistics()

                        training_start_time = timeit.default_timer()
                        # Validation loss is the same as training loss if trained with validate = False
                        yield status, epoch_statistics
                        if status.stop:
                            return

                # Perform validation at an epoch step if step-level validation is disabled
                if step_size is None:
                    # Clear output/move line upwards
                    progress_bar.reset()
                    _clear_cache()
                    status, epoch_statistics = self._end_step(
                        epoch_position,
                        training_device,
                        training_start_time,
                        progress_statistics.running_mean_training_losses,
                        use_mixed_precision,
                    )
                    _clear_cache()

                    # Reset progress statistics
                    progress_statistics = TrainingProgressStatistics()

                    training_start_time = timeit.default_timer()
                    # Validation loss is the same as training loss if trained with validate = False
                    yield status, epoch_statistics
                    if status.stop:
                        return

        progress_bar.close()


@dataclass
class Estimator:
    """
    Wrapper around an acoustic model, supporting training and
    prediction as well as saving and restoring checkpoints
    """

    config: Config
    feature_size: int
    sample_rate: int
    attribute_graph: AttributeGraph
    model: Allophant
    loss_functions: Dict[str, LossWrapper]
    history: List[Tuple[TrainingStatus, EpochStatistics]] = field(default_factory=list)
    epoch: EpochPosition = field(default_factory=EpochPosition)
    dataset_meta_data: List[SplitMetaData] = field(default_factory=list)

    def __post_init__(self):
        self._scaler = None

    @classmethod
    def from_config(
        cls: Type[EstimatorCls],
        config: Config,
        feature_size: int,
        sample_rate: int,
        attribute_graph: AttributeGraph,
        attribute_indexer: PhoneticAttributeIndexer | None = None,
        device: torch.device | str = "cuda",
        load_pretrained_weights: bool = True,
    ) -> EstimatorCls:
        model = Allophant.from_config(
            config.nn,
            feature_size,
            sample_rate,
            attribute_graph,
            attribute_indexer,
            load_pretrained_weights,
        )
        model = model.to(device)

        return cls(
            config,
            feature_size,
            sample_rate,
            attribute_graph,
            model,
            config.nn.projection.loss_functions(),
        )

    def create_optimizer(self) -> OptimizerWrapper:
        wrapper = self.config.nn.optimizer.get_optimizer(self.model.parameters(), WarmupInfo(self.model.d_model))
        wrapper.add_schedulers(self.config.nn.lr_schedule)

        return wrapper

    def _prepare_optimizer(
        self, optimizer: OptimizerWrapper, optimization_states: Optional[OptimizationStates] = None
    ) -> OptimizerWrapper:
        # Add plateau schedule if necessary before restoring the optimizer state if required
        optimizer.add_schedulers(self.config.nn.lr_schedule)

        # Restore states for the optimizer and the gradient scaler
        if optimization_states is not None:
            optimizer.load_state_dict(optimization_states.optimizer)
            if self._scaler is not None and optimization_states.grad_scaler is not None:
                self._scaler.load_state_dict(optimization_states.grad_scaler)

        return optimizer

    def train(
        self,
        dataset_manager: DatasetManager,
        optimizer: OptimizerWrapper,
        stopping_criterion: Optional[StoppingCriterion] = None,
        max_iterations: Optional[int] = None,
        validate: bool = True,
        progress: bool = True,
        optimization_states: Optional[OptimizationStates] = None,
        summary_writer: Optional[SummaryWriter] = None,
    ) -> TrainingRun:
        """
        Generator which performs one step or epoch of training and validation each iteration.

        :param dataset_manager: Dataset manager containing a data reader, a phoneme preprocessing function and batchers
        :param optimizer: `OptimizerWrapper` used for training
        :param stopping_criterion: A stopping criterion which supplies the `TrainingStatus` for each iteration.
            If `TrainingStatus.stop` is `True`, the training generator will stop
        :param max_iterations: Maximum number of epochs. If not given training will only stop based on the `stopping_criterion`
        :param validate: When set to `False` validation is disabled and the
            stopping criterion will be applied to training losses instead

        :return: Yields a tuple of the `TrainingStatus` and `EpochStatistics` for each step or epoch
        """
        # Track the last dataset meta data used for training
        self.dataset_meta_data.append(dataset_manager.splits.meta_data())

        return TrainingRun(
            self,
            dataset_manager,
            self._prepare_optimizer(optimizer, optimization_states),
            TrainingParameters(stopping_criterion, max_iterations, validate, progress, summary_writer),
        )

    @torch.inference_mode()
    def predict(
        self, batch: Batch, target_feature_indices: Tensor | None = None, log_probabilities: bool = True
    ) -> Predictions:
        with evaluation(self.model):
            predictions = self.model(batch, target_feature_indices, predict=True)
            if log_probabilities:
                return Predictions(
                    {name: self.model.log_probabilities(logits) for name, logits in predictions.outputs.items()},
                    predictions.lengths,
                )
            return predictions

    def map_allophones(self, phone_logits: Tensor, language_ids: Tensor) -> Tensor:
        return self.model.map_allophones(phone_logits, language_ids)

    def save(
        self,
        file: PathOrFileBinary,
        optimizer_state: Dict[str, Tensor],
        phonetic_indexer_state: PhoneticIndexerState,
        additional_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves a checkpoint containing model parameters and configuration to a file.
        Checkpoints can be restored using `Estimator.restore`

        :param file: Path or file object the current checkpoint will be saved to
        :param optimizer_state: The `state_dict` of the optimizer used during training
        :param additional_parameters: Dictionary of additional parameters which
            can later be retrieved from the saved `Checkpoint`
        """
        Checkpoint(
            config=self.config,
            feature_size=self.feature_size,
            sample_rate=self.sample_rate,
            attribute_graph=self.attribute_graph,
            epoch=self.epoch,
            phonetic_indexer_state=phonetic_indexer_state,
            dataset_meta_data=self.dataset_meta_data,
            allophant_version=version(__package__),
            model_state=self.model.state_dict(),
            optimization_states=OptimizationStates(
                optimizer_state,
                self._scaler.state_dict() if self._scaler is not None else None,
            ),
            additional=additional_parameters,
            history=self.history,
        ).save(file)

    @classmethod
    def restore(
        cls: Type[EstimatorCls], checkpoint: Union[Checkpoint, PathOrFile], device: torch.device | str = "cuda", **kwargs
    ) -> Tuple[EstimatorCls, PhoneticAttributeIndexer]:
        """
        Restores an `Estimator` from a checkpoint previously saved using
        `Estimator.save`. Additional keyword arguments are passed to
        :py:func:`transformers.utils.cached_file`.

        :param checkpoint_or_file: A `Checkpoint` which was previously saved
            using `Estimator.save` from which the model state and config are
            restored. It can either be:
            - a `Checkpoint` instance
            - a huggingface *model id*
            - a local *path* or *file object*
        :param device: Target device for model weights, specify "cuda" for
            loading weights on the GPU

        :return: A tuple of the `Estimator` restored from the given checkpoint and the phonetic attributes used by the model
        """
        # Restore parameters
        if not isinstance(checkpoint, Checkpoint):
            checkpoint = Checkpoint.restore(checkpoint, device=device, **kwargs)

        phoneme_indexer = PhoneticAttributeIndexer.from_config(
            checkpoint.config, state_dict=checkpoint.phonetic_indexer_state
        )

        estimator = cls.from_config(
            checkpoint.config,
            checkpoint.feature_size,
            checkpoint.sample_rate,
            checkpoint.attribute_graph,
            phoneme_indexer,
            device,
            load_pretrained_weights=False,
        )
        estimator.model.load_state_dict(checkpoint.model_state)
        estimator.epoch = checkpoint.epoch
        estimator.history = checkpoint.history

        return estimator, phoneme_indexer
