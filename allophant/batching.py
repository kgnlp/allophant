from collections.abc import Callable, Iterable, Sequence
import random
import typing
from typing import Any, Iterator, List, TypeVar

from torch import Generator, Tensor, LongTensor
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler
from torch.nn.utils import rnn
import numpy as np
from allophant import MAIN_LOGGER, utils

from allophant.config import BatchingMode
from allophant.dataset_processing import Batch, BatchType, LabeledBatch, PhonemeDataset, RawLabeledBatch
from allophant.datasets.speech_corpus import MultilingualCorpus


class _SequentialSamplerWrapper:
    """Wrapper class for the _LanguageBinSampler to get a consistent signature with `RandomSampler`"""

    def __init__(self, indices: range, generator: Generator | None = None) -> None:
        self._sampler = SequentialSampler(indices)

    def __iter__(self) -> Iterator[int]:
        yield from self._sampler


class _LanguageBinSampler:
    def __init__(self, language_indices: range, shuffle: bool = False, generator: Generator | None = None) -> None:
        self._sampler_type = RandomSampler if shuffle else _SequentialSamplerWrapper
        self._generator = generator
        self._language_indices = language_indices
        self._sampler = iter(self._sampler_type(language_indices, generator=self._generator))
        self._start_offset = language_indices.start

    def _next_sample(self) -> int:
        return next(self._sampler) + self._start_offset

    def sample(self) -> int:
        try:
            return self._next_sample()
        except StopIteration:
            # Recreate the sample without replacement from the start again
            self._sampler = iter(self._sampler_type(self._language_indices, generator=self._generator))
            return self._next_sample()


class LanguageOversamplingSampler(Sampler[int]):
    """
    Sampler using language oversampling based on XLS-R pre-training (Babu et al., 2021)

    .. references::
        Babu, Arun, Changhan Wang, Andros Tjandra, Kushal Lakhotia,
        Qiantong Xu, Naman Goyal, Kritika Singh et al. "XLS-R:
        Self-supervised cross-lingual speech representation learning at
        scale." arXiv preprint arXiv:2111.09296 (2021).
    """

    def __init__(
        self,
        corpus: MultilingualCorpus,
        oversampling_factor: float = 0.5,
        shuffle: bool = False,
        generator: Generator | None = None,
    ) -> None:
        self._language_bins = []
        utterance_counts = []
        for language in corpus.languages:
            indices = corpus.monolingual_index_range(language)
            self._language_bins.append(_LanguageBinSampler(indices, shuffle, generator))
            utterance_counts.append(len(indices))

        utterance_counts = torch.tensor(utterance_counts, dtype=torch.float32)
        utterance_weights = (utterance_counts / utterance_counts.sum()) ** oversampling_factor
        highest_resource_index = utterance_weights.argmax()
        highest_resource_factor = utterance_counts[highest_resource_index] / utterance_weights[highest_resource_index]
        # Gets the number of samples by ensuring that all utterances of the highest resource language are sampled at least once
        # and utterances from all other languages according to the distribution
        expected_samples = (highest_resource_factor * utterance_weights).round().long()
        bin_indices = torch.repeat_interleave(torch.arange(len(expected_samples)), expected_samples)

        self._samples_per_epoch = int(expected_samples.sum().item())
        self._bin_indices = bin_indices
        self._language_sampler = RandomSampler(bin_indices, generator=generator)

    def __iter__(self) -> Iterator[int]:
        for language_index in self._language_sampler:
            yield self._language_bins[self._bin_indices[language_index]].sample()

    def __len__(self) -> int:
        return self._samples_per_epoch


class MaxFrameBatchSampler(BatchSampler):
    """
    :py:class:`~torch.utils.data.BatchSampler` implementation for batching
    tensors together until the product of their sequence and batch dimension
    would grow larger than a maximum number of frames. This ensures consistent memory usage
    where batches use as much memory as possible below a maximum
    """

    def __init__(self, sampler: Sampler[int] | Iterable[int], batch_size: int, frame_lengths: Tensor) -> None:
        """
        Creates a batch sampler with the given batch size in maximum number of
        frames and a tensor of lengths for every sequence in the dataset.

        :param sampler: A :py:class:`~torch.utils.data.Sampler` for generating (random) data indices
        :param batch_size: The maximum number of frames that the product of the batch and sequence dimension cannot exceed
        :param frame_lengths: A 1d tensor of lengths for every sample in the dataset.
            The tensor should be at least as long as the largest index that can be generated from the `sampler`
        """
        self._sampler = sampler
        self._batch_size = batch_size
        self._frame_lengths = frame_lengths

    def __iter__(self) -> Iterator[List[int]]:
        batch_indices = []
        max_length = 0

        for index in self._sampler:
            length = self._frame_lengths[index]
            if length > max_length:
                max_length = length

            # Predicted batch size dependent on the dense size if the current word was included
            batch_size = len(batch_indices)
            new_batch_size = (batch_size + 1) * max_length
            # Return the current batch if the batch size would be too large with the current word
            if new_batch_size > self._batch_size:
                yield batch_indices
                # New batch with the current word which wasn't included in the last one
                max_length = length
                batch_indices = [index]
            else:
                batch_indices.append(index)

        # Handle final batch if necessary
        if batch_indices:
            yield batch_indices


class SkipBatchSampler(BatchSampler):
    def __init__(self, sampler: BatchSampler, skip_count: int) -> None:
        self._sampler = sampler
        self._skip_count = skip_count

    def __iter__(self) -> Iterator[List[int]]:
        samples = iter(self._sampler)
        if self._skip_count == 0:
            return samples

        skipped = 0
        for indices, _ in zip(samples, range(self._skip_count)):
            skipped += len(indices)

        if skipped:
            MAIN_LOGGER.info(f"Skipped {skipped}")

        return samples


def _build_batch(batch_type: BatchType) -> Callable[[Sequence[Batch]], Batch]:
    """
    Builds a dense batch from a sequence of single entry batches

    :param entries: A sequence of batches where each batch only contains a single entry

    :return: A dense batch with all entries merged together in (padded) tensors
    """

    def _create_batch(entries: Sequence[T]) -> Batch:
        lengths = LongTensor([entry.lengths for entry in entries])
        language_ids = LongTensor([entry.language_ids for entry in entries])
        audio_features = rnn.pad_sequence([entry.audio_features for entry in entries], True)
        # Only transpose if there is a feature dimension
        if audio_features.ndim > 2:
            audio_features = audio_features.transpose(1, 2)
        match batch_type:
            case BatchType.UNLABELED:
                return Batch(audio_features, lengths, language_ids)
            case BatchType.RAW:
                # Redefine entries with a narrower type - required since Sequence isn't invariant
                labeled_entries = typing.cast(Sequence[RawLabeledBatch], entries)
                return RawLabeledBatch(
                    audio_features,
                    lengths,
                    language_ids,
                    [list(labels) for labels in zip(*(entry.raw_labels[0] for entry in labeled_entries))],
                    [entry.utterance_ids[0] for entry in labeled_entries],
                )

        # Redefine entries with a narrower type - required since Sequence isn't invariant
        labeled_entries = typing.cast(Sequence[LabeledBatch], entries)
        label_lengths = []
        attribute_indices = []

        if entries:
            num_engines = len(labeled_entries[0].attribute_indices)
            for engine in range(num_engines):
                label_lengths.append(torch.stack([entry.label_lengths[engine] for entry in labeled_entries], 1))
                attribute_indices.append(
                    {
                        indices[0][0]: rnn.pad_sequence([tensor for _, tensor in indices], True)
                        for indices in zip(*(entry.attribute_indices[engine].items() for entry in labeled_entries))
                    }
                )

        return LabeledBatch(
            audio_features,
            lengths,
            language_ids,
            attribute_indices,
            label_lengths,
            labeled_entries[0].label_length_indices if entries else {},
        )

    return _create_batch


def _seed_worker(_worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


T = TypeVar("T", bound=Batch)


class Batcher:
    """
    Manages generation of batches for prediction and validation using utterance or frame
    count based batch sizes to ensure consistent and efficient memory usage
    """

    def __init__(
        self,
        batch_size: int,
        batching_mode: BatchingMode,
        language_oversampling_factor: float | None = None,
        data_workers: int | None = 0,
    ):
        """
        Initializes a `Batcher`

        :param batch_size: The maximum number of utterance per batch in "utterances" mode
            or the maximum number of speech frames per batch including padding in "frames" mode
        :param batching_mode: A :py:class:`~allophant.config.BatchingMode` for choosing between utterance based or frames based batch sizes
        :param language_oversampling_factor: Factor to use for oversampling from lower resource languages. Oversampling is based on XLS-R pre-training (Babu et al., 2021)
        :param data_workers: Number of worker threads used in batch generation - if `None` is
            specified the number of threads is inferred from the number available in the CPU

        .. references::
            Babu, Arun, Changhan Wang, Andros Tjandra, Kushal Lakhotia,
            Qiantong Xu, Naman Goyal, Kritika Singh et al. "XLS-R:
            Self-supervised cross-lingual speech representation learning at
            scale." arXiv preprint arXiv:2111.09296 (2021).
        """
        self._batch_size = batch_size
        self._batching_mode = batching_mode
        self._language_oversampling_factor = language_oversampling_factor
        self._data_workers = utils.get_worker_count(data_workers)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def batches(
        self,
        data: PhonemeDataset[Any, Any, T],
        data_lengths: Tensor | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        skip_batches: int = 0,
    ) -> Iterator[T]:
        """
        Generates batches given the size and mode defined in the :py:class:`Batcher`. During batching, the data is optionally shuffled.
        If the batching mode is set to use the maximum number of frames as the criterion, lengths for each utterance in the dataset need to be provided.

        :param data: The dataset for which to generate batches
        :param data_lengths: A :py:class:`~torch.Tensor` which is only required by the frame count based batching strategy
        :param shuffle: Whether to shuffle the data while batching
        :param seed: Sets a seed for the random shuffling of data when `shuffle` is enabled
        :param skip_batches: Skips the first `n` batches

        :return: A generator over labeled or unlabeled batches depending on the properties of the :py:class:`Batcher`
        """
        corpus = data.corpus
        random_generator = Generator()
        if seed is not None:
            random_generator.manual_seed(seed)

        if self._batching_mode == BatchingMode.UTTERANCES:
            if self._language_oversampling_factor is None:
                sampler = SequentialSampler(corpus)
            else:
                sampler = LanguageOversamplingSampler(
                    corpus,
                    self._language_oversampling_factor,
                    shuffle,
                    random_generator,
                )
            yield from DataLoader(
                data,
                shuffle=shuffle,
                batch_sampler=SkipBatchSampler(BatchSampler(sampler, self._batch_size, drop_last=False), skip_batches),
                collate_fn=_build_batch(data.batch_type),
                num_workers=self._data_workers,
                persistent_workers=True,
                worker_init_fn=None if seed is None else _seed_worker,
                generator=random_generator,
            )
            return

        if data_lengths is None:
            raise ValueError("Frame Lengths for each utterance are required for using max frame batching")

        if self._language_oversampling_factor is None:
            sampler = RandomSampler(corpus, generator=random_generator) if shuffle else SequentialSampler(corpus)
        else:
            sampler = LanguageOversamplingSampler(
                corpus,
                self._language_oversampling_factor,
                shuffle,
                random_generator,
            )

        yield from DataLoader(
            data,
            batch_sampler=SkipBatchSampler(
                MaxFrameBatchSampler(
                    sampler,
                    self._batch_size,
                    data_lengths,
                ),
                skip_batches,
            ),
            collate_fn=_build_batch(data.batch_type),
            num_workers=self._data_workers,
            persistent_workers=True,
            worker_init_fn=None if seed is None else _seed_worker,
            generator=random_generator,
        )
