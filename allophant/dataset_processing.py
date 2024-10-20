from abc import ABCMeta
from collections.abc import Iterable, Iterator, Set
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generic, List, Tuple, TypeVar

from torch import LongTensor, Tensor
import torch
from torch.utils.data import Dataset
import numpy as np

from allophant.config import Config
from allophant.datasets.speech_corpus import (
    IndexedEntry,
    LanguageInfo,
    MultilingualCorpus,
    PhoneticallySegmentedUtterance,
    PhoneticallyTranscribedUtterance,
)
from allophant.phonetic_features import PhonemeIndexer, PhoneticAttributeIndexer, PhoneticIndexerState
from allophant.preprocessing import FeatureFunction


@dataclass
class SamplesProcessor:
    feature_function: FeatureFunction
    attribute_indexer: PhoneticAttributeIndexer

    @property
    def feature_size(self) -> int:
        return self.feature_function.feature_size

    def indexer_state(self) -> PhoneticIndexerState:
        return self.attribute_indexer.state()

    @classmethod
    def from_config(
        cls,
        config: Config,
        sampling_rate: int,
        attribute_indexer: PhoneticAttributeIndexer,
    ):
        return cls(
            FeatureFunction.from_config(config, sampling_rate),
            attribute_indexer,
        )


@dataclass
class Batch:
    audio_features: Tensor
    lengths: Tensor
    language_ids: Tensor

    def pin_memory(self):
        self.audio_features = self.audio_features.pin_memory()
        self.lengths = self.lengths.pin_memory()
        self.language_ids = self.language_ids.pin_memory()
        return self

    def _inputs_to(
        self, device: torch.device | str | None, non_blocking: bool = False, copy: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.audio_features.to(device, non_blocking=non_blocking, copy=copy),
            self.lengths.to(device, non_blocking=non_blocking, copy=copy),
            self.language_ids.to(device, non_blocking=non_blocking, copy=copy),
        )

    def to(self, device: torch.device | str | None, non_blocking: bool = False, copy: bool = False):
        return self.__class__(*self._inputs_to(device, non_blocking, copy))

    def cuda(self, non_blocking: bool = False, copy: bool = False):
        return self.to("cuda", non_blocking, copy)

    def size(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return self.lengths.numel()

    def __repr__(self) -> str:
        return "{}(Features: ({}; {}))".format(
            self.__class__.__name__, self.audio_features.shape, self.audio_features.dtype
        )


RawLabeledBatchCls = TypeVar("RawLabeledBatchCls", bound="RawLabeledBatch")


@dataclass
class RawLabeledBatch(Batch):
    raw_labels: List[List[List[str]]]
    utterance_ids: List[str]

    def to(self, device: torch.device | str | None, non_blocking: bool = False, copy: bool = False):
        return self.__class__(
            *self._inputs_to(device, non_blocking, copy),
            self.raw_labels,
            self.utterance_ids,
        )

    def split_by_language(self: RawLabeledBatchCls) -> Iterator[Tuple[int, RawLabeledBatchCls]]:
        split_ids, split_indices = self.language_ids.unique_consecutive(return_counts=True)
        # Turn sizes into end indices for tensor split
        split_indices.cumsum_(0)

        offset = 0
        for split_id, split_index, features, lengths, language_ids in zip(
            split_ids,
            split_indices,
            self.audio_features.tensor_split(split_indices),
            self.lengths.tensor_split(split_indices),
            self.language_ids.tensor_split(split_indices),
        ):
            yield (
                split_id,
                self.__class__(
                    features[..., : lengths.max()],
                    lengths,
                    language_ids,
                    [labels[offset:split_index] for labels in self.raw_labels],
                    self.utterance_ids[offset:split_index],
                ),
            )
            offset = split_index


LabeledBatchCls = TypeVar("LabeledBatchCls", bound="LabeledBatch")


@dataclass
class LabeledBatch(Batch):
    attribute_indices: List[Dict[str, Tensor]]
    label_lengths: List[Tensor]
    label_length_indices: Dict[str, int]

    def pin_memory(self):
        super().pin_memory()
        for lengths in self.label_lengths:
            lengths.pin_memory()

        for engine in self.attribute_indices:
            for indices in engine.values():
                indices.pin_memory()

        return self

    def to(self, device: torch.device | str | None, non_blocking: bool = False, copy: bool = False):
        return self.__class__(
            *self._inputs_to(device, non_blocking, copy),
            [
                {
                    name: labels.to(device, non_blocking=non_blocking, copy=copy)
                    for name, labels in engine.items()
                    if isinstance(labels, Tensor)
                }
                for engine in self.attribute_indices
            ],
            [tensor.to(device, non_blocking=non_blocking, copy=copy) for tensor in self.label_lengths],
            label_length_indices=self.label_length_indices,
        )


class BatchType(Enum):
    UNLABELED = 0
    RAW = 1
    INDEXED = 2


I = TypeVar("I", bound=LanguageInfo)
T = TypeVar("T", bound=PhoneticallyTranscribedUtterance | PhoneticallySegmentedUtterance)
B = TypeVar("B", bound=Batch)


class PhonemeDataset(Dataset, Generic[I, T, B], metaclass=ABCMeta):
    _corpus: MultilingualCorpus[I, T]
    _processor: SamplesProcessor
    _index_start_offset: int
    _features: List[np.ndarray] | None
    _indexer: PhonemeIndexer
    _batch_type: BatchType
    _inventories: Dict[int, Set[str]] | None = None

    def phoneme_count(self) -> int:
        return len(self._processor.attribute_indexer)

    @property
    def corpus(self) -> MultilingualCorpus[I, T]:
        return self._corpus

    @property
    def unrestricted_inventory(self) -> bool:
        return True

    @property
    def indexer(self) -> PhonemeIndexer:
        return self._indexer

    @property
    def batch_type(self) -> BatchType:
        return self._batch_type

    def __len__(self) -> int:
        return len(self._corpus)

    def _filter_with_inventory(self, language_id: int, segmented_entries: Iterable[List[str]]) -> Iterable[List[str]]:
        inventory = self._inventories
        if inventory is None:
            return segmented_entries

        inventory = inventory[language_id]
        return (
            [phoneme for phoneme in segmented_sentence if phoneme in inventory]
            for segmented_sentence in segmented_entries
        )

    def _create_processed_labeled_batch(
        self, language_id: int, segmented_entries: Iterable[List[str]], features: Tensor, feature_size: Tensor
    ) -> LabeledBatch:
        engine_attribute_indices = []
        for segmented_sentence in segmented_entries:
            # Add start offset e.g. for CTC blanks
            engine_attribute_indices.append(
                self._indexer.get_named(self._indexer.phoneme_indices(segmented_sentence), self._index_start_offset)
            )

        return LabeledBatch(
            features,
            feature_size,
            torch.tensor(language_id),
            engine_attribute_indices,
            [LongTensor(list(map(len, attribute_indices.values()))) for attribute_indices in engine_attribute_indices],
            {name: index for index, name in enumerate(engine_attribute_indices[0])} if engine_attribute_indices else {},
        )

    def _generate_sample(
        self,
        index: int,
        transcribed_item: IndexedEntry[T],
        segmented_entries: Iterable[List[str]],
    ) -> B:
        language_id = transcribed_item.language_id

        if self._features is None:
            features = self._processor.feature_function(self._corpus.audio_for(transcribed_item)[0])
        else:
            # Uses torch.tensor instead of torch.from_numpy since creating tensors from non-writable numpy arrays can result in undefined behavior
            features = torch.tensor(self._features[index])

        feature_size = torch.tensor(features.shape[0])

        match self._batch_type:
            case BatchType.UNLABELED:
                return Batch(features, feature_size, torch.tensor(language_id))  # type: ignore
            case BatchType.INDEXED:
                return self._create_processed_labeled_batch(  # type: ignore
                    language_id, self._filter_with_inventory(language_id, segmented_entries), features, feature_size
                )
            case BatchType.RAW:
                return RawLabeledBatch(  # type: ignore
                    features,
                    feature_size,
                    torch.tensor(language_id),
                    [
                        [entry]
                        for entry in self._filter_with_inventory(
                            language_id,
                            segmented_entries,
                        )
                    ],
                    [transcribed_item.entry.utterance_id],
                )


P = TypeVar("P", bound=PhoneticallyTranscribedUtterance)


class TranscribedDataset(Generic[I, P, B], PhonemeDataset[I, P, B]):
    def __init__(
        self,
        batch_type: BatchType,
        corpus: MultilingualCorpus[I, P],
        processor: SamplesProcessor,
        index_start_offset: int = 0,
        features: List[np.ndarray] | None = None,
        inventories: Dict[int, Set[str]] | None = None,
        unrestricted_inventory: bool = False,
    ) -> None:
        self._batch_type = batch_type
        self._corpus = corpus
        self._processor: SamplesProcessor = processor
        self._index_start_offset = index_start_offset
        self._features = features
        self._inventories = inventories
        self._unrestricted_inventory = unrestricted_inventory
        self._indexer = (
            self._processor.attribute_indexer.full_subset_attributes
            if unrestricted_inventory
            else self._processor.attribute_indexer
        )

    @property
    def unrestricted_inventory(self) -> bool:
        return self._unrestricted_inventory

    def __getitem__(self, index: int) -> B:
        transcribed_item = self._corpus[index]
        # Flattens phonemes from each subsequence of the transcription
        transcribed = transcribed_item.entry.phonemes.flattened_transcriptions()

        return self._generate_sample(
            index,
            transcribed_item,
            transcribed,
        )


S = TypeVar("S", bound=PhoneticallySegmentedUtterance)


class PhoneticallySegmentedDataset(Generic[I, S, B], PhonemeDataset[I, S, B]):
    def __init__(
        self,
        batch_type: BatchType,
        corpus: MultilingualCorpus[I, S],
        processor: SamplesProcessor,
        index_start_offset: int = 0,
        features: List[np.ndarray] | None = None,
        inventories: Dict[int, Set[str]] | None = None,
    ) -> None:
        self._batch_type = batch_type
        self._corpus = corpus
        self._processor = processor
        self._index_start_offset = index_start_offset
        self._features = features
        self._inventories = inventories
        # Always use unrestricted target inventory for already segment datasets
        self._indexer = self._processor.attribute_indexer.full_subset_attributes

    def __getitem__(self, index: int) -> B:
        transcribed_item = self._corpus[index]
        return self._generate_sample(
            index,
            transcribed_item,
            (transcribed_item.entry.phonemes,),
        )
