from abc import ABCMeta
from dataclasses import dataclass
from io import StringIO
import itertools
import json
import logging
import typing
from typing import Dict, List, Literal, Optional, Tuple, overload
from collections.abc import Callable, Sequence, Iterator, Iterable
from argparse import ArgumentParser, FileType
from importlib import resources
import sys
import warnings
import unicodedata

import panphon
import pandas as pd
from pandas import CategoricalDtype, DataFrame, Index, Series
from pandas.io.parsers.readers import ReadCsvBuffer
from torch import LongTensor, Tensor
import torch
import numpy as np

from allophant.config import Config, PhonemeLayerType, ProjectionEntryConfig
from allophant.package_data import ALLOPHOIBLE_PATH, DEFAULT_DIALECTS_PATH
from allophant.phonemes import MissingSegmentError, IpaSegmenter
from allophant.language_codes import LanguageCode
from allophant.config import ProjectionEntryConfig, FeatureSet
from allophant import utils
from allophant.utils import AnyPath
from allophant.phoneme_segmentation import IpaSentenceSegmenter
from allophant import phoneme_segmentation
from allophant import language_codes


# IPA tie character
TIE = "\u0361"


@dataclass
class LanguageAllophoneMappings:
    allophones: Dict[int, Dict[int, List[int]]]
    languages: List[str]
    shared_phones: List[str]

    def iso6393_inventories(self, shared_phoneme_inventory: Sequence[str]) -> Dict[str, List[str]]:
        return {
            LanguageCode.from_str(language).alpha3: [
                shared_phoneme_inventory[phoneme_index] for phoneme_index in self.allophones[language_id].keys()
            ]
            for language_id, language in enumerate(self.languages)
        }

    @classmethod
    def from_allophone_data(
        cls,
        attribute_indexer: "PhoneticAttributeIndexer",
        languages: List[str],
    ):
        allophone_data = attribute_indexer.allophone_data
        if allophone_data is None:
            raise ValueError("No allophone data is available in the indexer")

        allophone_inventories = allophone_data.inventories
        shared_phone_indexer = allophone_data.shared_phone_indexer
        standardized_codes = [LanguageCode.from_str(code).alpha3 for code in languages]
        allophones = {}

        for language_id, language in enumerate(standardized_codes):
            allophone_inventory = (
                allophone_inventories.loc[allophone_inventories.ISO6393 == language, "Allophones"]
                .str.split(" ")
                .to_dict()
            )
            allophones[language_id] = {
                attribute_indexer.phoneme_index(phoneme): list(
                    map(int, shared_phone_indexer.phoneme_indices(allophones))
                )
                for phoneme, allophones in allophone_inventory.items()
            }

        return cls(allophones, languages, allophone_data.shared_phone_indexer.phonemes.tolist())


@dataclass
class LanguageInventories:
    inventories: Dict[int, List[str]]
    languages: List[str]

    def shared_inventory(self) -> List[str]:
        # Sorted for better reproducibility
        return sorted(set(itertools.chain.from_iterable(self.inventories.values())))

    def iso6393_inventories(self) -> Dict[str, List[str]]:
        return {
            LanguageCode.from_str(language).alpha3: self.inventories[language_id]
            for language_id, language in enumerate(self.languages)
        }

    def map_allophones(self, attribute_indexer: "PhonemeIndexer") -> LanguageAllophoneMappings:
        return LanguageAllophoneMappings(
            {
                language_id: {phoneme: [phoneme] for phoneme in map(int, attribute_indexer.phoneme_indices(inventory))}
                for language_id, inventory in self.inventories.items()
            },
            self.languages,
            attribute_indexer.phonemes.tolist(),
        )


@dataclass
class PhoneticIndexerState:
    phoneme_inventory: List[str]
    language_allophones: Optional[LanguageAllophoneMappings] = None
    table_file: Optional[str] = None


class PhonemeIndexer(metaclass=ABCMeta):
    _phoneme_data: DataFrame
    _feature_table: np.ndarray
    _feature_columns: Index
    _feature_names: List[str]
    _feature_categories: Dict[str, List[str]]

    @property
    def feature_table(self) -> np.ndarray:
        return self._feature_table

    @property
    def phoneme_data(self) -> DataFrame:
        return self._phoneme_data

    @property
    def phonemes(self) -> Index:
        return self._phoneme_data.index

    @property
    def feature_columns(self) -> Index:
        return self._feature_columns

    def phoneme_indices(self, phonemes: Sequence[str]) -> Tensor:
        phoneme_indices = self._phoneme_data.index.get_indexer(phonemes)
        if -1 in phoneme_indices:
            raise ValueError(f"Missing phonemes: {[phonemes[index] for index in np.where(phoneme_indices == -1)[0]]}")
        return LongTensor(phoneme_indices)

    def phoneme_indices_with_missing(self, phonemes: Sequence[str]) -> Tuple[Tensor, List[str]]:
        phoneme_indices = self._phoneme_data.index.get_indexer(phonemes)
        return LongTensor(phoneme_indices), [phonemes[index] for index in map(int, np.where(phoneme_indices == -1)[0])]

    def phoneme_index(self, phoneme: str) -> int:
        return typing.cast(int, self._phoneme_data.index.get_loc(phoneme))

    @overload
    def phoneme(self, index: int) -> str: ...

    @overload
    def phoneme(self, index: np.ndarray) -> Index: ...

    def phoneme(self, index: int | np.ndarray) -> str | Index:
        # Only compatible with np.ndarray and not torch tensors
        return typing.cast(str | Index, self._phoneme_data.index[index])

    def state(self) -> PhoneticIndexerState:
        return PhoneticIndexerState(self.phonemes.tolist())

    def segmenter(self) -> IpaSentenceSegmenter:
        return IpaSentenceSegmenter(self.phonemes.to_list())

    def __getitem__(self, index_or_name: int | Tensor | Tuple[int, Sequence[int] | int]) -> List[Tensor]:
        if isinstance(index_or_name, Tensor):
            index_or_name = index_or_name.numpy()

        feature_columns = self._feature_table[index_or_name].T
        # Special case for empty phoneme sequences
        if feature_columns.shape[1] == 0:
            return [torch.empty(0) for _ in range(len(feature_columns))]
        return [torch.from_numpy(np.concatenate(column)) for column in feature_columns]

    def get_named(
        self, index_or_name: List[str] | str | int | Tensor | np.ndarray, attribute_index_offset: int = 0
    ) -> Dict[str, Tensor]:
        if isinstance(index_or_name, list):
            indices = self.phoneme_indices(index_or_name).numpy()
        elif isinstance(index_or_name, str):
            indices = np.array([self.phoneme_index(index_or_name)])
        else:
            indices = index_or_name.numpy() if isinstance(index_or_name, Tensor) else index_or_name

        feature_matrix = self._feature_table[indices].T
        # Special case for empty phoneme sequences
        if feature_matrix.shape[1] == 0:
            return {name: torch.empty(0) for name in self._feature_columns}
        return {
            name: torch.from_numpy(np.concatenate(column)) + attribute_index_offset
            for name, column in zip(self._feature_columns, feature_matrix)
        }

    def feature_categories(self, feature: str) -> List[str]:
        return self._feature_categories[feature]

    def feature_category_index(self, name: str) -> int:
        return typing.cast(int, self._feature_columns.get_loc(name))

    def feature_values(self, name: str, feature_indices: Tensor) -> List[str]:
        categories = self._feature_categories[name]
        return [categories[index] for index in feature_indices]

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def __len__(self) -> int:
        # Phoneme count
        return self._feature_table.shape[0]


def _closest_phonemes(distance_matrix: Tensor, target_inventory: List[str]) -> Iterator[Tuple[str, int]]:
    """
    Gets the phonemes with the lowest distance for each row from a distance matrix

    :param distance_matrix: A distance matrix of size SxT where S is the number
        of source phonemes and T is the number of target phonemes. Note that T has
        to be equivalent toe the length of the `target_inventory`
    :param target_inventory: A list of phonemes in the target inventory where
        each element corresponds to a column position in the `distance_matrix`

    :return: An iterator over tuples of the closest target phonemes and their
        distance for each row in the distance matrix
    """
    return (
        (target_inventory[column_index], int(distance_matrix[row_index, column_index].item()))
        for row_index, column_index in enumerate(distance_matrix.argmin(1))
    )


PhonemeMapping = Dict[str, List[str]]


@dataclass
class DistanceMatrices:
    main: DataFrame
    splits: Dict[str, DataFrame]


class ArticulatoryAttributes(PhonemeIndexer):
    def __init__(
        self, phoneme_features: DataFrame, feature_categories: Dict[str, List[str]], reindex_phonemes: bool = True
    ):
        #  Removes phoneme from the feature set if present and keep_phonemes is False
        self._phoneme_data = phoneme_features

        if reindex_phonemes and "phoneme" in self._phoneme_data:
            # Reindex phonemes in subset order
            self._phoneme_data["phoneme"] = [np.array([index]) for index in range(len(self._phoneme_data))]
            feature_categories["phoneme"] = self._phoneme_data.index.tolist()

        self._feature_table = self._phoneme_data.values
        self._feature_columns = self._phoneme_data.columns
        # Filter feature categories with available columns, this also ensures
        # that the order should be the same as the columns in the matrix
        self._feature_categories = {name: feature_categories[name] for name in self._feature_columns}
        self._feature_names = self._feature_columns.tolist()

        # Extract the first feature of every contour for the hamming distance
        self._dense_feature_table = Tensor([contour[0] for contour in self._feature_table.flatten()]).reshape(
            self._feature_table.shape
        )

    @property
    def dense_feature_table(self) -> Tensor:
        return self._dense_feature_table

    def _phoneme_subset_fallback(self, subset_phonemes: DataFrame, phonemes: Sequence[str]) -> DataFrame:
        segmenter = self.segmenter().word_segmenter
        subset_indices, missing = self.phoneme_indices_with_missing(phonemes)
        # Replace missing indices with the indices of the first subsegments of missing phonemes
        missing_mask = subset_indices == -1
        if missing:
            subset_indices[missing_mask] = self.phoneme_indices(
                [segmenter.segment_checked(phoneme)[0] for phoneme in missing]
            )
        # Select subset
        subset_phonemes = subset_phonemes.iloc[subset_indices]
        # Replace phoneme representation with missing phonemes
        if missing:
            index = subset_phonemes.index.values.copy()
            index[missing_mask] = missing
            subset_phonemes.index = index

        return subset_phonemes

    def subset(
        self,
        phonemes: Sequence[str] | None = None,
        attribute_subset: Sequence[str] | None = None,
        reindex_phonemes: bool = True,
        missing_feature_fallback: bool = False,
    ):
        subset_phonemes = self._phoneme_data.copy()
        if phonemes is not None:
            if missing_feature_fallback:
                subset_phonemes = self._phoneme_subset_fallback(subset_phonemes, phonemes)
            else:
                subset_phonemes = subset_phonemes.iloc[self.phoneme_indices(phonemes)]
        if attribute_subset is not None:
            subset_phonemes = subset_phonemes[attribute_subset]
        # Copy subset to allow for further modification without affecting the superset
        return self.__class__(subset_phonemes, self._feature_categories.copy(), reindex_phonemes)

    def _hamming_distance(self, features: Tensor) -> Tensor:
        """
        Computes the hamming distances between a batch of feature vectors and
        the features of all available phonemes in the set

        :param feature: A matrix of phonetic feature vectors

        :return: A matrix of pairwise hamming distances between `features` and
            the phonetic feature vectors of all available phonemes
        """
        assert features.numel() > 0, "Cannot compute hamming distance without phonetic features"
        assert self._dense_feature_table.numel() > 0, "No features defined in the indexer"
        # cdist with p=0 corresponds to hamming distance
        return torch.cdist(features, self._dense_feature_table, 0)

    def _inventory_distance_matrix(
        self, source_inventory: List[str], target_inventory: List[str], add_base_count: bool = False
    ) -> Tensor:
        return torch.cdist(
            self._simplified_inventory_features(source_inventory, add_base_count),
            self._simplified_inventory_features(target_inventory, add_base_count),
            0,
        )

    @overload
    def map_inventories_closest(
        self,
        source_inventory: Sequence[str],
        target_inventory: Sequence[str],
        split_non_matching_complex: bool,
        return_distance_matrices: Literal[True],
        distance_threshold: int | None,
    ) -> Tuple[PhonemeMapping, DistanceMatrices]: ...

    @overload
    def map_inventories_closest(
        self,
        source_inventory: Sequence[str],
        target_inventory: Sequence[str],
        split_non_matching_complex: bool,
        return_distance_matrices: bool = False,
        distance_threshold: int | None = None,
    ) -> PhonemeMapping: ...

    def map_inventories_closest(
        self,
        source_inventory: Sequence[str],
        target_inventory: Sequence[str],
        split_non_matching_complex: bool = False,
        return_distance_matrices: bool = False,
        distance_threshold: int | None = None,
    ) -> Tuple[PhonemeMapping, DistanceMatrices] | PhonemeMapping:
        """
        Maps phonemes from the source inventory to the closest phonemes in the
        target inventory. Optionally splits complex segments and maps
        subsegments individually if the source phoneme is a complex segment and
        the closest target phoneme doesn't have the same number of subsegments.
        For instance, a diphthong that would be mapped to a single vowel will
        be split into its constituent vowels and each of its sub-vowels is
        mapped individually. For analysis purposes, distance matrices for the
        main inventory mapping and individual subsegment mappings can also be returned.

        :param source_inventory: A sequence of source phonemes to be mapped to the target
        :param target_inventory: The target phoneme inventory to map to
        :param split_non_matching_complex: Splits complex segments and maps
            subsegments individually if the source phoneme is a complex segment and
            the closest target phoneme doesn't have the same number of subsegments
        :param return_distance_matrices: Returns :py:class:`DistanceMatrices`
            in a tuple after phoneme mappings for the main inventory mapping and
            individual subsegment mappings if `split_non_matching_complex` is `True`
        :param distance_threshold: A phoneme mapping reverts to an identity mapping if the distance is higher or equal to the threshold.
            If complex segment splitting is active, the threshold applies to the subsegments after splitting
        :return: A :py:class:`PhonemeMapping` from the source to the target
            inventory or a tuple of a :py:class:`PhonemeMapping` and
            :py:class:`DistanceMatrices` if `return_distance_matrices` is `True`
        """
        # Map phonemes that match exactly first since different phonemes could
        # have the same features depending on the feature set and might map differently by chance otherwise
        matching = set(source_inventory).intersection(target_inventory)
        mapping = {phoneme: [phoneme] for phoneme in matching}
        # Only match source phonemes but use the full target inventory to allow for one-to-many mappings
        source_inventory = [phoneme for phoneme in source_inventory if phoneme not in matching]
        target_inventory = list(target_inventory)
        distance_matrix = self._inventory_distance_matrix(source_inventory, target_inventory, add_base_count=True)
        split_matrices = {}

        replacements = {}
        for phoneme, (target, distance) in zip(source_inventory, _closest_phonemes(distance_matrix, target_inventory)):
            phoneme_base = list(phoneme_segmentation.base_phonemes(phoneme))
            target_base = list(phoneme_segmentation.base_phonemes(target))
            # If splitting complex segments is disabled or the source and
            # target phonemes are both complex but with the same number of
            # subsegments (e.g. diphthong mapped to another diphthong)
            if not (split_non_matching_complex and len(phoneme_base) != len(target_base)):
                # Revert to an identity mapping if the distance is higher or equal to the threshold
                if distance_threshold is not None and distance >= distance_threshold:
                    target = phoneme
                # Direct source to target mapping
                replacements[phoneme] = [target]
                continue

            # Split complex segment if the target isn't a complex segment as well
            # Map subsegments individually instead
            subsegments = phoneme_segmentation.split_complex_segment(phoneme)
            split_matrix = self._inventory_distance_matrix(subsegments, target_inventory, add_base_count=True)
            # Store matrix for mapping subsegments if requested
            if return_distance_matrices:
                split_matrices[phoneme] = DataFrame(split_matrix, index=subsegments, columns=target_inventory)
            # Assign the replacement target phonemes with the lowest distance to each subsegment
            # Applies the threshold to each subsegment if provided
            replacements[phoneme] = [
                target if distance_threshold is None or distance < distance_threshold else subsegment
                for subsegment, (target, distance) in zip(
                    subsegments, _closest_phonemes(split_matrix, target_inventory)
                )
            ]

        # Merge distance based mappings with identity mappings for matching phonemes
        mapping.update(replacements)
        # Collect and warn about all target phonemes that weren't the closest phoneme to any source phoneme
        unmapped_from_target = set(target_inventory) - set(
            phoneme for phonemes in mapping.values() for phoneme in phonemes
        )
        if unmapped_from_target:
            logging.warn(f"{len(unmapped_from_target)} unmapped from target: {unmapped_from_target}")

        # Returns the general and split specific distance matrices if requested
        if return_distance_matrices:
            return (
                mapping,
                DistanceMatrices(
                    DataFrame(distance_matrix, index=source_inventory, columns=target_inventory), split_matrices
                ),
            )
        return mapping

    def closest_phone(self, features: Tensor) -> int:
        return int(self._hamming_distance(features.unsqueeze(0)).argmin())

    def closest_phone_for(self, phone: str, features: Tensor) -> int:
        # In cases where multiple phones have the same hamming distance and for performance
        if phone in self.phonemes:
            return self.phoneme_index(phone)
        return self.closest_phone(features.unsqueeze(0))

    def feature_vector(self, phone: str | int) -> np.ndarray:
        if isinstance(phone, str):
            phone = self.phoneme_index(phone)
        return self._feature_table[phone]

    def simplified_feature_vector(self, phone: str | int) -> Tensor:
        if isinstance(phone, str):
            phone = self.phoneme_index(phone)
        return self._dense_feature_table[phone]

    def _simplified_inventory_features(self, inventory: Sequence[str], add_base_count: bool = False) -> Tensor:
        features = self._dense_feature_table[self.phoneme_indices(inventory)]
        if not add_base_count:
            return features

        return torch.cat(
            (
                features,
                torch.tensor(
                    [utils.iterator_length(phoneme_segmentation.base_phonemes(segment)) for segment in inventory]
                ).unsqueeze(1),
            ),
            1,
        )

    def k_nearest_phones(self, phone_or_features: str | int | Tensor, k: int) -> Tensor:
        if not isinstance(phone_or_features, Tensor):
            phone_or_features = self.simplified_feature_vector(phone_or_features)

        # Stacks distance in the first row and indices in the second row
        return torch.stack(self._hamming_distance(phone_or_features.unsqueeze(0)).topk(k, largest=False))

    def missing_inventory_mappings(
        self, shared_inventory: Sequence[str], segment_missing: bool = False
    ) -> Dict[str, str]:
        """
        Generates a mapping between phonemes that are missing from the
        inventory due to differences in their unicode representation.

        Optionally, the mapping attempts to segment missing phonemes and chooses
        their first sub-segment only if the inventory contains all of its
        sub-segments, such as for unseen consonant clusters
        """
        segmenter = self.segmenter().word_segmenter
        mapping = {}
        for phoneme in shared_inventory:
            if phoneme not in self.phonemes:
                # Handles complications with unicode normalization
                combined_phoneme = unicodedata.normalize("NFC", phoneme)
                if combined_phoneme in self.phonemes:
                    mapping[phoneme] = combined_phoneme
                    continue

                if not segment_missing:
                    raise ValueError(f"No suitable mapping found for segment {phoneme!r}")

                # Segment phoneme and choose the first sub-segment if all
                # sub-segments are in the segmenter's phoneme inventory
                try:
                    mapping[phoneme] = segmenter.segment_checked(phoneme)[0]
                except MissingSegmentError as error:
                    # When no mapping with the other rules is possible
                    raise ValueError(f"No suitable mapping found for segment {phoneme!r}") from error

        return mapping


@dataclass
class AllophoneData:
    inventories: DataFrame
    shared_phone_indexer: ArticulatoryAttributes


def _extract_contours(column: Series):
    return column.str.split(",")


def _collect_vocabulary(column: Series, start_offset: int = 0) -> Dict[str, int]:
    return {value: index for index, value in enumerate(sorted(column.explode().unique()), start_offset)}


def _binarize_vocabulary(column: Series, vocabulary: Series) -> Series:
    current_vocabulary = vocabulary[column.name]

    def binarize_row(row: List[str]) -> np.ndarray:
        return np.array([current_vocabulary[element] for element in row], dtype=np.int64)

    return typing.cast(Series, column.apply(binarize_row))


FeatureTableInput = AnyPath | ReadCsvBuffer | str


def _binarize_contours(data: DataFrame, feature_start_column: str, vocabularies: Series | None = None) -> Series:
    attribute_contours = data.loc[:, feature_start_column:].apply(_extract_contours)
    if vocabularies is None:
        vocabularies = typing.cast(Series, attribute_contours.apply(_collect_vocabulary))

    data.loc[:, feature_start_column:] = attribute_contours.apply(_binarize_vocabulary, args=(vocabularies,))
    return vocabularies


LanguageInventoryTypes = LanguageInventories | LanguageAllophoneMappings | Sequence[str] | None


def generate_allophone_data(
    language_inventories: LanguageInventoryTypes,
    feature_table: DataFrame,
    attribute_subset: Sequence[str] | None = None,
    phoneme_subset: Sequence[str] | None = None,
) -> Tuple[DataFrame, Sequence[str]]:
    match language_inventories:
        case LanguageInventories():
            languages = language_inventories.languages
            inventories = language_inventories.iso6393_inventories()
        case LanguageAllophoneMappings():
            languages = language_inventories.languages
            if phoneme_subset is None:
                raise ValueError(
                    "allophone inventories can only be restored from LanguageAllophoneMappings if a correct phoneme_subset is provided"
                )
            inventories = language_inventories.iso6393_inventories(phoneme_subset)
        case None:
            languages = None
            inventories = None
        case language_codes:
            languages = language_codes
            inventories = None

    allophone_data = extract_allophone_inventories(
        feature_table.reset_index(),
        languages,
        attribute_subset,
        inventories,
        prefer_default_dialects=True,
        remove_zero_phoneme=True,
    ).set_index("phoneme")

    if phoneme_subset is None:
        phonemes: DataFrame = allophone_data[allophone_data["InventoryID"] != 0]
        phoneme_subset = phonemes.index.unique().tolist()

    return allophone_data, phoneme_subset


class PhoneticAttributeIndexer(PhonemeIndexer):
    def __init__(
        self,
        feature_set: FeatureSet,
        attribute_table_file: FeatureTableInput | None = None,
        attribute_subset: Sequence[str] | None = None,
        phoneme_subset: Sequence[str] | None = None,
        language_inventories: LanguageInventoryTypes = None,
        allophones_from_allophoible: bool = False,
    ):
        self._allophone_data = None

        if feature_set == FeatureSet.PHOIBLE:
            original_feature_table: DataFrame = read_allophoible(attribute_table_file, index_column="Phoneme")
            self._allophone_data, phoneme_subset = generate_allophone_data(
                language_inventories,
                original_feature_table,
                phoneme_subset=phoneme_subset,
            )

            feature_table = original_feature_table.copy(deep=True)
            feature_table.index.rename("phoneme", inplace=True)
            feature_start_column = "tone"
            phoneme_attributes = feature_table.loc[
                ~feature_table.index.duplicated(keep="first"),
                ["SegmentClass", *feature_table.columns[feature_table.columns.get_loc(feature_start_column) :]],
            ]
        elif feature_set == FeatureSet.PANPHON:
            if allophones_from_allophoible:
                raise NotImplementedError("Allophone handling is not implemented for Panphon features")
            original_feature_table = read_panphon(attribute_table_file, index_column="ipa")
            feature_start_column = "syl"
            # Removes erroneous duplicates vowels with delrel, the first strategy should only select those with delrel 0
            # See: https://github.com/dmort27/panphon/issues/26
            phoneme_attributes = original_feature_table.loc[
                ~original_feature_table.index.duplicated(keep="first"),
                original_feature_table.columns[original_feature_table.columns.get_loc(feature_start_column) :],
            ]
            # Add phoneme segments with their ties removed to handle G2P models that don't include ties
            rows_with_ties = phoneme_attributes[phoneme_attributes.index.str.contains(TIE)].copy()
            rows_with_ties.index = [segment.replace(TIE, "") for segment in rows_with_ties.index]
            phoneme_attributes = pd.concat([phoneme_attributes, rows_with_ties], verify_integrity=True)
            phoneme_attributes.index.rename("phoneme", inplace=True)
        else:
            raise ValueError(f"Unsupported feature set: {feature_set}")

        self._table_file = original_feature_table.to_csv()

        # Add phonemes as an extra column make all features categorical
        phoneme_attributes["phoneme"] = phoneme_attributes.index
        attribute_vocabularies = _binarize_contours(phoneme_attributes, feature_start_column)

        feature_categories = typing.cast(Dict[str, List[str]], attribute_vocabularies.apply(list).to_dict())
        # Full feature set used for general phoneme hamming distance
        self._full_attributes = ArticulatoryAttributes(
            phoneme_attributes.loc[:, feature_start_column:], feature_categories
        )
        # Features used for hamming distance which only includes the features
        # that are also used by the classifier - e.g. for handling inventory mapping
        # during zero-shot transfer
        self._subset_attributes = self._full_attributes.subset(phoneme_subset, attribute_subset)
        # Features with the reduced feature space but for all phonemes for generating evaluation labels with remapping
        if attribute_subset is None or "phoneme" in attribute_subset:
            full_subset = attribute_subset
        else:
            full_subset = [*attribute_subset, "phoneme"]
        self._full_phoneme_subset_attributes = self._full_attributes.subset(attribute_subset=full_subset)

        self._phoneme_data = self._subset_attributes.phoneme_data.copy()

        # Create indexing information for the subset used during training and validation
        self._feature_categories = self._subset_attributes._feature_categories
        self._feature_table = self._subset_attributes.feature_table
        self._feature_columns = self._subset_attributes.feature_columns
        self._feature_names = self._subset_attributes.feature_names

        self._feature_counts = LongTensor([len(self._feature_categories[name]) for name in self._feature_columns])
        self._total_size = int(self._feature_counts.sum().item())

        # Start column for the feature subset
        start_column = "tone" if feature_set == FeatureSet.PHOIBLE else "syl"
        feature_start_column = typing.cast(
            str,
            self._full_attributes._feature_columns[
                typing.cast(int, self._full_attributes._feature_columns.get_loc(start_column)) + 1
            ],
        )
        varying_feature_categories = self._full_attributes._feature_categories.copy()
        if feature_set == FeatureSet.PHOIBLE:
            del varying_feature_categories[start_column]

        # Keep track of all phonetic features required for embedding
        # composition by removing the "phoneme" column
        features_only = varying_feature_categories.copy()
        features_only.pop("phoneme", None)
        self._composition_features = list(features_only)

        if self._allophone_data is not None:
            _binarize_contours(self._allophone_data, feature_start_column, attribute_vocabularies)

            self._allophone_data = AllophoneData(
                self._allophone_data,
                ArticulatoryAttributes(
                    self._allophone_data.loc[~self._allophone_data.index.duplicated(keep="first"), feature_start_column:],
                    varying_feature_categories,
                ),
            )

        # Create mappings from inventories if they are given
        match language_inventories:
            case LanguageAllophoneMappings():
                self._language_allophones = language_inventories
            case LanguageInventories():
                if allophones_from_allophoible:
                    self._language_allophones = LanguageAllophoneMappings.from_allophone_data(
                        self,
                        language_inventories.languages,
                    )
                else:
                    self._language_allophones = language_inventories.map_allophones(self)
            case _:
                self._language_allophones = None

        if self._language_allophones is not None:
            self._feature_categories["phone"] = self._language_allophones.shared_phones

    def state(self) -> PhoneticIndexerState:
        return PhoneticIndexerState(self.phonemes.tolist(), self._language_allophones, self._table_file)

    @classmethod
    def from_state(
        cls,
        feature_set: FeatureSet,
        state: PhoneticIndexerState,
        feature_subset: List[str] | None = None,
    ):
        return cls(
            feature_set,
            state.table_file,
            feature_subset,
            state.phoneme_inventory,
            # Always initialize with allophone data if phoible features are used
            allophones_from_allophoible=feature_set == FeatureSet.PHOIBLE,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        attribute_table_file: FeatureTableInput | None = None,
        language_inventories: LanguageInventories | None = None,
        state_dict: PhoneticIndexerState | None = None,
    ):
        # Makes use of the guarantee that dict preserves insertion order to keep attributes ordered while set doesn't preserve order
        existing_entries = {}
        for entry in config.nn.projection.classes:
            existing_entries[entry.name] = None
            existing_entries.update((attribute, None) for attribute in entry.dependencies)

        # Remove output dependency tag since it is not a valid feature
        existing_entries.pop(ProjectionEntryConfig.OUTPUT_DEPENDENCY, None)
        for attribute in list(existing_entries):
            if ProjectionEntryConfig.OUTPUT_PATTERN.match(attribute):
                del existing_entries[attribute]

        match state_dict:
            case PhoneticIndexerState(
                phoneme_inventory, table_file=table_file, language_allophones=LanguageAllophoneMappings()
            ):
                language_allophone_mappings = state_dict.language_allophones
                phoneme_subset = phoneme_inventory
                attribute_table_file = table_file
            case _ if language_inventories is not None:
                language_allophone_mappings = language_inventories
                phoneme_subset = sorted(language_inventories.shared_inventory())
            case _:
                language_allophone_mappings = phoneme_subset = None

        return cls(
            config.nn.projection.feature_set,
            attribute_table_file,
            list(existing_entries.keys()),
            phoneme_subset,
            language_allophone_mappings,
            config.nn.projection.phoneme_layer == PhonemeLayerType.ALLOPHONES,
        )

    @property
    def composition_features(self) -> List[str]:
        return self._composition_features

    @property
    def language_allophones(self) -> LanguageAllophoneMappings | None:
        return self._language_allophones

    @property
    def attributes(self) -> ArticulatoryAttributes:
        return self._subset_attributes

    @property
    def full_attributes(self) -> ArticulatoryAttributes:
        return self._full_attributes

    @property
    def full_subset_attributes(self) -> ArticulatoryAttributes:
        return self._full_phoneme_subset_attributes

    def composition_feature_matrix(self, inventory: list[str]) -> Tensor:
        """
        Constructs the feature matrix used by the :py:class:`allophant.network.acoustic_model.EmbeddingCompositionLayer`
        for recognizing phones from a given inventory

        :param inventory: An inventory of phones supported by the Allophoible database

        :return: A matrix of feature vectors for each phone in the inventory.
            In the case of contour features, the first feature value in the contour is used
        """
        return self._full_attributes.subset(inventory, self._composition_features).dense_feature_table.long()

    def allophone_inventory(self, language_code: str) -> DataFrame:
        if self._allophone_data is None:
            raise ValueError(
                'Allophone inventories can only be accessed if features were extracted from Allophoible'
            )

        return self._allophone_data.inventories[
            self._allophone_data.inventories.ISO6393 == language_codes.standardize_to_iso6393(language_code)
        ]

    def phoneme_inventory(self, languages: Sequence[str] | str) -> list[str]:
        """
        Constructs a (shared) phoneme inventory for the given language or
        multiple languages. If a sequence of language codes is given, the union
        of phoneme inventories is returned.

        :param languages: Either a single language code or a sequence of
            language codes.

        :return: A list of phonemes from the given language or languages. If a
            sequence of language codes is given, the list contains the union of
            phoneme inventories.
        """
        if self._allophone_data is None:
            raise ValueError(
                'Allophone inventories can only be accessed if features were extracted from Allophoible'
            )

        if isinstance(languages, str):
            selection = self._allophone_data.inventories.ISO6393 == language_codes.standardize_to_iso6393(languages)
        else:
            selection = self._allophone_data.inventories.ISO6393.isin(
                {language_codes.standardize_to_iso6393(language_code) for language_code in languages}
            )

        return self._allophone_data.inventories[selection].index.unique().to_list()

    @overload
    def map_language_inventory(
        self,
        inventories: Iterable[List[str]],
        language: str,
        return_distance_matrices: Literal[True],
        distance_threshold: int | None,
    ) -> List[Tuple[PhonemeMapping, DistanceMatrices]]: ...

    @overload
    def map_language_inventory(
        self,
        inventories: Iterable[List[str]],
        language: str,
        return_distance_matrices: bool = False,
        distance_threshold: int | None = None,
    ) -> List[PhonemeMapping]: ...

    def map_language_inventory(
        self,
        inventories: Iterable[List[str]],
        language: str,
        return_distance_matrices: bool = False,
        distance_threshold: int | None = None,
    ) -> List[Tuple[PhonemeMapping, DistanceMatrices]] | List[PhonemeMapping]:
        phoneme_inventory = self.allophone_inventory(language).index.tolist()
        # Remap all phoneme inventories with the full allophoible
        # If the target inventory is smaller than the source inventory,
        # remaining phonemes are mapped to the closest one in the target inventory
        return [
            self.full_attributes.map_inventories_closest(
                inventory,
                phoneme_inventory,
                split_non_matching_complex=True,
                return_distance_matrices=return_distance_matrices,
                distance_threshold=distance_threshold,
            )
            for inventory in inventories
        ]

    @property
    def allophone_data(self) -> AllophoneData | None:
        return self._allophone_data

    def size(self, column: int | str | None = None) -> int:
        if column is None:
            return self._total_size
        if isinstance(column, str):
            column = self.feature_category_index(column)
        return int(self._feature_counts[column].item())

    def map_to_subset(self, inventory: Sequence[str]) -> Dict[str, str]:
        current_segments = self._subset_attributes
        inventory_segments = self._full_attributes.subset(inventory)

        return {
            phoneme: current_segments.phoneme(
                current_segments.closest_phone_for(phoneme, inventory_segments.simplified_feature_vector(phoneme))
            )
            for phoneme in inventory
        }

    def _phoneme_fallback(self, segmenter: IpaSegmenter, phoneme: str) -> str:
        if phoneme in self.phonemes:
            return phoneme

        return segmenter.segment_checked(phoneme)[0]

    def map_target_inventory(
        self,
        inventory: Sequence[str],
        map_uncovered_target_phonemes: bool = True,
        missing_feature_fallback: bool = False,
    ) -> Dict[str, str]:
        """
        Maps a given source inventory to the closest phones in the indexer
        based on the "tr2tgt" scheme from Xu et al. (2022).

        :param inventory: The source inventory to map to the closest phones in the indexer
        :param map_uncovered_target_phonemes: Enables mapping the closest phone
            in the source inventory to any uncovered target phone after the initial
            mapping phase as in the original "tr2tgt" scheme
        :param missing_feature_fallback: Falls back to features of the first
            subsegment of complex phones for which no features are available to
            complete the mapping

        :return: A mapping from the given source inventory to the indexer's target inventory

        .. references::
            Xu, Q., Baevski, A., Auli, M. (2022) Simple and Effective Zero-shot
            Cross-lingual Phoneme Recognition. Proc. Interspeech 2022,
            2113-2117, doi: 10.21437/Interspeech.2022-60
        """
        current_segments = self._subset_attributes
        inventory_segments = self._full_attributes.subset(inventory, missing_feature_fallback=missing_feature_fallback)
        remaining_phonemes = set(inventory)

        source_mapping = {}
        for phoneme in self.phonemes:
            target_phoneme = inventory_segments.phoneme(
                inventory_segments.closest_phone_for(phoneme, current_segments.simplified_feature_vector(phoneme))
            )
            source_mapping[phoneme] = target_phoneme
            remaining_phonemes.discard(target_phoneme)

        if map_uncovered_target_phonemes:
            for target_phoneme in remaining_phonemes:
                closest_in_source = current_segments.phoneme(
                    current_segments.closest_phone_for(
                        target_phoneme, inventory_segments.simplified_feature_vector(target_phoneme)
                    )
                )
                source_mapping[closest_in_source] = target_phoneme

        return source_mapping


def read_panphon(
    file: FeatureTableInput | None = None, index_column: str | None = None, use_categorical: bool = False
) -> DataFrame:
    column_dtype = CategoricalDtype(ordered=True) if use_categorical else str
    if file is not None:
        if isinstance(file, str):
            file = StringIO(file)
        panphon_features = pd.read_csv(file, dtype=column_dtype, index_col=index_column)
    else:
        with (resources.files(panphon) / "data/ipa_all.csv").open("r", encoding="utf-8") as default_file:
            panphon_features = pd.read_csv(default_file, dtype=column_dtype, index_col=index_column)

    return panphon_features


def read_allophoible(file: FeatureTableInput | None = None, index_column: str | None = None) -> DataFrame:
    if file is not None:
        if isinstance(file, str):
            file = StringIO(file)
        allophoible = pd.read_csv(file, dtype=str, index_col=index_column)
    else:
        with ALLOPHOIBLE_PATH.open("r", encoding="utf-8") as default_file:
            allophoible = pd.read_csv(default_file, dtype=str, index_col=index_column)

    return allophoible.astype(
        {"InventoryID": int, "SegmentClass": CategoricalDtype(["consonant", "vowel", "tone", "null"])}
    )


class LanguageMappingWarning(UserWarning):
    """Warns about languages being remapped to a closely related variant"""


class SingletonFeatureWarning(UserWarning):
    """Warns about features not varying"""


warnings.simplefilter("always", LanguageMappingWarning)
warnings.simplefilter("always", SingletonFeatureWarning)


_SOURCE_AND_LANGUAGE = ["Source", "ISO6393", "SpecificDialect"]


def _select_largest_inventories(
    non_marginal_allophones: DataFrame, preferred_dialects: Dict[str, str] | None = None
) -> DataFrame:
    data = non_marginal_allophones[_SOURCE_AND_LANGUAGE]
    if preferred_dialects is not None:
        # Selects preferred dialects and remove other dialects with the same language code
        data = pd.concat(
            [
                *(
                    data[(data.ISO6393 == language) & (data.SpecificDialect == dialect)]
                    for language, dialect in preferred_dialects.items()
                ),
                data[~data.ISO6393.isin(preferred_dialects)],
            ]
        )

    # Select the largest phoneme inventory per language
    return (
        data.groupby(_SOURCE_AND_LANGUAGE, dropna=False)
        .size()
        .sort_values(ascending=False)  # type: ignore
        .reset_index()
        .drop_duplicates("ISO6393")
    )


def _filter_inventory(
    phoible: DataFrame, remapped_inventories: Dict[str, List[str]]
) -> Callable[[DataFrame], DataFrame]:
    def inventory_filter(inventory: DataFrame) -> DataFrame:
        expected_inventory = set(remapped_inventories[inventory.name])
        inventory_subset = inventory[inventory["Phoneme"].isin(expected_inventory)]
        remaining_phonemes = expected_inventory - set(inventory_subset["Phoneme"])
        if not remaining_phonemes:
            return inventory_subset

        remaining_inventory = phoible[phoible["Phoneme"].isin(remaining_phonemes)].drop_duplicates("Phoneme").copy()
        # Assign only the phonemes as allophones
        remaining_inventory["Allophones"] = remaining_inventory["Phoneme"]
        remaining_inventory.loc[:, "InventoryID":"SpecificDialect"] = inventory_subset.loc[
            [inventory_subset.index[0]] * len(remaining_inventory), "InventoryID":"SpecificDialect"
        ].values
        remaining_inventory["Marginal"] = None
        assert len(remaining_inventory) == len(remaining_phonemes), "Inventory mismatch detected"
        return pd.concat((inventory_subset, remaining_inventory))

    return inventory_filter


def extract_allophone_inventories(
    phoible: DataFrame,
    language_codes: Sequence[str] | None = None,
    attribute_subset: Sequence[str] | None = None,
    remapped_inventories: Dict[str, List[str]] | None = None,
    prefer_default_dialects: bool = False,
    remove_zero_phoneme: bool = False,
) -> DataFrame:
    non_marginal_allophones: DataFrame = phoible[~phoible["Allophones"].isna() & (phoible["Marginal"] != "TRUE")]
    if language_codes is not None:
        language_codes_iso6393 = {LanguageCode.from_str(code).alpha3 for code in language_codes}
        filtered: DataFrame = non_marginal_allophones[non_marginal_allophones.ISO6393.isin(language_codes_iso6393)]
    else:
        language_codes_iso6393 = None
        filtered = non_marginal_allophones.copy()

    if prefer_default_dialects:
        with DEFAULT_DIALECTS_PATH.open("r", encoding="utf-8") as file:
            default_dialects = json.load(file)
    else:
        default_dialects = None

    languages = _select_largest_inventories(filtered, default_dialects)

    # Try resolving language codes with no inventories from PHOIBLE to their
    # macro language codes to find the correct inventory
    if language_codes_iso6393 is not None and len(languages) != len(language_codes_iso6393):
        phoible_language_codes = non_marginal_allophones.ISO6393.unique()
        missing_languages = {
            LanguageCode.from_str(language, True, True).alpha3_t: language
            for language in set(language_codes_iso6393) - set(languages.ISO6393)
        }
        missing_mappings = {}

        for language in phoible_language_codes:
            macro = LanguageCode.from_str(language, True, True).alpha3_t
            if macro in missing_languages:
                missing_mappings[missing_languages.pop(macro)] = language
            # Prefer languages that are already macro languages over variants even if a variant was already mapped
            elif language == macro and macro in missing_mappings:
                missing_mappings[missing_mappings[macro]] = language

        if missing_languages:
            raise ValueError(
                f"Some of the requested languages don't contain allophone data: {sorted(missing_languages.values())}"
            )

        warnings.warn(
            f"Remapped some languages to a variant within the same macro language: {missing_mappings}",
            LanguageMappingWarning,
        )
        languages = pd.concat(
            (
                languages,
                _select_largest_inventories(
                    non_marginal_allophones[non_marginal_allophones.ISO6393.isin(missing_mappings.values())],
                    default_dialects,
                ),
            )
        )
    else:
        missing_mappings = {}

    # Filter rows for all source and language pairs that were filtered in the previous step
    filtered = phoible[
        phoible.set_index(_SOURCE_AND_LANGUAGE).index.isin(languages.set_index(_SOURCE_AND_LANGUAGE).index)
    ].copy()

    # Replaces remapped languages
    filtered.ISO6393 = filtered.ISO6393.replace({macro: language for language, macro in missing_mappings.items()})

    # Filter again using remapped inventories if they are provided
    if remapped_inventories is not None:
        filtered = filtered.groupby("ISO6393").apply(_filter_inventory(phoible, remapped_inventories))

    if remove_zero_phoneme:
        # Filters out all null phonemes including surrounding whitespace in a list
        filtered["Allophones"].replace(r"( ?∅|∅ ?)", "", regex=True, inplace=True)

    unique_allophones = filtered["Allophones"].str.split(" ").explode().unique()
    unique_phonemes = set(filtered["Phoneme"].unique())
    missing_phonemes = set(unique_allophones) - unique_phonemes
    additional_phones = phoible[phoible["Phoneme"].isin(missing_phonemes)].drop_duplicates("Phoneme")

    missing_features = missing_phonemes - set(additional_phones["Phoneme"])
    if missing_features:
        raise ValueError(
            "Missing pre-computed feature definitions for", len(missing_features), "allophones:", missing_features
        )

    # Set inventory for additional phones to the unused 0 index and remove language specific information that is left over as artifacts
    additional_phones["InventoryID"] = 0
    additional_phones.loc[:, "Glottocode":"SpecificDialect"] = pd.NA
    additional_phones.loc[:, ["Source", "Allophones"]] = pd.NA

    phoible_subset = pd.concat((additional_phones, filtered))
    # Remove columns that aren't required for the feature table
    phoible_subset = phoible_subset.drop(["Marginal"], axis=1)

    # Set phonemes as the index and rename for compatibility with other lower case phonetic feature names
    phoible_subset.rename(columns={"Phoneme": "phoneme"}, inplace=True)

    last_non_feature_column = typing.cast(int, phoible_subset.columns.get_loc("Source"))
    # Filter attribute subset
    if attribute_subset is not None:
        initial_columns = phoible_subset.columns[: last_non_feature_column + 1].tolist()
        if "phoneme" in initial_columns:
            try:
                initial_columns.remove("phoneme")
            except ValueError:
                pass

        initial_columns.extend(attribute_subset)
        phoible_subset = phoible_subset.loc[:, initial_columns]

    singleton_columns = phoible_subset.iloc[:, last_non_feature_column + 1 :].nunique() <= 1
    if singleton_columns.any():
        warnings.warn(
            f"Only one feature variant found in {phoible_subset.iloc[:, last_non_feature_column + 1:].columns[singleton_columns].tolist()}",
            SingletonFeatureWarning,
        )

    return phoible_subset


def main(args: Sequence[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    parser = ArgumentParser(description="Extracts the most suitable PHOIBLE inventories for every language or a subset")
    parser.add_argument(
        "language_codes",
        nargs="?",
        type=lambda codes: codes.split(","),
        help="ISO639 language codes for which to extract inventories",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=FileType("w", encoding="utf-8"),
        default=sys.stdout,
        help="Output paths for the processed PHOIBLE CSV",
    )
    parser.add_argument(
        "-p",
        "--feature-path",
        type=FileType("r", encoding="utf-8"),
        help="Path to a custom PHOIBLE version to process instead of the included version",
    )
    parser.add_argument(
        "-r", "--remove-zero", action="store_true", help="Removes the zero phoneme from allophone positions"
    )
    parser.add_argument(
        "-d",
        "--prefer-allophant-dialects",
        action="store_true",
        help=(
            "Selects the same dialect used during Allophant pre-training over using the largest inventory "
            "for some languages with inventories for multiple dialects"
        ),
    )

    arguments = parser.parse_args(args)

    with ALLOPHOIBLE_PATH.open("r", encoding="utf-8") if not arguments.feature_path else arguments.feature_path as file:
        allophoible = read_allophoible(file)

    with arguments.out as file:
        extract_allophone_inventories(
            allophoible,
            arguments.language_codes,
            None,
            None,
            arguments.prefer_allophant_dialects,
            arguments.remove_zero,
        ).to_csv(file, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
