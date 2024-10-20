from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
import re
from typing import Dict, List, Literal, Union, overload
import typing

import zarr
from zarr import Group
import numpy as np
import torch
from torch import Tensor

from allophant import utils
from allophant.language_codes import LanguageCodeMap
from allophant.datasets.mozilla_common_voice import CommonVoiceCorpus, CommonVoiceSplits
from allophant.datasets.speech_corpus import MultilingualSplits
from allophant.datasets.ucla_phonetic_corpus import UCLAPhoneticCorpus
from allophant import utils
from allophant.utils import AnyPath


def _map_languages(directory: AnyPath, languages: Sequence[str]) -> List[str]:
    language_map = LanguageCodeMap(CommonVoiceCorpus.read_meta_from(directory).language_codes())
    return [language_map[code] for code in languages]


@overload
def load_corpus(
    path: AnyPath,
    dataset_type: Literal["ucla-phonetic"],
    resample: int | None = None,
    languages: Sequence[str] | None = None,
    validation_limits: None = None,
    only_primary_script: Literal[False] = False,
    progress_bar: bool = False,
) -> MultilingualSplits[UCLAPhoneticCorpus]: ...


@overload
def load_corpus(
    path: AnyPath,
    dataset_type: Literal["common-voice"],
    resample: int | None = None,
    languages: Sequence[str] | None = None,
    validation_limits: Dict[str, int] | int | None = None,
    only_primary_script: bool = False,
    progress_bar: bool = False,
) -> CommonVoiceSplits: ...


def load_corpus(
    path: AnyPath,
    dataset_type: Union[Literal["common-voice"], Literal["ucla-phonetic"]],
    resample: int | None = None,
    languages: Sequence[str] | None = None,
    validation_limits: Dict[str, int] | int | None = None,
    only_primary_script: bool = False,
    progress_bar: bool = False,
) -> MultilingualSplits:
    if dataset_type == "common-voice":
        # Loads the full corpus if a directory is given or a single split from a transcription file otherwise
        if os.path.isdir(path):
            if languages is not None:
                languages = _map_languages(path, languages)
            return CommonVoiceSplits.from_file(
                path,
                resample,
                languages,
                validation_limits=validation_limits,
                only_primary_script=only_primary_script,
                progress_bar=progress_bar,
            )

        data_directory = os.path.dirname(path)
        if languages is not None:
            languages = _map_languages(data_directory, languages)

        if (
            split_match := re.match(
                utils.format_parse_pattern(CommonVoiceCorpus.TRANSCRIPTION_PATTERN), os.path.basename(path)
            )
        ) is None:
            raise ValueError(f"Path is not a valid transcription path: {path!r}")

        (split,) = split_match.groups()
        if split not in CommonVoiceSplits.SPLIT_NAMES:
            raise ValueError(f"{split} is not a valid Split, must be one of {CommonVoiceSplits.SPLIT_NAMES}")

        # Manually cast since literal inference is not yet supported
        split = typing.cast(Literal["train", "dev", "test"], split)
        return CommonVoiceSplits.single(
            CommonVoiceCorpus.from_file(
                data_directory,
                split,
                resample,
                languages,
                only_primary_script=only_primary_script,
                progress_bar=progress_bar,
            ),
            split,
        )

    elif dataset_type == "ucla-phonetic":
        # Loaded as a test split
        return MultilingualSplits.single(UCLAPhoneticCorpus.load(path, resample, languages, progress_bar))

    raise ValueError(f"Corpus of type {dataset_type} is not supported")


@dataclass
class PreprocessedSplitData:
    lengths: Tensor
    features: List[np.ndarray] | None = None


def _reshape_to_features(feature_dimension: int) -> Callable[[np.ndarray], np.ndarray]:
    def reshape(array: np.ndarray) -> np.ndarray:
        return array.reshape((-1, feature_dimension))

    return reshape


def _load_features_and_lengths(
    split_data: Group, languages: Sequence[str], feature_size: int = 1
) -> PreprocessedSplitData:
    features = []
    lengths = []
    for language in languages:
        language_data = split_data[language]
        features.extend(map(_reshape_to_features(feature_size), language_data["features"]))
        lengths.append(language_data["lengths"])

    return PreprocessedSplitData(torch.from_numpy(np.concatenate(lengths)), features)


def _lengths_for_split(split_data: Group, languages: Sequence[str]) -> PreprocessedSplitData:
    return PreprocessedSplitData(
        torch.from_numpy(np.concatenate([split_data[language]["lengths"] for language in languages]))  # type: ignore
    )


def preprocessed_features_or_lengths(
    data_path: str, split_languages: Dict[str, Sequence[str]], lengths_only: bool = True
) -> Dict[str, PreprocessedSplitData]:
    feature_data = zarr.open_group(data_path, "r")
    if feature_data is None:
        raise IOError("Feature data couldn't be loaded")

    feature_size = None if lengths_only else feature_data.attrs["feature_size"]
    split_groups = []
    split_data = {}
    for split in split_languages:
        split_group = feature_data[split]
        assert isinstance(split_group, Group), f"Invalid feature format - split {split} is not a group"
        if feature_size is None:
            split_data[split] = _lengths_for_split(split_group, split_languages[split])
        else:
            split_data[split] = _load_features_and_lengths(split_group, split_languages[split], feature_size)
        split_groups.append(split_group)

    return split_data
