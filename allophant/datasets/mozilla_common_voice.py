from collections.abc import Callable, KeysView, Iterable, Iterator, Sequence
import contextlib
import csv
from dataclasses import dataclass, field
from datetime import date
from importlib.metadata import version
import itertools
import json
import os
from os import path
from typing import Any, ClassVar, Dict, Generic, List, Literal, Optional, Tuple, Type, TypeVar, Union, overload

from marshmallow import EXCLUDE, Schema
from marshmallow.fields import Raw
import marshmallow_dataclass
import requests
import msgpack
from msgpack import ExtType
from mashumaro.mixins.msgpack import DataClassMessagePackMixin
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

from allophant import utils
import allophant
from allophant.config import FeatureSet
from allophant.datasets.phonemes import PhonemeTranscriber, PhonemeTranscription, PhonemeSource, TaggedTranscription
from allophant.datasets.speech_corpus import (
    LanguageData,
    LanguageInfo,
    MultilingualCorpus,
    MultilingualSplits,
    PhoneticallyTranscribedUtterance,
)
from allophant import csv_validator
from allophant.csv_validator import CsvSchema
from allophant.phoneme_segmentation import SEGMENTATION_LOGGER
from allophant.phonetic_features import PhoneticAttributeIndexer
from allophant.utils import AnyPath, CamelCasingSchema, MarshmallowDataclassLoadMixin


def _custom_data_key(key: str) -> Any:
    return field(metadata={"data_key": key})


def _optional_zero_field() -> Any:
    return field(default=0, metadata={"allow_none": True})


@dataclass
class CategoryStatistics:
    accent: Dict[str, float]
    age: Dict[str, float]
    gender: Dict[str, float]


@dataclass
class SplitSizes:
    dev: int = 0
    invalidated: int = 0
    other: int = 0
    reported: int = 0
    test: int = 0
    train: int = 0
    validated: int = 0


@dataclass
class LocaleMeta:
    buckets: SplitSizes
    clips: int
    splits: CategoryStatistics
    users: int
    size: int
    checksum: Optional[str]
    avg_duration_secs: float = _optional_zero_field()
    valid_duration_secs: float = _optional_zero_field()
    total_hrs: float = _optional_zero_field()
    valid_hrs: float = _optional_zero_field()
    duration: int = _optional_zero_field()
    reported_sentences: int = 0


@marshmallow_dataclass.add_schema(base_schema=CamelCasingSchema)
@dataclass
class ReleaseMeta(MarshmallowDataclassLoadMixin):
    date: date
    name: str
    multilingual: bool
    locales: Dict[str, LocaleMeta]
    total_duration: int
    total_valid_duration_secs: int
    total_hrs: int
    total_valid_hrs: int
    bundle_url_template: str = _custom_data_key("bundleURLTemplate")
    Schema: ClassVar[Type[Schema]]

    def __getitem__(self, locale: str) -> LocaleMeta:
        return self.locales[locale]

    def __iter__(self) -> Iterator[Tuple[str, LocaleMeta]]:
        yield from self.locales.items()

    def language_codes(self) -> KeysView[str]:
        return self.locales.keys()

    @classmethod
    def download(cls, version: str):
        return cls.from_json(download_release_meta(version))

    class Meta:
        unknown = EXCLUDE


LANGUAGE_META_URL_TEMPLATE = "https://commonvoice.mozilla.org/dist/releases/cv-corpus-{}.json"


def download_release_meta(version: str) -> Any:
    return requests.get(LANGUAGE_META_URL_TEMPLATE.format(version)).json()


@dataclass
class RawTranscription(DataClassMessagePackMixin):
    raw_sentence: str
    utterance_id: str
    client_id: str
    age: Optional[str]
    gender: Optional[str]
    accents: Optional[str]

    _EXT_ID = 2


@dataclass
class Transcription(RawTranscription, DataClassMessagePackMixin):
    phonemes: PhonemeTranscription

    Schema: ClassVar[Type[Schema]]
    _EXT_ID = 4


_TRANSCRIPTION_TYPES = (Transcription, RawTranscription)


def transcription_ext_encoder(obj: Any) -> ExtType:
    for transcription_type in _TRANSCRIPTION_TYPES:
        if isinstance(obj, transcription_type):
            return ExtType(transcription_type._EXT_ID, obj.to_msgpack())
    raise TypeError(f"Received object of unknown type: {type(obj)}")


def transcription_ext_decoder(code: int, data: bytes):
    for transcription_type in _TRANSCRIPTION_TYPES:
        if code == transcription_type._EXT_ID:
            return transcription_type.from_msgpack(data)
    return ExtType(code, data)


CommonVoiceRawData = LanguageData[LanguageInfo, RawTranscription]
CommonVoicePhonemeData = LanguageData[LanguageInfo, Transcription]


T = TypeVar("T", bound=Union[RawTranscription, Transcription])


@marshmallow_dataclass.add_schema
@dataclass
class CommonVoiceCorpusMeta(MarshmallowDataclassLoadMixin):
    # Uses Union and Optional instead of Python 3.10 union syntax as a workaround: https://github.com/lidatong/dataclasses-json/issues/349
    corpus_name: str
    phoneme_sources: Optional[List[str]] = None
    feature_set: Union[FeatureSet, str, None] = None
    limits: Union[Dict[str, int], int, None] = None
    utterance_counts: Dict[str, int] = field(default_factory=dict)
    package_version: str = version(allophant.__package__)


@marshmallow_dataclass.add_schema
@dataclass
class CommonVoiceCorpusSplit(Generic[T], MarshmallowDataclassLoadMixin):
    meta: CommonVoiceCorpusMeta
    inventory_mappings: Dict[str, List[Dict[str, List[str]]]]
    transcriptions: Dict[str, List[T]] = utils.schema_field(Raw())


class CommonVoiceCorpus(MultilingualCorpus, Generic[T]):
    UTTERANCE_TYPE = PhoneticallyTranscribedUtterance

    CORPUS_PATH_FILE = ".corpus_path"
    META_FILE = "meta.json"
    TRANSCRIPTION_PATTERN = "{split}_transcriptions.bin"
    INVENTORY_PATTERN = "{split}_inventories.json"

    # Sample rates identified in CommonVoice version 10.0
    _SAMPLE_RATES = [8000, 16000, 24000, 32000, 44100, 48000]
    _AUDIO_DIRECTORY = "clips"

    def __init__(
        self,
        base_directory: AnyPath,
        languages: Iterable[LanguageData[LanguageInfo, T]],
        meta_data: CommonVoiceCorpusMeta,
        resample: int | None = None,
        phoneme_sources: Sequence[PhonemeSource] | None = None,
        limits: Dict[str, int] | int | None = None,
        data_directory: AnyPath | None = None,
    ) -> None:
        super().__init__(base_directory, languages, self._AUDIO_DIRECTORY, "mp3", limits, resample, phoneme_sources)
        # Set and update meta data after corpus has been fully loaded
        meta_data.utterance_counts = {
            language_subset.info.code: len(language_subset.transcribed_samples) for language_subset in self
        }
        self._original_meta_data = meta_data
        self._meta_data = meta_data.to_json()
        self._data_directory = data_directory

        self._limits = limits

    def __str__(self) -> str:
        return f"Mozilla Common Voice Corpus containing {len(self._languages)} languages with {self._num_utterances} utterances"

    @property
    def data_directory(self) -> AnyPath | None:
        return self._data_directory

    @classmethod
    def write_corpus_path(cls, directory: AnyPath, corpus_directory: AnyPath) -> None:
        with open(path.join(directory, cls.CORPUS_PATH_FILE), "w", encoding="utf-8") as file:
            file.write(str(corpus_directory))

    @classmethod
    def get_corpus_path(cls, directory: AnyPath) -> AnyPath:
        path_file_path = path.join(directory, cls.CORPUS_PATH_FILE)
        if not path.isfile(path_file_path):
            return directory
        with open(path_file_path, "r", encoding="utf-8") as file:
            return file.read().strip("\r\n")

    @classmethod
    def _read_meta_directly(cls, direct_directory: AnyPath) -> ReleaseMeta:
        return ReleaseMeta.load(path.join(direct_directory, cls.META_FILE))

    @classmethod
    def read_meta_from(cls, directory: AnyPath) -> ReleaseMeta:
        return cls._read_meta_directly(cls.get_corpus_path(directory))

    def read_meta(self) -> ReleaseMeta:
        return self._read_meta_directly(self._base_directory)

    @classmethod
    def load_split(
        cls,
        directory: AnyPath,
        split: Literal["train"] | Literal["dev"] | Literal["test"],
        g2p_engine: PhonemeTranscriber | None = None,
        feature_set: FeatureSet | str | None = None,
        include_single_upvote_other: bool = False,
        batch_size: int = 1,
        language_codes: Sequence[str] | None = None,
        map_to_allophoible: bool = False,
        limits: Dict[str, int] | int | None = None,
        use_progress_bar: bool = False,
        mapping_threshold: int | None = None,
    ):
        # Ignore files in the top level directory like meta.json and transcriptions.json
        language_codes = (
            [sub_dir for sub_dir in os.listdir(directory) if path.isdir(path.join(directory, sub_dir))]
            if language_codes is None
            else language_codes
        )

        inventory_indexer = (
            PhoneticAttributeIndexer(
                FeatureSet.PHOIBLE,
                language_inventories=language_codes,
                allophones_from_allophoible=True,
            )
            if map_to_allophoible
            else None
        )

        if g2p_engine is not None:
            for language in language_codes:
                if not g2p_engine.supports(language):
                    raise ValueError(f"Language {language!r} not supported by all G2P engines")
                if not g2p_engine.supports_tokenization(language):
                    raise ValueError(f"No available tokenizer model for {language!r}")

        if use_progress_bar:
            # Reset logger to get warnings for each invocation
            SEGMENTATION_LOGGER.reset()

        phoneme_sources = None if g2p_engine is None else g2p_engine.phoneme_sources

        with logging_redirect_tqdm() if use_progress_bar else contextlib.nullcontext():
            return cls(
                directory,
                _load_common_voice_splits(
                    directory,
                    language_codes,
                    [split, _OTHER_SPLIT] if include_single_upvote_other else [split],
                    g2p_engine,
                    batch_size,
                    inventory_indexer,
                    limits,
                    use_progress_bar,
                    mapping_threshold,
                ),
                CommonVoiceCorpusMeta(
                    path.basename(path.normpath(directory)),
                    None if phoneme_sources is None else [source.value for source in phoneme_sources],
                    feature_set,
                    limits,
                ),
                phoneme_sources=phoneme_sources,
            )

    def transcriptions_to_json(self) -> Dict[str, Dict[str, Any]]:
        transcriptions = {}
        inventory_mappings = {}
        for language in self.languages:
            subset = self.monolingual_subset(language)
            transcriptions[language] = subset.transcribed_samples
            inventory_mappings[language] = subset.info.phoneme_mappings

        return CommonVoiceCorpusSplit(
            self._original_meta_data,
            inventory_mappings,
            transcriptions,
        ).to_json()

    def save(self, corpus_directory: AnyPath, split: str, output_directory: Optional[AnyPath] = None) -> None:
        if output_directory is None:
            output_directory = corpus_directory
        else:
            os.mkdir(output_directory)
            self.write_corpus_path(output_directory, corpus_directory)

        with open(
            path.join(output_directory, self.TRANSCRIPTION_PATTERN.format(split=split)), "wb"
        ) as transcription_file:
            msgpack.pack(self.transcriptions_to_json(), transcription_file, default=transcription_ext_encoder)
        with open(
            path.join(output_directory, self.INVENTORY_PATTERN.format(split=split)), "w", encoding="utf-8"
        ) as inventory_file:
            json.dump(
                {language: self.monolingual_subset(language).info.phoneme_inventory for language in self.languages},
                inventory_file,
            )

    @classmethod
    def load(
        cls,
        transcriptions: Dict[str, Any],
        inventories: Dict[str, List[str]],
        data_directory: AnyPath,
        base_directory: AnyPath,
        resample: Optional[int] = None,
        languages: Optional[Sequence[str]] = None,
        limits: Dict[str, int] | int | None = None,
        only_primary_script: bool = False,
    ):
        split = CommonVoiceCorpusSplit.from_json(transcriptions)
        transcribed_utterances = split.transcriptions
        inventory_mappings = split.inventory_mappings

        if languages is None:
            allowed_languages = None
        else:
            # NOTE: Assumes language codes are premapped
            allowed_languages = set(languages)
            missing_languages = allowed_languages - transcribed_utterances.keys()
            if missing_languages:
                raise KeyError(
                    f"Languages are missing form the locally stored Common Voice dataset: {missing_languages}"
                )

        return cls(
            base_directory,
            (
                LanguageData(
                    LanguageInfo(language, inventories[language], inventory_mappings[language]),
                    # Keep only transcriptions that consist of a single script that corresponds to the primary script of the language if `only_primary_script` is enabled
                    (
                        [
                            transcription
                            for transcription in transcriptions
                            if transcription.phonemes.only_primary_script()
                        ]
                        if only_primary_script
                        else transcriptions
                    ),
                )
                for language, transcriptions in transcribed_utterances.items()
                if allowed_languages is None or language in allowed_languages
            ),
            CommonVoiceCorpusMeta(
                split.meta.corpus_name,
                split.meta.phoneme_sources,
                split.meta.feature_set,
                # Either store the current limits in the metadata or limits set during initial data preprocessing
                (limits or split.meta.limits),
                split.meta.utterance_counts,
                split.meta.package_version,
            ),
            resample,
            (
                None
                if split.meta.phoneme_sources is None
                else [PhonemeSource(variant) for variant in split.meta.phoneme_sources]
            ),
            limits,
            data_directory,
        )

    @classmethod
    def from_file(
        cls,
        data_directory: AnyPath,
        split: Union[Literal["train"], Literal["dev"], Literal["test"]],
        resample: Optional[int] = None,
        languages: Optional[Sequence[str]] = None,
        corpus_directory: Optional[AnyPath] = None,
        limits: Dict[str, int] | int | None = None,
        only_primary_script: bool = False,
        progress_bar: bool = False,
    ):
        if corpus_directory is None:
            corpus_directory = cls.get_corpus_path(data_directory)
        with (
            utils.file_and_path_wrapper(
                path.join(data_directory, cls.TRANSCRIPTION_PATTERN.format(split=split)), "rb"
            ) as file,
            (
                tqdm.wrapattr(file, "read", total=path.getsize(file.name), desc=f"Loading transcriptions for {split}")
                if progress_bar
                else file
            ) as wrapped,
        ):
            transcriptions = msgpack.unpack(wrapped, ext_hook=transcription_ext_decoder)
        with utils.file_and_path_wrapper(
            path.join(data_directory, cls.INVENTORY_PATTERN.format(split=split)), "r", encoding="utf-8"
        ) as file:
            inventories = json.load(file)

        return cls.load(
            transcriptions,
            inventories,
            data_directory,
            corpus_directory,
            resample,
            languages,
            limits,
            only_primary_script,
        )

    def load_inventories_for(self, split: str) -> Dict[str, List[str]]:
        with utils.file_and_path_wrapper(
            path.join(self._data_directory or self._base_directory, self.INVENTORY_PATTERN.format(split=split)),
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)


@dataclass
class CommonVoiceSplits(Generic[T], MultilingualSplits[CommonVoiceCorpus[T]]):
    def transcriptions_to_json(self) -> Dict[str, Any]:
        return {
            "train": self.train.transcriptions_to_json(),
            "dev": self.dev.transcriptions_to_json(),
            "test": self.test.transcriptions_to_json(),
        }

    def save(self, corpus_directory: AnyPath, output_directory: Optional[AnyPath] = None) -> None:
        if output_directory is None:
            output_directory = corpus_directory
        else:
            os.mkdir(output_directory)
            CommonVoiceCorpus.write_corpus_path(output_directory, corpus_directory)

        for split in self.SPLIT_NAMES:
            getattr(self, split).save(output_directory, split)

    @classmethod
    def load(
        cls,
        transcriptions: Dict[str, Any],
        inventories: Dict[str, List[str]],
        data_directory: AnyPath,
        base_directory: AnyPath,
        resample: int | None = None,
        languages: Sequence[str] | None = None,
        validation_limits: Dict[str, int] | int | None = None,
        only_primary_script: bool = False,
    ):
        train, dev, test = cls.SPLIT_NAMES
        return cls(
            CommonVoiceCorpus.load(
                transcriptions[train],
                inventories,
                data_directory,
                base_directory,
                resample,
                languages,
                only_primary_script=only_primary_script,
            ),
            CommonVoiceCorpus.load(
                transcriptions[dev],
                inventories,
                data_directory,
                base_directory,
                resample,
                languages,
                validation_limits,
                only_primary_script,
            ),
            CommonVoiceCorpus.load(
                transcriptions[test],
                inventories,
                data_directory,
                base_directory,
                resample,
                languages,
                only_primary_script=only_primary_script,
            ),
        )

    @classmethod
    def from_file(
        cls,
        directory: AnyPath,
        resample: Optional[int] = None,
        languages: Optional[Sequence[str]] = None,
        corpus_directory: Optional[AnyPath] = None,
        validation_limits: Dict[str, int] | int | None = None,
        only_primary_script: bool = False,
        progress_bar: bool = False,
    ):
        if corpus_directory is None:
            corpus_directory = CommonVoiceCorpus.get_corpus_path(directory)

        train, dev, test = cls.SPLIT_NAMES
        return cls(
            CommonVoiceCorpus.from_file(
                directory,
                train,
                resample,
                languages,
                corpus_directory,
                only_primary_script=only_primary_script,
                progress_bar=progress_bar,
            ),
            CommonVoiceCorpus.from_file(
                directory,
                dev,
                resample,
                languages,
                corpus_directory,
                validation_limits,
                only_primary_script,
                progress_bar,
            ),
            CommonVoiceCorpus.from_file(
                directory,
                test,
                resample,
                languages,
                corpus_directory,
                only_primary_script=only_primary_script,
                progress_bar=progress_bar,
            ),
        )


_OTHER_SPLIT = "other"


@dataclass
class _CommonVoiceEntry:
    client_id: str
    path: str
    sentence: str
    up_votes: int
    down_votes: int
    age: Optional[str]
    gender: Optional[str]
    accents: Optional[str]
    locale: str
    segment: Optional[str]

    @classmethod
    def schema(cls):
        return csv_validator.make_schema(_CommonVoiceEntry)


def _extract_entries(
    language_path: AnyPath,
    splits: Sequence[str],
    split_filters: Sequence[Callable[[_CommonVoiceEntry], bool]],
    schema: CsvSchema[_CommonVoiceEntry],
    limit: Optional[int] = None,
    use_progress_bar: bool = False,
) -> Iterator[_CommonVoiceEntry]:
    processed_utterances = 0

    for split, split_filter in zip(splits, split_filters):
        with open(path.join(language_path, split + ".tsv"), "r", encoding="utf-8") as file:
            if use_progress_bar:
                # Get the number of all entries without the header
                entry_count = sum(1 for _ in file) - 1
                file.seek(0)
                file = tqdm(
                    file,
                    # Get the number of remaining entries until the limit or the total number of entries if no limit is given
                    total=entry_count if limit is None else min(limit - processed_utterances, entry_count),
                    unit=" utterances",
                    desc=f"Processing {path.basename(language_path)}/{split}",
                    position=1,
                    leave=False,
                )
                processed_utterances += entry_count

            tsv_file = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
            # Skip header
            next(tsv_file, None)
            for i, (line, _) in enumerate(zip(tsv_file, utils.limit_indices(limit))):
                try:
                    entry = schema.convert_line(line)
                    if not split_filter(entry):
                        continue

                    yield entry
                except csv.Error as error:
                    # Schema validation errors with more details
                    raise csv.Error(f"In line {i}: {line}") from error


@overload
def _load_common_voice_splits(
    directory: AnyPath,
    languages: Sequence[str],
    splits: Sequence[str],
    g2p_engine: PhonemeTranscriber,
    batch_size: int,
    inventory_indexer: Optional[PhoneticAttributeIndexer] = None,
    limits: Optional[Union[int, Dict[str, int]]] = None,
    use_progress_bar: bool = False,
    mapping_threshold: int | None = None,
) -> Iterator[CommonVoicePhonemeData]: ...


@overload
def _load_common_voice_splits(
    directory: AnyPath,
    languages: Sequence[str],
    splits: Sequence[str],
    g2p_engine: None,
    batch_size: int,
    inventory_indexer: Optional[PhoneticAttributeIndexer] = None,
    limits: Optional[Union[int, Dict[str, int]]] = None,
    use_progress_bar: bool = False,
    mapping_threshold: int | None = None,
) -> Iterator[CommonVoiceRawData]: ...


def _load_common_voice_splits(
    directory: AnyPath,
    languages: Sequence[str],
    splits: Sequence[str],
    g2p_engine: Optional[PhonemeTranscriber] = None,
    batch_size: int = 1,
    inventory_indexer: Optional[PhoneticAttributeIndexer] = None,
    limits: Optional[Union[int, Dict[str, int]]] = None,
    use_progress_bar: bool = False,
    mapping_threshold: int | None = None,
) -> Iterator[Union[CommonVoiceRawData, CommonVoicePhonemeData]]:
    def true(_: _CommonVoiceEntry) -> Literal[True]:
        return True

    def positive_score(entry: _CommonVoiceEntry) -> bool:
        return (entry.up_votes - entry.down_votes) > 1

    schema = _CommonVoiceEntry.schema()

    split_filters = [positive_score if split == _OTHER_SPLIT else true for split in splits]
    for language in tqdm(languages, position=0, unit=" languages") if use_progress_bar else languages:
        SEGMENTATION_LOGGER.context_language(language)
        language_path = path.join(directory, language)
        phoneme_inventories = [set() for _ in range(g2p_engine.num_engines if g2p_engine is not None else 0)]
        transcribed = []
        language_entries = _extract_entries(
            language_path,
            splits,
            split_filters,
            schema,
            utils.global_or_local_limit(limits, language),
            use_progress_bar,
        )

        if g2p_engine is None:
            for entry in language_entries:
                transcribed.append(
                    RawTranscription(
                        entry.sentence,
                        path.splitext(entry.path)[0],
                        entry.client_id,
                        entry.age,
                        entry.gender,
                        entry.accents,
                    )
                )
        else:
            for entries, transcription_batches in g2p_engine.extractor(language).auto_batch_g2p_transcribe(
                language_entries, batch_size
            ):
                for entry, transcriptions in zip(entries, transcription_batches):
                    for inventory, transcription in zip(
                        phoneme_inventories, transcriptions.flattened_primary_transcriptions()
                    ):
                        inventory.update(transcription)
                    transcribed.append(
                        Transcription(
                            entry.sentence,
                            path.splitext(entry.path)[0],
                            entry.client_id,
                            entry.age,
                            entry.gender,
                            entry.accents,
                            transcriptions,
                        )
                    )

        # Sort phoneme inventories for consistency
        phoneme_inventories = [sorted(inventory) for inventory in phoneme_inventories]
        # Create phoneme mapping and remap transcriptions for each engine
        if inventory_indexer is None:
            phoneme_inventory = sorted(phoneme for inventory in phoneme_inventories for phoneme in inventory)
            inventory_mappings = []
        else:
            inventory_mappings = inventory_indexer.map_language_inventory(
                phoneme_inventories, language, distance_threshold=mapping_threshold
            )

            # Only store phonemes in the inventory that occur after remapping
            phoneme_inventory = sorted(
                set(
                    mapped_phoneme
                    for mapping, inventory in zip(inventory_mappings, phoneme_inventories)
                    for phoneme in inventory
                    for mapped_phoneme in mapping[phoneme]
                )
            )

            # Remap phonemes from all transcriptions
            for transcription in transcribed:
                transcription.phonemes.phonemes = [
                    [
                        (
                            TaggedTranscription(
                                list(
                                    itertools.chain.from_iterable(mapping[phoneme] for phoneme in segment.transcription)
                                ),
                                segment.language,
                            )
                            if segment.language is None
                            else segment
                        )
                        for segment in phoneme_transcriptions
                    ]
                    for mapping, phoneme_transcriptions in zip(inventory_mappings, transcription.phonemes.phonemes)
                ]

        yield LanguageData(
            LanguageInfo(
                language,
                # Merge original inventories from multiple engines together
                phoneme_inventory,
                inventory_mappings,
            ),
            transcribed,
        )


@overload
def load_common_voice(
    directory: AnyPath,
    g2p_engine: PhonemeTranscriber,
    feature_set: FeatureSet | str,
    include_single_upvote_other: bool,
    batch_size: int,
    languages: Optional[Sequence[str]],
    map_to_allophoible: bool = False,
    training_limits: Optional[Union[int, Dict[str, int]]] = None,
    use_progress_bar: bool = False,
    mapping_threshold: int | None = None,
) -> CommonVoiceSplits[Transcription]: ...


@overload
def load_common_voice(
    directory: AnyPath,
    g2p_engine: None,
    feature_set: None,
    include_single_upvote_other: bool,
    batch_size: int,
    languages: Optional[Sequence[str]],
    map_to_allophoible: bool = False,
    training_limits: Optional[Union[int, Dict[str, int]]] = None,
    use_progress_bar: bool = False,
    mapping_threshold: int | None = None,
) -> CommonVoiceSplits[RawTranscription]: ...


def load_common_voice(
    directory: AnyPath,
    g2p_engine: Optional[PhonemeTranscriber] = None,
    feature_set: FeatureSet | str | None = None,
    include_single_upvote_other: bool = False,
    batch_size: int = 1,
    languages: Optional[Sequence[str]] = None,
    map_to_allophoible: bool = False,
    training_limits: Optional[Union[int, Dict[str, int]]] = None,
    use_progress_bar: bool = False,
    mapping_threshold: int | None = None,
) -> CommonVoiceSplits:
    if use_progress_bar:
        # Share logger memory over all three slits
        SEGMENTATION_LOGGER.suppress_resets(3)

    train_split = CommonVoiceCorpus.load_split(
        directory,
        "train",
        g2p_engine,
        feature_set,
        include_single_upvote_other,
        batch_size,
        languages,
        map_to_allophoible,
        training_limits,
        use_progress_bar,
        mapping_threshold,
    )
    return CommonVoiceSplits(
        train_split,
        CommonVoiceCorpus.load_split(
            directory,
            "dev",
            g2p_engine,
            feature_set,
            map_to_allophoible=map_to_allophoible,
            batch_size=batch_size,
            language_codes=train_split.languages,
            use_progress_bar=use_progress_bar,
            mapping_threshold=mapping_threshold,
        ),
        # Don't remap test phonemes regardless of settings for more freedom at test time
        CommonVoiceCorpus.load_split(
            directory,
            "test",
            g2p_engine,
            feature_set,
            map_to_allophoible=False,
            batch_size=batch_size,
            language_codes=train_split.languages,
            use_progress_bar=use_progress_bar,
        ),
    )
