from argparse import Action, ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Iterable, Sequence, Iterator
import dataclasses
from enum import Enum
import itertools
import json
import os
import re
from re import Match
import typing
from typing import Callable, ClassVar, Type, TypeVar, Dict, Generic, IO, Any, BinaryIO
from contextlib import contextmanager
from os import PathLike

from marshmallow import Schema, fields
import torch
from torch import nn
from torch import Tensor
from marshmallow.fields import Field

from allophant import MAIN_LOGGER


T = TypeVar("T")


class classproperty(Generic[T]):
    """
    Equivalent of the :py:func:`property` decorator for class variables.
    Can only be applied to class methods.
    """

    def __init__(self, getter_function: Callable[[Type], T]):
        """
        Turns the given class method into a getter function

        :param getter_function: A class method
        """
        self._getter_function = getter_function

    def __get__(self, _, outer_cls: Type) -> T:
        return self._getter_function(outer_cls)


def mask_sequence(
    lengths: Tensor, max_length: int | None = None, start: int = 0, inverse: bool = False, batch_first: bool = True
) -> Tensor:
    """
    Returns a boolean `torch.Tensor` of the shape `batch size` x `max length`
    for masking elements of variable length sequences in a dense matrix

    :param lengths: `torch.Tensor` containing the lengths of every sequence in a batch
    :param max_length: Optionally the maximum length in the batch which is allowed to be larger or smaller
        than the minimum value of `lengths` and will truncate or pad as necessary.
        Defaults to the maximum of `lengths`
    :param start: Start offset for all sequences. The resulting batch will be of shape `batch size` x (`max length` - `start`)
    :param inverse: If `True` sets all valid position to `False` instead of `True`
    :param batch_first: If `False` the batch dimension is transposed to the second dimension of the mask

    :return: A boolean sequence mask for all elements of variable length sequences in a dense matrix
    """
    if max_length is None:
        max_length = typing.cast(int, lengths.max())

    if batch_first:
        batch_indices = [0, 1]
    else:
        batch_indices = [1, 0]

    if inverse:
        return torch.arange(start, max_length, device=lengths.device).unsqueeze(batch_indices[0]) >= lengths.unsqueeze(
            batch_indices[1]
        )
    return torch.arange(start, max_length, device=lengths.device).unsqueeze(batch_indices[0]) < lengths.unsqueeze(
        batch_indices[1]
    )


AnyPath = str | PathLike[str]
PathOrFile = str | PathLike[str] | IO[Any]
PathOrFileBinary = str | PathLike[str] | BinaryIO


@contextmanager
def file_and_path_wrapper(path_or_file: PathOrFile, mode: str, encoding: str | None = None) -> Iterator[IO[Any]]:
    """
    Context manager unifying string paths and file objects. If a path is given
    it is opened using the given `mode` and `encoding` while file objects are returned directly

    :param path_or_file: Either a string path or a file object
    :param mode: Mode used when opening a file if a path is given (See `open`)
    :param encoding: Encoding used when opening a file if a path is given (See `open`)

    :return: A file object in a generator context either directly corresponding to `path_or_file` or newly opened if a string path is given
    """
    if isinstance(path_or_file, (str, PathLike)):
        with open(path_or_file, mode, encoding=encoding) as file:
            yield file
    else:
        yield path_or_file


def file_from(path_or_file: PathOrFile, mode: str, encoding: str | None = None) -> IO[Any]:
    """
    Either directly returns the provided file object or opens a file if `path_or_file` is a path

    :param path_or_file: Either a string path or a file object
    :param mode: Mode used when opening a file if a path is given (See `open`)
    :param encoding: Encoding used when opening a file if a path is given (See `open`)

    :return: A file object either directly corresponding to `path_or_file` or newly opened if a string path is given
    """
    if isinstance(path_or_file, (str, PathLike)):
        return open(path_or_file, mode, encoding=encoding)
    else:
        return path_or_file


def get_filepath(path_or_file: PathOrFile) -> AnyPath:
    return path_or_file if isinstance(path_or_file, (str, PathLike)) else path_or_file.name


def _handle_group_match(match: Match[str]) -> str:
    # NOTE: Handles only plain named or empty placeholders without format specifiers
    (name,) = match.groups()
    return rf"(?P<{name}>.*)" if name else r"(.*)"


def format_parse_pattern(format_string: str) -> str:
    return re.sub(r"{(.*?)}", _handle_group_match, format_string)


def schema_field(field: Field, metadata: Dict[str, Any] | None = None, **kwargs):
    metadata = {"marshmallow_field": field}
    if metadata is not None:
        metadata.update(metadata)

    return dataclasses.field(metadata=metadata, **kwargs)


T = TypeVar("T")


def argparse_type_wrapper(type_function: Callable[[str], T]) -> Callable[[str], T | IO[Any]]:
    def _wrapper(argument: str) -> T:
        try:
            return type_function(argument)
        except Exception as error:
            raise ArgumentTypeError(error)

    return _wrapper


class EnumAction(Action):
    def __init__(self, type: Type[Enum], **kwargs) -> None:
        self._enum = type
        # Only overrides choices if they aren't explicitly provided
        kwargs.setdefault("choices", tuple(variant.value for variant in self._enum))
        super().__init__(**kwargs)

    def __call__(
        self,
        _parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        _option_string: str | None = None,
    ) -> None:
        if values is None or isinstance(values, str):
            setattr(namespace, self.dest, self._enum(values))
        else:
            setattr(namespace, self.dest, [self._enum(value) for value in values])


class OnlineMean:
    def __init__(self):
        self._mean = 0.0
        self._total = 0

    def __iadd__(self, element: float) -> "OnlineMean":
        self._total += 1
        self._mean += (element - self._mean) / self._total
        return self

    def add_sum(self, element_sum: float, num_elements: int) -> "OnlineMean":
        self._total += num_elements
        self._mean += (element_sum - self._mean) / self._total
        return self

    def __float__(self) -> float:
        return self._mean


class Unsqueeze(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.unsqueeze(self.dimension)


class Squeeze(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.squeeze(self.dimension)


class LogCompression(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return (inputs + 1e-5).log()


def get_worker_count(workers: int | None = None) -> int:
    """
    Gets either the worker count if provided or infers and returns the number of available CPU cores.
    Not that this function currently only works on UNIX

    :param workers: Either a fixed number of workers that is kept or `None` for inferring the worker count
        from the number of available CPU cores

    :return: Either the provided number of workers or the inferred worker count if `workers` is `None`
    """
    if workers is None:
        # Infer number of available CPU cores for the process
        workers = len(os.sched_getaffinity(0))
        MAIN_LOGGER.info(f"Using {workers} workers - autodetected")

    return workers


class CamelCasingSchema(Schema):
    def on_bind_field(self, field_name: str, field: fields.Field) -> None:
        if not field.data_key:
            segments = field_name.split("_")
            field.data_key = segments[0] + "".join(segment.title() for segment in segments[1:])


MarshmallowDataclassLoadMixinCls = TypeVar("MarshmallowDataclassLoadMixinCls", bound="MarshmallowDataclassLoadMixin")


class MarshmallowDataclassLoadMixin:
    Schema: ClassVar[Type[Schema]]

    def to_json(self) -> Dict[str, Any]:
        return self.Schema().dump(self)  # type: ignore

    def dump(self, path_or_file: PathOrFile) -> None:
        with file_and_path_wrapper(path_or_file, "w", encoding="utf-8") as file:
            json.dump(self.to_json(), file)

    def dumps(self) -> str:
        return json.dumps(self.to_json())

    @classmethod
    def from_json(
        cls: Type[MarshmallowDataclassLoadMixinCls], json: Dict[str, Any]
    ) -> MarshmallowDataclassLoadMixinCls:
        return cls.Schema().load(json)  # type: ignore

    @classmethod
    def loads(cls: Type[MarshmallowDataclassLoadMixinCls], json_string: str) -> MarshmallowDataclassLoadMixinCls:
        return cls.Schema().load(json.loads(json_string))  # type: ignore

    @classmethod
    def load(cls, path_or_file: PathOrFile):
        with file_and_path_wrapper(path_or_file, "r", encoding="utf-8") as file:
            return cls.from_json(json.load(file))


def limit_indices(limits: int | None) -> Iterable[int]:
    return itertools.count() if limits is None else range(limits)


def global_or_local_limit(limits: Dict[T, int] | int | None, key: T) -> int | None:
    return limits if limits is None or isinstance(limits, int) else limits[key]


def iterator_length(iterator: Iterator) -> int:
    return sum(1 for _ in iterator)
