import typing
from typing import Any, Callable, Generic, Sequence, Type, TypeVar, Union


class CsvSchemaError(Exception):
    """Raised when a CSV row doesn't match the given schema"""


T = TypeVar("T")


class CsvSchema(Generic[T]):
    def __init__(self, output_class: Type[T], converters: Sequence[Callable[[str], Any]]) -> None:
        self._converters = converters
        self._output_class = output_class

    def convert_line(self, line: Sequence[str]) -> T:
        if len(line) != len(self._converters):
            raise CsvSchemaError(f"Number of columns doesn't match, expected {len(self._converters)}, got {len(line)}")

        return self._output_class(*(converter(column) for converter, column in zip(self._converters, line)))


def _optional_string(output_class: Type[T]) -> Callable[[str], T | None]:
    def converter(string: str) -> T | None:
        return output_class(string) if string else None

    return converter


def make_schema(dataclass: Type[T]) -> CsvSchema[T]:
    converters = []
    for type_hint in typing.get_type_hints(dataclass).values():
        if (
            typing.get_origin(type_hint) is Union
            and len(args := typing.get_args(type_hint)) == 2
            and args[1] == type(None)
        ):
            converters.append(_optional_string(args[0]))
        else:
            converters.append(type_hint)

    return CsvSchema(dataclass, converters)
