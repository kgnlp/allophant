from dataclasses import dataclass
from importlib.metadata import version
from typing import TypeVar, Type, List, Dict, ClassVar, TypeVar, Type
import marshmallow_dataclass

from marshmallow import fields, ValidationError, Schema

from allophant import utils
from allophant.utils import MarshmallowDataclassLoadMixin
import allophant
from allophant.phonemes import EditStatistics


class EditStatisticsField(fields.Field):
    KEYS = {"insertions", "deletions", "substitutions", "correct"}

    def _serialize(self, value, _attr, _obj, **kwargs):
        return {
            "insertions": value.insertions,
            "deletions": value.deletions,
            "substitutions": value.substitutions,
            "correct": value.correct,
        }

    def _deserialize(self, value, _attr, _data, **kwargs):
        if set(value.keys()) != self.KEYS:
            raise ValidationError("EditStatistics field mismatch, either missing or superfluous fields present")
        return EditStatistics(value["insertions"], value["deletions"], value["substitutions"], value["correct"])


@marshmallow_dataclass.add_schema
@dataclass
class EvaluationResults(MarshmallowDataclassLoadMixin):
    Schema: ClassVar[Type[Schema]]

    properties: List[str]
    error_rates: Dict[str, float]
    error_statistics: Dict[str, EditStatistics] = utils.schema_field(
        fields.Dict(keys=fields.Str(), values=EditStatisticsField)
    )

    def __format__(self, format_spec: str) -> str:
        strings = []
        for name in self.properties:
            strings.append(
                f"{name}: | {self.error_statistics[name]} | {self.error_rates[name] * 100:{format_spec + 'f'}}"
            )

        return "\n".join(strings)

    def __str__(self) -> str:
        return f"{self:.4}"


@marshmallow_dataclass.add_schema
@dataclass
class MultilingualEvaluationResults(MarshmallowDataclassLoadMixin):
    Schema: ClassVar[Type[Schema]]

    evaluation_arguments: str
    results: Dict[str, EvaluationResults]
    package_version: str = version(allophant.__package__)

    def __format__(self, format_spec: str) -> str:
        strings = []
        strings.append(f"Command: {self.evaluation_arguments}\nVersion: {self.package_version}")
        for language, results in self.results.items():
            strings.append(f"{language}:\n{results:{format_spec}}")

        return "\n".join(strings)

    def __str__(self) -> str:
        return f"{self:.4}"
