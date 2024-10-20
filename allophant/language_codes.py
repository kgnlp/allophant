from dataclasses import dataclass
from typing import Dict, Iterable, Union

import langcodes
from langcodes import Language


@dataclass
class LanguageCode:
    language: str
    alpha3_t: str
    alpha3_b: str
    variant: str | None

    @classmethod
    def from_str(cls, language_code: str, standardize: bool = False, macro: bool = False):
        if standardize:
            language_code = langcodes.standardize_tag(language_code, macro=macro)
        elif macro:
            raise ValueError("Retrieving the macro language requires standardization")

        language = Language.get(language_code)
        if language.language is None:
            raise ValueError(f"{language_code!r} does not contain a valid language code")

        variants = []
        if language.territory is not None:
            variants.append(language.territory)
        if language.variants is not None:
            variants.extend(language.variants)

        if not language.is_valid():
            # Language codes for e.g. constructed languages have no ISO 639-2 codes
            alpha3_t = alpha3_b = language.language
        else:
            alpha3_b = language.to_alpha3("T")
            alpha3_t = language.to_alpha3("B")

        return cls(
            language.language,
            alpha3_b,
            alpha3_t,
            "-".join(variants) if variants else None,
        )

    @property
    def alpha3(self) -> str:
        # Use T form as the standard alpha3 variant
        return self.alpha3_t

    def __str__(self) -> str:
        return self.language if self.variant is None else f"{self.language}-{self.variant}"


def standardize_to_iso6393(language_code: str) -> str:
    return LanguageCode.from_str(language_code, True).alpha3


LanguageCodeAny = Union[str, LanguageCode]


def to_language_code(language_code: LanguageCodeAny) -> LanguageCode:
    if isinstance(language_code, str):
        return LanguageCode.from_str(language_code)

    return language_code


class LanguageCodeMap:
    def __init__(self, language_codes: Iterable[str], defaults: Dict[str, str] | None = None) -> None:
        if defaults is None:
            defaults = {}
        code_map = {}
        existing_codes = {}
        duplicates = set()
        has_default = set()

        for code in language_codes:
            # Standardizes to a macro language to also handle cases like cmn -> zh for Mandarin
            standardized = langcodes.standardize_tag(code, macro=True)
            language_code = LanguageCode.from_str(code)
            language = language_code.language
            # Always create an identity mapping
            code_map[standardized] = code

            if language_code.variant is None:
                # Handle simple codes without regional variants
                existing_codes[language] = (code, language_code)
                has_default.add(language)
            elif language in existing_codes:
                # Handle duplicates
                duplicates.add(language)
                code_map[str(language_code)] = code
            else:
                existing_codes[language] = (code, language_code)

        # Postprocess with full duplicate information
        for code, language_code in existing_codes.values():
            language = language_code.language
            # make a defaults for sets of language codes that all specify
            # regional variants if defaults were provided
            if language not in has_default and (default_variant := defaults.get(language)) is not None:
                code_map[language] = default_variant
                has_default.add(code)
            # Create mappings from standardized tags to regional variants or
            # simply a direct mapping from a language code to a variant if only
            # one variant is present in the map
            if language in duplicates:
                code_map[str(language_code)] = code
            else:
                code_map[language] = code

        self._code_map = code_map

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._code_map})"

    def __getitem__(self, code: LanguageCodeAny) -> str:
        standardized = langcodes.standardize_tag(str(code))
        # Use full language code and fall back to discarding variant information
        return self._code_map.get(standardized) or self._code_map[LanguageCode.from_str(standardized).language]

    def __contains__(self, code: LanguageCodeAny) -> bool:
        standardized = langcodes.standardize_tag(str(code))
        # Use full language code and fall back to discarding variant information
        return standardized in self._code_map or LanguageCode.from_str(standardized).language in self._code_map

    def __len__(self) -> int:
        return len(self._code_map)
