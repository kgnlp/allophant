#!/usr/bin/env python
from collections.abc import Sequence
from argparse import ArgumentParser, FileType
import sys
from typing import Dict, List, Tuple

import numpy as np

from allophant.evaluation import MultilingualEvaluationResults


def _print_single_category_error_rates(
    title: str, results: List[Tuple[str, Dict[str, float]]], category_error_rates: Sequence[float]
) -> None:
    print(title)

    for (language, _), error_rates in zip(results, category_error_rates):
        print(language, error_rates, sep=",")

    print("Average", sum(category_error_rates) / len(category_error_rates), sep=",")


def main(args: Sequence[str] | None = None) -> None:
    if args is None:
        args = sys.argv[1:]

    parser = ArgumentParser()
    parser.add_argument(
        "results_file", type=FileType("r", encoding="utf-8"), help="Path to a results file from the evaluation command"
    )
    parser.add_argument(
        "-l",
        "--languages",
        type=lambda codes: set(codes.split(",")),
        help="Comma separated list of language codes to display the results and averages for",
    )

    arguments = parser.parse_args(args)

    language_subset = arguments.languages

    results_data = MultilingualEvaluationResults.load(arguments.results_file)
    # Don't include the total as a language in the report
    del results_data.results["total"]
    results = [
        (language, language_results.error_rates)
        for language, language_results in results_data.results.items()
        if language_subset is None or language in language_subset
    ]

    if language_subset is not None and len(language_subset) > len(results):
        raise ValueError(
            f"Languages from -l/--languages are missing from the results file: [{', '.join(language_subset - {language for language, _ in results})}]"
        )

    print("Evaluation Arguments", results_data.evaluation_arguments)
    print("Package Version", results_data.package_version)

    first_result = results[0][1]

    phone = [] if "phone" in first_result else None
    phoneme = [] if "phoneme" in first_result else None
    attribute = []

    for _, error_rates in results:
        if phone is not None:
            phone.append(error_rates.pop("phone") * 100)
        if phoneme is not None:
            phoneme.append(error_rates.pop("phoneme") * 100)
        attribute.append(list(error_rates.values()))

    if phone is not None:
        _print_single_category_error_rates("phoneme", results, phone)

    if phoneme is not None:
        _print_single_category_error_rates("phoneme", results, phoneme)

    attribute_error_rates = np.array(attribute) * 100
    if not attribute_error_rates.size:
        print("No Attribute Error Rates")
        return

    for (language, _), error_rates in zip(results, attribute_error_rates.mean(1)):
        print(language, error_rates, sep=",")

    for attribute, error_rates in zip(results[0][1], attribute_error_rates.mean(0)):
        print(attribute, error_rates, sep=",")

    print("Average", attribute_error_rates.mean(), sep=",")


if __name__ == "__main__":
    main(sys.argv[1:])
