"""
Compile KGQA navigation results into a compact CSV.

Example:
    python compile_navigation_results.py --dataset kinship_v2

Output:
    results/navigation/compiled/kinship_v2.csv

Each run contributes:
    - one Overall row
    - one row per available hop

Only paper-relevant metrics are included.
"""

import argparse
import csv
import json
from pathlib import Path


METRICS = [
    "PED",
    "RED",
    "F1_SG",
    "F1_REL",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile navigation experiment results."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. kinship_v2 or mquake_multi.",
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("./results/navigation"),
        help="Navigation results root directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )

    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include runs created with --max-questions.",
    )

    return parser.parse_args()


def load_result(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_name(config):
    """
    Produce the same readable model name used by the runner.
    """
    name = config["model"]

    if config.get("use_instruct"):
        name += "-instruct"

    if config.get("use_quantized"):
        bits = config.get("quantization_bits")
        name += f"-q{bits}"

    return name


def output_format(config):
    return (
        "structured"
        if config.get("structured_output")
        else "unstructured"
    )


def make_row(
    config,
    section_name,
    section_stats,
    source_file,
):
    path_stats = section_stats.get("path_fidelity", {})
    answer_stats = section_stats.get("final_entity", {})

    accuracy = answer_stats.get("Hits1")

    # Fallback in case Hits1 is unavailable.
    if accuracy is None:
        total = section_stats.get("running_count", 0)
        correct = section_stats.get("accuracy", 0)

        accuracy = (
            correct / total
            if total
            else 0.0
        )

    row = {
        "Model": model_name(config),
        "Prompting": config.get(
            "prompting_approach",
            config.get("requested_prompting_approach"),
        ),
        "Format": output_format(config),
        "Hop": section_name,
        "N": section_stats.get("running_count"),
        "Accuracy": 100 * accuracy,
    }

    for metric in METRICS:
        row[metric] = path_stats.get(metric)

    row["Source"] = source_file.name

    return row


def hop_sort_key(section):
    """
    Overall first, then numeric hops.
    """
    if section == "overall":
        return -1

    try:
        return int(section)
    except ValueError:
        return 999


def compile_file(path, dataset):
    try:
        payload = load_result(path)
    except (json.JSONDecodeError, OSError):
        print(f"[WARN] Skipping unreadable file: {path}")
        return []

    config = payload.get("config", {})
    statistics = payload.get("statistics", {})

    if config.get("dataset") != dataset:
        return []

    rows = []

    for section in sorted(
        statistics.keys(),
        key=hop_sort_key,
    ):
        section_stats = statistics[section]

        if not isinstance(section_stats, dict):
            continue

        label = (
            "Overall"
            if section == "overall"
            else str(section)
        )

        rows.append(
            make_row(
                config=config,
                section_name=label,
                section_stats=section_stats,
                source_file=path,
            )
        )

    return rows

def find_result_files(dataset_dir: Path):
    """
    Find experiment result JSON files.

    Excludes:
        - JSON files directly in the dataset root
        - anything inside a directory named 'test'
    """
    result_files = []

    for path in dataset_dir.rglob("*.json"):
        relative = path.relative_to(dataset_dir)

        # Ignore files directly under:
        # results/navigation/<dataset>/
        if len(relative.parts) == 1:
            continue

        # Ignore any test directory.
        directory_parts = relative.parts[:-1]
        if any(part.lower() == "test" for part in directory_parts):
            continue

        result_files.append(path)

    return sorted(result_files)

def main():
    args = parse_args()

    dataset_dir = args.results_root / args.dataset

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Result directory not found: {dataset_dir}"
        )

    result_files = find_result_files(dataset_dir)

    rows = []

    for path in result_files:
        payload = None

        try:
            payload = load_result(path)
        except (json.JSONDecodeError, OSError):
            print(f"[WARN] Skipping {path}")
            continue

        config = payload.get("config", {})

        if config.get("dataset") != args.dataset:
            continue

        # Skip sample/debug runs by default.
        if (
            not args.include_partial
            and config.get("max_questions") is not None
        ):
            continue

        statistics = payload.get("statistics", {})

        for section in sorted(
            statistics.keys(),
            key=hop_sort_key,
        ):
            section_stats = statistics[section]

            if not isinstance(section_stats, dict):
                continue

            label = (
                "Overall"
                if section == "overall"
                else str(section)
            )

            rows.append(
                make_row(
                    config=config,
                    section_name=label,
                    section_stats=section_stats,
                    source_file=path,
                )
            )

    if not rows:
        raise RuntimeError(
            f"No matching results found for {args.dataset}"
        )

    #
    # Sort:
    #   prompting
    #   model
    #   overall -> hop2 -> hop3 -> ...
    #
    def row_sort_key(row):
        hop = row["Hop"]

        if hop == "Overall":
            hop_value = -1
        else:
            try:
                hop_value = int(hop)
            except ValueError:
                hop_value = 999

        return (
            row["Prompting"],
            row["Model"],
            row["Format"],
            hop_value,
        )

    rows.sort(key=row_sort_key)

    output = args.output

    if output is None:
        output = (
            args.results_root
            / "compiled"
            / f"{args.dataset}.csv"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "Model",
        "Prompting",
        "Format",
        "Hop",
        "N",
        "Accuracy",
        "PED",
        "RED",
        "F1_SG",
        "F1_REL",
        "Source",
    ]

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset: {args.dataset}")
    print(f"Runs found: {len(result_files)}")
    print(f"Rows written: {len(rows)}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()