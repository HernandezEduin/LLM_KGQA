#!/usr/bin/env python3
"""Summarize errors/timeouts in KGQA result JSON files."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple
from model.constants import valid_models
from collections import defaultdict


def _count_value(val: Any) -> int:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, list):
        return len(val)
    if isinstance(val, dict):
        # Attempt to count nested items if they look like error entries
        if "count" in val and isinstance(val["count"], (int, float)):
            return int(val["count"])
        return len(val)
    return 0


def _as_float(val: Any) -> float | None:
    if isinstance(val, bool):
        return float(int(val))
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _extract_sections(data: Any) -> List[Tuple[str, Dict[str, Any]]]:
    if isinstance(data, dict):
        sections = []
        for k, v in data.items():
            if isinstance(v, dict) and ("errors" in v or "timeouts" in v or "response_tokens" in v):
                sections.append((str(k), v))
        if sections:
            return sections
    return []


def _summarize_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections = _extract_sections(data)
    summary = {
        "file": path,
        "sections": [],
        "total_errors": 0,
        "total_timeouts": 0,
        "response_tokens_gt_10": False,
        "max_response_tokens": None,
    }

    for name, section in sections:
        err = _count_value(section.get("errors", 0))
        tout = _count_value(section.get("timeouts", 0))
        response_tokens = _as_float(section.get("response_tokens"))
        summary["sections"].append({
            "name": name,
            "errors": err,
            "timeouts": tout,
            "response_tokens": response_tokens,
        })
        # avoid double-counting if overall is present
        if name != "overall":
            summary["total_errors"] += err
            summary["total_timeouts"] += tout
            if response_tokens is not None:
                summary["response_tokens_gt_10"] = summary["response_tokens_gt_10"] or response_tokens > 10
                summary["max_response_tokens"] = max(response_tokens, summary["max_response_tokens"] or response_tokens)

    # If overall exists, prefer it as total
    overall = next((s for s in summary["sections"] if s["name"] == "overall"), None)
    if overall:
        summary["total_errors"] = overall["errors"]
        summary["total_timeouts"] = overall["timeouts"]
        if overall["response_tokens"] is not None:
            summary["response_tokens_gt_10"] = overall["response_tokens"] > 10
            summary["max_response_tokens"] = overall["response_tokens"]

    return summary


def _format_summary(s: Dict[str, Any], show_all: bool) -> str:
    lines = [f"File: {s['file']}"]

    overall = next((sec for sec in s["sections"] if sec["name"] == "overall"), None)
    if overall:
        lines.append(
            f"  overall: errors={overall['errors']} timeouts={overall['timeouts']} "
            f"response_tokens={overall['response_tokens']}"
        )

    if show_all:
        for sec in s["sections"]:
            if sec["name"] == "overall":
                continue
            lines.append(
                f"  {sec['name']}: errors={sec['errors']} timeouts={sec['timeouts']} "
                f"response_tokens={sec['response_tokens']}"
            )

    if not s["sections"]:
        lines.append("  no error/timeout sections found")

    return "\n".join(lines)


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description="Check result JSONs for errors/timeouts")
    parser.add_argument('--dataset', type=str, default='mquake',
                        help='Name of the dataset to process.')
    parser.add_argument('--hops', type=str, default='n',
                        help='Number of hops for subgraph extraction.')
    
    parser.add_argument('--llm-models', type=str, nargs='+', default=valid_models,
                        help='Models to include in the plot.')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed used in the experiments.')
    
    parser.add_argument('--subgraph-size', type=int, nargs='+', default=[None, 5, 10, 50, 100, 250, 500, 700, 1000, 1500, 1700, 2000, 2500, 3000],
                        help='Number of triplets in the extracted subgraph.')
    
    parser.add_argument('--sampling-method', type=str, default='neighborhood',
                        choices=['random', 'neighborhood'],
                        help='Method for subgraph sampling.')
    
    parser.add_argument('-r', '--retrieve', action='store_true',
                        help='Non-oracle subgraph retrieval method.')

    parser.add_argument('--result-dir', type=str, default='./results',
                        help='Directory to save the results.')
    parser.add_argument('--plot-dir', type=str, default='./plots',
                        help='Directory to save the plots.')
    
    parser.add_argument('--use-instruct', action='store_true',
                        help='Whether to use the instruction-tuned version of the model.')
    parser.add_argument('--use-quantized', action='store_true',
                        help='Whether to use the quantized version of the model.')
    parser.add_argument('--quantization-bits', type=int, default=4,
                        help='Number of bits for quantization (if using quantized model).')
    parser.add_argument('--out', type=str,
                        help='Write summary to this file (text).')
    args = parser.parse_args(list(argv))

    result_path = os.path.join(args.result_dir, args.dataset)
    os.makedirs(args.plot_dir, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for llm_model in args.llm_models:
        model_name = llm_model
        if args.use_instruct:
            model_name += "-instruct"
            if args.use_quantized:
                model_name += f"-q{args.quantization_bits}"
        for size in args.subgraph_size:
            sampling_str = args.sampling_method if size is not None else 'evidence'
            results_file = os.path.join(
                result_path,
                f"results_{args.hops}hop_{model_name}_subgraph{size}_{'retrieve' if args.retrieve else 'oracle'}_{args.sampling_method}_seed{args.seed}.json"
            )
            if os.path.exists(results_file):
                summaries.append(_summarize_file(results_file))
            else:
                print(f"Warning: Result file {results_file} does not exist.")

    any_fail = False
    output_lines: List[str] = []
    for s in summaries:
        line = (
            f"{os.path.basename(s['file'])}: errors={s['total_errors']} "
            f"timeouts={s['total_timeouts']} "
            f"response_tokens_gt_10={s['response_tokens_gt_10']}"
        )
        if s["max_response_tokens"] is not None:
            line += f" max_response_tokens={s['max_response_tokens']}"
        print(line)
        output_lines.append(line)
        if s["total_errors"] or s["total_timeouts"] or s["response_tokens_gt_10"]:
            any_fail = True

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines).rstrip() + "\n")

    return 1 if (any_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
