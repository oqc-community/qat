# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import argparse
import json
import os
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape


def get_directory(dir):
    """
    Save directory for benchmarks depends on the environment: determine this.
    """
    subdir = [name for name in os.listdir(dir) if os.path.isdir(dir + name)]
    return dir + subdir[0]


def round_sf(x, sf=4):
    return np.round(x, -int(np.floor(np.log10(abs(x)))) + sf - 1)


def compare_tests(
    warn_threshold=1.2,
    fail_threshold=1.5,
    improve_threshold=0.9,
    benchmark_name="benchmark",
    return_successes=False,
    return_improvements=True,
    dir=".benchmarks/",
):
    """
    Generate a dictonary of tests that contains the key information for the report.
    """
    # load in the two benchmarks
    dir = get_directory(dir)
    with open(f"{dir}/0001_{benchmark_name}.json", "r") as f:
        benchmark_before = json.load(f)["benchmarks"]
    with open(f"{dir}/0002_{benchmark_name}.json", "r") as f:
        benchmark_after = json.load(f)["benchmarks"]

    # compare the benchmarks
    benchmarks = {}
    for bm_after in benchmark_after:
        name = bm_after["name"]
        data = {
            "min_after": round_sf(bm_after["stats"]["min"]),
            "min_before": "-",
            "rel_diff": "-",
            "outcome": "success",
        }
        for bm_before in benchmark_before:
            if bm_before["name"] == name:
                data["min_before"] = round_sf(bm_before["stats"]["min"])
                data["rel_diff"] = round_sf(
                    bm_after["stats"]["min"] / bm_before["stats"]["min"]
                )
                if data["rel_diff"] > fail_threshold:
                    data["outcome"] = "fail"
                elif data["rel_diff"] > warn_threshold:
                    data["outcome"] = "warning"
                elif data["rel_diff"] < improve_threshold:
                    data["outcome"] = "improvement"
                break

        # if not warn/fail, decide if it should be added
        if not (data["outcome"] == "success" and return_successes == False) and not (
            data["outcome"] == "improvement" and return_improvements == False
        ):
            benchmarks[name] = data
    return benchmarks


def create_report(
    benchmarks,
    input_path="benchmarks/report_template.md",
    output_path=".benchmarks/summary.md",
):
    """
    Create the report using the template
    """
    env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
    template = env.get_template(input_path)
    summary = template.render(tests=benchmarks)
    Path(output_path).write_text(summary)


def get_args():
    """
    Allow arguments to be parsed to the program.
    """

    parser = argparse.ArgumentParser(
        prog="Performance regression report",
        description="Generates a report that compares performance between two benchmarks.",
    )
    parser.add_argument("benchmark_name")
    parser.add_argument("output_path")
    parser.add_argument(
        "--template",
        default="benchmarks/report_template.md",
        type=str,
        help="Path to the template file",
    )
    parser.add_argument(
        "--warn_threshold", default=1.2, type=float, help="Slow down needed to warn"
    )
    parser.add_argument(
        "--fail_threshold", default=1.5, type=float, help="Slow down needed to fail"
    )
    parser.add_argument(
        "--improvement",
        default=True,
        type=bool,
        help="Should performance improvements be shown",
    )
    parser.add_argument(
        "--improvement_threshold",
        default=0.9,
        type=float,
        help="The speed-up needed to notify the improvement",
    )
    parser.add_argument(
        "--success",
        default=False,
        type=bool,
        help="Should successes be shown (tests that do not give a performance regression)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    report = compare_tests(
        args.warn_threshold,
        args.fail_threshold,
        args.improvement_threshold,
        args.benchmark_name,
        args.success,
        args.improvement,
    )
    create_report(report, input_path=args.template, output_path=args.output_path)
