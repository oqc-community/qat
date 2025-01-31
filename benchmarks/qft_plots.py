# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def get_directory(dir):
    """
    Save directory for benchmarks depends on the environment: determine this.
    """
    subdir = [name for name in os.listdir(dir) if os.path.isdir(dir + name)]
    return dir + subdir[0]


def load_results(benchmark_name, dir=".benchmarks/"):
    """
    Load only the QFT results.
    """
    dir = get_directory(dir)
    with open(f"{dir}/{benchmark_name}.json", "r") as f:
        benchmarks = json.load(f)["benchmarks"]

    results = {
        "Legacy": {"compile": {}, "execute": {}},
        "Experimental": {"compile": {}, "execute": {}},
    }

    for bm in benchmarks:
        if bm["group"] != "QFT":
            continue

        mode = bm["params"]["mode"]
        qubits = bm["params"]["qubits"]
        if bm["name"] == f"test_compile_qasm[{qubits}-{mode}]":
            results[mode]["compile"][qubits] = bm["stats"]["mean"]
        else:
            results[mode]["execute"][qubits] = bm["stats"]["mean"]

    return results


def get_args():
    """
    Allow arguments to be parsed to the program.
    """

    parser = argparse.ArgumentParser(
        prog="Plotting tool for QFT benchmarks",
        description=(
            "Plot performance times for compilation, serialization and execution of QFT "
            "circuits.",
        ),
    )
    parser.add_argument(
        "benchmark_file",
        help="File location for benchmarks generated using pytest-benchmark.",
    )
    parser.add_argument(
        "--output_file", default="plot.pdf", help="Where should the plot be saved?"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    results = load_results(args.benchmark_file)

    width = 0.4

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))

    modes = ["compile", "execute"]
    scales = ["linear", "log"]
    for i, scale in enumerate(scales):
        for j, mode in enumerate(modes):
            legacy = results["Legacy"][mode]
            experimental = results["Experimental"][mode]

            axes[i, j].bar(
                np.array(list(legacy.keys())) - width / 2,
                legacy.values(),
                width=width,
                label="purr",
                color="teal",
            )
            axes[i, j].bar(
                np.array(list(experimental.keys())) + width / 2,
                experimental.values(),
                width=width,
                label="pipelines",
                color="maroon",
            )
            axes[i, j].set_yscale(scale)

    compile_legacy = results["Legacy"]["compile"]
    compile_experimental = results["Experimental"]["compile"]

    axes[1, 0].set_xlabel("Number of qubits")
    axes[1, 1].set_xlabel("Number of qubits")
    axes[0, 0].set_ylabel("Execution time (s)")
    axes[1, 0].set_ylabel("Execution time (s)")
    axes[0, 0].set_title("qat.compile() + serialize")
    axes[0, 1].set_title("deserialize + qat.execute()")

    lines, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, loc=(0.6, 0.82))
    plt.tight_layout()

    fig.savefig(args.output_file)
