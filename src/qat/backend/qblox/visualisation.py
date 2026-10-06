# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Oxford Quantum Circuits Ltd

import numpy as np
from matplotlib import pyplot as plt

from qat.backend.qblox.acquisition import Acquisition
from qat.backend.qblox.execution import QbloxProgram
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def plot_program(program: QbloxProgram):
    packages = program.packages

    if not packages:
        return

    max_length = max([len(pkg.timeline) for pkg in packages.values()])
    if max_length <= 0:
        return

    # Padding short timelines with zeros
    for pkg in packages.values():
        length = len(pkg.timeline)
        if length < max_length:
            pkg.timeline = np.append(
                pkg.timeline, np.zeros(max_length - length, dtype=pkg.timeline.dtype)
            )

    t = np.linspace(0, max_length, max_length)
    fig, axes = plt.subplots(
        nrows=len(packages),
        ncols=1,
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=(10, 5),
    )
    fig.suptitle("Timeline plots")
    for i, pkg in enumerate(packages.values()):
        axes[i][0].plot(t, pkg.timeline.real, label="I")
        axes[i][0].plot(t, pkg.timeline.imag, label="Q")
        axes[i][0].set_title(pkg.pulse_channel_id)
        axes[i][0].set_xlabel("Time (ns)")
        axes[i][0].set_ylabel("Digital offset")
        axes[i][0].autoscale()
        axes[i][0].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_playback(
    playback: dict[str, list[Acquisition]],
    figure_width: float = 10,
    row_height: float = 2,
) -> None:
    if not playback:
        return

    for pulse_channel_id, acquisitions in playback.items():
        for acquisition in acquisitions:
            scope_data = acquisition.acquisition.scope
            scope_pairs = []
            for path_ids in ((0, 1), (2, 3)):
                populated_paths = [
                    (path_id, path)
                    for path_id in path_ids
                    if (path := getattr(scope_data, f"path{path_id}")) is not None
                    and np.asarray(path.data).size
                ]
                if populated_paths:
                    scope_pairs.append((path_ids, populated_paths))

            nrows = len(scope_pairs) + 2
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=1,
                sharex=False,
                sharey=False,
                squeeze=False,
                figsize=(figure_width, row_height * nrows),
            )
            fig.suptitle(f"Playback plots for {acquisition.name} on {pulse_channel_id}")

            integ_data = acquisition.acquisition.bins.integration
            thrld_data = acquisition.acquisition.bins.threshold

            # Scope data
            for row, (path_ids, populated_paths) in enumerate(scope_pairs):
                axis = axes[row, 0]
                for path_id, path in populated_paths:
                    axis.plot(path.data, label=f"path{path_id}")
                axis.set_xlabel("Sample (ns)")
                axis.set_ylabel("Value")
                axis.autoscale()
                axis.legend()
                axis.title.set_text(
                    f"Scope acquisition (paths {path_ids[0]}/{path_ids[1]})"
                )

            # Integration data
            integration_axis = axes[len(scope_pairs), 0]
            integration_axis.plot(integ_data.path0, label="I")
            integration_axis.plot(integ_data.path1, label="Q")
            integration_axis.set_xlabel("Iteration")
            integration_axis.set_ylabel("Value")
            integration_axis.autoscale()
            integration_axis.legend()
            integration_axis.title.set_text("Integrated acquisition")

            # Threshold data
            threshold_axis = axes[len(scope_pairs) + 1, 0]
            threshold_axis.plot(thrld_data, label="I")
            threshold_axis.set_xlabel("Iteration")
            threshold_axis.set_ylabel("Value")
            threshold_axis.autoscale()
            threshold_axis.title.set_text("Thresholded acquisition")

        plt.tight_layout()
        plt.show()
