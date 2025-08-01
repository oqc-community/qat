# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt

from qat.backend.qblox.acquisition import Acquisition
from qat.backend.qblox.execution import QbloxExecutable
from qat.purr.compiler.devices import PulseChannel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def plot_executable(executable: QbloxExecutable):
    packages = executable.packages

    if not packages:
        return

    max_length = max([len(pkg.timeline) for pkg in packages])
    if max_length <= 0:
        return

    # Padding short timelines with zeros
    for pkg in packages:
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
    fig.suptitle("Target timeline plots")
    for i, pkg in enumerate(packages):
        axes[i][0].plot(t, pkg.timeline.real, label="I")
        axes[i][0].plot(t, pkg.timeline.imag, label="Q")
        axes[i][0].set_title(pkg.target)
        axes[i][0].set_xlabel("Time (ns)")
        axes[i][0].set_ylabel("Digital offset")
        axes[i][0].autoscale()
        axes[i][0].legend()

    plt.tight_layout()
    plt.show()


def plot_acquisitions(playback: Dict[PulseChannel, List[Acquisition]]):
    if not playback:
        return

    for i, (target, acquisitions) in enumerate(playback.items()):
        for acquisition in acquisitions:
            fig, axes = plt.subplots(
                nrows=3,
                ncols=1,
                sharex=False,
                sharey=False,
                squeeze=False,
                figsize=(10, 5),
            )
            fig.suptitle(f"Playback plots for {acquisition.name} on {target}")

            scope_data = acquisition.acq_data.scope
            integ_data = acquisition.acq_data.bins.integration
            thrld_data = acquisition.acq_data.bins.threshold

            # Scope data
            axes[0, 0].plot(scope_data.i.data, label="I")
            axes[0, 0].plot(scope_data.q.data, label="Q")
            axes[0, 0].set_xlabel("Sample (ns)")
            axes[0, 0].set_ylabel("Value")
            axes[0, 0].autoscale()
            axes[0, 0].legend()
            axes[0, 0].title.set_text("Scope acquisition")

            # Integration data
            axes[1, 0].plot(integ_data.i, label="I")
            axes[1, 0].plot(integ_data.q, label="Q")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].autoscale()
            axes[1, 0].legend()
            axes[1, 0].title.set_text("Integrated acquisition")

            # Threshold data
            axes[2, 0].plot(thrld_data, label="I")
            axes[2, 0].set_xlabel("Iteration")
            axes[2, 0].set_ylabel("Value")
            axes[2, 0].autoscale()
            axes[2, 0].title.set_text("Thresholded acquisition")

        plt.tight_layout()
        plt.show()
