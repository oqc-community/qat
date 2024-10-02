from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt

from qat.purr.backends.qblox.codegen import QbloxPackage
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def plot_packages(packages: List[QbloxPackage]):
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


def plot_acquisitions(acquisitions: Dict, *args, **kwargs):
    if not acquisitions:

        return

    integration_length: int = kwargs["integration_length"]

    for i, (acq_name, acq) in enumerate(acquisitions.items()):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=False,
            sharey=False,
            squeeze=False,
            figsize=(10, 5),
        )
        fig.suptitle(f"Scope and Acquisition plots for {acq_name}")

        # Scope acquisition
        scope_acq_i = np.array(acq["acquisition"]["scope"]["path0"]["data"])
        scope_acq_q = np.array(acq["acquisition"]["scope"]["path1"]["data"])
        max_length = scope_acq_i.size
        if max_length > 0:
            t = np.linspace(0, max_length, max_length)
            axes[0, 0].plot(t, scope_acq_i, label="I")
            axes[0, 0].plot(t, scope_acq_q, label="Q")
            axes[0, 0].set_xlabel("Sample (ns)")
            axes[0, 0].set_ylabel("Scope signal")
            axes[0, 0].autoscale()
            axes[0, 0].legend()

        # Binned acquisition
        bin_acq_i = (
            np.array(acq["acquisition"]["bins"]["integration"]["path0"])
            / integration_length
        )
        bin_acq_q = (
            np.array(acq["acquisition"]["bins"]["integration"]["path1"])
            / integration_length
        )
        max_length = bin_acq_i.size
        if max_length > 0 and not np.isnan(np.append(bin_acq_i, bin_acq_q)).any():
            t = np.linspace(0, max_length, max_length)
            axes[1, 0].plot(t, bin_acq_i, label="I")
            axes[1, 0].plot(t, bin_acq_q, label="Q")
            axes[1, 0].set_xlabel("Shot")
            axes[1, 0].set_ylabel("Acq Signal")
            axes[1, 0].autoscale()
            axes[1, 0].legend()

        plt.tight_layout()
        plt.show()
