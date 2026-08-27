# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Shared utilities for waveform documentation plot scripts."""

import shutil
from argparse import ArgumentParser, Namespace

import numpy as np
from matplotlib import pyplot as plt, rc

LABEL_SIZE: int = 12
TICK_SIZE: int = 11


def configure_matplotlib() -> None:
    """Configure matplotlib for documentation plots (LaTeX if available)."""
    if shutil.which("latex"):
        rc("text", usetex=True)
    else:
        rc("text", usetex=False)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})


def sample_time(time_width: float, samples: int = 700) -> np.ndarray:
    """Return a time array centred around zero."""
    return np.linspace(-time_width / 2.0, time_width / 2.0, samples)


def parse_args(description: str) -> Namespace:
    """Parse ``--output`` and ``--no-show`` CLI options."""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        help="Path to save the generated figure, e.g. docs/source/.../image.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive Matplotlib window.",
    )
    return parser.parse_args()


def draw_span(
    ax: "plt.Axes",
    start: float,
    end: float,
    y: float,
    color: str,
    linestyle: str = "-",
) -> None:
    """Draw a horizontal ``<->`` span annotation between *start* and *end*."""
    ax.annotate(
        "",
        xy=(start, y),
        xytext=(end, y),
        arrowprops={
            "arrowstyle": "<->",
            "connectionstyle": "arc3",
            "linestyle": linestyle,
            "linewidth": 0.8,
            "color": color,
        },
    )


def draw_amp_axis(
    ax: "plt.Axes",
    y_axis_x: float,
    x_max: float,
    y_bottom: float = 0.0,
    y_top: float = 1.05,
    label_offset: float = 4.0,
    hlines_ext: float = 4.0,
) -> None:
    """Draw the amplitude y-axis with tick marks and an ``amp`` label.

    :param y_axis_x: x position of the vertical axis line.
    :param x_max: right extent for the horizontal baseline.
    :param y_bottom: lower extent of the vertical line.
    :param y_top: upper extent of the vertical line.
    :param label_offset: horizontal distance left of *y_axis_x* for the ``amp`` label.
    :param hlines_ext: additional pixels past *x_max* for the baseline hline.
    """
    ax.hlines(0.0, y_axis_x, x_max + hlines_ext, color="black", linewidth=1.0)
    ax.vlines(y_axis_x, y_bottom, y_top, color="black", linewidth=1.0)
    ax.annotate(
        "",
        xy=(y_axis_x, 1.0),
        xytext=(y_axis_x, 0.0),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "black"},
    )
    ax.hlines(0.0, y_axis_x - 1.5, y_axis_x + 1.5, color="black", linewidth=1.2)
    ax.hlines(1.0, y_axis_x - 1.5, y_axis_x + 1.5, color="black", linewidth=1.2)
    ax.text(
        y_axis_x - label_offset,
        0.5,
        "amp",
        ha="center",
        va="center",
        fontsize=LABEL_SIZE,
    )


def save_and_show(fig: "plt.Figure", args: Namespace) -> None:
    """Save *fig* to ``args.output`` (if set) and show unless ``--no-show``."""
    if args.output:
        fig.savefig(args.output, bbox_inches="tight", dpi=300)
    if not args.no_show:
        plt.show()
