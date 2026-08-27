# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Gaussian waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_gaussian_waveform.py

The :func:`draw_figure` function is importable for use in Sphinx ``.. plot::``
directives without triggering the CLI argument parser.
"""

import numpy as np
from _plot_utils import (
    LABEL_SIZE,
    TICK_SIZE,
    configure_matplotlib,
    draw_amp_axis,
    draw_span,
    parse_args,
    sample_time,
    save_and_show,
)
from matplotlib import pyplot as plt

from qat.utils.waveform import GaussianFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw the annotated dual-Gaussian waveform figure.

    Creates a new figure when *ax* is ``None`` (the default). Passing an
    existing :class:`~matplotlib.axes.Axes` allows the caller to embed the
    drawing inside a larger layout.

    :param ax: Optional axes to draw into. A new figure and axes are created
        when omitted.
    :returns: The :class:`~matplotlib.figure.Figure` containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    width1, rise1, color1 = 40e-9, 0.13, "#2980b9"
    width2, rise2, color2 = 80e-9, 0.25, "grey"

    time1 = sample_time(width1)
    waveform1 = GaussianFunction(width=width1, rise=rise1).eval(time1).real
    time2 = sample_time(width2)
    waveform2 = GaussianFunction(width=width2, rise=rise2).eval(time2).real

    ax.plot(time1 * 1e9, waveform1, color=color1)
    ax.plot(time2 * 1e9, waveform2, color=color2)

    label_size, tick_size = LABEL_SIZE, TICK_SIZE
    ax.text(
        5.3,
        0.62,
        r"$\mathbf{G_1}$",
        color=color1,
        ha="center",
        va="center",
        fontsize=label_size,
    )
    ax.text(
        13.1,
        0.72,
        r"$\mathbf{G_2}$",
        color=color2,
        ha="center",
        va="center",
        fontsize=label_size,
    )

    rise_y = np.exp(-1.0)
    rise_scale1 = width1 * rise1 * 1e9
    rise_scale2 = width2 * rise2 * 1e9

    draw_span(ax, -rise_scale1, 0.0, rise_y, color1)
    draw_span(ax, 0.0, rise_scale2, rise_y, color2)
    ax.annotate(
        r"$\mathbf{w_1 r_1}$",
        xy=(-rise_scale1 / 2.0, rise_y),
        xytext=(-rise_scale1 / 2.0 + 1.0, rise_y + 0.08),
        ha="center",
        color=color1,
        fontsize=label_size,
    )
    ax.annotate(
        r"$\mathbf{w_2 r_2}$",
        xy=(rise_scale2 / 2.0, rise_y),
        xytext=(rise_scale2 / 2.0, rise_y + 0.08),
        ha="center",
        color=color2,
        fontsize=label_size,
    )
    ax.text(
        -rise_scale1 / 2.0,
        rise_y - 0.085,
        r"$\mathbf{k_1}$",
        color=color1,
        ha="center",
        va="center",
        fontsize=label_size,
    )
    ax.text(
        rise_scale2 / 2.0,
        rise_y - 0.085,
        r"$\mathbf{k_2}$",
        color=color2,
        ha="center",
        va="center",
        fontsize=label_size,
    )

    x_min = -width2 * 1e9 / 2.0
    x_max = width2 * 1e9 / 2.0
    y_axis_x = x_min - 6.0

    draw_amp_axis(ax, y_axis_x, x_max, label_offset=5.0)
    ax.text(
        y_axis_x + 2.0,
        1.03,
        r"$\mathbf{r_i}$: rise"
        "\n"
        r"$\mathbf{w_i}$: width"
        "\n"
        r"$\mathbf{G_i}$: waveform",
        ha="left",
        va="top",
        fontsize=label_size,
    )

    tick_labels = {
        -width2 * 1e9 / 2.0: (r"$-w_2/2$", color2),
        -width1 * 1e9 / 2.0: (r"$-w_1/2$", color1),
        0.0: (r"$0$", "black"),
        width1 * 1e9 / 2.0: (r"$w_1/2$", color1),
        width2 * 1e9 / 2.0: (r"$w_2/2$", color2),
    }
    for tick, (label, color) in sorted(tick_labels.items()):
        ax.vlines(tick, -0.035, 0.035, color="black", linewidth=1.4)
        ax.text(
            tick,
            -0.075,
            label,
            ha="center",
            va="top",
            color=color,
            fontsize=tick_size,
        )

    mid_x = (y_axis_x + x_max) / 2.0 + 3
    ax.text(mid_x, -0.14, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.18, 1.1)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated Gaussian waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
