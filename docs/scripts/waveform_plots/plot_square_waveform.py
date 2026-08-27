# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Square waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_square_waveform.py

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

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-Square waveform figure.

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

    width1, color1 = 40e-9, "#2980b9"
    width2, color2 = 80e-9, "grey"

    display_width = width2 * 1.35
    time = sample_time(display_width)
    waveform1 = np.where(np.abs(time) <= width1 / 2.0, 1.0, 0.0)
    waveform2 = np.where(np.abs(time) <= width2 / 2.0, 1.0, 0.0)

    time_ns = time * 1e9
    ax.plot(time_ns, waveform2, color=color2, zorder=2)
    ax.plot(time_ns, waveform1, color=color1, linestyle="--", linewidth=2.0, zorder=3)

    label_size, tick_size = LABEL_SIZE, TICK_SIZE
    ax.text(-18.0, 0.72, r"$\mathbf{S_1}$", color=color1, fontsize=label_size)
    ax.text(-38.0, 0.62, r"$\mathbf{S_2}$", color=color2, fontsize=label_size)

    x_min = -display_width * 1e9 / 2.0
    x_max = display_width * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    w1_ns = width1 * 1e9
    w2_ns = width2 * 1e9

    w1_y = 0.30
    w2_y = 0.14
    draw_span(ax, -w1_ns / 2.0, w1_ns / 2.0, w1_y, color1)
    draw_span(ax, -w2_ns / 2.0, w2_ns / 2.0, w2_y, color2)

    ax.text(
        0.0, w1_y + 0.05, r"$\mathbf{w_1}$", color=color1, ha="center", fontsize=label_size
    )
    ax.text(
        0.0, w2_y + 0.05, r"$\mathbf{w_2}$", color=color2, ha="center", fontsize=label_size
    )

    draw_amp_axis(ax, y_axis_x, x_max)

    ax.text(
        (x_min + x_max) / 2.0,
        1.10,
        r"$\mathbf{w_i}$: width    $\mathbf{S_i}$: waveform",
        ha="center",
        va="top",
        fontsize=label_size,
    )

    tick_labels = {
        -width2 * 1e9 / 2.0: r"$-w_2/2$",
        -width1 * 1e9 / 2.0: r"$-w_1/2$",
        0.0: r"$0$",
        width1 * 1e9 / 2.0: r"$w_1/2$",
        width2 * 1e9 / 2.0: r"$w_2/2$",
    }
    for tick, label in sorted(tick_labels.items()):
        ax.vlines(tick, -0.035, 0.035, color="black", linewidth=1.4)
        ax.text(tick, -0.075, label, ha="center", va="top", fontsize=tick_size)

    mid_x = (y_axis_x + x_max) / 2.0 + 3.0
    ax.text(mid_x, -0.14, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.18, 1.16)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated Square waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
