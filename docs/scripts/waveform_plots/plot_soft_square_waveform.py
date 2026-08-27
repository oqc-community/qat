# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""SoftSquare waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_soft_square_waveform.py

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

from qat.utils.waveform import SoftSquareFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-SoftSquare waveform figure.

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

    width1, rise1, color1 = 80e-9, 5e-9, "#2980b9"
    width2, rise2, color2 = 80e-9, 12e-9, "grey"

    time = sample_time(width2)
    waveform1 = SoftSquareFunction(width=width1, rise=rise1).eval(time).real
    waveform2 = SoftSquareFunction(width=width2, rise=rise2).eval(time).real

    ax.plot(time * 1e9, waveform1, color=color1)
    ax.plot(time * 1e9, waveform2, color=color2)

    label_size, tick_size = LABEL_SIZE, TICK_SIZE
    ax.text(-35.0, 0.93, r"$\mathbf{S_1}$", color=color1, fontsize=label_size)
    ax.text(-23.0, 0.80, r"$\mathbf{S_2}$", color=color2, fontsize=label_size)

    x_min = -width2 * 1e9 / 2.0
    x_max = width2 * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    rise1_ns = rise1 * 1e9
    rise2_ns = rise2 * 1e9

    time_ns = time * 1e9
    rise1_left_edge = x_min + rise1_ns
    rise1_right_edge = x_max - rise1_ns
    rise2_left_edge = x_min + rise2_ns
    rise2_right_edge = x_max - rise2_ns

    draw_span(ax, x_min, x_min + rise1_ns, 0.72, color1)
    draw_span(ax, x_min, x_min + rise2_ns, 0.57, color2)

    r1_edge1_y = float(np.interp(rise1_left_edge, time_ns, waveform1))
    r1_edge2_y = float(np.interp(rise1_right_edge, time_ns, waveform1))
    r2_edge1_y = float(np.interp(rise2_left_edge, time_ns, waveform2))
    r2_edge2_y = float(np.interp(rise2_right_edge, time_ns, waveform2))

    for x, y, color in [
        (rise1_left_edge, r1_edge1_y, color1),
        (rise1_right_edge, r1_edge2_y, color1),
        (rise2_left_edge, r2_edge1_y, color2),
        (rise2_right_edge, r2_edge2_y, color2),
    ]:
        ax.vlines(x, 0.0, y, color=color, linewidth=1.1, linestyles="dotted", alpha=0.7)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.0,
        )

    ax.text(
        x_min + rise1_ns / 2.0,
        0.77,
        r"$\mathbf{r_1}$",
        color=color1,
        ha="center",
        fontsize=label_size,
    )
    ax.text(
        x_min + rise2_ns / 2.0,
        0.62,
        r"$\mathbf{r_2}$",
        color=color2,
        ha="center",
        fontsize=label_size,
    )

    draw_amp_axis(ax, y_axis_x, x_max, label_offset=5.0)
    ax.text(
        x_max - 14.0,
        0.40,
        r"$\mathbf{r_i}$: rise"
        "\n"
        r"$\mathbf{w}$: width"
        "\n"
        r"$\mathbf{S_i}$: waveform",
        ha="right",
        va="top",
        fontsize=label_size,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
        },
    )

    tick_labels = {
        -width2 * 1e9 / 2.0: r"$-w/2$",
        0.0: r"$0$",
        width2 * 1e9 / 2.0: r"$w/2$",
    }
    for tick, label in sorted(tick_labels.items()):
        ax.vlines(tick, -0.035, 0.035, color="black", linewidth=1.4)
        ax.text(tick, -0.075, label, ha="center", va="top", fontsize=tick_size)

    mid_x = (y_axis_x + x_max) / 2.0 + 3
    ax.text(mid_x, -0.14, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.18, 1.1)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated SoftSquare waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
