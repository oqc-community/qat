# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""GaussianSquare waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_gaussian_square_waveform.py

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

from qat.utils.waveform import GaussianSquareFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-GaussianSquare waveform figure.

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

    width = 90e-9
    std_dev = 6e-9
    square_width1, color1 = 35e-9, "#2980b9"
    square_width2, color2 = 55e-9, "grey"

    time = sample_time(width)
    waveform1 = (
        GaussianSquareFunction(
            width=width,
            square_width=square_width1,
            std_dev=std_dev,
            zero_at_edges=True,
        )
        .eval(time)
        .real
    )
    waveform2 = (
        GaussianSquareFunction(
            width=width,
            square_width=square_width2,
            std_dev=std_dev,
            zero_at_edges=True,
        )
        .eval(time)
        .real
    )

    ax.plot(time * 1e9, waveform1, color=color1)
    ax.plot(time * 1e9, waveform2, color=color2)

    label_size, tick_size = LABEL_SIZE, TICK_SIZE
    ax.text(-21.0, 0.8, r"$\mathbf{G_1}$", color=color1, fontsize=label_size)
    ax.text(-32.0, 0.66, r"$\mathbf{G_2}$", color=color2, fontsize=label_size)

    x_min = -width * 1e9 / 2.0
    x_max = width * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    s1_ns = square_width1 * 1e9
    s2_ns = square_width2 * 1e9

    s1_y = 0.30
    s2_y = 0.14
    draw_span(ax, -s1_ns / 2.0, s1_ns / 2.0, s1_y, color1)
    draw_span(ax, -s2_ns / 2.0, s2_ns / 2.0, s2_y, color2)

    time_ns = time * 1e9
    s1_edges = (-s1_ns / 2.0, s1_ns / 2.0)
    s2_edges = (-s2_ns / 2.0, s2_ns / 2.0)

    s1_edge_y = [float(np.interp(x, time_ns, waveform1)) for x in s1_edges]
    s2_edge_y = [float(np.interp(x, time_ns, waveform2)) for x in s2_edges]

    for x, y in zip(s1_edges, s1_edge_y, strict=True):
        ax.vlines(x, 0.0, y, color=color1, linewidth=1.1, linestyles="dotted")
        ax.plot(
            x,
            y,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=color1,
            markeredgewidth=1.2,
        )
    for x, y in zip(s2_edges, s2_edge_y, strict=True):
        ax.vlines(x, 0.0, y, color=color2, linewidth=1.1, linestyles="dotted")
        ax.plot(
            x,
            y,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=color2,
            markeredgewidth=1.2,
        )

    ax.text(
        0.0,
        s1_y + 0.05,
        r"$\mathbf{s_1}$",
        color=color1,
        ha="center",
        fontsize=label_size,
    )
    ax.text(
        0.0,
        s2_y + 0.05,
        r"$\mathbf{s_2}$",
        color=color2,
        ha="center",
        fontsize=label_size,
    )

    draw_amp_axis(ax, y_axis_x, x_max)
    ax.text(
        (x_min + x_max) / 2.0,
        1.18,
        r"$\mathbf{s_i}$: square_width    $\mathbf{\sigma}$: std_dev  zero_at_edges: True   $\mathbf{w}$: width    $\mathbf{G_i}$: waveform",
        ha="center",
        va="top",
        fontsize=label_size,
    )

    tick_labels = {
        -width * 1e9 / 2.0: r"$-w/2$",
        0.0: r"$0$",
        width * 1e9 / 2.0: r"$w/2$",
    }
    for tick, label in sorted(tick_labels.items()):
        ax.vlines(tick, -0.035, 0.035, color="black", linewidth=1.4)
        ax.text(tick, -0.075, label, ha="center", va="top", fontsize=tick_size)

    s_ns = std_dev * 1e9
    square_tick_labels = {
        -s1_ns / 2.0 - s_ns: (r"$-s_1/2{-}\sigma$", "black", -0.075),
        -s1_ns / 2.0 + s_ns: (r"$-s_1/2{+}\sigma$", "black", -0.075),
        s1_ns / 2.0 - s_ns: (r"$s_1/2{-}\sigma$", "black", -0.075),
        s1_ns / 2.0 + s_ns: (r"$s_1/2{+}\sigma$", "black", -0.075),
    }
    for tick, (label, color, y) in sorted(square_tick_labels.items()):
        ax.vlines(tick, -0.028, 0.028, color=color, linewidth=1.2)
        ax.text(tick, y, label, ha="center", va="top", color=color, fontsize=10)

    mid_x = (y_axis_x + x_max) / 2.0 + 3.0
    ax.text(mid_x, -0.14, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.18, 1.28)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated GaussianSquare waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
