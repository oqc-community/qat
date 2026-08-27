# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Sech waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_sech_waveform.py

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

from qat.utils.waveform import SechFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-Sech waveform figure.

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

    width = 80e-9
    std_dev1, color1 = 7e-9, "#2980b9"
    std_dev2, color2 = 13e-9, "grey"

    time = sample_time(width)
    waveform1 = SechFunction(std_dev=std_dev1).eval(time).real
    waveform2 = SechFunction(std_dev=std_dev2).eval(time).real

    time_ns = time * 1e9
    ax.plot(time_ns, waveform1, color=color1, linewidth=2.0)
    ax.plot(time_ns, waveform2, color=color2, linewidth=2.0)

    for x, y in [(time_ns[0], waveform1[0]), (time_ns[-1], waveform1[-1])]:
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor=color1,
            markeredgewidth=1.4,
            zorder=5,
        )
    for x, y in [(time_ns[0], waveform2[0]), (time_ns[-1], waveform2[-1])]:
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor=color2,
            markeredgewidth=1.4,
            zorder=5,
        )

    label_size, tick_size = LABEL_SIZE, TICK_SIZE
    ax.text(-15.0, 0.35, r"$\mathbf{S_1}$", color=color1, fontsize=label_size)
    ax.text(-19.0, 0.56, r"$\mathbf{S_2}$", color=color2, fontsize=label_size)

    x_min = -width * 1e9 / 2.0
    x_max = width * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    s1_ns = std_dev1 * 1e9
    s2_ns = std_dev2 * 1e9

    s1_y = 0.30
    s2_y = 0.14
    draw_span(ax, -s1_ns / 2.0, s1_ns / 2.0, s1_y, color1)
    draw_span(ax, -s2_ns / 2.0, s2_ns / 2.0, s2_y, color2)

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
        r"$\mathbf{\sigma_1}$",
        color=color1,
        ha="center",
        fontsize=label_size,
    )
    ax.text(
        0.0,
        s2_y + 0.02,
        r"$\mathbf{\sigma_2}$",
        color=color2,
        ha="center",
        fontsize=label_size,
    )

    draw_amp_axis(ax, y_axis_x, x_max, y_bottom=-0.02, y_top=1.06)

    ax.text(
        (x_min + x_max) / 2.0,
        1.14,
        r"$\mathbf{\sigma_i}$: std_dev    $\mathbf{w}$: width    $\mathbf{S_i}$: waveform",
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

    sigma_tick_labels = {
        -s2_ns / 2.0: r"$-\sigma_2/2$",
        s2_ns / 2.0: r"$\sigma_2/2$",
    }

    for tick in (-s1_ns / 2.0, s1_ns / 2.0):
        ax.vlines(tick, -0.028, 0.028, color="black", linewidth=1.2)

    for tick, label in sorted(sigma_tick_labels.items()):
        ax.vlines(tick, -0.028, 0.028, color="black", linewidth=1.2)
        ax.text(tick, -0.05, label, ha="center", va="top", color="black", fontsize=10)

    mid_x = (y_axis_x + x_max) / 2.0 + 3.0
    ax.text(mid_x, -0.19, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.24, 1.20)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated Sech waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
