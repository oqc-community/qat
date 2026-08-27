# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""RoundedSquare waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_rounded_square_waveform.py

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

from qat.utils.waveform import RoundedSquareFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-RoundedSquare waveform figure.

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
    rise = 6e-9
    std_dev1, color1 = 50e-9, "#2980b9"
    std_dev2, color2 = 34e-9, "grey"

    time = sample_time(width)
    eval_time = time + (width / 2.0)
    waveform1 = (
        RoundedSquareFunction(width=width, rise=rise, std_dev=std_dev1).eval(eval_time).real
    )
    waveform2 = (
        RoundedSquareFunction(width=width, rise=rise, std_dev=std_dev2).eval(eval_time).real
    )

    time_ns = time * 1e9
    ax.plot(time_ns, waveform1, color=color1)
    ax.plot(time_ns, waveform2, color=color2)

    label_size, tick_size = LABEL_SIZE, TICK_SIZE
    ax.text(-24.0, 0.84, r"$\mathbf{S_1}$", color=color1, fontsize=label_size)
    ax.text(-14.0, 0.70, r"$\mathbf{S_2}$", color=color2, fontsize=label_size)

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
        s2_y + 0.05,
        r"$\mathbf{\sigma_2}$",
        color=color2,
        ha="center",
        fontsize=label_size,
    )

    draw_amp_axis(ax, y_axis_x, x_max)

    # Rise annotation — erf transition centred on ±σ/2; span ±r around each edge
    # S1 annotated on left edge, S2 on right edge
    r_ns = rise * 1e9
    rise_arrow_y = 1.14

    for std_ns, waveform, arrow_y_offset, side in [
        (s1_ns, waveform1, 0.0, "left"),
        (s2_ns, waveform2, -0.12, "right"),
    ]:
        if side == "left":
            edge_x = -std_ns / 2.0
            x_left, x_right = edge_x - r_ns, edge_x + r_ns
        else:
            edge_x = std_ns / 2.0
            x_left, x_right = edge_x - r_ns, edge_x + r_ns
        arr_y = rise_arrow_y + arrow_y_offset
        w_left = float(np.interp(x_left, time_ns, waveform))
        w_right = float(np.interp(x_right, time_ns, waveform))

        ax.plot(
            [x_left, x_left],
            [w_left, arr_y],
            color="black",
            linewidth=0.9,
            linestyle="--",
            alpha=0.6,
        )
        ax.plot(
            [x_right, x_right],
            [w_right, arr_y],
            color="black",
            linewidth=0.9,
            linestyle="--",
            alpha=0.6,
        )
        ax.plot(
            x_left,
            w_left,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=6,
        )
        ax.plot(
            x_right,
            w_right,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=6,
        )
        ax.annotate(
            "",
            xy=(x_right, arr_y),
            xytext=(x_left, arr_y),
            arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "black"},
        )
        ax.text(
            (x_left + x_right) / 2.0,
            arr_y + 0.03,
            r"$r$",
            ha="center",
            va="bottom",
            color="black",
            fontsize=label_size,
        )

    ax.text(
        (x_min + x_max) / 2.0 + 14,
        1.32,
        r"$\mathbf{r}$: rise    $\mathbf{\sigma_i}$: std_dev    $\mathbf{w}$: width    $\mathbf{S_i}$: waveform",
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
        -s1_ns / 2.0: (r"$-\sigma_1/2$", -0.075),
        -s2_ns / 2.0: (r"$-\sigma_2/2$", -0.108),
        s2_ns / 2.0: (r"$\sigma_2/2$", -0.108),
        s1_ns / 2.0: (r"$\sigma_1/2$", -0.075),
    }
    for tick, (label, y) in sorted(sigma_tick_labels.items()):
        ax.vlines(tick, -0.028, 0.028, color="black", linewidth=1.2)
        ax.text(tick, y, label, ha="center", va="top", color="black", fontsize=10)

    mid_x = (y_axis_x + x_max) / 2.0 + 3.0
    ax.text(mid_x, -0.16, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.22, 1.42)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated RoundedSquare waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
