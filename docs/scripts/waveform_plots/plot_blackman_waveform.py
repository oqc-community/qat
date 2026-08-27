# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Blackman waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_blackman_waveform.py

The :func:`draw_figure` function is importable for use in Sphinx ``.. plot::``
directives without triggering the CLI argument parser.
"""

from _plot_utils import (
    LABEL_SIZE,
    TICK_SIZE,
    configure_matplotlib,
    draw_amp_axis,
    parse_args,
    sample_time,
    save_and_show,
)
from matplotlib import pyplot as plt

from qat.utils.waveform import BlackmanFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-Blackman waveform figure.

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

    width1, color1 = 20e-9, "#2980b9"
    width2, color2 = 80e-9, "grey"

    time = sample_time(width2)
    waveform1 = BlackmanFunction(width=width1).eval(time).real
    waveform2 = BlackmanFunction(width=width2).eval(time).real

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
    ax.text(-36.0, 0.86, r"$\mathbf{B_1}$", color=color1, fontsize=label_size)
    ax.text(-15.0, 0.80, r"$\mathbf{B_2}$", color=color2, fontsize=label_size)

    x_min = -width2 * 1e9 / 2.0
    x_max = width2 * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    draw_amp_axis(ax, y_axis_x, x_max, y_bottom=-0.02, y_top=1.06)

    ax.text(
        (x_min + x_max) / 2.0,
        1.14,
        r"$\mathbf{w_i}$: width    $\mathbf{B_i}$: waveform",
        ha="center",
        va="top",
        fontsize=label_size,
    )

    tick_labels = {
        -width2 * 1e9 / 2.0: (r"$-w_2/2$", "black"),
        width2 * 1e9 / 2.0: (r"$w_2/2$", "black"),
        -width1 * 1e9 / 2.0: (r"$-w_1/2$", color1),
        0.0: (r"$0$", "black"),
        width1 * 1e9 / 2.0: (r"$w_1/2$", color1),
    }
    for tick, (label, col) in sorted(tick_labels.items()):
        ax.vlines(tick, -0.035, 0.035, color=col, linewidth=1.4)
        ax.text(tick, -0.075, label, ha="center", va="top", fontsize=tick_size, color=col)

    mid_x = (y_axis_x + x_max) / 2.0 + 3.0
    ax.text(mid_x, -0.2, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.18, 1.20)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated Blackman waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
