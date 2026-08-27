# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Sin waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_sin_waveform.py

The :func:`draw_figure` function is importable for use in Sphinx ``.. plot::``
directives without triggering the CLI argument parser.
"""

import numpy as np
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

from qat.utils.waveform import Sin

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-Sin waveform figure.

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
    frequency1, color1 = 14e6, "#2980b9"
    frequency2, color2 = 33e6, "grey"
    phase1, phase2 = 0.0, np.pi / 3

    time = sample_time(width)
    waveform1 = Sin(frequency=frequency1, internal_phase=phase1).eval(time).real
    waveform2 = Sin(frequency=frequency2, internal_phase=phase2).eval(time).real

    time_ns = time * 1e9
    ax.plot(time_ns, waveform1, color=color1, linewidth=2.0)
    ax.plot(time_ns, waveform2, color=color2, alpha=0.6, linewidth=2.0)

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
            alpha=0.6,
            zorder=5,
        )

    label_size, tick_size = LABEL_SIZE, TICK_SIZE
    ax.text(-43.0, 0.45, r"$\mathbf{S_1}$", color=color1, fontsize=label_size)
    ax.text(-43.0, -0.65, r"$\mathbf{S_2}$", color=color2, fontsize=label_size)

    x_min = -width * 1e9 / 2.0
    x_max = width * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    draw_amp_axis(ax, y_axis_x, x_max, y_bottom=-1.06, y_top=1.06)

    ax.text(
        (x_min + x_max) / 2.0,
        1.65,
        r"$\mathbf{f_i}$: frequency    $\mathbf{\varphi_i}$: internal_phase    "
        r"$\mathbf{w}$: width    $\mathbf{S_i}$: waveform",
        ha="center",
        va="top",
        fontsize=label_size,
    )

    # --- Frequency annotation: period arrows below x-axis ---
    T1_ns = 1.0 / frequency1 * 1e9  # ~71.4 ns
    T2_ns = 1.0 / frequency2 * 1e9  # ~30.3 ns

    period_y1 = -1.22
    T1_left, T1_right = -T1_ns / 2.0, T1_ns / 2.0
    ax.annotate(
        "",
        xy=(T1_right, period_y1),
        xytext=(T1_left, period_y1),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": color1},
    )
    ax.vlines(T1_left, period_y1 - 0.04, period_y1 + 0.04, color=color1, linewidth=1.2)
    ax.vlines(T1_right, period_y1 - 0.04, period_y1 + 0.04, color=color1, linewidth=1.2)
    ax.text(
        0.0,
        period_y1 + 0.07,
        r"$T_1 = 1/f_1$",
        ha="center",
        va="bottom",
        fontsize=tick_size,
        color=color1,
    )

    period_y2 = -1.52
    T2_left, T2_right = -T2_ns / 2.0, T2_ns / 2.0
    ax.annotate(
        "",
        xy=(T2_right, period_y2),
        xytext=(T2_left, period_y2),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": color2},
    )
    ax.vlines(T2_left, period_y2 - 0.04, period_y2 + 0.04, color=color2, linewidth=1.2)
    ax.vlines(T2_right, period_y2 - 0.04, period_y2 + 0.04, color=color2, linewidth=1.2)
    ax.text(
        0.0,
        period_y2 + 0.07,
        r"$T_2 = 1/f_2$",
        ha="center",
        va="bottom",
        fontsize=tick_size,
        color=color2,
    )

    # --- Phase annotation ---
    # S1 (phi1=0): upward zero-crossing at t=0
    # S2 (phi2=pi/3): upward zero-crossing at t = -phi2 / (2*pi*f2) ~= -5.05 ns
    zero_S2_ns = -phase2 / (2.0 * np.pi * frequency2) * 1e9

    ax.vlines(0.0, -0.06, 1.12, color="black", linewidth=0.8, linestyle="--")
    ax.vlines(zero_S2_ns, -0.06, 1.12, color=color2, linewidth=0.9, linestyle="--")
    phase_y = 1.12
    ax.annotate(
        "",
        xy=(0.0, phase_y),
        xytext=(zero_S2_ns, phase_y),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": color2},
    )
    ax.text(
        zero_S2_ns / 2.0,
        phase_y + 0.03,
        r"$\varphi_2$",
        ha="center",
        va="bottom",
        fontsize=tick_size,
        color=color2,
    )

    tick_labels = {
        -width * 1e9 / 2.0: r"$-w/2$",
        0.0: r"$0$",
        width * 1e9 / 2.0: r"$w/2$",
    }
    for tick, label in sorted(tick_labels.items()):
        ax.vlines(tick, -0.06, 0.06, color="black", linewidth=1.4)
        ax.text(tick, -0.12, label, ha="center", va="top", fontsize=tick_size)

    mid_x = (y_axis_x + x_max) / 2.0 + 3.0
    ax.text(mid_x, -1.80, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-1.94, 1.36)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated Sin waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
