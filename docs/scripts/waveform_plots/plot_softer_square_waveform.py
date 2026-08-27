# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""SofterSquare waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_softer_square_waveform.py

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

from qat.utils.waveform import SofterSquareFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-SofterSquare waveform figure.

    Two waveforms sharing the same ``width`` are overlaid with different
    ``std_dev`` values, showing how ``std_dev`` controls the flat-top width.
    A second pair illustrates the effect of ``rise`` on edge softness.

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
    std_dev1, rise1, color1 = 50e-9, 5e-9, "#2980b9"
    std_dev2, rise2, color2 = 50e-9, 18e-9, "grey"

    time = sample_time(width)
    waveform1 = SofterSquareFunction(std_dev=std_dev1, rise=rise1).eval(time).real
    waveform2 = SofterSquareFunction(std_dev=std_dev2, rise=rise2).eval(time).real

    time_ns = time * 1e9
    ax.plot(time_ns, waveform1, color=color1, linewidth=2.0)
    ax.plot(time_ns, waveform2, color=color2, linewidth=2.0)

    label_size, tick_size = LABEL_SIZE, TICK_SIZE

    ax.text(18, 0.85, r"$\mathbf{S_1}$", color=color1, fontsize=label_size)
    ax.text(13.0, 0.70, r"$\mathbf{S_2}$", color=color2, fontsize=label_size)

    x_min = -width * 1e9 / 2.0
    x_max = width * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    s_ns = std_dev1 * 1e9  # shared std_dev

    # Dotted verticals at ±std_dev/2 (same for both waveforms), up to waveform only
    for x_pos in (-s_ns / 2.0, s_ns / 2.0):
        for wf, color in [(waveform1, color1), (waveform2, color2)]:
            y_val = float(np.interp(x_pos, time_ns, wf))
            ax.vlines(x_pos, 0.0, y_val, color=color, linewidth=1.1, linestyles="dotted")
            ax.plot(
                x_pos,
                y_val,
                marker="o",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.2,
            )

    # Rise annotation — same style as ExtraSoftSquare:
    # dashed guides from waveform up to arrow, open circles at contact, <-> arrow.
    # The tanh transition is centred on -std_dev/2; bracket by ±rise.
    r1_ns = rise1 * 1e9
    r2_ns = rise2 * 1e9

    for r_ns, label, color, waveform, arr_y, edge in [
        (r1_ns, r"$r_1$", color1, waveform1, 0.84, -s_ns / 2.0),
        (r2_ns, r"$r_2$", color2, waveform2, 0.52, s_ns / 2.0),
    ]:
        x_left = edge - r_ns
        x_right = edge + r_ns
        w_left = float(np.interp(x_left, time_ns, waveform))
        w_right = float(np.interp(x_right, time_ns, waveform))

        # Dashed vertical guides from waveform to arrow height
        ax.plot(
            [x_left, x_left],
            [w_left, arr_y],
            color=color,
            linewidth=0.9,
            linestyle="--",
            alpha=0.6,
        )
        ax.plot(
            [x_right, x_right],
            [w_right, arr_y],
            color=color,
            linewidth=0.9,
            linestyle="--",
            alpha=0.6,
        )
        # Open circles at waveform contact points
        for xc, yc in [(x_left, w_left), (x_right, w_right)]:
            ax.plot(
                xc,
                yc,
                marker="o",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.2,
                zorder=6,
            )
        # Spanning arrow
        ax.annotate(
            "",
            xy=(x_right, arr_y),
            xytext=(x_left, arr_y),
            arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": color},
        )
        ax.text(
            edge,
            arr_y + 0.03,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontsize=label_size,
        )

    draw_amp_axis(ax, y_axis_x, x_max, y_bottom=-0.02, y_top=1.06)

    ax.text(
        (x_min + x_max) / 2.0,
        1.13,
        r"$\mathbf{\sigma}$: std_dev    $\mathbf{r_i}$: rise    "
        r"$\mathbf{w}$: width    $\mathbf{S_i}$: waveform",
        ha="center",
        va="top",
        fontsize=label_size,
    )

    tick_labels = {
        x_min: (r"$-w/2$", "black"),
        0.0: (r"$0$", "black"),
        x_max: (r"$w/2$", "black"),
        -s_ns / 2.0: (r"$-\sigma/2$", "black"),
        s_ns / 2.0: (r"$\sigma/2$", "black"),
    }
    for tick, (label, color) in sorted(tick_labels.items()):
        ax.vlines(tick, -0.035, 0.035, color="black", linewidth=1.4)
        ax.text(tick, -0.075, label, ha="center", va="top", color=color, fontsize=tick_size)

    mid_x = (y_axis_x + x_max) / 2.0 + 3.0
    ax.text(mid_x, -0.16, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.22, 1.20)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated SofterSquare waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
