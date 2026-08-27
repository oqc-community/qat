# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""ExtraSoftSquare waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_extra_soft_square_waveform.py

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

from qat.utils.waveform import ExtraSoftSquareFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-ExtraSoftSquare waveform figure.

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
    rise = 6e-9
    # rise = 2e-9
    std_dev1, color1 = 40e-9, "#2980b9"
    std_dev2, color2 = 60e-9, "grey"

    time = sample_time(width)
    waveform1 = ExtraSoftSquareFunction(std_dev=std_dev1, rise=rise).eval(time).real
    waveform2 = ExtraSoftSquareFunction(std_dev=std_dev2, rise=rise).eval(time).real

    ax.plot(time * 1e9, waveform1, color=color1)
    ax.plot(time * 1e9, waveform2, color=color2)

    time_ns = time * 1e9

    label_size, tick_size = LABEL_SIZE, TICK_SIZE
    ax.text(-4.0, 0.80, r"$\mathbf{S_1}$", color=color1, fontsize=label_size)
    ax.text(-20.0, 0.78, r"$\mathbf{S_2}$", color=color2, fontsize=label_size)

    x_min = -width * 1e9 / 2.0
    x_max = width * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    sigma1_ns = std_dev1 * 1e9
    sigma2_ns = std_dev2 * 1e9

    sigma1_edges = (-sigma1_ns / 2.0, sigma1_ns / 2.0)
    sigma2_edges = (-sigma2_ns / 2.0, sigma2_ns / 2.0)

    sigma1_y = 0.32
    sigma2_y = 0.16
    draw_span(ax, sigma1_edges[0], sigma1_edges[1], sigma1_y, color1, linestyle=":")
    draw_span(ax, sigma2_edges[0], sigma2_edges[1], sigma2_y, color2, linestyle=":")

    ax.text(
        0.0,
        sigma1_y + 0.05,
        r"$\mathbf{\sigma_1}$",
        color=color1,
        ha="center",
        fontsize=label_size,
    )
    ax.text(
        0.0,
        sigma2_y + 0.05,
        r"$\mathbf{\sigma_2}$",
        color=color2,
        ha="center",
        fontsize=label_size,
    )

    # Rise annotation — r is the tanh length scale; inflection at -σ/2 + 2r
    # Shown on both waveforms' left edges — same r, same arrow length
    rise_ns = rise * 1e9
    rise_arrow_y = 1.14

    for std_ns, waveform, arrow_y_offset, side in [
        (sigma1_ns, waveform1, 0.0, "left"),
        (sigma2_ns, waveform2, -0.12, "right"),
    ]:
        if side == "left":
            x_infl = -std_ns / 2.0 + 2.0 * rise_ns
            x_infl_r = x_infl + rise_ns
        else:
            x_infl = std_ns / 2.0 - 2.0 * rise_ns
            x_infl_r = x_infl - rise_ns
        arr_y = rise_arrow_y + arrow_y_offset
        w_infl = float(np.interp(x_infl, time_ns, waveform))
        w_infl_r = float(np.interp(x_infl_r, time_ns, waveform))

        ax.plot(
            [x_infl, x_infl],
            [w_infl, arr_y],
            color="black",
            linewidth=0.9,
            linestyle="--",
            alpha=0.6,
        )
        ax.plot(
            [x_infl_r, x_infl_r],
            [w_infl_r, arr_y],
            color="black",
            linewidth=0.9,
            linestyle="--",
            alpha=0.6,
        )
        ax.plot(
            x_infl,
            w_infl,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=6,
        )
        ax.plot(
            x_infl_r,
            w_infl_r,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=6,
        )
        ax.annotate(
            "",
            xy=(x_infl_r, arr_y),
            xytext=(x_infl, arr_y),
            arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "black"},
        )
        ax.text(
            (x_infl + x_infl_r) / 2.0,
            arr_y + 0.03,
            r"$r$",
            ha="center",
            va="bottom",
            color="black",
            fontsize=label_size,
        )

    draw_amp_axis(ax, y_axis_x, x_max, label_offset=5.0)

    ax.text(
        x_max - 1.5,
        1.03,
        r"$\mathbf{\sigma_i}$: std_dev"
        "\n"
        r"$\mathbf{r}$: rise"
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
        -width * 1e9 / 2.0: r"$-w/2$",
        0.0: r"$0$",
        width * 1e9 / 2.0: r"$w/2$",
    }
    for tick, label in sorted(tick_labels.items()):
        ax.vlines(tick, -0.035, 0.035, color="black", linewidth=1.4)
        ax.text(tick, -0.075, label, ha="center", va="top", fontsize=tick_size)

    sigma_tick_labels = {
        -sigma2_ns / 2.0: (r"$-\sigma_2/2$", color2, -0.108),
        -sigma1_ns / 2.0: (r"$-\sigma_1/2$", color1, -0.075),
        sigma1_ns / 2.0: (r"$\sigma_1/2$", color1, -0.075),
        sigma2_ns / 2.0: (r"$\sigma_2/2$", color2, -0.108),
    }
    for tick, (label, color, y) in sorted(sigma_tick_labels.items()):
        ax.vlines(tick, -0.028, 0.028, color=color, linewidth=1.2)
        ax.text(tick, y, label, ha="center", va="top", color=color, fontsize=10)

    mid_x = (y_axis_x + x_max) / 2.0 + 3
    ax.text(mid_x, -0.14, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.22, 1.32)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated ExtraSoftSquare waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
