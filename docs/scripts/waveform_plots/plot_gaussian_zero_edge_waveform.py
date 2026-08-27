# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""GaussianZeroEdge waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_gaussian_zero_edge_waveform.py

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
from matplotlib.patches import ConnectionPatch

from qat.utils.waveform import GaussianZeroEdgeFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-GaussianZeroEdge waveform figure.

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
    std_dev = 12e-9
    color_false = "#2980b9"
    color_true = "grey"

    time = sample_time(width)
    waveform_false = (
        GaussianZeroEdgeFunction(
            std_dev=std_dev,
            width=width,
            zero_at_edges=False,
        )
        .eval(time)
        .real
    )
    waveform_true = (
        GaussianZeroEdgeFunction(
            std_dev=std_dev,
            width=width,
            zero_at_edges=True,
        )
        .eval(time)
        .real
    )

    # Ideal un-clipped Gaussian for reference — clipped where curve is still visible
    _sigma_clip = 3.2 * std_dev
    time_wide = np.linspace(-_sigma_clip, _sigma_clip, 700)
    ideal_gaussian = np.exp(-(time_wide**2) / (2.0 * std_dev**2))
    time_wide_ns = time_wide * 1e9

    time_ns = time * 1e9
    ax.plot(
        time_wide_ns,
        ideal_gaussian,
        color="black",
        linestyle=":",
        linewidth=1.2,
        alpha=0.25,
        zorder=1,
    )
    ax.plot(time_ns, waveform_true, color=color_true, zorder=2)
    ax.plot(time_ns, waveform_false, color=color_false, linestyle="--", zorder=3)

    label_size, tick_size = LABEL_SIZE, TICK_SIZE
    x_min = -width * 1e9 / 2.0
    x_max = width * 1e9 / 2.0
    y_axis_x = x_min - 8.0
    sigma_ns = std_dev * 1e9

    draw_amp_axis(ax, y_axis_x, x_max, label_offset=5.0)
    ax.text(y_axis_x - 1.0, 0.0, "0", ha="right", va="center", fontsize=10)

    ax.text(
        x_max + 1.5,
        1.03,
        r"$\mathbf{\sigma}$: std_dev"
        "\n"
        r"$\mathbf{z}$: zero_at_edges"
        "\n"
        r"$\mathbf{w}$: width"
        "\n"
        r"$\mathbf{G}$: waveform",
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
        -sigma_ns / 2.0: r"$-\sigma/2$",
        sigma_ns / 2.0: r"$\sigma/2$",
    }
    for tick, label in sorted(sigma_tick_labels.items()):
        ax.vlines(tick, -0.028, 0.028, color="black", linewidth=1.2)
        ax.text(tick, -0.108, label, ha="center", va="top", color="black", fontsize=10)

    edge_inset = ax.inset_axes([0.15, 0.5, 0.26 / 1.3, 0.46 / 1.3])
    edge_inset.plot(
        time_wide_ns,
        ideal_gaussian,
        color="black",
        linestyle=":",
        linewidth=1.2,
        alpha=0.25,
        zorder=1,
    )
    edge_inset.plot(time_ns, waveform_true, color=color_true, zorder=2)
    edge_inset.plot(time_ns, waveform_false, color=color_false, linestyle="--", zorder=3)
    edge_inset.set_xlim(x_min, x_min + 3.5)
    edge_inset.set_ylim(0, 0.01)
    edge_inset.set_xticks([])
    edge_inset.set_yticks([])
    edge_inset.set_facecolor("white")
    edge_inset.patch.set_alpha(0.94)

    bottom_connector = ConnectionPatch(
        xyA=(y_axis_x, 0.0),
        coordsA=ax.transData,
        xyB=(0.0, 0.0),
        coordsB=edge_inset.transAxes,
        axesA=ax,
        axesB=edge_inset,
        color=color_true,
        linewidth=1.0,
        linestyle=":",
        alpha=0.45,
        zorder=1,
    )
    top_connector = ConnectionPatch(
        xyA=(y_axis_x, 0.01),
        coordsA=ax.transData,
        xyB=(0.0, 1.0),
        coordsB=edge_inset.transAxes,
        axesA=ax,
        axesB=edge_inset,
        color=color_true,
        linewidth=1.0,
        linestyle=":",
        alpha=0.45,
        zorder=1,
    )
    ax.add_artist(bottom_connector)
    ax.add_artist(top_connector)
    edge_inset.text(
        x_min + 0.10,
        0.007,
        r"$\mathbf{G}(z=\mathrm{false})$",
        color=color_false,
        fontsize=9,
    )
    edge_inset.text(
        x_min + 0.20,
        0.003,
        r"$\mathbf{G}(z=\mathrm{true})$",
        color=color_true,
        fontsize=9,
    )

    mid_x = (y_axis_x + x_max) / 2.0 + 3
    ax.text(mid_x, -0.205, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 5.0)
    ax.set_ylim(-0.29, 1.1)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated GaussianZeroEdge waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
