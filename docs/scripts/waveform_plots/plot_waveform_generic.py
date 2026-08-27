# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Generic Waveform parameter plot generator for documentation.

Illustrates the shared parameters common to all :class:`~qat.ir.waveforms.Waveform`
subclasses — ``amp``, ``width``, ``scale_factor``, ``drag``, and ``phase`` — using
an example Gaussian envelope.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_waveform_generic.py

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

from qat.utils.waveform import GaussianFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated generic Waveform parameter figure.

    Uses a Gaussian envelope to illustrate ``amp``, ``width``,
    ``scale_factor``, ``drag``, and ``phase``.

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
    rise = 0.35  # dimensionless rise for GaussianFunction: k = width * rise

    color_base = "#2980b9"
    color_scaled = "grey"

    time = sample_time(width)
    time_ns = time * 1e9

    fn = GaussianFunction(width=width, rise=rise)
    waveform_base = fn.eval(time).real

    # scale_factor = 0.6 applied to the base waveform
    scale_factor = 0.6
    waveform_scaled = waveform_base * scale_factor

    # DRAG correction: imaginary component proportional to the time-derivative
    # Normalised so its peak is clearly visible
    dt = time[1] - time[0]
    drag_coeff = 0.45
    drag1 = drag_coeff * np.gradient(waveform_base, dt) * (width * rise)
    drag2 = drag_coeff * np.gradient(waveform_scaled, dt) * (width * rise)

    ax.plot(time_ns, waveform_base, color=color_base, linewidth=2.0, label="_base")
    ax.plot(
        time_ns,
        waveform_scaled,
        color=color_scaled,
        linewidth=2.0,
        linestyle="-",
        label="_scaled",
    )
    ax.plot(time_ns, drag1, color=color_base, linewidth=1.5, linestyle=":", label="_drag1")
    ax.plot(
        time_ns, drag2, color=color_scaled, linewidth=1.5, linestyle=":", label="_drag2"
    )

    label_size, tick_size = LABEL_SIZE, TICK_SIZE

    x_min = -width * 1e9 / 2.0
    x_max = width * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    # ── waveform / scale labels ───────────────────────────────────────────
    ax.text(-18.0, 0.82, r"$\mathbf{W_1}$", color=color_base, fontsize=label_size)
    ax.text(-18, 0.50, r"$\mathbf{W_2}$", color=color_scaled, fontsize=label_size)

    # ── drag: open markers on both drag curves right of zero, both → single d label
    marker_x = 8.0  # ns — right of zero where both curves are still distinct
    m1_y = float(np.interp(marker_x, time_ns, drag1))
    m2_y = float(np.interp(marker_x, time_ns, drag2))
    label_x = marker_x + 14.0
    label_y = (m1_y + m2_y) / 2.0 + 0.08
    for my in (m1_y, m2_y):
        ax.plot(
            marker_x,
            my,
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.2,
            zorder=7,
        )
        ax.annotate(
            "",
            xy=(marker_x, my),
            xytext=(label_x, label_y),
            arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "black"},
        )
    ax.text(
        label_x + 1,
        label_y - 0.03,
        r"$\mathbf{d}$",
        fontsize=label_size,
        ha="center",
        va="bottom",
        color="black",
    )

    # ── y-axis: amp ───────────────────────────────────────────────────────
    draw_amp_axis(ax, y_axis_x, x_max, y_bottom=-0.55, y_top=1.06, label_offset=4.5)

    # ── scale_factor: arrow between peaks, ratio label at midpoint ─────────
    peak_x = 2.0
    ax.annotate(
        "",
        xy=(peak_x, scale_factor),
        xytext=(peak_x, 1.0),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "black"},
    )
    ax.text(
        peak_x + 1.5,
        (1.0 + scale_factor) / 2.0,
        r"$s_2/s_1$",
        ha="left",
        va="center",
        fontsize=label_size - 1,
        color="black",
    )

    # ── legend ────────────────────────────────────────────────────────────
    ax.text(
        (x_min + x_max) / 2.0,
        1.22,
        r"$\mathbf{w}$: width    "
        r"$\mathbf{\phi}$: phase=0    "
        r"$\mathbf{d}$: drag    "
        r"$\mathbf{s_i}$: scale_factor    "
        r"$\mathbf{W_i}$: waveform",
        ha="center",
        va="top",
        fontsize=label_size,
    )

    # ── x-axis ticks ──────────────────────────────────────────────────────
    tick_labels = {
        x_min: r"$-w/2$",
        0.0: r"$0$",
        x_max: r"$w/2$",
    }
    for tick, label in sorted(tick_labels.items()):
        ax.vlines(tick, -0.035, 0.035, color="black", linewidth=1.4)
        ax.text(tick, -0.075, label, ha="center", va="top", fontsize=tick_size)

    mid_x = (y_axis_x + x_max) / 2.0 + 3.0
    ax.text(mid_x, -0.50, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 8.0, x_max + 18.0)
    ax.set_ylim(-0.60, 1.30)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated generic Waveform parameter diagram.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
