# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""SofterGaussian waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_softer_gaussian_waveform.py

The :func:`draw_figure` function is importable for use in Sphinx ``.. plot::``
directives without triggering the CLI argument parser.

:class:`~qat.ir.waveforms.SofterGaussianWaveform` is a Gaussian that has been
min/max normalised so that the pulse is *exactly* zero at the window edges and
peaks at 1, unlike a raw Gaussian which only asymptotically approaches zero.
The plot overlays the raw :class:`~qat.utils.waveform.GaussianFunction`
(dashed) against the normalised :class:`~qat.utils.waveform.SofterGaussianFunction`
(solid) to make the edge-zero property immediately visible.
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

from qat.utils.waveform import GaussianFunction, SofterGaussianFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated SofterGaussian waveform figure.

    Two ``rise`` values are shown. For each, the raw (unnormalised) Gaussian
    is drawn as a **dashed** line and the SofterGaussian (min/max normalised)
    as a **solid** line, highlighting the zero-at-edges correction.

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
    rise1, color1 = 0.20, "#2980b9"
    rise2, color2 = 0.35, "grey"

    time = sample_time(width)
    time_ns = time * 1e9

    softer1 = SofterGaussianFunction(width=width, rise=rise1).eval(time).real
    softer2 = SofterGaussianFunction(width=width, rise=rise2).eval(time).real
    raw2 = GaussianFunction(width=width, rise=rise2).eval(time).real

    # Draw raw (dashed) then softer (solid) so solid sits on top
    ax.plot(time_ns, raw2, color=color2, linewidth=1.4, linestyle="--", alpha=0.6)
    ax.plot(time_ns, softer1, color=color1, linewidth=2.0)
    ax.plot(time_ns, softer2, color=color2, linewidth=2.0)

    label_size, tick_size = LABEL_SIZE, TICK_SIZE

    # Waveform labels
    ax.text(0.7, 0.7, r"$\mathbf{G_1}$", color=color1, fontsize=label_size)
    ax.text(8.0, 0.96, r"$\mathbf{G_2}$", color=color2, fontsize=label_size)

    x_min = -width * 1e9 / 2.0
    x_max = width * 1e9 / 2.0
    y_axis_x = x_min - 8.0

    # Open circle markers where each softer waveform touches zero at the edges
    for x_edge in (x_min, x_max):
        for color in (color1, color2):
            ax.plot(
                x_edge,
                0.0,
                marker="o",
                markersize=6,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.2,
                zorder=5,
            )

    # rise span annotations for both waveforms
    k1_ns = width * 1e9 * rise1
    k2_ns = width * 1e9 * rise2
    for k_ns, label, color, span_y in [
        (k1_ns, r"$k_1 = w\,r_1$", color1, 0.38),
        (k2_ns, r"$k_2 = w\,r_2$", color2, 0.18),
    ]:
        draw_span(ax, -k_ns / 2.0, k_ns / 2.0, span_y, color)
        ax.text(
            0.0,
            span_y + 0.03,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontsize=tick_size,
        )
        for x_tick in (-k_ns / 2.0, k_ns / 2.0):
            ax.vlines(x_tick, span_y - 0.03, span_y + 0.03, color=color, linewidth=1.2)

    # Inline "raw" annotation — tip near the right edge where softer≈0 but raw is still visible,
    # so the open circle sits unambiguously on the dashed curve only
    raw_label_x, raw_label_y = 27.0, 0.55
    raw_tip_x = 36.0
    raw_tip_y = float(np.interp(raw_tip_x, time_ns, raw2))
    ax.annotate(
        "raw",
        xy=(raw_tip_x, raw_tip_y),
        xytext=(raw_label_x, raw_label_y),
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.3",
            "linewidth": 0.9,
            "color": "black",
        },
        color="black",
        fontsize=tick_size,
        ha="left",
        va="center",
    )
    ax.plot(
        raw_tip_x,
        raw_tip_y,
        marker="o",
        markersize=5,
        color="black",
        markerfacecolor="none",
        markeredgewidth=1.2,
        zorder=6,
    )

    draw_amp_axis(
        ax, y_axis_x, x_max, y_bottom=-0.14, y_top=1.12, label_offset=4.5, hlines_ext=2.0
    )

    ax.text(
        (x_min + x_max) / 2.0 - 10.0,
        1.14,
        r"$\mathbf{r_i}$: rise    $\mathbf{w}$: width    "
        r"$\mathbf{G_i}$: waveform",
        ha="center",
        va="top",
        fontsize=label_size,
    )

    tick_labels = {x_min: r"$-w/2$", 0.0: r"$0$", x_max: r"$w/2$"}
    for tick, label in sorted(tick_labels.items()):
        ax.vlines(tick, -0.035, 0.035, color="black", linewidth=1.4)
        ax.text(tick, -0.075, label, ha="center", va="top", fontsize=tick_size)

    # std dev ticks at ±k/2 for each waveform
    for k_ns, color, label_neg, label_pos in [
        (k1_ns, color1, r"$-\sigma_1$", r"$\sigma_1$"),
        (k2_ns, color2, r"$-\sigma_2$", r"$\sigma_2$"),
    ]:
        for x_tick, label in [(-k_ns / 2.0, label_neg), (k_ns / 2.0, label_pos)]:
            ax.vlines(x_tick, -0.035, 0.035, color=color, linewidth=1.2)
            ax.text(
                x_tick,
                -0.075,
                label,
                ha="center",
                va="top",
                color=color,
                fontsize=tick_size - 1,
            )

    mid_x = (y_axis_x + x_max) / 2.0 + 3.0
    ax.text(mid_x, -0.18, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(y_axis_x - 10.0, x_max + 34.0)
    ax.set_ylim(-0.28, 1.22)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated SofterGaussian waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
