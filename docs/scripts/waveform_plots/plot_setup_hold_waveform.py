# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""SetupHold waveform plot generator for documentation.

Run directly::

    poetry run python docs/scripts/waveform_plots/plot_setup_hold_waveform.py

The :func:`draw_figure` function is importable for use in Sphinx ``.. plot::``
directives without triggering the CLI argument parser.
"""

import numpy as np
from _plot_utils import (
    LABEL_SIZE,
    TICK_SIZE,
    configure_matplotlib,
    draw_span,
    parse_args,
    save_and_show,
)
from matplotlib import pyplot as plt

from qat.utils.waveform import SetupHoldFunction

configure_matplotlib()


def draw_figure(ax: "plt.Axes | None" = None) -> plt.Figure:
    """Draw an annotated dual-SetupHold waveform figure.

    Two waveforms are overlaid, one with a longer setup section (``rise``)
    and one with a shorter setup section, making the setup/hold split clearly
    visible.  The hold amplitude is normalised to 1 (``amp = amp``); the
    setup section is drawn at ``amp_setup / amp`` relative to that.

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
    amp = 1.0

    rise1, amp_setup1, color1 = 20e-9, 1.8, "#2980b9"
    rise2, amp_setup2, color2 = 35e-9, 1.5, "grey"

    # Pulse runs from t=0 to t=w; pad only the right side so the waveform
    # emerges directly from the y-axis at t=0.
    pad_ns = 15.0
    pad_samples = 140
    time_core = np.linspace(0.0, width, 700)
    time_ns = np.concatenate(
        [
            time_core * 1e9,
            time_core[-1] * 1e9 + np.linspace(1, pad_ns, pad_samples),
        ]
    )

    waveform1_core = (
        SetupHoldFunction(width=width, rise=rise1, amp_setup=amp_setup1, amp=amp)
        .eval(time_core)
        .real
    )
    waveform2_core = (
        SetupHoldFunction(width=width, rise=rise2, amp_setup=amp_setup2, amp=amp)
        .eval(time_core)
        .real
    )

    zeros_right = np.zeros(pad_samples)
    waveform1 = np.concatenate([waveform1_core, zeros_right])
    waveform2 = np.concatenate([waveform2_core, zeros_right])

    ax.plot(time_ns, waveform1, color=color1, linewidth=2.0)
    ax.plot(time_ns, waveform2, color=color2, linewidth=2.0)

    label_size, tick_size = LABEL_SIZE, TICK_SIZE

    x_min = 0.0
    x_max = width * 1e9
    y_axis_x = x_min  # y-axis sits exactly at t=0

    r1_ns = rise1 * 1e9
    r2_ns = rise2 * 1e9

    # Waveform labels: H1 sits inside its setup region, H2 inside its hold region
    ax.text(
        r1_ns / 2.0,
        amp_setup1 + 0.06,
        r"$\mathbf{H_1}$",
        ha="center",
        color=color1,
        fontsize=label_size,
    )
    ax.text(
        r2_ns + (x_max - r2_ns) / 2.0,
        amp + 0.06,
        r"$\mathbf{H_2}$",
        ha="center",
        color=color2,
        fontsize=label_size,
    )

    span1_y = 0.40
    span2_y = 0.22
    draw_span(ax, x_min, r1_ns, span1_y, color1)
    draw_span(ax, x_min, r2_ns, span2_y, color2)

    ax.text(
        r1_ns / 2.0,
        span1_y + 0.05,
        r"$\mathbf{r_1}$",
        ha="center",
        va="bottom",
        color=color1,
        fontsize=label_size,
    )
    ax.text(
        r2_ns / 2.0,
        span2_y + 0.05,
        r"$\mathbf{r_2}$",
        ha="center",
        va="bottom",
        color=color2,
        fontsize=label_size,
    )

    # Vertical dotted line and marker at the setup→hold transition
    for x_rise, color in [(r1_ns, color1), (r2_ns, color2)]:
        ax.vlines(x_rise, 0.0, amp, color=color, linewidth=1.1, linestyles="dotted")
        ax.plot(
            x_rise,
            amp,
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=1.2,
        )

    # Horizontal dotted reference lines: extend rightward from transition
    for x_rise_ns, amp_s, color in [
        (r1_ns, amp_setup1, color1),
        (r2_ns, amp_setup2, color2),
    ]:
        ax.hlines(
            amp_s,
            x_rise_ns,
            x_max + pad_ns + 1.0,
            color=color,
            linewidth=1.0,
            linestyles="dotted",
            alpha=0.7,
        )

    # amp_setup double-headed arrows: staggered x positions to avoid overlap
    def _draw_vspan(x: float, y0: float, y1: float, color: str) -> None:
        ax.annotate(
            "",
            xy=(x, y0),
            xytext=(x, y1),
            arrowprops={
                "arrowstyle": "<->",
                "connectionstyle": "arc3",
                "linestyle": "-",
                "linewidth": 0.8,
                "color": color,
            },
        )

    # as1/as2 arrows inside the hold region, left of w
    arrow1_x = x_max - 8.0
    arrow2_x = x_max - 16.0
    _draw_vspan(arrow1_x, amp, amp_setup1, color1)
    ax.text(
        arrow1_x + 1.0,
        (amp + amp_setup1) / 2.0,
        r"$a_{\mathrm{s,1}}$",
        ha="left",
        va="center",
        color=color1,
        fontsize=tick_size,
    )
    _draw_vspan(arrow2_x, amp, amp_setup2, color2)
    ax.text(
        arrow2_x + 1.0,
        (amp + amp_setup2) / 2.0,
        r"$a_{\mathrm{s,2}}$",
        ha="left",
        va="center",
        color=color2,
        fontsize=tick_size,
    )

    ax.hlines(0.0, x_min, x_max + pad_ns + 2.0, color="black", linewidth=1.0)
    ax.vlines(y_axis_x, 0.0, amp_setup1 + 0.14, color="black", linewidth=1.0)
    ax.annotate(
        "",
        xy=(y_axis_x, 1.0),
        xytext=(y_axis_x, 0.0),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "black"},
    )
    ax.hlines(0.0, y_axis_x - 1.5, y_axis_x + 1.5, color="black", linewidth=1.2)
    ax.hlines(1.0, y_axis_x - 1.5, y_axis_x + 1.5, color="black", linewidth=1.2)
    ax.text(
        -4.0,
        0.5,
        "amp",
        ha="center",
        va="center",
        fontsize=label_size,
    )

    # amp (hold level) tick on y-axis
    ax.hlines(amp, y_axis_x - 0.5, y_axis_x + 0.5, color="black", linewidth=1.2)
    ax.text(
        y_axis_x - 1.5,
        amp,
        r"$a$",
        ha="right",
        va="center",
        fontsize=tick_size,
    )

    ax.text(
        (x_min + x_max) / 2.0 + 5,
        amp_setup1 + 0.4,
        r"$\mathbf{r_i}$: rise    $\mathbf{a_{\mathrm{s},i}}$: amp_setup    "
        r"$\mathbf{w}$: width    $\mathbf{H_i}$: waveform",
        ha="center",
        va="top",
        fontsize=label_size,
    )

    # x-axis ticks: 0, r1, r2, w
    ax.vlines(x_min, -0.035, 0.035, color="black", linewidth=1.4)
    ax.text(x_min, -0.075, r"$0$", ha="center", va="top", fontsize=tick_size)
    ax.vlines(x_max, -0.035, 0.035, color="black", linewidth=1.4)
    ax.text(x_max, -0.075, r"$w$", ha="center", va="top", fontsize=tick_size)
    for r_ns, label, color in [
        (r1_ns, r"$r_1$", color1),
        (r2_ns, r"$r_2$", color2),
    ]:
        ax.vlines(r_ns, -0.028, 0.028, color=color, linewidth=1.2)
        ax.text(r_ns, -0.075, label, ha="center", va="top", color=color, fontsize=tick_size)

    mid_x = (x_min + x_max) / 2.0
    ax.text(mid_x + 13, -0.10, r"Time / ns", ha="center", va="top", fontsize=label_size)

    ax.set_xlim(-8.0, x_max + pad_ns + 3.0)
    ax.set_ylim(-0.18, amp_setup1 + 0.38)
    ax.axis("off")
    return fig


if __name__ == "__main__":
    args = parse_args("Plot annotated SetupHold waveform examples.")
    fig, ax = plt.subplots(figsize=(8, 4))
    draw_figure(ax=ax)
    save_and_show(fig, args)
