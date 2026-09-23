# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from types import SimpleNamespace

import numpy as np
import pytest

from qat.purr.backends.qblox import ir
from qat.purr.backends.qblox.visualisation import plot_packages, plot_playback


def test_purr_sequence_builder_manages_data_tables():
    builder = ir.SequenceBuilder()

    builder.add_waveform("pulse", 0, [0.25, 0.5])
    builder.add_weight("weight", 1, [1.0, -1.0])
    builder.add_acquisition("readout", 2, 4)

    assert builder.lookup_waveform_by_data(np.array([0.25, 0.5])) == 0
    assert builder.lookup_waveform_by_data(np.array([0.25])) is None

    for method, arguments in [
        ("add_waveform", ("pulse", 3, [0.0])),
        ("add_weight", ("weight", 3, [0.0])),
        ("add_acquisition", ("readout", 3, 1)),
    ]:
        with pytest.raises(ValueError, match="already exists"):
            getattr(builder, method)(*arguments)


def test_purr_sequence_builder_adds_all_instruction_types():
    builder = ir.SequenceBuilder()
    calls = [
        ("nop", ()),
        ("stop", ()),
        ("label", ("target",)),
        ("jmp", ("target",)),
        ("jmp", (4,)),
        ("jge", ("R0", 1, "target")),
        ("jge", ("R0", 1, 4)),
        ("jlt", ("R0", 1, "target")),
        ("jlt", ("R0", 1, 4)),
        ("loop", ("R0", "target")),
        ("loop", ("R0", 4)),
        ("move", (1, "R0")),
        ("add", ("R0", 1, "R1")),
        ("sub", ("R0", 1, "R1")),
        ("logic_not", ("R0", "R1")),
        ("logic_and", ("R0", 1, "R1")),
        ("logic_or", ("R0", 1, "R1")),
        ("logic_xor", ("R0", 1, "R1")),
        ("set_mrk", (1,)),
        ("set_freq", (2,)),
        ("set_ph", (3,)),
        ("set_ph_delta", (4,)),
        ("reset_ph", ()),
        ("set_awg_gain", (5, 6)),
        ("set_awg_offs", (7, 8)),
        ("set_cond", (1, 2, 3, 4)),
        ("upd_param", (4,)),
        ("play", (0, 1, 4)),
        ("acquire", (0, "R0", 4)),
        ("acquire_weighed", (0, "R0", 1, 2, 4)),
        ("acquire_ttl", (0, "R0", 1, 4)),
        ("set_latch_en", (1, 4)),
        ("latch_rst", (4,)),
        ("wait", (4,)),
        ("wait_trigger", (1, 4)),
        ("wait_sync", (4,)),
    ]

    for method, arguments in calls:
        assert getattr(builder, method)(*arguments, comment="comment") is builder

    sequence = builder.build()
    assert len(builder.q1asm_instructions) == len(calls)
    assert len(sequence.program.splitlines()) == len(calls)
    assert repr(builder.q1asm_instructions[0]) == "nop # comment"
    assert str(ir.Q1asmInstruction(ir.Opcode.ADDRESS, "target")) == "target:"
    assert str(ir.Q1asmInstruction(ir.Opcode.STOP)) == "stop"


@pytest.mark.parametrize(
    "packages",
    [
        pytest.param([], id="no-packages"),
        pytest.param(
            [SimpleNamespace(timeline=np.array([]))],
            id="empty-timeline",
        ),
    ],
)
def test_purr_plot_packages_ignores_empty_programs(mocker, packages):
    show = mocker.patch("qat.purr.backends.qblox.visualisation.plt.show")

    plot_packages(packages)

    show.assert_not_called()


def test_purr_plot_packages_pads_and_plots_timelines(mocker):
    short_package = SimpleNamespace(
        timeline=np.array([1 + 2j]),
        pulse_channel_id="q0.drive",
    )
    long_package = SimpleNamespace(
        timeline=np.array([3 + 4j, 5 + 6j]),
        pulse_channel_id="q1.drive",
    )
    figure = mocker.Mock()
    axes = np.array([[mocker.Mock()], [mocker.Mock()]])
    subplots = mocker.patch(
        "qat.purr.backends.qblox.visualisation.plt.subplots",
        return_value=(figure, axes),
    )
    tight_layout = mocker.patch("qat.purr.backends.qblox.visualisation.plt.tight_layout")
    show = mocker.patch("qat.purr.backends.qblox.visualisation.plt.show")

    plot_packages([short_package, long_package])

    np.testing.assert_array_equal(short_package.timeline, np.array([1 + 2j, 0]))
    subplots.assert_called_once_with(
        nrows=2,
        ncols=1,
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=(10, 5),
    )
    figure.suptitle.assert_called_once_with("Timeline plots")
    assert axes[0, 0].plot.call_count == 2
    assert axes[1, 0].plot.call_count == 2
    axes[0, 0].set_title.assert_called_once_with("q0.drive")
    axes[1, 0].set_title.assert_called_once_with("q1.drive")
    axes[0, 0].legend.assert_called_once_with(loc="upper right")
    axes[1, 0].legend.assert_called_once_with(loc="upper right")
    tight_layout.assert_called_once_with()
    show.assert_called_once_with()


def test_purr_plot_playback_ignores_empty_results(mocker):
    show = mocker.patch("qat.purr.backends.qblox.visualisation.plt.show")

    plot_playback({})

    show.assert_not_called()


def test_purr_plot_playback_plots_all_acquisition_data(mocker):
    acquisition = SimpleNamespace(
        name="readout",
        acquisition=SimpleNamespace(
            scope=SimpleNamespace(
                path0=SimpleNamespace(data=np.array([1.0, 2.0])),
                path1=SimpleNamespace(data=np.array([3.0, 4.0])),
            ),
            bins=SimpleNamespace(
                integration=SimpleNamespace(
                    path0=np.array([5.0]),
                    path1=np.array([6.0]),
                ),
                threshold=np.array([1]),
            ),
        ),
    )
    figure = mocker.Mock()
    axes = np.array([[mocker.Mock()], [mocker.Mock()], [mocker.Mock()]])
    subplots = mocker.patch(
        "qat.purr.backends.qblox.visualisation.plt.subplots",
        return_value=(figure, axes),
    )
    tight_layout = mocker.patch("qat.purr.backends.qblox.visualisation.plt.tight_layout")
    show = mocker.patch("qat.purr.backends.qblox.visualisation.plt.show")

    plot_playback({"q0.readout": [acquisition]})

    subplots.assert_called_once_with(
        nrows=3,
        ncols=1,
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=(10, 5),
    )
    figure.suptitle.assert_called_once_with("Playback plots for readout on q0.readout")
    assert axes[0, 0].plot.call_count == 2
    assert axes[1, 0].plot.call_count == 2
    axes[2, 0].plot.assert_called_once_with(np.array([1]), label="I")
    axes[0, 0].title.set_text.assert_called_once_with("Scope acquisition")
    axes[1, 0].title.set_text.assert_called_once_with("Integrated acquisition")
    axes[2, 0].title.set_text.assert_called_once_with("Thresholded acquisition")
    tight_layout.assert_called_once_with()
    show.assert_called_once_with()
