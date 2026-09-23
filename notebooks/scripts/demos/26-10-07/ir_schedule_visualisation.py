# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Pulse and Q1 Schedule Visualisation
#
# This notebook demonstrates the experimental schedule builders and the generic
# visualisation layer. Pulse schedules use seconds, while Q1 schedules use nanoseconds.

# %% [markdown]
# ## 1) Build and visualise a Pulse schedule
#
# The Pulse module contains two hardware-independent frames. Each operation consumes one
# frame SSA value and produces its successor while preserving the initial frame frequency.
#
# > **TODO:** [COMPILER-1472](https://oxfordquantumcircuits.atlassian.net/browse/COMPILER-1472)
# > tracks tighter operand typing and more explicit SSA construction for Pulse IR.

# %%
import numpy as np
from matplotlib import pyplot as plt
from xdsl.dialects import func
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region

from qat.experimental.dialect.pulse.ir.attributes import (
    FrequencyAttr,
    PhaseAttr,
    SampledWaveformAttr,
    TimeAttr,
)
from qat.experimental.dialect.pulse.ir.ops import (
    AcquireOp,
    ConstantOp,
    CreateFrameOp,
    PhaseSetOp,
    PulseOp,
    WaitOp,
)
from qat.experimental.dialect.pulse.ir.schedule import build_pulse_schedule
from qat.experimental.tools.schedule import visualise_schedule

frequency_q0 = ConstantOp(FrequencyAttr(5e9))
frequency_q1 = ConstantOp(FrequencyAttr(5.1e9))
frame_q0 = CreateFrameOp(frequency_q0, StringAttr("q0.drive"))
frame_q1 = CreateFrameOp(frequency_q1, StringAttr("q1.drive"))
phase = ConstantOp(PhaseAttr(np.pi / 2))
duration = ConstantOp(TimeAttr(4e-9))
waveform = ConstantOp(
    SampledWaveformAttr(
        [0.25 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.25 + 0.0j],
        width=TimeAttr(4e-9),
        sample_time=TimeAttr(1e-9),
    )
)
phase_set = PhaseSetOp(frame_q0, phase)
wait = WaitOp(phase_set, duration)
pulse = PulseOp(wait, waveform)
wait_q1 = WaitOp(frame_q1, duration)
acquire = AcquireOp(pulse, duration, label="readout")

pulse_module = ModuleOp(
    [
        func.FuncOp(
            "main",
            ((), ()),
            Region(
                Block(
                    [
                        frequency_q0,
                        frequency_q1,
                        frame_q0,
                        frame_q1,
                        phase,
                        duration,
                        waveform,
                        phase_set,
                        wait,
                        pulse,
                        wait_q1,
                        acquire,
                        func.ReturnOp(),
                    ]
                )
            ),
        )
    ]
)

# %%
pulse_schedule = build_pulse_schedule(pulse_module)
pulse_figure, pulse_axes = visualise_schedule(pulse_schedule)
pulse_figure

# %%
assert pulse_schedule.resources == ("q0.drive", "q1.drive")
assert [event.label for event in pulse_schedule.timelines["q0.drive"]] == [
    "wait",
    "pulse",
    "readout",
]
assert [event.label for event in pulse_schedule.timelines["q1.drive"]] == ["wait"]
assert len(pulse_axes) == 4
assert all(axis.get_xlabel() == "Time (s)" for axis in pulse_axes)
assert pulse_axes[0].get_ylabel() == "DAC/ADC range"
assert pulse_axes[0].get_ylim() == (-1.0, 1.0)
assert pulse_axes[1].get_ylabel() == "Phase (rad)"

# %% [markdown]
# ## 2) Build and visualise a Q1 schedule
#
# Q1 instructions use nanoseconds. The builder materialises one `sequence` resource
# and records runtime instructions in module order. Parameter instructions update
# latched state and become active only when `play`, `acquire`, or `upd_param` executes.
# The two real waveform-memory paths are sampled on the target's 1 ns grid, independently
# scaled and offset, combined as a complex envelope, and NCO-modulated with Qblox's
# documented `1 / sqrt(2)` normalisation.
#
# The plotted amplitude is the normalised digital DAC full-scale value. It is not a
# connector voltage. Analogue gain, attenuation, mixer conversion, and loading determine
# the physical output. `acquire` advances the runtime schedule but does not synthesise ADC
# samples, since this notebook has no simulated device or acquisition response.

# %%
from qat.experimental.dialect.q1.ir.imm_desc import (
    DurationImm,
    NcoPhaseImm,
    SI16Imm,
    SI32Imm,
    UI5Imm,
    UI10Imm,
    UI24Imm,
)
from qat.experimental.dialect.q1.ir.ops import (
    AcquireImmImmImmOp,
    NopOp,
    PlayImmImmImmOp,
    ResetPhOp,
    SetAwgGainImmImmOp,
    SetAwgOffsImmImmOp,
    SetFreqImmOp,
    SetPhDeltaImmOp,
    SetPhImmOp,
    StopOp,
    UpdParamImmOp,
    WaitImmOp,
    WaitSyncImmOp,
)
from qat.experimental.dialect.q1.ir.schedule import build_q1_schedule
from qat.experimental.dialect.q1_sequence.ir.attrs import make_waveform
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp

q1_waveform_length = 40
q1_waveforms = {
    "sinusoid": {
        "data": np.sin(
            2 * np.pi * np.arange(q1_waveform_length) / q1_waveform_length
        ).tolist(),
        "index": 0,
    },
    "block": {
        "data": np.ones(q1_waveform_length).tolist(),
        "index": 1,
    },
    "gaussian": {
        "data": np.exp(
            -0.5
            * (
                (np.arange(q1_waveform_length) - q1_waveform_length / 2)
                / (0.12 * q1_waveform_length)
            )
            ** 2
        ).tolist(),
        "index": 2,
    },
    "zero": {
        "data": np.zeros(q1_waveform_length).tolist(),
        "index": 3,
    },
}
q1_waveform_table = ArrayAttr(
    [
        make_waveform(name, waveform["index"], waveform["data"])
        for name, waveform in q1_waveforms.items()
    ]
)

# %%
q1_waveform_figure, q1_waveform_axis = plt.subplots(figsize=(10, 3))
q1_waveform_time = np.arange(q1_waveform_length)
for name, waveform in q1_waveforms.items():
    if name != "zero":
        q1_waveform_axis.plot(q1_waveform_time, waveform["data"], label=name)
q1_waveform_axis.set_xlabel("Time (ns)")
q1_waveform_axis.set_ylabel("Waveform amplitude")
q1_waveform_axis.legend()
q1_waveform_axis.grid(alpha=0.1)
q1_waveform_figure.tight_layout()
plt.close(q1_waveform_figure)
q1_waveform_figure

q1_module = ModuleOp(
    [
        SequenceOp(
            "sequence",
            [
                SetFreqImmOp(SI32Imm(1_000_000_000)),
                SetPhImmOp(NcoPhaseImm(125_000_000)),
                SetAwgGainImmImmOp(SI16Imm(32767), SI16Imm(32767)),
                SetAwgOffsImmImmOp(SI16Imm(2048), SI16Imm(-1024)),
                UpdParamImmOp(DurationImm(4)),
                NopOp(),
                PlayImmImmImmOp(UI10Imm(0), UI10Imm(0), DurationImm(4)),
                WaitImmOp(DurationImm(36)),
                WaitSyncImmOp(DurationImm(4)),
                SetPhDeltaImmOp(NcoPhaseImm(250_000_000)),
                UpdParamImmOp(DurationImm(4)),
                PlayImmImmImmOp(UI10Imm(1), UI10Imm(1), DurationImm(40)),
                SetAwgGainImmImmOp(SI16Imm(16384), SI16Imm(16384)),
                SetAwgOffsImmImmOp(SI16Imm(0), SI16Imm(0)),
                UpdParamImmOp(DurationImm(4)),
                PlayImmImmImmOp(UI10Imm(2), UI10Imm(2), DurationImm(4)),
                AcquireImmImmImmOp(UI5Imm(0), UI24Imm(0), DurationImm(36)),
                ResetPhOp(),
                UpdParamImmOp(DurationImm(4)),
                WaitImmOp(DurationImm(8)),
                StopOp(),
            ],
            waveforms=q1_waveform_table,
        )
    ]
)

# %%
q1_schedule = build_q1_schedule(q1_module)
q1_signal_interpolation = "linear"  # Or "zero_order_hold" for native 1 ns steps.
q1_figure, q1_axes = visualise_schedule(
    q1_schedule,
    signal_interpolation=q1_signal_interpolation,
)
q1_figure

# %%
assert q1_schedule.resources == ("sequence",)
assert [event.label for event in q1_schedule.records] == [
    "upd_param",
    "nop",
    "play",
    "wait",
    "wait_sync",
    "upd_param",
    "play",
    "upd_param",
    "play",
    "acquire",
    "upd_param",
    "wait",
]
assert q1_schedule.timelines["sequence"][-1].end == 152
assert q1_axes[0].get_xlabel() == "Time (ns)"
assert q1_axes[0].get_ylabel() == "DAC/ADC range"
assert q1_axes[1].get_ylabel() == "NCO phase (steps)"

# %% [markdown]
# ## 3) Qblox-inspired Q1 experiment patterns
#
# These small schedules mirror common Qblox scheduler examples. State-setting
# operations do not consume time. Their effects become visible when the next
# latch-updating runtime instruction starts.

# %% [markdown]
# ### NCO frequency sweep and acquisition
#
# A spectroscopy-style sequence changes the NCO frequency, plays a waveform, and
# integrates an acquisition. The two frequency points are kept in separate
# sequences so their effects can be compared directly.

# %%
q1_spectroscopy_modules = [
    ModuleOp(
        [
            SequenceOp(
                "freq_sweep_readout",
                [
                    SetFreqImmOp(SI32Imm(80_000_000)),
                    UpdParamImmOp(DurationImm(4)),
                    SetAwgGainImmImmOp(SI16Imm(32767), SI16Imm(32767)),
                    PlayImmImmImmOp(UI10Imm(0), UI10Imm(0), DurationImm(32)),
                    AcquireImmImmImmOp(UI5Imm(0), UI24Imm(0), DurationImm(64)),
                    StopOp(),
                ],
                waveforms=q1_waveform_table,
            )
        ]
    ),
    ModuleOp(
        [
            SequenceOp(
                "freq_sweep_control",
                [
                    SetFreqImmOp(SI32Imm(160_000_000)),
                    UpdParamImmOp(DurationImm(4)),
                    SetAwgGainImmImmOp(SI16Imm(32767), SI16Imm(32767)),
                    PlayImmImmImmOp(UI10Imm(1), UI10Imm(1), DurationImm(32)),
                    AcquireImmImmImmOp(UI5Imm(0), UI24Imm(1), DurationImm(64)),
                    StopOp(),
                ],
                waveforms=q1_waveform_table,
            )
        ]
    ),
]
q1_spectroscopy_schedules = [
    build_q1_schedule(module) for module in q1_spectroscopy_modules
]
q1_spectroscopy_figures = [
    visualise_schedule(schedule)[0] for schedule in q1_spectroscopy_schedules
]
q1_spectroscopy_figures[0]

# %%
q1_spectroscopy_figures[1]

# %% [markdown]
# ### Ramsey-style phase evolution
#
# A Ramsey sequence uses an absolute phase, a free-evolution wait, a phase kick,
# and a second pulse before acquisition. The event metadata exposes the phase
# used by each timed instruction.

# %%
q1_ramsey_module = ModuleOp(
    [
        SequenceOp(
            "ramsey",
            [
                SetFreqImmOp(SI32Imm(750_000_000)),
                UpdParamImmOp(DurationImm(4)),
                SetPhImmOp(NcoPhaseImm(0)),
                UpdParamImmOp(DurationImm(4)),
                PlayImmImmImmOp(UI10Imm(2), UI10Imm(2), DurationImm(16)),
                WaitImmOp(DurationImm(80)),
                SetPhDeltaImmOp(NcoPhaseImm(250_000_000)),
                UpdParamImmOp(DurationImm(4)),
                PlayImmImmImmOp(UI10Imm(3), UI10Imm(3), DurationImm(16)),
                AcquireImmImmImmOp(UI5Imm(0), UI24Imm(2), DurationImm(32)),
                StopOp(),
            ],
            waveforms=q1_waveform_table,
        )
    ]
)
q1_ramsey_schedule = build_q1_schedule(q1_ramsey_module)
q1_ramsey_figure, q1_ramsey_axes = visualise_schedule(q1_ramsey_schedule)
q1_ramsey_figure

# %%
assert [event.label for event in q1_ramsey_schedule.records] == [
    "upd_param",
    "upd_param",
    "play",
    "wait",
    "upd_param",
    "play",
    "acquire",
]
np.testing.assert_allclose(
    [event.phase for event in q1_ramsey_schedule.records],
    [0.0, 1.5 * np.pi, np.pi, np.pi, 1.5 * np.pi, np.pi, np.pi],
    atol=1e-12,
)
assert q1_ramsey_axes[0].get_xlabel() == "Time (ns)"
