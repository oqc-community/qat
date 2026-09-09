# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :class:`~qat.experimental.middleend.middleend.PulseLevelMiddleend`."""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr

from qat.experimental.conversion.pulse.lower_kernels_to_arrays import (
    LowerKernelsToResultsArrays,
)
from qat.experimental.dialect.pulse.ir import (
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    PhaseAttr,
    PhaseShiftOp,
    Pulse,
    TimeAttr,
    WaitOp,
)
from qat.experimental.dialect.pulse.ir.attributes import SampledWaveformAttr
from qat.experimental.dialect.pulse.ir.ops import PulseOp, SquareWaveformOp
from qat.experimental.dialect.pulse.transforms.constants import OrderedCanonicalizePass
from qat.experimental.middleend.middleend import PulseLevelMiddleend
from qat.experimental.passes.pass_ordering import OrderedPassPipeline
from qat.middleend.base import BaseMiddleend

from tests.unit.utils.ir import (
    build_module_from_ops,
    create_context,
    get_operations_with_type,
)

pytest_plugins = ("tests.unit.experimental.utils.canonical",)


class TestPulseLevelMiddleend:
    def test_is_a_base_middleend(self, canonical_model):
        """The middleend conforms to the ``BaseMiddleend`` interface."""
        middleend = PulseLevelMiddleend(model=canonical_model)

        assert isinstance(middleend, BaseMiddleend)
        assert middleend.model is canonical_model

    def test_builds_default_pulse_pipeline_from_canonical_model(self, canonical_model):
        """The constructor derives the default pulse pipeline from canonical data."""
        middleend = PulseLevelMiddleend(model=canonical_model)

        assert isinstance(middleend._pulse_pipeline, OrderedPassPipeline)
        pass_types = [type(p) for p in middleend._pulse_pipeline.passes]
        # The middleend runs the kernel-lowering pass before the final cleanup, ahead of
        # the (not-yet-implemented) Q1 lowering handled by the backend.
        assert LowerKernelsToResultsArrays in pass_types
        assert pass_types[-1] is OrderedCanonicalizePass

    def test_emit_applies_pipeline_and_returns_same_module(self, canonical_model, mocker):
        """``emit`` runs the pulse pipeline in place and returns the same module."""
        middleend = PulseLevelMiddleend(model=canonical_model)
        # ``OrderedPassPipeline`` is a frozen dataclass, so swap the whole attribute for a
        # stub rather than patching a method on the frozen instance.
        pipeline_stub = mocker.Mock()
        middleend._pulse_pipeline = pipeline_stub
        module = ModuleOp([])

        result = middleend.emit(module)

        assert result is module
        pipeline_stub.apply.assert_called_once()
        context_arg, ir_arg = pipeline_stub.apply.call_args.args
        assert isinstance(context_arg, Context)
        assert ir_arg is module

    def test_emit_creates_context_when_none_provided(self, canonical_model, mocker):
        """``emit`` constructs a fresh ``Context`` when one is not supplied."""
        middleend = PulseLevelMiddleend(model=canonical_model)
        pipeline_stub = mocker.Mock()
        middleend._pulse_pipeline = pipeline_stub

        middleend.emit(ModuleOp([]), context=None)

        context_arg, _ = pipeline_stub.apply.call_args.args
        assert isinstance(context_arg, Context)

    def test_emit_uses_supplied_context(self, canonical_model, mocker):
        """``emit`` forwards an explicitly provided ``Context`` to the pipeline."""
        middleend = PulseLevelMiddleend(model=canonical_model)
        pipeline_stub = mocker.Mock()
        middleend._pulse_pipeline = pipeline_stub
        context = Context()

        middleend.emit(ModuleOp([]), context=context)

        context_arg, _ = pipeline_stub.apply.call_args.args
        assert context_arg is context

    def test_emit_ignores_result_and_metrics_managers(self, canonical_model, mocker):
        """``emit`` accepts (and ignores) the standard middleend managers."""
        middleend = PulseLevelMiddleend(model=canonical_model)
        pipeline_stub = mocker.Mock()
        middleend._pulse_pipeline = pipeline_stub
        module = ModuleOp([])

        result = middleend.emit(
            module,
            res_mgr=None,
            met_mgr=None,
            compiler_config=None,
        )

        assert result is module
        pipeline_stub.apply.assert_called_once()


def _sampled_waveform_constants(module: ModuleOp) -> list[ConstantOp]:
    """Return every ``ConstantOp`` in *module* holding a sampled-waveform payload."""
    return [
        op
        for op in get_operations_with_type(module, ConstantOp)
        if isinstance(op.value, SampledWaveformAttr)
    ]


class TestPulseLevelMiddleendTransformsExampleIR:
    """End-to-end check that the middleend actually transforms an example pulse program."""

    def _build_example_pulse_ir(self, port: str) -> ModuleOp:
        """Build a small pulse program exercising several pipeline stages.

        The program contains a no-op (modulo-2pi zero) phase shift and a zero-duration wait
        that canonicalization should remove, plus an analytical square waveform that
        waveform-evaluation should replace with sampled constants.
        """
        freq = ConstantOp(FrequencyAttr(5e9))
        frame = CreateFrameOp(freq, StringAttr(port))
        # A modulo-2pi zero phase shift -> folded away by canonicalization.
        zero_phase = ConstantOp(PhaseAttr(0.0))
        shifted_frame = PhaseShiftOp(frame, zero_phase)
        # A zero-duration wait -> folded away by canonicalization.
        zero_duration = ConstantOp(TimeAttr(0.0))
        waited_frame = WaitOp(shifted_frame, zero_duration)
        # An analytical square waveform (80 ns) -> sampled by waveform evaluation. 80 ns is
        # a whole multiple of the 8 ns granularity and the 1 ns sample time of "port0".
        width = ConstantOp(TimeAttr(80e-9))
        amplitude = ConstantOp(AmplitudeAttr(0.5))
        waveform = SquareWaveformOp(width, amplitude)
        pulse = PulseOp(waited_frame, waveform)

        return build_module_from_ops(
            [
                freq,
                frame,
                zero_phase,
                shifted_frame,
                zero_duration,
                waited_frame,
                width,
                amplitude,
                waveform,
                pulse,
            ]
        )

    def test_emit_applies_transformations_to_example_ir(self, canonical_model):
        """Running the middleend over example pulse IR applies the pipeline transforms."""
        # ``canonical_model`` exposes a single port "port0" (1 ns sample time, 8 ns
        # granularity, no native waveform shapes), so square waveforms are sampled.
        module = self._build_example_pulse_ir(port="port0")

        # Pre-conditions: the analytical/no-op ops are present, nothing sampled yet.
        assert len(get_operations_with_type(module, PhaseShiftOp)) == 1
        assert len(get_operations_with_type(module, WaitOp)) == 1
        assert len(get_operations_with_type(module, SquareWaveformOp)) == 1
        assert _sampled_waveform_constants(module) == []

        middleend = PulseLevelMiddleend(model=canonical_model)
        result = middleend.emit(module, context=create_context(Pulse))

        # The IR is transformed in place and returned.
        assert result is module

        # No-op phase shift and zero wait are eliminated by canonicalization.
        assert get_operations_with_type(module, PhaseShiftOp) == []
        assert get_operations_with_type(module, WaitOp) == []

        # The analytical square waveform is replaced by a sampled-waveform constant.
        assert get_operations_with_type(module, SquareWaveformOp) == []
        sampled_constants = _sampled_waveform_constants(module)
        assert len(sampled_constants) == 1

        # The surviving pulse now reads the sampled waveform constant.
        pulse_ops = get_operations_with_type(module, PulseOp)
        assert len(pulse_ops) == 1
        assert pulse_ops[0].waveform is sampled_constants[0].result

    def test_emit_uses_default_context_when_none_supplied(self, canonical_model):
        """The middleend can transform example IR without an explicit context."""
        module = self._build_example_pulse_ir(port="port0")

        middleend = PulseLevelMiddleend(model=canonical_model)
        result = middleend.emit(module)

        assert result is module
        assert get_operations_with_type(module, PhaseShiftOp) == []
        assert get_operations_with_type(module, WaitOp) == []
        assert get_operations_with_type(module, SquareWaveformOp) == []
        assert len(_sampled_waveform_constants(module)) == 1
