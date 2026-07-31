# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Block, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import (
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    FrequencyType,
    PhaseAttr,
    PhaseSetOp,
    PhaseShiftOp,
    PhaseType,
    TimeAttr,
    TimeType,
    WaitOp,
)
from qat.experimental.dialect.pulse.utils import (
    extract_frequency_hz,
    extract_phase_radians,
    extract_time_seconds,
    pulse_entry_block,
    require_constant_operand,
)


@irdl_op_definition
class _DynamicPhaseSourceOp(IRDLOperation):
    name = "test.dynamic_phase_source"
    result = result_def(PhaseType)

    def __init__(self):
        super().__init__(result_types=[PhaseType()])


def _frame(channel_id: str = "q0/drive") -> tuple[ConstantOp, CreateFrameOp]:
    freq = ConstantOp(FrequencyAttr(4.8e9))
    return freq, CreateFrameOp(freq, StringAttr(channel_id))


def test_pulse_entry_block_returns_top_level_block_for_flat_module():
    freq, frame = _frame()
    module = ModuleOp([freq, frame])

    assert pulse_entry_block(module) is module.body.block


def test_pulse_entry_block_returns_function_body_for_single_function_module():
    freq, frame = _frame()
    body = Block([freq, frame, func.ReturnOp()])
    fn = func.FuncOp("main", ((), ()), Region(body))
    module = ModuleOp([fn])

    assert pulse_entry_block(module) is body


def test_pulse_entry_block_rejects_mixed_module_shape():
    freq, frame = _frame()
    body = Block([freq, frame, func.ReturnOp()])
    fn = func.FuncOp("main", ((), ()), Region(body))
    module = ModuleOp([fn, ConstantOp(TimeAttr(8e-9))])

    with pytest.raises(PassFailedException, match="either a flat module or a module"):
        pulse_entry_block(module)


def test_pulse_entry_block_rejects_multiple_entry_functions():
    fn_a = func.FuncOp("main_a", ((), ()), Region(Block([func.ReturnOp()])))
    fn_b = func.FuncOp("main_b", ((), ()), Region(Block([func.ReturnOp()])))
    module = ModuleOp([fn_a, fn_b])

    with pytest.raises(PassFailedException, match="single entry function"):
        pulse_entry_block(module)


def test_require_constant_operand_returns_owner_for_constant_operand():
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(0.125))
    phase_set = PhaseSetOp(frame, phase)

    assert require_constant_operand(phase_set.name, "phase", phase_set.phase) is phase


def test_require_constant_operand_rejects_dynamic_operand():
    freq, frame = _frame()
    dynamic_phase = _DynamicPhaseSourceOp()
    phase_set = PhaseSetOp(frame, dynamic_phase)

    with pytest.raises(PassFailedException, match="requires constant phase"):
        require_constant_operand(phase_set.name, "phase", phase_set.phase)


def test_extract_time_seconds_returns_literal_seconds():
    freq, frame = _frame()
    duration = ConstantOp(TimeAttr(16e-9))
    wait = WaitOp(frame, duration)

    assert extract_time_seconds(wait) == pytest.approx(16e-9)


def test_extract_time_seconds_rejects_non_time_constant():
    freq, frame = _frame()
    malformed_duration = ConstantOp(PhaseAttr(0.5), result_type=TimeType())
    wait = WaitOp(frame, malformed_duration)

    with pytest.raises(PassFailedException, match="expects pulse.constant time operand"):
        extract_time_seconds(wait)


def test_extract_phase_radians_returns_literal_radians():
    freq, frame = _frame()
    phase = ConstantOp(PhaseAttr(0.75))
    phase_shift = PhaseShiftOp(frame, phase)

    assert extract_phase_radians(phase_shift) == pytest.approx(0.75)


def test_extract_phase_radians_rejects_non_phase_constant():
    freq, frame = _frame()
    malformed_phase = ConstantOp(TimeAttr(4e-9), result_type=PhaseType())
    phase_set = PhaseSetOp(frame, malformed_phase)

    with pytest.raises(PassFailedException, match="expects pulse.constant phase operand"):
        extract_phase_radians(phase_set)


def test_extract_frequency_hz_returns_literal_hertz():
    freq, frame = _frame()

    assert extract_frequency_hz(frame) == pytest.approx(4.8e9)


def test_extract_frequency_hz_rejects_non_frequency_constant():
    malformed_frequency = ConstantOp(PhaseAttr(0.25), result_type=FrequencyType())
    frame = CreateFrameOp(malformed_frequency, StringAttr("q0/drive"))

    with pytest.raises(
        PassFailedException, match="expects pulse.constant frequency operand"
    ):
        extract_frequency_hz(frame)
