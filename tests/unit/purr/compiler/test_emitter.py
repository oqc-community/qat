# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.instructions import Repeat
from qat.purr.compiler.runtime import get_builder

models = [get_default_echo_hardware(), get_default_RTCS_hardware()]


@pytest.mark.parametrize("model", models)
def test_emitter_adds_repeat(model):
    builder = get_builder(model)
    assert (
        next(iter(inst for inst in builder.instructions if isinstance(inst, Repeat)), None)
        is None
    )
    qat_file = InstructionEmitter().emit(builder.instructions, model)
    repeat_inst = qat_file.repeat
    assert repeat_inst is not None
    assert repeat_inst.repeat_count == model.default_repeat_count
    assert repeat_inst.repetition_period == model.default_repetition_period


@pytest.mark.parametrize("model", models)
def test_qat_file_corrects_repeats_count_and_period(model):
    builder = get_builder(model).repeat(repeat_value=None, repetition_period=None)
    repeat_inst = next(
        iter(inst for inst in builder.instructions if isinstance(inst, Repeat)), None
    )
    assert repeat_inst is not None
    assert repeat_inst.repeat_count is None
    assert repeat_inst.repetition_period is None
    qat_file = InstructionEmitter().emit(builder.instructions, model)
    repeat_inst = qat_file.repeat
    assert repeat_inst is not None
    assert repeat_inst.repeat_count == model.default_repeat_count
    assert repeat_inst.repetition_period == model.default_repetition_period


@pytest.mark.parametrize("model", models)
def test_use_qat_file_repeats_count_and_period(model):
    repeat_count = int(model.default_repeat_count * 1.2)
    repetition_period = model.default_repetition_period * 1.2
    builder = get_builder(model).repeat(
        repeat_value=repeat_count, repetition_period=repetition_period
    )
    repeat_inst = next(
        iter(inst for inst in builder.instructions if isinstance(inst, Repeat)), None
    )
    assert repeat_inst is not None
    assert repeat_inst.repeat_count == repeat_count
    assert repeat_inst.repetition_period == repetition_period
    qat_file = InstructionEmitter().emit(builder.instructions, model)
    repeat_inst = qat_file.repeat
    assert repeat_inst is not None
    assert repeat_inst.repeat_count == repeat_count
    assert repeat_inst.repetition_period == repetition_period
