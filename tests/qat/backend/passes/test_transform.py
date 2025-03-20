# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

import pytest

from qat.backend.passes.analysis import TriagePass, TriageResult
from qat.backend.passes.transform import (
    DesugaringPass,
    PydReturnSanitisation,
    ReturnSanitisation,
)
from qat.backend.passes.validation import (
    PydReturnSanitisationValidation,
    ReturnSanitisationValidation,
)
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.ir.instructions import Return as PydReturn
from qat.ir.measure import Acquire as PydAcquire
from qat.purr.backends.echo import get_default_echo_hardware
from qat.utils.hardware_model import generate_hw_model

from tests.qat.utils.builder_nuggets import resonator_spect


class TestTransformPasses:
    def test_return_sanitisation_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        with pytest.raises(ValueError):
            ReturnSanitisationValidation().run(builder, res_mgr)

        ReturnSanitisation().run(builder, res_mgr)
        ReturnSanitisationValidation().run(builder, res_mgr)

    def test_desugaring_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        TriagePass().run(builder, res_mgr)
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)

        assert len(triage_result.sweeps) == 1
        sweep = next(iter(triage_result.sweeps))
        assert len(sweep.variables) == 1

        DesugaringPass().run(builder, res_mgr)
        assert len(sweep.variables) == 2
        assert f"sweep_{hash(sweep)}" in sweep.variables


class TestPydReturnSanitisation:
    hw = generate_hw_model(8)

    def test_empty_builder(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        res_mgr = ResultManager()

        with pytest.raises(ValueError):
            PydReturnSanitisationValidation().run(builder, res_mgr)

        PydReturnSanitisation().run(builder, res_mgr)
        PydReturnSanitisationValidation().run(builder, res_mgr)

        return_instr: PydReturn = builder._ir.tail
        assert len(return_instr.variables) == 0

    def test_single_return(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.returns(variables=["test"])
        ref_nr_instructions = builder.number_of_instructions

        res_mgr = ResultManager()
        PydReturnSanitisationValidation().run(builder, res_mgr)
        PydReturnSanitisation().run(builder, res_mgr)

        assert builder.number_of_instructions == ref_nr_instructions
        assert builder.instructions[0].variables == ["test"]

    def test_multiple_returns_squashed(self):
        q0 = self.hw.qubit_with_index(0)
        q1 = self.hw.qubit_with_index(1)

        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.measure_single_shot_z(target=q0, output_variable="out_q0")
        builder.measure_single_shot_z(target=q1, output_variable="out_q1")

        output_vars = [
            instr.output_variable for instr in builder if isinstance(instr, PydAcquire)
        ]
        assert len(output_vars) == 2

        builder.returns(variables=[output_vars[0]])
        builder.returns(variables=[output_vars[1]])

        res_mgr = ResultManager()
        # Two returns in a single IR should raise an error.
        with pytest.raises(ValueError):
            PydReturnSanitisationValidation().run(builder, res_mgr)

        # Compress the two returns to a single return and validate.
        PydReturnSanitisation().run(builder, res_mgr)
        PydReturnSanitisationValidation().run(builder, res_mgr)

        return_instr = builder._ir.tail
        assert isinstance(return_instr, PydReturn)
        for var in return_instr.variables:
            assert var in output_vars
