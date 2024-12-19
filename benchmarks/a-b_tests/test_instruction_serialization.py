# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import pytest

from qat.ir.instruction_list import InstructionList
from qat.ir.serializers import LegacyIRSerializer
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.utils.ir_converter import IRConverter

from benchmarks.utils.helpers import load_experiments

experiments = load_experiments(
    return_circuit=False, mock_live_hardware=False, rtcs_hardware=False
)


@pytest.mark.benchmark(
    disable_gc=True, max_time=2, min_rounds=10, group="Instruction Serialization:"
)
@pytest.mark.parametrize("key", experiments.keys())
@pytest.mark.parametrize("mode", ["Legacy", "Pydantic", "Pydantic-converter"])
class TestInstructionSerialization:
    def test_serialization(self, benchmark, key, mode):
        """
        Tests three modes of instruction serialization:
          - Legacy: serialization of the legacy QuantumInstructionBuilder
          - Pydantic: serialization of the Pydantic InstructionList
          - Pydantic-converter: serialization of Pydantic InstructionList, combined with
            the IRConverter.
        """
        hw = experiments[key]["hardware"]
        builder = experiments[key]["builder"]
        inst_list = IRConverter(hw).legacy_to_pydantic_instructions(builder._instructions)

        # Wrapper functions for benchmarking
        if mode == "Legacy":
            run = lambda: builder.serialize()
        elif mode == "Pydantic":
            run = lambda: inst_list.serialize()
        else:
            run = lambda: LegacyIRSerializer(hw).serialize(builder)
        benchmark(run)
        assert True

    def test_deserialization(self, benchmark, key, mode):
        """
        Tests three modes of instruction deserialization:
          - Legacy: deserialization of the legacy QuantumInstructionBuilder
          - Pydantic: deserialization of the Pydantic InstructionList
          - Pydantic-converter: deserialization of Pydantic InstructionList, combined with
            the IRConverter.
        """
        hw = experiments[key]["hardware"]
        builder = experiments[key]["builder"]
        inst_list = IRConverter(hw).legacy_to_pydantic_instructions(builder._instructions)
        legacy_blob = builder.serialize()
        pydantic_blob = inst_list.serialize()

        # Wrapper functions for benchmarking
        if mode == "Legacy":
            run = lambda: QuantumInstructionBuilder.deserialize(legacy_blob)
        elif mode == "Pydantic":
            run = lambda: InstructionList.deserialize(pydantic_blob)
        else:
            run = lambda: LegacyIRSerializer(hw).deserialize(pydantic_blob)
        benchmark(run)
        assert True
