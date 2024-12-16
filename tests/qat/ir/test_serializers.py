import pytest
from compiler_config.config import CompilerConfig, Tket, TketOptimizations

from qat.core import QAT
from qat.ir.serializers import LegacyIRSerializer
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import Instruction, InstructionBlock, Variable

from tests.qat.test_core import (
    qasm2_files,
    qasm3_files,
    qir_files,
    short_file_name,
    skip_exec,
)

skip_serialization = [
    *skip_exec,
    # skip because they use undefined pulse channels
    "qasm3-arb_waveform.qasm",
    "qasm3-redefine_defcal.qasm",
    # skip because of bug in legacy that gives "phase" as a bool
    "qasm3-complex_gates_test.qasm",
]

serialization_files = [
    p
    for p in [*qasm2_files, *qasm3_files, *qir_files]
    if short_file_name(p) not in skip_serialization
]


def equivalent_array(self, other):
    if isinstance(self, type(other)):
        vars_self = vars(self)
        vars_other = vars(other)
        if len(vars_self) != len(vars_other):
            return False
        for key in vars_self.keys():
            if not key in vars_other:
                return False
            check = vars_self[key] == vars_other[key]
            if not isinstance(check, bool):
                check = all(check)
            if not check:
                return False
        return True
    return False


@pytest.mark.parametrize(
    "compiler_config",
    [
        pytest.param(None, id="No_config"),
        pytest.param(CompilerConfig(), id="Default_config"),
        pytest.param(
            CompilerConfig(optimizations=Tket(TketOptimizations.Two)),
            id="TketOptimizations.Two",
        ),
    ],
)
@pytest.mark.parametrize(
    "hardware_model", [pytest.param(get_default_echo_hardware(32), id="Echo32Q")]
)
class TestLegacySerialization:

    @pytest.mark.parametrize("source_file", serialization_files, ids=short_file_name)
    def test_instructions_equal_after_serialize_deserialize_roundtrip(
        self, source_file, hardware_model, compiler_config, monkeypatch
    ):
        # Equal for instructions are defined at an instance level: we monkey patch to compare
        # the attributes of instructions.
        monkeypatch.setattr(Instruction, "__eq__", equivalent_array)
        monkeypatch.setattr(InstructionBlock, "__eq__", equivalent_array)
        monkeypatch.setattr(Variable, "__eq__", equivalent_array)
        qat = QAT(hardware_model)
        qat_ib, _ = qat.compile(str(source_file), compiler_config=compiler_config)
        serializer = LegacyIRSerializer(hardware_model)
        blob = serializer.serialize(qat_ib)
        new_qat_ib = serializer.deserialize(blob)
        assert qat_ib._instructions == new_qat_ib._instructions
