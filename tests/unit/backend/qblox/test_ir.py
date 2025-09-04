import pytest

from qat.backend.qblox.ir import Opcode, Q1asmInstruction, SequenceBuilder


class TestQ1asmInstruction:
    @pytest.mark.parametrize(
        ("op_code", "operands", "comment"),
        [
            (Opcode.NOP, (), "Stalling"),
            (Opcode.STOP, (), "Stop the sequencer"),
            (Opcode.ADDRESS, ("label",), "Marked address as label"),
            (Opcode.ADD, ("R0", "R1", "R0"), "R0 <- R0 + R1"),
            (Opcode.JUMP, ("label",), "Unconditionally jump to address marked by label"),
            (
                Opcode.JUMP_LESS_THAN,
                (
                    "R0",
                    1000,
                    "label",
                ),
                "If R0 < 1000 then jump to address marked by label",
            ),
        ],
    )
    def test_asm_string(self, op_code, operands, comment):
        instruction = Q1asmInstruction(op_code, *operands, comment=comment)

        args = ",".join([str(op) for op in operands])
        if op_code == Opcode.ADDRESS:
            assert str(instruction) == f"{args}: # {comment}"
        elif args:
            assert str(instruction) == f"{op_code.value} {args} # {comment}"
        else:
            assert str(instruction) == f"{op_code.value} # {comment}"


def test_sequence_builder():
    builder = SequenceBuilder()

    builder.label("foo", "Section for function foo")
    sequence = builder.build()
    assert sequence.program
    assert not sequence.waveforms
    assert not sequence.weights
    assert not sequence.acquisitions

    instruction = Q1asmInstruction(
        Opcode.ADDRESS, "foo", comment="Section for function foo"
    )
    assert sequence.program == str(instruction)
    assert str(instruction) == "foo: # Section for function foo"
