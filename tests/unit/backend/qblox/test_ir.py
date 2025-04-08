from qat.purr.backends.qblox.ir import Opcode, Q1asmInstruction, SequenceBuilder


class TestQ1asmInstruction:
    def test_address_instruction(self):
        without_comment = Q1asmInstruction(Opcode.ADDRESS, "label")
        assert str(without_comment) == "label:"

        with_comment = Q1asmInstruction(
            Opcode.ADDRESS, "label", comment="Marked address for jumping here"
        )
        assert str(with_comment) == "label: # Marked address for jumping here"


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
