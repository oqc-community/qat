from qat.ir.instruction_list import InstructionList
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.utils.ir_converter import IRConverter


class LegacyIRSerializer:

    def __init__(self, hardware_model: QuantumHardwareModel):
        """
        Serializes and deserializers legacy QatIR by converting between legacy and
        Pydantic QatIR, and uses Pydantic's built-in serialization for high performance.

        The legacy way of serializing IR was to convert a QuantumInstructionBuilder into
        a JSON blob that had details of the instructions, the hardware model, and any
        builder attributes. Because of this, the input for serialization is a builder,
        and the output of deserialization is also a builder. However, the builders will
        not be indentical as any additional attributes will be lost (e.g. entanglement
        map).
        """

        self.model = hardware_model
        self.converter = IRConverter(hardware_model)

    def serialize(self, builder: QuantumInstructionBuilder) -> str:
        """
        Serializes legacy QatIR to a JSON blob by converting the instructions to
        Pydantic, and using its built-in serialization.
        """
        pydantic_instructions = self.converter.legacy_to_pydantic_instructions(
            builder.instructions
        )
        return pydantic_instructions.serialize()

    def deserialize(self, blob: str) -> QuantumInstructionBuilder:
        """
        Deserialize a JSON blob of instructions to QatIR.

        The blob must have been generated using a Pydantic InstructionList. However,
        the deserialized instructions are return as the legacy instructions inside a
        QuantumInstructionBuilder. This process makes use of the hardware model to
        repopulate the instructions with the necessary PulseChannels.
        """
        pydantic_instructions = InstructionList.deserialize(blob)
        instructions = self.converter.pydantic_to_legacy_instructions(
            pydantic_instructions,
        )
        return QuantumInstructionBuilder(self.model, instructions)
