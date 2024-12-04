from typing import List, Union

from pydantic import BaseModel, Field
from pydantic_core import from_json
from typing_extensions import Annotated

from qat.ir.instructions import Instruction
from qat.ir.measure import Acquire, MeasureBlock, PostProcessing
from qat.ir.waveforms import Pulse


def find_all_instructions(start_types=[Instruction]):
    """
    Creates a tuple of all possible instructions, used in seralization.
    """
    instructions = set()
    check = start_types
    while check:
        parent = check.pop()
        for child in parent.__subclasses__():
            if child not in instructions:
                check.append(child)
                instructions.add(child)

    return tuple(instructions)


# An annotated type used in instruction lists so python can correctly
# identify instruction types during deserialization.
all_instructions = find_all_instructions(
    [Instruction, MeasureBlock, Acquire, PostProcessing, Pulse]
)
InstList = List[
    Annotated[
        Union[all_instructions],
        Field(discriminator="inst"),
    ]
]


class InstructionList(BaseModel):
    """
    A Pydantic wrapper around a list of instructions that can be used
    to (de)serialize.
    """

    instruction_list: InstList = []

    @property
    def instructions(self):
        return self.instruction_list

    @instructions.setter
    def instructions(self, value):
        self.instruction_list = value

    def add(
        self,
        instructions: Union[
            "InstructionList", Instruction, List[Union["InstructionList", Instruction]]
        ],
    ):
        """
        Adds an instruction(s) to the instruction list.
        """
        if isinstance(instructions, InstructionList):
            instructions = instructions.instruction_list
        if not isinstance(instructions, List):
            instructions = [instructions]
        self.instruction_list.extend(instructions)
        return self

    def insert(
        self,
        instructions: Union["InstructionList", Instruction, List[Instruction]],
        index: int,
    ):
        """
        Inserts one or more instruction(s) into this list, starting at index.
        """
        if instructions is None:
            return self

        if isinstance(instructions, InstructionList):
            instructions = instructions.instructions

        if not isinstance(instructions, List):
            instructions = [instructions]

        for instruction in instructions:
            self.instruction_list.insert(index, instruction)
            index += 1
        return self

    def serialize(self):
        """
        Serialize the instruction list to a JSON blob.
        """
        return self.model_dump_json(indent=4, exclude_none=True)

    @staticmethod
    def deserialize(blob):
        """
        Takes a JSON blob and returns an instruction list.
        """
        blob = from_json(blob)
        return InstructionList(**blob)

    def __getitem__(self, index):
        return self.instruction_list[index]
