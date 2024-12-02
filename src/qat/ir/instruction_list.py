from typing import List, Union

from pydantic import BaseModel, Field
from pydantic_core import from_json
from typing_extensions import Annotated

from qat.ir.instructions import (
    Instruction,
    InstructionBlock,
    MeasureBlock,
    find_all_instructions,
)
from qat.ir.waveforms import AbstractWaveform

InstBlockList = List[
    Annotated[
        Union[find_all_instructions([Instruction, InstructionBlock, AbstractWaveform])],
        Field(discriminator="inst"),
    ]
]


class InstructionList(BaseModel):
    instruction_list: InstBlockList = []

    @property
    def instructions(self):
        flat_list = []
        for inst in self.instruction_list:
            if isinstance(inst, MeasureBlock):
                flat_list.extend(inst.instructions)
            else:
                flat_list.append(inst)
        return flat_list

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
