from pydantic import BaseModel
from typing import Literal

class Instruction(BaseModel):
    pass

class Add(Instruction):
    inst : Literal['add'] = 'add'

class Subtract(Instruction):
    inst : Literal['subtract'] = 'subtract'