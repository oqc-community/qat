# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ImportString


class HardwareTypeEnum(str, Enum):
    rtcs = "rtcs"
    echo = "echo"
    qiskit = "qiskit"


class HardwareDescription(BaseModel):
    qubit_count: Optional[int] = None  # This there should be a class for each type
    hardware_type: HardwareTypeEnum


class PipelineDescription(BaseModel):
    name: str
    compile: ImportString
    execute: ImportString
    postprocess: ImportString
    hardware: HardwareDescription
    default: bool = False
