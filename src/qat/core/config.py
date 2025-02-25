# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from pydantic import BaseModel, ImportString


class PipelineImportDescription(BaseModel):
    name: str
    pipeline: ImportString
    default: bool = False
