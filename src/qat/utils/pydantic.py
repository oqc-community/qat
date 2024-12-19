# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from pydantic import BaseModel, ConfigDict

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class NoExtraFieldsModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, extra="forbid"
    )
