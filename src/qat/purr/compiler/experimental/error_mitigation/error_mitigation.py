from typing import Optional

from pydantic import BaseModel

from qat.purr.compiler.experimental.error_mitigation.readout_mitigation import (
    ReadoutMitigation,
)


class ErrorMitigation(BaseModel, validate_assignment=True):
    readout_mitigation: Optional[ReadoutMitigation] = None