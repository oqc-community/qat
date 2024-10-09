from typing import Optional

from pydantic import ConfigDict

from qat.purr.compiler.experimental.error_mitigation.readout_mitigation import (
    ReadoutMitigation,
)
from qat.purr.utils.pydantic import WarnOnExtraFieldsModel


class ErrorMitigation(WarnOnExtraFieldsModel):
    model_config = ConfigDict(validate_assignment=True)

    readout_mitigation: Optional[ReadoutMitigation] = None
