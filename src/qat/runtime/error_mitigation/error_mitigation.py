from typing import Optional

from qat.runtime.error_mitigation.readout_mitigation import ReadoutMitigation
from qat.utils.pydantic import WarnOnExtraFieldsModel


class ErrorMitigation(WarnOnExtraFieldsModel):
    readout_mitigation: Optional[ReadoutMitigation] = None
