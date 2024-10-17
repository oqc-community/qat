from typing import Optional

from qat.purr.utils.pydantic import WarnOnExtraFieldsModel
from qat.runtime.error_mitigation.readout_mitigation import ReadoutMitigation


class ErrorMitigation(WarnOnExtraFieldsModel):
    readout_mitigation: Optional[ReadoutMitigation] = None
