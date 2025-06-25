from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from qat.backend.qblox.config import SequencerConfig
from qat.backend.qblox.ir import Sequence
from qat.model.device import PulseChannel


@dataclass
class QbloxPackage:
    target: PulseChannel = None
    sequence: Sequence = None
    sequencer_config: SequencerConfig = field(default_factory=lambda: SequencerConfig())
    timeline: np.ndarray = None


class QbloxExecutable(BaseModel):
    pass
