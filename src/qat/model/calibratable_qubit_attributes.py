from enum import Enum

from qat.utils.pydantic import WarnOnExtraFieldsModel


class PulseShapeType(Enum):
    SQUARE = "square"
    GAUSSIAN = "gaussian"
    SOFT_SQUARE = "soft_square"
    BLACKMAN = "blackman"
    SETUP_HOLD = "setup_hold"
    SOFTER_SQUARE = "softer_square"
    EXTRA_SOFT_SQUARE = "extra_soft_square"
    SOFTER_GAUSSIAN = "softer_gaussian"
    ROUNDED_SQUARE = "rounded_square"
    GAUSSIAN_DRAG = "gaussian_drag"
    GAUSSIAN_ZERO_EDGE = "gaussian_zero_edge"
    GAUSSIAN_SQUARE = "gaussian_square"
    SECH = "sech"
    SIN = "sin"
    COS = "cos"


class CalibratablePulse(WarnOnExtraFieldsModel):
    shape: PulseShapeType
    width: float
    amp: float = 1.0
    phase: float = 0.0
    drag: float = 0.0
    rise: float = 0.0
    amp_setup: float = 0.0


class CalibratableAcquire(WarnOnExtraFieldsModel):
    delay: float = 0.0
    width: float = 0.0
    sync: bool = True
    weights: list[float]
    use_weights: bool = False
