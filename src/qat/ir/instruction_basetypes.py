from __future__ import annotations

from enum import Enum


class PostProcessType(Enum):
    """
    `PostProcessType` is used in a `PostProcessing` instruction to determine how readouts
    should be processed. It describes a type of classical post-processing:

    - `DOWN_CONVERT`: Down-converts the readout measurement, should only be used along the  axis `ProcessAxis.TIME`.
    - `MEAN`: Takes an average over the given axis.
    - `LINEAR_MAP_COMPLEX_TO_REAL`: Maps the (complex) measured value onto a (real) z-projection using a supplied
       linear mapping.
    - `DISCRIMINATE`: Converts a z-projection to a classical bit by comparison with a supplied discrimination threshold.
    """

    DOWN_CONVERT = "down_convert"
    MEAN = "mean"
    LINEAR_MAP_COMPLEX_TO_REAL = "linear_map_real"
    DISCRIMINATE = "discriminate"
    MUL = "mul"

    def __repr__(self):
        return self.name


class ProcessAxis(Enum):
    """
    `ProcessAxis` is used during classical post-processing of readouts. It specifies the axis
    which the post-processing should occur on. Often used in conjunction with the
    `AcquireMode` to determine the correct method for post-processing.

    - `TIME`: Instructs the post-processing to be performed over the time-series data returned from a readout.
    - `SEQUENCE`: Instructs the post-processing to be performed over the shots.
    """

    TIME = "time"
    SEQUENCE = "sequence"

    def __repr__(self):
        return self.name


class AcquireMode(Enum):
    """
    The `AcquireMode` is used to specify the type of acquisition at the level of the control
    hardware.

    Note that different backends will only allow selected acquisition modes.

    - `RAW`: Returns the time-series acquisition data for each shot
    - `SCOPE`: Returns the time-series acquisition data where each point is averaged over the number of shots
    - `INTEGRATOR`: Down-conversion and integration of the acquisition data has occurred on the hardware so that a
       single data point is returned for each shot
    """

    RAW = "raw"
    SCOPE = "scope"
    INTEGRATOR = "integrator"

    def __repr__(self):
        return self.name
