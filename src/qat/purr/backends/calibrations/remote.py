# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from typing import Dict, Union

from compiler_config.config import CalibrationArguments

from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.runtime import QuantumRuntime, RemoteCalibration, get_runtime

RuntimeOrModel = Union[QuantumHardwareModel, QuantumRuntime]


def find_calibration(args: "CalibrationArguments"):
    calibration = None
    for cali in _standard_calibrations.values():
        if cali.arguments_type() == args.__class__:
            calibration = cali
            break

    if calibration is None:
        raise ValueError(f"Calibration with an argument of {str(args)} couldn't be found.")

    return calibration


def run_calibration(runtime: RuntimeOrModel, args: "CalibrationArguments"):
    calibration = find_calibration(args)

    # If we're just a basic model, use auto-detection.
    if isinstance(runtime, QuantumHardwareModel):
        runtime = get_runtime(runtime)

    calibration.run(runtime.model, runtime, args)


_standard_calibrations: Dict[str, "BuiltinRemoteCalibration"] = dict()


class CustomCalibration(RemoteCalibration):
    """Bespoke calibration built by external users."""

    pass


class BuiltinRemoteCalibration(RemoteCalibration):
    """
    Identifier for any built-in calibrations. Enables automatic running of calibrations
    remotely for any class that inherits this.
    """

    def __init_subclass__(cls, **kwargs):
        _standard_calibrations[cls.__name__] = cls()
