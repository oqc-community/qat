# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import warnings


class MismatchingHardwareModelError(Exception):
    """Raised when the hardware model does not match in the compile and execute modes."""


def __getattr__(name):
    if name == "MismatchingHardwareModelException":
        warnings.warn(
            "MismatchingHardwareModelException is deprecated; use MismatchingHardwareModelError",
            DeprecationWarning,
            stacklevel=2,
        )
        return MismatchingHardwareModelError
    raise AttributeError(name)
