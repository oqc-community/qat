# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


class MismatchingHardwareModelException(Exception):
    """Raised when the hardware model does not match in the compile and execute modes."""

    ...
