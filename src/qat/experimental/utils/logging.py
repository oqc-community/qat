# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from qat.purr.utils.logger import get_default_logger


def get_logger(name: str | None = None):
    """Get a logger for the PuRR materialisation domain."""
    return get_default_logger()
