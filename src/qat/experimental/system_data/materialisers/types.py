# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Shared types for the experimental materialisation boundary."""

from enum import Enum


class SourceType(str, Enum):
    """Supported external source types."""

    PURR = "purr"
