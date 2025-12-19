# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import os

from piny import StrictMatcher


class VeryStrictMatcher(StrictMatcher):
    """
    Expand an environment variable of form ${VAR} with its value

    If value is not found, raises a ValueError
    """

    @staticmethod
    def constructor(loader, node):
        match = StrictMatcher.matcher.match(node.value)
        key = match.groups()[0]
        if key not in os.environ:
            raise ValueError(f"Environment Variable {key} not found")
        return os.environ[key]  # type: ignore
