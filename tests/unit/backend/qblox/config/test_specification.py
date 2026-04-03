# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

import dataclasses
import inspect

import qat.backend.qblox.config.specification as specification


def test_config_classes_are_dataclasses():
    config_classes = [
        obj
        for name, obj in inspect.getmembers(specification, inspect.isclass)
        if obj.__module__ == specification.__name__
    ]

    for clazz in config_classes:
        assert dataclasses.is_dataclass(clazz), (
            f"Class {clazz.__name__} must be a dataclass!"
        )
