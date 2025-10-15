# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from functools import singledispatchmethod


class BuilderFactory:
    @singledispatchmethod
    @staticmethod
    def create_builder(model):
        raise TypeError(
            f"Cannot find a builder for hardware model with type {type(model)}."
        )
