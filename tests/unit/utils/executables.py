# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.executables import AbstractProgram


class MockProgram(AbstractProgram):
    shapes: dict[str, tuple[int, ...]]

    @property
    def acquire_shapes(self) -> dict[str, tuple[int, ...]]:
        return self.shapes


class MockProgram2(AbstractProgram):
    shapes: dict[str, tuple[int, ...]]

    @property
    def acquire_shapes(self) -> dict[str, tuple[int, ...]]:
        return self.shapes
