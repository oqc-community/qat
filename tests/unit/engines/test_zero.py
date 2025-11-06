# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.engines.zero import ZeroEngine
from qat.executables import AbstractProgram


class MockProgram(AbstractProgram):
    acquires: dict[str, tuple[int, ...]]

    @property
    def acquire_shapes(self) -> dict[str, tuple[int, ...]]:
        return self.acquires


class TestZeroEngine:
    def test_execute_returns_zeros(self):
        engine = ZeroEngine()
        acquires = {"acquire1": (10,), "acquire2": (5, 20), "acquire3": (3, 4, 5)}
        program = MockProgram(acquires=acquires)

        results = engine.execute(program)
        for key, shape in acquires.items():
            assert key in results
            assert results[key].shape == shape
