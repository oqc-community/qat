# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from tests.unit.utils.loaders import MockModelLoader


class TestModelUpdateChecker:
    loader = MockModelLoader(num_qubits=2)
    model1 = loader.load()
    model2 = loader.load()

    def test_model_is_not_up_to_date(self):
        assert not self.loader.is_up_to_date(self.model1)

    def test_model_is_up_to_date(self):
        assert self.loader.is_up_to_date(self.model2)
