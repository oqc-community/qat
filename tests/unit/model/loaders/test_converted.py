# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.converted import JaggedEchoModelLoader


class TestJaggedEchoModelLoader:
    def test_seed_reproducibility(self):
        loader1 = JaggedEchoModelLoader(
            qubit_count=32,
            random_seed=42,
        )
        model1 = loader1.load()

        loader2 = JaggedEchoModelLoader(
            qubit_count=32,
            random_seed=42,
        )
        model2 = loader2.load()

        assert model1.error_mitigation == model2.error_mitigation
        assert model1.physical_connectivity == model2.physical_connectivity
        assert model1.logical_connectivity == model2.logical_connectivity

    def test_different_seeds_give_different_model(self):
        loader1 = JaggedEchoModelLoader(
            qubit_count=32,
            random_seed=42,
        )
        model1 = loader1.load()

        loader2 = JaggedEchoModelLoader(
            qubit_count=32,
            random_seed=254,
        )
        model2 = loader2.load()

        assert model1.error_mitigation != model2.error_mitigation or (
            model1.physical_connectivity != model2.physical_connectivity
            or model1.logical_connectivity != model2.logical_connectivity
        )
