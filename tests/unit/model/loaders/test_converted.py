# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.converted import JaggedEchoModelLoader
from qat.utils.hardware_model import random_error_mitigation


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

    def test_serialisation_with_mitigation(self):
        jagged_indices = {2, 3, 6, 8, 9, 10, 11, 12}
        error_mit = random_error_mitigation(jagged_indices, seed=42)
        hw = JaggedEchoModelLoader(
            qubit_count=8,
            random_seed=42,
            qubit_indices=jagged_indices,
            error_mitigation=error_mit,
        ).load()
        assert hw.error_mitigation is not None
        hw2 = PhysicalHardwareModel.model_validate_json(hw.model_dump_json())
        assert hw2.error_mitigation == hw.error_mitigation
