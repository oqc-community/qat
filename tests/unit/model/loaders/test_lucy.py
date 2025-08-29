# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.model.loaders.lucy import (
    LucyCouplingDirection,
    LucyCouplingQuality,
    LucyModelLoader,
)


class TestLucyModelLoader:
    def test_model_is_calibrated(self):
        loader = LucyModelLoader()
        model = loader.load()
        assert model.is_calibrated

    @pytest.mark.parametrize("qubit_count", [1, 2, 3, 8, 9])
    def test_qubit_count(self, qubit_count):
        loader = LucyModelLoader(qubit_count=qubit_count)
        model = loader.load()
        assert len(model.qubits) == qubit_count

    def test_left_coupling_direction(self):
        qubit_count = 8
        loader = LucyModelLoader(
            qubit_count=qubit_count, coupling_direction=LucyCouplingDirection.LEFT
        )
        model = loader.load()
        coupling_directions = model.logical_connectivity
        for i in range(qubit_count):
            assert i in coupling_directions
            assert coupling_directions[i] == {(i - 1) % qubit_count}

    def test_right_coupling_direction(self):
        qubit_count = 8
        loader = LucyModelLoader(
            qubit_count=qubit_count, coupling_direction=LucyCouplingDirection.RIGHT
        )
        model = loader.load()
        coupling_directions = model.logical_connectivity
        for i in range(qubit_count):
            assert i in coupling_directions
            assert coupling_directions[i] == {(i + 1) % qubit_count}

    def test_random_coupling_direction(self):
        qubit_count = 8
        loader = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_direction=LucyCouplingDirection.RANDOM,
            random_seed=42,
        )
        model = loader.load()
        coupling_directions = model.logical_connectivity
        encountered_couplings = []
        for qubit, couplings in coupling_directions.items():
            for coupling in couplings:
                pair = (coupling, qubit) if coupling < qubit else (qubit, coupling)
                encountered_couplings.append(pair)
        assert len(set(encountered_couplings)) == qubit_count
        assert set(encountered_couplings) == {
            tuple(sorted((i, (i + 1) % qubit_count))) for i in range(qubit_count)
        }

    def test_random_seed(self):
        qubit_count = 8
        loader1 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_direction=LucyCouplingDirection.RANDOM,
            random_seed=254,
        )
        model1 = loader1.load()
        loader2 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_direction=LucyCouplingDirection.RANDOM,
            random_seed=254,
        )
        model2 = loader2.load()
        assert model1.logical_connectivity == model2.logical_connectivity

    def test_random_gives_different_directions(self):
        qubit_count = 8
        loader1 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_direction=LucyCouplingDirection.RANDOM,
            random_seed=42,
        )
        model1 = loader1.load()
        loader2 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_direction=LucyCouplingDirection.RANDOM,
            random_seed=43,
        )
        model2 = loader2.load()
        assert model1.logical_connectivity != model2.logical_connectivity

    def test_same_seed_gives_same_directions(self):
        qubit_count = 8
        loader1 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_direction=LucyCouplingDirection.RANDOM,
            random_seed=42,
        )
        model1 = loader1.load()
        loader2 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_direction=LucyCouplingDirection.RANDOM,
            random_seed=42,
        )
        model2 = loader2.load()
        assert model1.logical_connectivity == model2.logical_connectivity

    def test_uniform_qualities(self):
        qubit_count = 8
        loader = LucyModelLoader(
            qubit_count=qubit_count, coupling_qualities=LucyCouplingQuality.UNIFORM
        )
        model = loader.load()
        assert np.allclose(list(model.logical_connectivity_quality.values()), 1.0)

    def test_random_qualities(self):
        qubit_count = 8
        loader = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_qualities=LucyCouplingQuality.RANDOM,
            random_seed=42,
        )
        model = loader.load()
        qualities = list(model.logical_connectivity_quality.values())
        assert len(qualities) == qubit_count
        assert all(0.0 <= q <= 1.0 for q in qualities)
        assert len(set(qualities)) == len(qualities)

    def test_custom_qualities(self):
        qubit_count = 8
        qualities = {
            (i, (i + 1) % qubit_count): i / qubit_count for i in range(qubit_count)
        }
        loader = LucyModelLoader(qubit_count=qubit_count, coupling_qualities=qualities)
        model = loader.load()
        assert model.logical_connectivity_quality == qualities

    def test_custom_qualities_are_ordered(self):
        qubit_count = 8
        qualities = {
            (i, (i + 1) % qubit_count): i / qubit_count for i in range(qubit_count)
        }
        loader = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_qualities=qualities,
            coupling_direction=LucyCouplingDirection.LEFT,
        )
        model = loader.load()
        assert model.logical_connectivity_quality == {
            ((i + 1) % qubit_count, i): i / qubit_count for i in range(qubit_count)
        }

    def test_invalid_custom_couplings_raises(self):
        qubit_count = 8
        qualities = {
            (i, (i + 1) % qubit_count): i / qubit_count for i in range(qubit_count)
        }
        qualities[(0, 2)] = 0.5  # Invalid coupling
        with pytest.raises(ValueError):
            LucyModelLoader(
                qubit_count=qubit_count,
                coupling_qualities=qualities,
                coupling_direction=LucyCouplingDirection.LEFT,
            ).load()

    def test_missing_coupling_raises_validaiton_error(self):
        qubit_count = 8
        qualities = {
            (i, (i + 1) % qubit_count): i / qubit_count for i in range(qubit_count)
        }
        del qualities[(0, 1)]
        with pytest.raises(IndexError):
            LucyModelLoader(
                qubit_count=qubit_count,
                coupling_qualities=qualities,
                coupling_direction=LucyCouplingDirection.LEFT,
            ).load()

    def test_random_gives_different_qualities(self):
        qubit_count = 8
        loader1 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_qualities=LucyCouplingQuality.RANDOM,
            random_seed=42,
        )
        model1 = loader1.load()
        loader2 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_qualities=LucyCouplingQuality.RANDOM,
            random_seed=43,
        )
        model2 = loader2.load()
        assert model1.logical_connectivity_quality != model2.logical_connectivity_quality

    def test_same_seed_gives_same_qualities(self):
        qubit_count = 8
        loader1 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_qualities=LucyCouplingQuality.RANDOM,
            random_seed=42,
        )
        model1 = loader1.load()
        loader2 = LucyModelLoader(
            qubit_count=qubit_count,
            coupling_qualities=LucyCouplingQuality.RANDOM,
            random_seed=42,
        )
        model2 = loader2.load()
        assert model1.logical_connectivity_quality == model2.logical_connectivity_quality

    def test_set_qubit_baseband_frequency(self):
        loader = LucyModelLoader(qubit_count=8, qubit_baseband_frequency=6.0e9)
        model = loader.load()
        for qubit in model.qubits.values():
            assert qubit.physical_channel.baseband.frequency == 6.0e9

    def test_set_resonator_baseband_frequency(self):
        loader = LucyModelLoader(qubit_count=8, resonator_baseband_frequency=9.0e9)
        model = loader.load()
        for qubit in model.qubits.values():
            assert qubit.resonator.physical_channel.baseband.frequency == 9.0e9

    def test_set_qubit_baseband_if_frequency(self):
        loader = LucyModelLoader(qubit_count=8, qubit_baseband_if_frequency=300e6)
        model = loader.load()
        for qubit in model.qubits.values():
            assert qubit.physical_channel.baseband.if_frequency == 300e6

    def test_set_resonator_baseband_if_frequency(self):
        loader = LucyModelLoader(qubit_count=8, resonator_baseband_if_frequency=300e6)
        model = loader.load()
        for qubit in model.qubits.values():
            assert qubit.resonator.physical_channel.baseband.if_frequency == 300e6

    def test_set_drive_frequency(self):
        loader = LucyModelLoader(qubit_count=8, drive_frequency=6.0e9)
        model = loader.load()
        for qubit in model.qubits.values():
            assert qubit.drive_pulse_channel.frequency == 6.0e9
            for channel in qubit.cross_resonance_pulse_channels.values():
                assert channel.frequency == 6.0e9
            for channel in qubit.cross_resonance_cancellation_pulse_channels.values():
                assert channel.frequency == 6.0e9

    def test_set_measure_frequency(self):
        loader = LucyModelLoader(qubit_count=8, measure_frequency=5.0e9)
        model = loader.load()
        for qubit in model.qubits.values():
            assert qubit.measure_pulse_channel.frequency == 5.0e9

    def test_set_cr_scale(self):
        loader = LucyModelLoader(qubit_count=8, cr_scale=0.5)
        model = loader.load()
        for qubit in model.qubits.values():
            for channel in qubit.cross_resonance_pulse_channels.values():
                assert channel.scale == 0.5

    def test_set_crc_scale(self):
        loader = LucyModelLoader(qubit_count=8, crc_scale=0.3)
        model = loader.load()
        for qubit in model.qubits.values():
            for channel in qubit.cross_resonance_cancellation_pulse_channels.values():
                assert channel.scale == 0.3
