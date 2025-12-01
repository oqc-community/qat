# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import random
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError

from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.device import PulseChannel
from qat.model.error_mitigation import ErrorMitigation, ReadoutMitigation
from qat.model.hardware_model import VERSION, PhysicalHardwareModel
from qat.model.loaders.lucy import LucyModelLoader
from qat.utils.hardware_model import (
    generate_connectivity_data,
    generate_random_linear,
    random_connectivity,
    random_quality_map,
)

from tests.unit.utils.models import get_toshiko_connectivity


class TestPhysicalHardwareModel:
    def test_models_are_equal(self):
        model = LucyModelLoader(qubit_count=8).load()
        new_model = deepcopy(model)
        assert new_model is not model
        assert model == new_model

    def test_models_not_equal(self):
        model1 = LucyModelLoader(qubit_count=8).load()
        model2 = LucyModelLoader(qubit_count=16).load()
        assert model1 != model2

    def test_unique_phys_chan_indices(self):
        model = LucyModelLoader(qubit_count=8).load()
        model_data = model.model_dump()
        model_data["qubits"][0]["resonator"]["physical_channel"]["name_index"] = model_data[
            "qubits"
        ][0]["physical_channel"]["name_index"]
        with pytest.raises(ValidationError):
            PhysicalHardwareModel(**model_data)

    def test_qubit_for_physical_channel_id_returns_qubits(self):
        model = LucyModelLoader(qubit_count=8).load()
        for qubit in model.qubits.values():
            returned_qubit = model.qubit_for_physical_channel_id(
                qubit.physical_channel.uuid
            )
            assert returned_qubit == qubit

            returned_qubit = model.qubit_for_physical_channel_id(
                qubit.resonator.physical_channel.uuid
            )
            assert returned_qubit == qubit

    def test_device_for_physical_channel_id_returns_correct_devices(self):
        model = LucyModelLoader(qubit_count=8).load()
        for qubit in model.qubits.values():
            returned_device = model.device_for_physical_channel_id(
                qubit.physical_channel.uuid
            )
            assert returned_device == qubit

            returned_device = model.device_for_physical_channel_id(
                qubit.resonator.physical_channel.uuid
            )
            assert returned_device == qubit.resonator

    def test_device_for_invalid_physical_channel_id_raises(self, caplog):
        model = LucyModelLoader(qubit_count=8).load()
        invalid_uuid = "00000000-0000-0000-0000-000000000000"
        with caplog.at_level("WARNING", logger="qat.purr.utils.logger"):
            model.device_for_physical_channel_id(invalid_uuid)
        assert "No device found for physical channel id" in caplog.text


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 10, 32])
@pytest.mark.parametrize("n_logical_qubits", [0, 2, 4])
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("as_json", [False, True])
class Test_HW_Serialisation:
    @staticmethod
    def reload_hw(hw, as_json):
        if as_json:
            return PhysicalHardwareModel.model_validate_json(hw.model_dump_json())
        return PhysicalHardwareModel(**hw.model_dump())

    def test_built_model_serialises(self, n_qubits, n_logical_qubits, seed, as_json):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        builder = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        )

        hw1 = builder.model
        hw2 = self.reload_hw(hw1, as_json)
        assert hw1 == hw2

    def test_built_model_with_default_error_mit_serialises(
        self, n_qubits, n_logical_qubits, seed, as_json
    ):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        builder = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        )

        hw1 = builder.model
        hw1.error_mitigation = ErrorMitigation()
        assert not hw1.error_mitigation.is_enabled

        hw2 = self.reload_hw(hw1, as_json)
        assert hw1 == hw2

    @pytest.mark.parametrize("m3_available", [True, False])
    def test_built_model_with_error_mit_serialises(
        self, n_qubits, n_logical_qubits, m3_available, seed, as_json
    ):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        builder = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        )
        hw1 = builder.model

        qubit_indices = list(hw1.qubits.keys())
        linear = generate_random_linear(qubit_indices)
        readout_mit = ReadoutMitigation(linear=linear, m3_available=m3_available)
        error_mit = ErrorMitigation(readout_mitigation=readout_mit)
        hw1.error_mitigation = error_mit

        assert hw1.error_mitigation.is_enabled
        assert hw1.error_mitigation.readout_mitigation == readout_mit

        hw2 = self.reload_hw(hw1, as_json)
        assert hw1 == hw2

    def test_built_logical_model_serialises(
        self, n_qubits, n_logical_qubits, seed, as_json
    ):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        builder = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        )

        hw1 = builder.model
        hw2 = self.reload_hw(hw1, as_json)
        assert hw1 == hw2

    def test_dump_load_eq(self, n_qubits, n_logical_qubits, seed, as_json):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        ).model

        hw2 = self.reload_hw(hw1, as_json)
        assert hw1 == hw2

        hw3 = PhysicalHardwareModelBuilder(
            physical_connectivity=random_connectivity(n=n_qubits, max_degree=3, seed=54389)
        ).model
        assert hw1 != hw3

    def test_dump_eq(self, n_qubits, n_logical_qubits, seed, as_json):
        if as_json:
            pytest.skip("List ordering in dicts is not guaranteed in JSON.")
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        ).model
        blob1 = hw1.model_dump()

        hw2 = self.reload_hw(hw1, as_json)
        blob2 = hw2.model_dump()
        assert blob1 == blob2

        hw3 = PhysicalHardwareModelBuilder(
            physical_connectivity=random_connectivity(n=n_qubits, max_degree=3, seed=seed)
        ).model
        blob3 = hw3.model_dump()
        assert blob1 != blob3

    def test_deep_equals(self, n_qubits, n_logical_qubits, seed, as_json):
        if as_json:
            pytest.skip("Deepcopy does not serialise to json.")
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        ).model
        hw2 = deepcopy(hw1)

        assert hw1 == hw2

        index = random.Random(seed).choice(list(hw2.qubits.keys()))
        hw2.qubit_with_index(index).drive_pulse_channel.frequency = random.Random(
            seed
        ).uniform(1e08, 1e10)
        assert hw1 != hw2

    def test_deserialise_version(self, n_qubits, n_logical_qubits, seed, as_json):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(n_logical_qubits, n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        ).model
        assert hw1.version == VERSION

        hw2 = self.reload_hw(hw1, as_json)
        assert hw2.version == VERSION


def randomly_calibrate(hardware_model: PhysicalHardwareModel, seed=42):
    for qubit in hardware_model.qubits.values():
        # Calibrate physical channel.
        for physical_channel in [qubit.physical_channel, qubit.resonator.physical_channel]:
            physical_channel.baseband.frequency = random.Random(seed).uniform(1e05, 1e07)
            physical_channel.baseband.if_frequency = random.Random(seed).uniform(1e05, 1e07)

        # Calibrate qubit and resonator pulse channels.
        for pulse_channels in [qubit.pulse_channels, qubit.resonator.pulse_channels]:
            for pulse_channel_name in pulse_channels.__class__.model_fields:
                pulse_channel = getattr(pulse_channels, pulse_channel_name)
                if isinstance(pulse_channel, PulseChannel):
                    pulse_channel.frequency = random.Random(seed).uniform(1e08, 1e10)
                elif isinstance(pulse_channel, tuple):
                    for sub_pulse_channel in pulse_channel:
                        sub_pulse_channel.frequency = random.Random(seed).uniform(
                            1e08, 1e10
                        )

    return hardware_model


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 10, 32])
@pytest.mark.parametrize("seed", [41, 42, 43])
class Test_HW_Calibration:
    def test_model_calibration(self, n_qubits, seed):
        hw = PhysicalHardwareModelBuilder(
            physical_connectivity=random_connectivity(n=n_qubits, max_degree=3, seed=seed)
        ).model
        assert hw.number_of_qubits == n_qubits
        assert not hw.is_calibrated

        hw2 = randomly_calibrate(hardware_model=hw, seed=seed)
        assert hw2.is_calibrated

    def test_model_calibration_serialises(self, n_qubits, seed):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(int(np.sqrt(n_qubits - 1)), n_qubits // 2), seed=seed
            )
        )

        hw1 = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        ).model
        hw1 = randomly_calibrate(hardware_model=hw1, seed=seed)

        hw2 = PhysicalHardwareModel(**hw1.model_dump())
        assert hw1 == hw2


@pytest.mark.parametrize("n_qubits", [8, 16, 32, 64])
@pytest.mark.parametrize("seed", [1, 2, 3])
class Test_HW_Connectivity:
    def test_constrained_connectivity_subgraph(self, n_qubits, seed):
        physical_connectivity, logical_connectivity, _ = generate_connectivity_data(
            n_qubits, min(int(np.sqrt(n_qubits - 1)), n_qubits // 2), seed=seed
        )

        hw = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
        ).model
        assert hw.physical_connectivity == physical_connectivity
        assert hw.logical_connectivity == logical_connectivity

        wrong_connectivity = deepcopy(physical_connectivity)
        wrong_connectivity[0].add(n_qubits)
        wrong_connectivity[n_qubits] = {0}
        with pytest.raises(ValidationError):
            PhysicalHardwareModelBuilder(
                physical_connectivity=physical_connectivity,
                logical_connectivity=wrong_connectivity,
            )

    def test_invalid_connectivity_quality(self, n_qubits, seed):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(int(np.sqrt(n_qubits - 1)), n_qubits // 2), seed=seed
            )
        )

        q1, q2 = random.Random(seed).sample(list(range(0, n_qubits)), 2)
        logical_connectivity_quality.update(
            {(q1, q2): random.Random(seed).uniform(-1.0, -0.001)}
        )

        with pytest.raises(ValueError):
            PhysicalHardwareModelBuilder(
                physical_connectivity=physical_connectivity,
                logical_connectivity=logical_connectivity,
                logical_connectivity_quality=logical_connectivity_quality,
            ).model

    def test_frozen_connectivity(self, n_qubits, seed):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(int(np.sqrt(n_qubits - 1)), n_qubits // 2), seed=seed
            )
        )

        hw = PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        ).model

        q = random.Random(seed).sample(list(range(0, n_qubits)), 1)[0]
        q1_altered = random.Random(seed + 1).sample(list(range(0, n_qubits)), 1)[0]
        q2_altered = random.Random(seed + 2).sample(list(range(0, n_qubits)), 1)[0]

        with pytest.raises(AttributeError):
            hw.physical_connectivity[q].add((q1_altered, q2_altered))

        with pytest.raises(AttributeError):
            hw.logical_connectivity[q].add((q1_altered, q2_altered))

        with pytest.raises(AttributeError):
            hw.physical_connectivity[q].discard(q1_altered)

        with pytest.raises(AttributeError):
            hw.logical_connectivity[q].discard(q2_altered)

    def test_invalid_connectivity_type(self, n_qubits, seed):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(
                n_qubits, min(int(np.sqrt(n_qubits - 1)), n_qubits // 2), seed=seed
            )
        )

        q = random.Random(seed).sample(list(range(0, n_qubits)), 1)[0]

        invalid_int = random.Random(seed).sample(list(range(-n_qubits, -1)), 1)[0]
        invalid_float = random.Random(seed).uniform(-10.0, 10.0)
        invalid_str = "abc"
        for invalid_value in [invalid_int, invalid_float, invalid_str]:
            physical_connectivity[q] = (invalid_value, invalid_value)
            logical_connectivity[q] = (invalid_value, invalid_value)

            with pytest.raises(ValueError):
                PhysicalHardwareModelBuilder(
                    physical_connectivity=physical_connectivity,
                    logical_connectivity=logical_connectivity,
                    logical_connectivity_quality=logical_connectivity_quality,
                ).model

    def test_mismatch_qubits_cr_crc(self, n_qubits, seed):
        physical_connectivity = random_connectivity(n_qubits, seed=seed)

        hw = PhysicalHardwareModelBuilder(physical_connectivity=physical_connectivity).model
        blob = hw.model_dump()

        q = random.Random(seed).sample(list(range(0, n_qubits)), 1)[0]
        for aux_qubit_id, pulse_channel in blob["qubits"][q]["pulse_channels"][
            "cross_resonance_channels"
        ].items():
            pulse_channel["auxiliary_qubit"] = aux_qubit_id + 12321

        with pytest.raises(ValidationError):
            PhysicalHardwareModel(**blob)

    def test_mismatch_qubits_vs_connectivity(self, n_qubits, seed):
        physical_connectivity = random_connectivity(n_qubits, seed=seed)

        hw = PhysicalHardwareModelBuilder(physical_connectivity=physical_connectivity).model
        blob = hw.model_dump()

        q = random.Random(seed).sample(list(range(0, n_qubits)), 1)[0]

        changed_connectivity = set(deepcopy(blob["physical_connectivity"][q]))
        changed_connectivity.update({100: {1, 2, 3}})
        blob["physical_connectivity"][q] = changed_connectivity
        with pytest.raises(ValidationError):
            PhysicalHardwareModel(**blob)

        changed_connectivity = set(deepcopy(blob["physical_connectivity"][q]))
        changed_connectivity.pop()
        blob["logical_connectivity"][q] = changed_connectivity
        with pytest.raises(ValidationError):
            PhysicalHardwareModel(**blob)

    def test_default_logical_topology(self, n_qubits, seed):
        physical_connectivity = random_connectivity(n_qubits, seed=seed)
        hw = PhysicalHardwareModelBuilder(physical_connectivity=physical_connectivity).model

        hw.logical_connectivity == physical_connectivity
        for quality in hw.logical_connectivity_quality.values():
            assert quality == 1.0


@pytest.mark.parametrize("seed", [500, 501, 502])
@pytest.mark.parametrize("n_removed_qubits", [1, 2, 3, 4])
class Test_OQC_Hardware:
    def test_lucy(self, seed, n_removed_qubits):
        n_qubits = 8
        qubit_indices = list(range(0, n_qubits))

        ring_graph = nx.cycle_graph(n_qubits)
        ring_architecture = {
            node: set(neighbors) for node, neighbors in ring_graph.adjacency()
        }

        # Randomly remove 3 qubits from the GPU connectivity.
        removed_qubits = set(
            random.Random(seed).sample(tuple(qubit_indices), n_removed_qubits)
        )
        logical_connectivity = {
            k: deepcopy(v) for k, v in ring_architecture.items() if k not in removed_qubits
        }
        for connected_qubits in logical_connectivity.values():
            connected_qubits -= removed_qubits

        logical_connectivity_quality = random_quality_map(logical_connectivity, seed=seed)

        hw = PhysicalHardwareModelBuilder(
            physical_connectivity=ring_architecture,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        ).model
        assert hw.number_of_qubits == n_qubits
        assert not hw.is_calibrated

        hw = randomly_calibrate(hardware_model=hw, seed=seed)
        assert hw.is_calibrated

    def test_toshiko(self, testpath, seed, n_removed_qubits):
        lattice_connectivity = defaultdict(set)
        qubit_indices = set()

        connections = get_toshiko_connectivity()
        for c in connections:
            lattice_connectivity[c[0]].add(c[1])
            lattice_connectivity[c[1]].add(c[0])
            qubit_indices.update([c[0], c[1]])
        lattice_connectivity = dict(lattice_connectivity)

        # Randomly remove 3 qubits from the GPU connectivity.
        removed_qubits = set(
            random.Random(seed).sample(tuple(qubit_indices), n_removed_qubits)
        )
        logical_connectivity = {
            k: deepcopy(v)
            for k, v in lattice_connectivity.items()
            if k not in removed_qubits
        }
        for connected_qubits in logical_connectivity.values():
            connected_qubits -= removed_qubits

        logical_connectivity_quality = random_quality_map(logical_connectivity, seed=seed)

        n_physical_qubits = len(
            set(lattice_connectivity.keys()).union(*lattice_connectivity.values())
        )
        n_logical_qubits = len(
            set(logical_connectivity.keys()).union(*logical_connectivity.values())
        )
        assert n_physical_qubits > n_logical_qubits

        builder = PhysicalHardwareModelBuilder(
            physical_connectivity=lattice_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        )
        hw = builder.model
        assert hw.number_of_qubits == n_physical_qubits
        assert hw.physical_connectivity == lattice_connectivity
        assert hw.logical_connectivity == logical_connectivity
        assert hw.logical_connectivity_quality == logical_connectivity_quality
