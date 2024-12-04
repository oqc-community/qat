import json
import random
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError

from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.device import PulseChannel
from qat.model.hardware_model import VERSION, PhysicalHardwareModel

from tests.qat.utils.hardware_models import (
    generate_connectivity_data,
    random_connectivity,
    random_quality_map,
)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 10, 32])
@pytest.mark.parametrize("n_logical_qubits", [0, 2, 4])
@pytest.mark.parametrize("seed", [1, 2, 3])
class Test_HW_Serialisation:
    def test_built_model_serialises(self, n_qubits, n_logical_qubits, seed):
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
        hw2 = PhysicalHardwareModel(**hw1.model_dump())
        assert hw1 == hw2

    def test_built_logical_model_serialises(self, n_qubits, n_logical_qubits, seed):
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
        hw2 = PhysicalHardwareModel(**hw1.model_dump())
        assert hw1 == hw2

    def test_dump_load_eq(self, n_qubits, n_logical_qubits, seed):
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
        blob = hw1.model_dump()

        hw2 = PhysicalHardwareModel(**blob)
        assert hw1 == hw2

        hw3 = PhysicalHardwareModelBuilder(
            physical_connectivity=random_connectivity(n=n_qubits, max_degree=3, seed=54389)
        ).model
        assert hw1 != hw3

    def test_dump_eq(self, n_qubits, n_logical_qubits, seed):
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

        hw2 = PhysicalHardwareModel(**blob1)
        blob2 = hw2.model_dump()
        assert blob1 == blob2

        hw3 = PhysicalHardwareModelBuilder(
            physical_connectivity=random_connectivity(n=n_qubits, max_degree=3, seed=seed)
        ).model
        blob3 = hw3.model_dump()
        assert blob1 != blob3

    def test_deep_equals(self, n_qubits, n_logical_qubits, seed):
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
        hw2.qubit_with_index(index).pulse_channels.drive.frequency = random.Random(
            seed
        ).uniform(1e08, 1e10)
        assert hw1 != hw2

    def test_deserialise_version(self, n_qubits, n_logical_qubits, seed):
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

        hw2 = PhysicalHardwareModel(**hw1.model_dump())
        assert hw2.version == VERSION


def randomly_calibrate(hardware_model: PhysicalHardwareModel, seed=42):
    for qubit in hardware_model.qubits.values():
        # Calibrate physical channel.
        for physical_channel in [qubit.physical_channel, qubit.resonator.physical_channel]:
            physical_channel.sample_time = random.Random(seed).uniform(1e-08, 1e-10)
            physical_channel.baseband.frequency = random.Random(seed).uniform(1e05, 1e07)
            physical_channel.baseband.if_frequency = random.Random(seed).uniform(1e05, 1e07)

        # Calibrate qubit and resonator pulse channels.
        for pulse_channels in [qubit.pulse_channels, qubit.resonator.pulse_channels]:
            for pulse_channel_name in pulse_channels.model_fields:
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

        q = random.Random(seed).sample(list(range(0, n_qubits)), 1)[0]
        logical_connectivity_quality.update({q: random.Random(seed).uniform(-1.0, -0.001)})

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
                hw = PhysicalHardwareModelBuilder(
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
        changed_connectivity = set(deepcopy(blob["logical_connectivity"][q]))
        changed_connectivity.pop()
        blob["logical_connectivity"][q] = changed_connectivity

        with pytest.raises(ValidationError):
            PhysicalHardwareModel(**blob)


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

    def test_toshiko(self, seed, n_removed_qubits):
        lattice_connectivity = defaultdict(set)
        qubit_indices = set()

        filepath = "tests/qat/files/hardware/toshiko_lattice_connections.json"
        with open(filepath, "r") as f:
            connections = json.load(f)
            for c in connections["connections"]:
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
