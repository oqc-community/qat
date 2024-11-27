import pytest

from tests.qat.utils.hardware_models import (
    apply_setup_to_hardware,
    random_directed_connectivity,
)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 10, 32])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
class TestLegacyHardwareModel:
    def test_echo(self, n_qubits, seed):
        logical_connectivity = random_directed_connectivity(n_qubits, seed=seed)
        logical_connectivity = [
            (q1_index, q2_index)
            for q1_index in logical_connectivity
            for q2_index in logical_connectivity[q1_index]
        ]

        hw_echo = apply_setup_to_hardware(
            qubit_count=n_qubits, connectivity=logical_connectivity
        )
        hw_pyd_echo = hw_echo.export_new()

        assert len(hw_echo.qubits) == len(hw_pyd_echo.qubits)
