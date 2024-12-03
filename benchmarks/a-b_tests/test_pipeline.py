import pytest

from qat import QAT
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.frontends import QASMFrontend

from benchmarks.utils.helpers import load_qasm
from benchmarks.utils.models import get_mock_live_hardware

experiments = {}


def prepare_experiment(hw_key, hw, circ):
    builder = QASMFrontend().parse(load_qasm(circ), hardware=hw)[0]
    experiments[f"{circ}[{hw_key}]"] = {"circuit": circ, "hardware": hw, "builder": builder}


# Two qubit benchmarks
hardware_two_qubits = {
    "echo": get_default_echo_hardware(2),
    # "rtcs": get_default_RTCS_hardware(), # Simulation is too slow
    "mock_live": get_mock_live_hardware(2),
}
circuits_two_qubits = ["bell_state", "2qb_random_cnot", "2qb_clifford"]
for circ in circuits_two_qubits:
    for hw_key, hw in hardware_two_qubits.items():
        prepare_experiment(hw_key, hw, circ)

# Ten qubit benchmarks
hardware_ten_qubits = {
    "echo": get_default_echo_hardware(10),
    "mock_live": get_mock_live_hardware(10),
}
circuits_ten_qubits = ["10qb_ghz", "10qb_random_cnot"]
for circ in circuits_ten_qubits:
    for hw_key, hw in hardware_ten_qubits.items():
        prepare_experiment(hw_key, hw, circ)


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10, group="Pipeline:")
@pytest.mark.parametrize("key", experiments.keys())
@pytest.mark.parametrize("mode", ["A", "B"])
class TestPipeline:
    def test_compile_qasm(self, benchmark, key, mode):
        hw = experiments[key]["hardware"]
        circuit = load_qasm(experiments[key]["circuit"])

        # Create a wrapper for the pipeline
        def run_a():
            QASMFrontend().parse(circuit, hw)

        def run_b():
            QAT(hw).compile(circuit)

        if mode == "A":
            benchmark(run_a)
        else:
            benchmark(run_b)
        assert True

    def test_execute_qasm(self, benchmark, key, mode):
        hw = experiments[key]["hardware"]
        builder = experiments[key]["builder"]

        # Create a wrapper for the pipeline
        def run_a():
            QASMFrontend().execute(builder, hw)

        def run_b():
            QAT(hw).execute(builder)

        if mode == "A":
            benchmark(run_a)
        else:
            benchmark(run_b)
        assert True
