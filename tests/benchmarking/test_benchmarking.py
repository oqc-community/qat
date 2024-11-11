import time

import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QASMFrontend

from tests.benchmarking.utils.models import get_mock_live_hardware
from tests.qat.utils.models import get_jagged_echo_hardware

experiments = {}

# Two qubit benchmarks
hardware_two_qubits = {
    "echo": get_default_echo_hardware(2),
    "rtcs": get_default_RTCS_hardware(),
    "mock_live": get_mock_live_hardware(2),
}
circuits_two_qubits = ["bell_state", "2qb_random_cnot", "2qb_clifford"]
for circ in circuits_two_qubits:
    for hw_key, hw in hardware_two_qubits.items():
        experiments[f"{circ}[{hw_key}]"] = (circ, hw)

# Ten qubit benchmarks
hardware_ten_qubits = {
    "echo": get_default_echo_hardware(10),
    "jagged": get_jagged_echo_hardware(10),
    "mock_live": get_mock_live_hardware(10),
}
circuits_ten_qubits = ["10qb_ghz", "10qb_random_cnot"]
for circ in circuits_ten_qubits:
    for hw_key, hw in hardware_ten_qubits.items():
        experiments[f"{circ}[{hw_key}]"] = (circ, hw)


# QASM2 Benchmarks
def load_qasm(qasm_string):
    with open(f"tests/benchmarking/qasm/{qasm_string}.qasm", "r") as f:
        return f.read()


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
@pytest.mark.parametrize("key", experiments.keys())
def test_benchmarks_qasm(benchmark, key):
    # Create the hw model
    circuit, hw = experiments[key]
    circuit = load_qasm(circuit)
    engine = hw.create_engine()

    # Create a wrapper for the pipeline
    def run():
        frontend = QASMFrontend()
        builder, _ = frontend.parse(circuit, hw)
        builder._instructions = engine.optimize(builder.instructions)
        engine.validate(builder.instructions)
        qatfile = InstructionEmitter().emit(builder.instructions, hw)
        engine.create_duration_timeline(qatfile.instructions)

    benchmark(run)
    assert True


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
def test_regression_report_warn(benchmark):
    benchmark(time.sleep, 0.1)


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
def test_regression_report_fail(benchmark):
    benchmark(time.sleep, 0.1)


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
def test_regression_report_improvement(benchmark):
    benchmark(time.sleep, 0.1)
