import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QASMFrontend

from tests.benchmarking.utils import bell_state, random_qasm_two_qubits

# Storing the hardware and experiments as dicts to make the saved benchmark
# names understandable
hardware_two_qubits = {
    "echo": get_default_echo_hardware(2),
    "rtcs": get_default_RTCS_hardware(),
}
experiments_two_qubits = {
    "qasm_bell_state": bell_state(),
    "qasm_random_two_qubits": random_qasm_two_qubits(10),
}


@pytest.mark.parametrize("hw", list(hardware_two_qubits.keys()))
@pytest.mark.parametrize("circuit", list(experiments_two_qubits.keys()))
def test_benchmarks(benchmark, hw, circuit):
    # Create the hw model
    circuit = experiments_two_qubits[circuit]
    hw = hardware_two_qubits[hw]
    engine = hw.create_engine()
    hw.create_builder()

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
