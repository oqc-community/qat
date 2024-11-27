import pytest

from qat.ir.instructions import InstructionList
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.frontends import QASMFrontend
from qat.utils.ir_converter import IRConverter

experiments = {}
# Two qubit benchmarks
hardware_two_qubits = {
    "echo": get_default_echo_hardware(2),
}
circuits_two_qubits = ["bell_state", "2qb_random_cnot", "2qb_clifford"]
for circ in circuits_two_qubits:
    for hw_key, hw in hardware_two_qubits.items():
        experiments[f"{circ}"] = (circ, hw)

# Ten qubit benchmarks
hardware_ten_qubits = {
    "echo": get_default_echo_hardware(10),
}
circuits_ten_qubits = ["10qb_ghz", "10qb_random_cnot"]
for circ in circuits_ten_qubits:
    for hw_key, hw in hardware_ten_qubits.items():
        experiments[f"{circ}"] = (circ, hw)


# QASM2 Benchmarks
def load_qasm(qasm_string):
    with open(f"benchmarks/qasm/{qasm_string}.qasm", "r") as f:
        return f.read()


ir_legacy = {}
for name, stuff in experiments.items():
    circuit, hw = experiments[name]
    circuit = load_qasm(circuit)
    engine = hw.create_engine()
    frontend = QASMFrontend()
    builder, _ = frontend.parse(circuit, hw)
    ir_legacy[name] = (builder, hw)


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
@pytest.mark.parametrize("key", ir_legacy.keys())
def test_benchmarks_legacy_serialize(benchmark, key):
    # Create the hw model
    builder, hw = ir_legacy[key]

    # Create a wrapper for the pipeline
    def run():
        builder.serialize()

    benchmark(run)
    assert True


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
@pytest.mark.parametrize("key", ir_legacy.keys())
def test_benchmarks_legacy_deserialize(benchmark, key):
    # Create the hw model
    builder, hw = ir_legacy[key]
    blob = builder.serialize()

    # Create a wrapper for the pipeline
    def run():
        QuantumInstructionBuilder.deserialize(blob)

    benchmark(run)
    assert True


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
@pytest.mark.parametrize("key", ir_legacy.keys())
def test_benchmarks_pydantic_serialize(benchmark, key):
    # Create the hw model
    builder, hw = ir_legacy[key]
    instructions = IRConverter().legacy_to_pydantic_instructions(builder.instructions)

    # Create a wrapper for the pipeline
    def run():
        instructions.serialize()

    benchmark(run)
    assert True


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
@pytest.mark.parametrize("key", ir_legacy.keys())
def test_benchmarks_pydantic_deserialize(benchmark, key):
    # Create the hw model
    builder, hw = ir_legacy[key]
    instructions = IRConverter().legacy_to_pydantic_instructions(builder.instructions)
    blob = instructions.serialize()

    # Create a wrapper for the pipeline
    def run():
        InstructionList.deserialize(blob)

    benchmark(run)
    assert True


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
@pytest.mark.parametrize("key", ir_legacy.keys())
def test_benchmarks_legacy_pydantic_serialize(benchmark, key):
    # Create the hw model
    builder, hw = ir_legacy[key]

    # Create a wrapper for the pipeline
    def run():
        instructions = IRConverter().legacy_to_pydantic_instructions(builder.instructions)
        instructions.serialize()

    benchmark(run)
    assert True


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
@pytest.mark.parametrize("key", ir_legacy.keys())
def test_benchmarks_legacy_pydantic_deserialize(benchmark, key):
    # Create the hw model
    builder, hw = ir_legacy[key]
    instructions = IRConverter().legacy_to_pydantic_instructions(builder.instructions)
    blob = instructions.serialize()

    # Create a wrapper for the pipeline
    def run():
        new_instructions = InstructionList.deserialize(blob)
        new_instructions = IRConverter(hw).pydantic_to_legacy_instructions(new_instructions)

    benchmark(run)
    assert True
