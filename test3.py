from time import perf_counter

from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import Qasm2Optimizations
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.integrations.qasm import Qasm2Parser
from tests.qat.qasm_utils import get_qasm2


def parse_and_apply_optimiziations(
    hardware, qasm_file_name, parser=None, opt_config=None
) -> InstructionBuilder:
    """
    Helper that builds a basic hardware, applies general optimizations, parses the QASM
    then returns the resultant builder.
    """
    qasm = get_qasm2(qasm_file_name)
    if opt_config is None:
        opt_config = Qasm2Optimizations()
    qasm = DefaultOptimizers().optimize_qasm(qasm, hardware, opt_config)
    if parser is None:
        parser = Qasm2Parser()
    return parser.parse(hardware.create_builder(), qasm)


hw = get_default_qiskit_hardware(35)
builder = parse_and_apply_optimiziations(hw, "15qb.qasm")
runtime = hw.create_runtime()

t = perf_counter()
results = runtime.execute(builder)
print(perf_counter() - t)
# print(results)
