from compiler_config.config import Qasm2Optimizations

from qat import qatconfig
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.integrations.qasm import Qasm2Parser

qiskitconfig = qatconfig.SIMULATION.QISKIT
# qiskitconfig.METHOD = "matrix_product_state"
qiskitconfig.ENABLE_METADATA = True
qasm_file = "tests/qat/files/qasm/qasm2/15qb.qasm"
with open(qasm_file, "r") as f:
    qasm = f.read()


hardware = get_default_qiskit_hardware(35)
opt_config = Qasm2Optimizations()
qasm = DefaultOptimizers().optimize_qasm(qasm, hardware, opt_config)

print(qasm)
parser = Qasm2Parser()
builder = parser.parse(hardware.create_builder(), qasm)
runtime = hardware.create_runtime()
results, metadata = runtime.execute(builder)
print(metadata)
