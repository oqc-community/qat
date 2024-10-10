from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QASMFrontend
from qat.purr.compiler.instructions import Acquire
from qat.purr.compiler.runtime import execute_instructions

# hw = get_default_echo_hardware(3)
hw = get_default_RTCS_hardware(repeats=10)


frame_measure = "r0_measure"
frame_aquire = "r0_acquire"

program = f"""
OPENQASM 3;
defcalgrammar "openpulse";

measure $0;
"""

program2 = f"""
OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
measure qr -> cr;
"""


config = CompilerConfig(
    results_format=QuantumResultsFormat().binary_count(),
    repeats=10,
)


parser = QASMFrontend()
builder, metrics = parser.parse(program, hw)
print("\n".join([str(inst) for inst in builder.instructions]))
# print([type(inst) for inst in builder.instructions])

for inst in builder.instructions:
    if isinstance(inst, Acquire):
        print(inst.delay)

qatfile = InstructionEmitter().emit(builder.instructions, hw)
print("--")
print("\n".join([str(inst) for inst in qatfile.instructions]))
print("--")
print(qatfile.meta_instructions)

engine = hw.create_engine()
results, metrics = execute_instructions(hw, builder, config)
print(results)
print(metrics.optimized_circuit)
