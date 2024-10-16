import sys

import numpy as np

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.config import CompilerConfig, QuantumResultsFormat
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QASMFrontend
from qat.purr.compiler.runtime import execute_instructions

hw = get_default_echo_hardware(3)
hw = get_default_RTCS_hardware(repeats=10)


frame_measure = "r0_measure"
frame_aquire = "r0_acquire"

program = f"""
OPENQASM 3;
defcalgrammar "openpulse";

cal {{extern frame q0_drive;}}
cal {{waveform wf1 = sech(0.2, 100dt, 10dt);}}
defcal puls $0 {{play(q0_drive, wf1);}}
puls $0;
bit[1] c;
c[0] = measure $0;
"""


config = CompilerConfig(
    results_format=QuantumResultsFormat().binary_count(),
    repeats=10,
)

print(np.arccosh(sys.float_info.max))

parser = QASMFrontend()
builder, metrics = parser.parse(program, hw)
# print(builder.instructions)
print("\n".join([str(inst) for inst in builder.instructions]))

# for inst in builder.instructions:
#    if isinstance(inst, Acquire):
#        inst.delay = 180e-9

qatfile = InstructionEmitter().emit(builder.instructions, hw)
print("--")
print("\n".join([str(inst) for inst in qatfile.instructions]))
print("--")
print(qatfile.meta_instructions)

print("--")
results, metrics = execute_instructions(hw, builder, config)
print(results)
print(metrics.optimized_circuit)
