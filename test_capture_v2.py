from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QASMFrontend
from qat.purr.compiler.instructions import Acquire

# hw = get_default_echo_hardware(3)
hw = get_default_RTCS_hardware(repeats=10)


frame_measure = "r0_measure"
frame_aquire = "r0_acquire"

program = f"""
OPENQASM 3;
defcalgrammar "openpulse";

cal {{
      extern frame {frame_measure};
      extern frame {frame_aquire};
      waveform wf1 = gaussian(1.0, 18ns, 0.20);

}}
defcal measure $0 {{
      barrier {frame_measure}, {frame_aquire};
      play({frame_measure}, wf1);
      capture_v2({frame_aquire},0.000001);
}}

measure $0;
"""


config = CompilerConfig(
    results_format=QuantumResultsFormat().binary_count(),
    repeats=10,
)


parser = QASMFrontend()
builder, metrics = parser.parse(program, hw)
# print(builder.instructions)
print("\n".join([str(inst) for inst in builder.instructions]))

for inst in builder.instructions:
    if isinstance(inst, Acquire):
        print(inst.delay)

qatfile = InstructionEmitter().emit(builder.instructions, hw)
print("--")
print("\n".join([str(inst) for inst in qatfile.instructions]))
print("--")
print(qatfile.meta_instructions)

runtime = hw.create_runtime()

# print("--")
# results, metrics = execute_instructions(hw, builder)
# print(results)
# print(metrics.optimized_circuit)
