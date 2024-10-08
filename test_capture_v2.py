from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.frontends import QASMFrontend

hw = get_default_echo_hardware(3)
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
print(builder.instructions[-1].results_processing)
