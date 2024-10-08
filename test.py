from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.qat import execute

hw = get_default_echo_hardware(3)
hw = get_default_RTCS_hardware(repeats=10)


frame_measure = "r1_measure"
frame_aquire = "r1_acquire"

program = f"""
OPENQASM 3;
defcalgrammar "openpulse";

cal {{
      extern frame {frame_measure};
      extern frame {frame_aquire};
      waveform wf1 = gaussian(1.0, 18ns, 0.20);

}}
defcal measure $1 {{
      barrier {frame_measure}, {frame_aquire};
      play({frame_measure}, wf1);
      capture_v1({frame_aquire},0.000001);
}}

h $0;
h $1;
measure $0;
measure $1;
"""

program2 = """
OPENQASM 2.0;
include "qelib1.inc";

creg c[2];
qreg q[2];

h q[0];
cx q[0], q[1];
measure q -> c;
"""

config = CompilerConfig(
    results_format=QuantumResultsFormat().binary_count(),
    repeats=10,
)
# result = execute(program, hw, config)
result = execute(program, hw)

# runtime = hw.create_runtime()
# result = runtime.execute(builder)
print(result[0, :, :])
print(result[1, :, :])

"""
defcal measure $1 {{
      barrier {frame_measure}, {frame_aquire};
      play({frame_measure}, wf1);
      capture_v1({frame_aquire},0.000001);
      return 1;
}}
"""
