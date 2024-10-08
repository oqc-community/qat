from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware

hw = get_default_echo_hardware(3)
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


# parser = QASMFrontend()
# builder, metrics = parser.parse(program, hw)
# print("\n".join([str(inst) for inst in builder.instructions]))
# print([type(inst) for inst in builder.instructions])


builder = hw.create_builder()
builder.measure(hw.get_qubit(0))
print("\n".join([str(inst) for inst in builder.instructions]))
print([type(inst) for inst in builder.instructions])
