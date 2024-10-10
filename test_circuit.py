from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.devices import ChannelType
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.instructions import Acquire
from qat.purr.compiler.runtime import execute_instructions

hw = get_default_echo_hardware(3)
hw = get_default_RTCS_hardware(repeats=10)


config = CompilerConfig(
    results_format=QuantumResultsFormat().binary_count(),
    repeats=10,
)

for pc in hw.pulse_channels.values():
    print(pc.channel_type)
    if pc.channel_type == ChannelType.acquire:
        print(hw.get_devices_from_pulse_channel(pc.full_id()))


builder = hw.create_builder()
builder.measure(hw.get_qubit(0))
print("\n".join([str(inst) for inst in builder.instructions]))
print([type(inst) for inst in builder.instructions])

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
