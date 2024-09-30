import cProfile
import os
from time import perf_counter

from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.instructions import Synchronize
from qatbm.circuit import random_circuit
from qatbm.models import echo_hardware

# parameters
nqubits = 10
ncnots = 100
profile_save = "profiles/N_qubit/"

if not os.path.exists(profile_save):
    os.makedirs(profile_save)

# Create hardware and scripts
hw = echo_hardware(nqubits)

# Create the circuit
t1 = perf_counter()
hw.create_builder()
builder = random_circuit(hw, ncnots)
print(f"Creating instructions: {perf_counter() - t1}")
cProfile.run("random_circuit(hw, ncnots)", profile_save + "build.prof")
num_before = len(builder.instructions)

# Optimise the pulse level instructions
hw_engine = hw.create_engine()
t1 = perf_counter()
instructions_opt = hw_engine.optimize(builder.instructions)
print(f"Pulse optimisation: {perf_counter() - t1}")
cProfile.run("hw_engine.optimize(builder.instructions)", profile_save + "optimise.prof")

# Validate the instructions
t1 = perf_counter()
hw_engine.validate(instructions_opt)
print(f"Validation: {perf_counter() - t1}")
cProfile.run("hw_engine.validate(instructions_opt)", profile_save + "validate.prof")

ctr = 0
comps = 0
for inst in instructions_opt:
    if isinstance(inst, Synchronize):
        comps += len(inst.quantum_targets)
        ctr += 1

# print(comps / ctr)

# Create an instruction timeline in absolute times
t1 = perf_counter()
# Remove the last instructions as they can't be dealt with?
res = hw_engine.create_duration_timeline(
    InstructionEmitter().emit(instructions_opt, hw).instructions
)
print(f"Instruction timeline: {perf_counter() - t1}")
cProfile.run(
    """res = hw_engine.create_duration_timeline(
    InstructionEmitter().emit(instructions_opt, hw)
    .instructions
    )""",
    profile_save + "timeline.prof",
)

# Instruction counts
# print(f"Pulse channels: {len(hw.pulse_channels)}")
# print(f"Pre-optimised instructions: {num_before}")
# print(f"Optimised instructions: {len(instructions_opt)}")
# print(f"Emitted instructions: {len(InstructionEmitter().emit(instructions_opt, hw).instructions)}")
# print(f"Execution time: {max([val[-1].end for val in res.values()])}")
inst_types = [type(inst).__name__ for inst in instructions_opt]
# print(Counter(inst_types))
