import cProfile
import os
from collections import Counter
from time import perf_counter

from qat.purr.compiler.config import CompilerConfig, Tket
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QASMFrontend
from qatbm.models import echo_hardware
from qatbm.qasm import random_qasm

# parameters
nqubits = 10
ncnots = 100
profile_save = "profiles/N_qubit_qasm/"
tket_optim = False

if not os.path.exists(profile_save):
    os.makedirs(profile_save)

# Create hardware and scripts
hw = echo_hardware(nqubits)
qasm = random_qasm(nqubits, ncnots)
optim = Tket()
if not tket_optim:
    optim.disable()
else:
    optim.default()
config = CompilerConfig(optimizations=optim)

# Parse the QASM script
t1 = perf_counter()

frontend = QASMFrontend()
builder, _ = frontend.parse(qasm, hw, config)
print(f"Parsing \& Tket optimisations: {perf_counter() - t1}")
cProfile.run("frontend.parse(qasm, hw, config)", profile_save + "parse.prof")

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

count = 0
for inst in instructions_opt:
    if hasattr(inst, "quantum_targets"):
        count += len(inst.quantum_targets)


# Instruction counts
print(f"Average targets: {count / len(instructions_opt)}")
print(f"Optimised instructions: {len(instructions_opt)}")
print(
    f"Emitted instructions: {len(InstructionEmitter().emit(instructions_opt, hw).instructions)}"
)
print(f"Execution time: {max([val[-1].end for val in res.values()])}")
inst_types = [type(inst).__name__ for inst in instructions_opt]
print(Counter(inst_types))
