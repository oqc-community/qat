from compiler_config.config import CompilerConfig, Tket

from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QASMFrontend

from tests.benchmarking.qatbm.qatbm import QatCollection


class BenchmarkingPass:
    def run(self, qatcol: QatCollection):
        pass

    def __hash__(self):
        return hash(self.__repr__())


class QasmBuilderPass(BenchmarkingPass):
    """
    A pass that parses a QASM string and builds the circuit in the builder.
    """

    def __init__(self, qasm_string: str):
        self.qasm_string = qasm_string

    def run(self, qatcol: QatCollection):
        # TODO: allow config to be passed...
        optim = Tket()
        config = CompilerConfig(optimizations=optim)

        frontend = QASMFrontend()
        builder, _ = frontend.parse(self.qasm_string, qatcol.model, config)
        qatcol.builder = builder

    def __repr__(self):
        return "QASM_Builder"


class CircuitBuilderPass(BenchmarkingPass):
    """
    A pass that takes a function which constructs a circuit directly in the
    circuit builder.

    The function must take the hardware model and a builder as function arguments,
    and mutate the builder directly.
    """

    def __init__(self, fn):
        self.fn = fn

    def run(self, qatcol: QatCollection):
        self.fn(qatcol.model, qatcol.builder)

    def __repr__(self):
        return "Circuit_Builder"


class OptimizationPass(BenchmarkingPass):
    """
    A pass that calls circuit optimization.
    """

    def run(self, qatcol: QatCollection):
        qatcol.builder._instructions = qatcol.engine.optimize(qatcol.builder.instructions)

    def __repr__(self):
        return "Optimization"


class ValidationPass(BenchmarkingPass):
    """
    A pass that calls circuit validation.
    """

    def run(self, qatcol: QatCollection):
        qatcol.engine.validate(qatcol.builder.instructions)

    def __repr__(self):
        return "Validation"


class EmitterPass(BenchmarkingPass):
    """
    A pass that calls the emiiter to emits classical instructions from the
    instruction list.
    """

    def run(self, qatcol: QatCollection):
        qatcol.qatfile = InstructionEmitter().emit(
            qatcol.builder.instructions, qatcol.model
        )

    def __repr__(self):
        return "Emitter"


class CreateTimelinePass(BenchmarkingPass):
    """
    A pass that calls create_duration_timeline.
    """

    def run(self, qatcol: QatCollection):
        qatcol.timeline = qatcol.engine.create_duration_timeline(
            qatcol.qatfile.instructions
        )

    def __repr__(self):
        return "Timeline"


class ExecutionPass(BenchmarkingPass):
    """
    A pass that executes the circuit using the hardware model.

    For now, runs engine.execute(), which incorporates passes such as EmitterPass,
    and CreateTimelinePass.
    """

    def run(self, qatcol: QatCollection):
        qatcol.results = qatcol.engine.execute(qatcol.builder.instructions)

    def __repr__(self):
        return "Execution"
