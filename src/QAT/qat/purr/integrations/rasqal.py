import math
from typing import Dict, Optional

from pytket._tket.passes import DefaultMappingPass, SequencePass

try:
    from rasqal.adaptors import BuilderAdaptor, RequiredFeatures, RuntimeAdaptor
    from rasqal.routing import TketBuilder, TketRuntime
    from rasqal.runtime import RasqalRunner

    rasqal_available = True
except:
    rasqal_available = False

    class BuilderAdaptor: ...

    class RuntimeAdaptor: ...

    class RasqalRunner: ...

    class TketRuntime: ...

    class RequiredFeatures: ...


from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import InlineResultsProcessing, QuantumResultsFormat
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Variable
from qat.purr.compiler.runtime import QuantumRuntime, get_builder, get_runtime
from qat.purr.integrations.tket import build_routing_architectures


class RasqalBuilder(BuilderAdaptor):
    def __init__(
        self,
        model: QuantumHardwareModel,
        results_format: InlineResultsProcessing,
    ):
        self.model: QuantumHardwareModel = model
        self.builder: InstructionBuilder = get_builder(model)
        self.results_format = results_format

        # As the results Rasqal desires is always a full QPU distribution just add a return to the end
        # of the builder, and only once in the case of multi-execution.
        self.acquire_variables = []
        self.added_return = False

    def _get_qubit(self, id_: int):
        return self.model.get_qubit(id_)

    def cx(self, controls, target, radii):
        # TODO: Hack as most builders have cnot but don't have generic controlled operations implemented yet.
        if len(controls) == 1 and radii == math.pi:
            self.builder.cnot(self._get_qubit(controls[0]), self._get_qubit(target))
        else:
            self.builder.cX(
                [self._get_qubit(c) for c in controls], self._get_qubit(target), radii
            )

    def cz(self, controls, target, radii):
        self.builder.cZ(
            [self._get_qubit(c) for c in controls], self._get_qubit(target), radii
        )

    def cy(self, controls, target, radii):
        self.builder.cY(
            [self._get_qubit(c) for c in controls], self._get_qubit(target), radii
        )

    def x(self, qubit, radii):
        self.builder.X(self._get_qubit(qubit), radii)

    def y(self, qubit, radii):
        self.builder.Y(self._get_qubit(qubit), radii)

    def z(self, qubit, radii):
        self.builder.Z(self._get_qubit(qubit), radii)

    def swap(self, qubit1, qubit2):
        self.builder.swap(self._get_qubit(qubit1), self._get_qubit(qubit2))

    def reset(self, qubit):
        self.builder.reset(self._get_qubit(qubit))

    def measure(self, qubit):
        # Create throwaway label and retrieve the name.
        result_var = self.builder.create_name()
        self.acquire_variables.append(result_var)
        self.builder.measure_single_shot_z(
            self._get_qubit(qubit), output_variable=result_var
        )
        self.builder.results_processing(result_var, self.results_format)

    def clear(self):
        self.builder.clear()
        self.acquire_variables.clear()


class RasqalRuntime(RuntimeAdaptor):
    def __init__(self, model: QuantumHardwareModel, results_format: QuantumResultsFormat):
        self.model = model
        self.runtime: QuantumRuntime = get_runtime(model)
        self.results_format: QuantumResultsFormat = results_format

    def execute(self, adaptor: RasqalBuilder) -> Dict[str, int]:
        if not adaptor.added_return:
            final_result_name = adaptor.builder.create_name()
            adaptor.builder.assign(
                final_result_name, [Variable(val) for val in adaptor.acquire_variables]
            )
            adaptor.builder.returns(final_result_name)
            adaptor.added_return = True

        results = self.runtime.execute(adaptor.builder, self.results_format.transforms)
        return results

    def create_builder(self) -> BuilderAdaptor:
        return RasqalBuilder(self.model, self.results_format.format)

    def has_features(self, required_features: RequiredFeatures):
        return True


class RasqalRouter(TketRuntime):
    """Wrapper around the Tket routing techniques to allow for incremental routing across a web of couplings."""

    def __init__(self, model: QuantumHardwareModel, forwarded_runtime: RuntimeAdaptor):
        # We don't use the couplings here anyway.
        super().__init__([(0, 1)], forwarded_runtime)
        self.model = model

    def execute(self, builder) -> Dict[str, int]:
        builder: TketBuilder
        potential_architectures = build_routing_architectures(self.model)

        optimizations_succeeded = False
        if any(potential_architectures):
            for subarchitecture in potential_architectures:
                if builder.circuit.n_qubits <= len(subarchitecture.nodes):
                    try:
                        SequencePass([DefaultMappingPass(subarchitecture)]).apply(
                            builder.circuit
                        )
                        optimizations_succeeded = True
                        break
                    except:
                        ...

            if not optimizations_succeeded:
                raise ValueError(
                    "Not able to route built circuit against appropriate mappings."
                )

        return self.forwarded.execute(self._forward_circuit(builder))


def create_runtime(
    model: QuantumHardwareModel,
    step_count: Optional[int] = 2500,
):
    if not rasqal_available:
        raise ValueError("Rasqal is not available.")

    # Apply routing with a custom router.
    runner = RasqalRunner(
        RasqalRouter(model, RasqalRuntime(model, QuantumResultsFormat().binary()))
    )

    # For now, we enforce a very limited step-count when building one with the default setup.
    if step_count is not None:
        runner.step_count_limit(step_count)

    return runner
