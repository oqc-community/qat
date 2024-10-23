# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from compiler_config.config import ErrorMitigationConfig, ResultsFormatting
from qiskit import QiskitError, QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import CheckMap
from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qat.purr.backends.echo import (
    Connectivity,
    add_direction_couplings_to_hardware,
    apply_setup_to_hardware,
    generate_connectivity,
)
from qat.purr.compiler.builders import Axis, InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, Qubit
from qat.purr.compiler.error_mitigation.readout_mitigation import get_readout_mitigation
from qat.purr.compiler.execution import InstructionExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Assign,
    Instruction,
    Return,
    Variable,
    is_generated_name,
)
from qat.purr.compiler.runtime import QuantumRuntime
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def get_default_qiskit_hardware(
    qubit_count=20,
    noise_model=None,
    strict_placement=True,
    connectivity: Optional[Union[Connectivity, List[Tuple[int, int]]]] = None,
) -> "QiskitHardwareModel":
    model = QiskitHardwareModel(qubit_count, noise_model)
    model.strict_placement = strict_placement

    if isinstance(connectivity, Connectivity):
        connectivity = generate_connectivity(connectivity, qubit_count)
    connectivity = connectivity or generate_connectivity(Connectivity.Ring, qubit_count)

    model = apply_setup_to_hardware(model, qubit_count, connectivity)
    return add_direction_couplings_to_hardware(model, connectivity)


class QiskitBuilder(InstructionBuilder):
    """Builder around QASM circuits."""

    def __init__(
        self,
        hardware_model: QuantumHardwareModel,
        qubit_count: int,
        instructions: InstructionBuilder = None,
    ):
        super().__init__(hardware_model=hardware_model, instructions=instructions)
        self.circuit = QuantumCircuit(qubit_count, qubit_count)
        self.shot_count = hardware_model.default_repeat_count
        self.bit_count = 0
        self.bit_ordering = {}

    def measure(self, target: Qubit, *args, **kwargs) -> "InstructionBuilder":
        self.circuit.measure(target.index, self.bit_count)
        # keep track of the ordering of measurements
        if len(args) >= 2:
            self.bit_ordering[args[1]] = self.bit_count
        self.bit_count = self.bit_count + 1
        return self

    def R(self, axis: Axis, target: Union[Qubit, PulseChannel], radii=None):
        qb, _ = self.model.resolve_qb_pulse_channel(target)
        if radii is None:
            radii = np.pi

        if axis == Axis.X:
            self.circuit.rx(radii, qb.index)
        elif axis == Axis.Y:
            self.circuit.ry(radii, qb.index)
        elif axis == Axis.Z:
            self.circuit.rz(radii, qb.index)

        return self

    def Z(self, target: Union[Qubit, PulseChannel], radii=np.pi):
        qb, _ = self.model.resolve_qb_pulse_channel(target)
        self.circuit.rz(radii, qb.index)
        return self

    def X(self, target: Union[Qubit, PulseChannel], radii=np.pi):
        qb, _ = self.model.resolve_qb_pulse_channel(target)
        self.circuit.rx(radii, qb.index)
        return self

    def Y(self, target: Union[Qubit, PulseChannel], radii=np.pi):
        qb, _ = self.model.resolve_qb_pulse_channel(target)
        self.circuit.ry(radii, qb.index)
        return self

    def swap(self, target: Qubit, destination: Qubit):
        self.circuit.swap(target.index, destination.index)
        return self

    def delay(self, target: Union[Qubit, PulseChannel], time: float):
        qb, _ = self.model.resolve_qb_pulse_channel(target)
        self.circuit.delay(time, qb.index)
        return self

    def cR(
        self,
        axis: Axis,
        controllers: Union[Qubit, List[Qubit]],
        target: Qubit,
        theta: float,
    ):
        if len(controllers) > 1:
            raise ValueError(
                "Cannot perform generic controlled operation across multiple qubits."
            )

        controllers = controllers[0]

        if axis == Axis.X:
            self.circuit.crx(theta, controllers, target.index)
        elif axis == Axis.Y:
            self.circuit.cry(theta, controllers, target.index)
        elif axis == Axis.Z:
            self.circuit.crz(theta, controllers, target.index)

        return self

    def cnot(self, control: Qubit, target_qubit: Qubit):
        self.circuit.cx(control.index, target_qubit.index)
        return self

    def ccnot(self, cone: Qubit, ctwo: Qubit, target_qubit: Qubit):
        self.circuit.ccx(cone.index, ctwo.index, target_qubit.index)
        return self

    def ECR(self, control: Qubit, target: Qubit):
        self.circuit.ecr(control.index, target.index)
        return self

    def had(self, qubit: Qubit):
        self.circuit.h(qubit.index)
        return self

    def reset(self, qubits: Union[Qubit, List[Qubit]]):
        if not isinstance(qubits, list):
            qubits = [qubits]

        for qb in qubits:
            self.circuit.reset(qb.index)

        return self

    def repeat(self, repeat_value: int = None, repetition_period=None):
        # Compiler config may attempt to pass None
        if repeat_value is not None:
            self.shot_count = repeat_value
        else:
            self.shot_count = self.model.default_repeat_count
        return self

    def clear(self):
        self.circuit.clear()
        self.bit_count = 0
        self.bit_ordering = {}

    def merge_builder(self, other_builder: InstructionBuilder):
        """
        overloaded merge_builder Ensure that circuits are copied too
        As well as the rest of the base merging operations.
        """
        self.circuit.compose(other_builder.circuit, inplace=True)
        self.bit_count = other_builder.bit_count
        self.bit_ordering = other_builder.bit_ordering

        return super().merge_builder(other_builder)


class QiskitHardwareModel(QuantumHardwareModel):
    def __init__(self, qubit_count, noise_model=None, strict_placement=True):
        self.qubit_count = qubit_count
        self.noise_model = noise_model
        self.strict_placement = strict_placement
        super().__init__()

    def create_runtime(self, existing_engine: InstructionExecutionEngine = None):
        # We aren't going to use the engine anyway.
        return QiskitRuntime(existing_engine or self.create_engine())

    def create_builder(self) -> InstructionBuilder:
        return QiskitBuilder(hardware_model=self, qubit_count=self.qubit_count)

    def create_engine(self) -> InstructionExecutionEngine:
        return QiskitEngine(hardware_model=self)


def verify_placement(coupling_map, circuit):
    """
    Check that the circuit can be directly mapped according to the
    wire map defined by the coupling map

    Raises if placement cannot happen without swaps
    """

    cmap = CouplingMap(coupling_map)
    checker = CheckMap(cmap)
    checker(circuit)
    if not checker.property_set["is_swap_mapped"]:
        raise RuntimeError(
            f"Cannot achieve placement on set couplings with operation:"
            f" {checker.property_set['check_map_msg']}"
        )


class QiskitEngine(InstructionExecutionEngine):
    def __init__(self, hardware_model: QiskitHardwareModel = None):
        super().__init__(hardware_model)

    def run_calibrations(self, qubits_to_calibrate=None):
        pass

    def execute(self, builder: QiskitBuilder) -> Dict[str, Any]:
        if not isinstance(builder, QiskitBuilder):
            raise ValueError(
                "Contravariance is not possible with QiskitEngine. "
                "execute input must be a QiskitBuilder"
            )

        coupling_map = None
        if any(self.model.qubit_direction_couplings):
            coupling_map = [
                list(coupling.direction)
                for coupling in self.model.qubit_direction_couplings
            ]

        circuit = builder.circuit
        if self.model.strict_placement:
            verify_placement(coupling_map, circuit)

        # With no coupling map the backend defaults to create couplings for qubit count, which
        # defaults to 30. So we change that.
        aer_config = AerBackendConfiguration.from_dict(
            AerSimulator._DEFAULT_CONFIGURATION | {"open_pulse": False}
        )
        aer_config.n_qubits = self.model.qubit_count
        qasm_sim = AerSimulator(aer_config, noise_model=builder.model.noise_model)

        try:
            job = qasm_sim.run(
                transpile(circuit, qasm_sim, coupling_map=coupling_map),
                shots=builder.shot_count,
            )
            results = job.result()
            distribution = results.get_counts(circuit)
        except QiskitError as e:
            raise ValueError(f"QiskitError while running Qiskit circuit: {str(e)}")

        # return the measurements in the correct order
        assigns = {}
        returns = []
        for inst in builder.instructions:
            if isinstance(inst, Assign):
                assigns[inst.name] = inst.value
            elif isinstance(inst, Return):
                returns.extend(inst.variables)
        # trim the qiskit returns to the number of measurements and revers the index order
        trimmed = {
            key[-builder.bit_count :][::-1]: value for key, value in distribution.items()
        }
        if len(returns) > 0:
            task_results = {}
            for creg in returns:
                creg_result = {}
                c_indices = [
                    builder.bit_ordering[ref.name] if isinstance(ref, Variable) else None
                    for ref in assigns[creg]
                ]
                for key, value in trimmed.items():
                    key_list = list(key)
                    new_key = "".join(
                        [
                            key_list[ind] if isinstance(ind, int) else "0"
                            for ind in c_indices
                        ]
                    )
                    if new_key in creg_result:
                        creg_result[new_key] += value
                    else:
                        creg_result[new_key] = value
                task_results[creg] = creg_result

            return task_results
        else:
            return trimmed

    def optimize(self, instructions: List[Instruction]) -> List[Instruction]:
        log.info("No optimize implemented for QiskitEngine")
        return instructions

    def validate(self, instructions: List[Instruction]):
        pass


cl2qu_index_pattern = re.compile(r"(.*)\[(?P<clbit_index>[0-9]+)\]_(?P<qubit_index>[0-9]+)")


def get_cl2qu_index_mapping(instructions):
    mapping = {}
    for instruction in instructions:
        if not isinstance(instruction, Assign):
            continue
        for value in instruction.value:
            result = cl2qu_index_pattern.match(value.name)
            if result is None:
                raise ValueError(
                    f"Could not extract cl register index from {instruction.output_variable}"
                )
            mapping[result.group("clbit_index")] = int(result.group("qubit_index"))
    return mapping


class QiskitRuntime(QuantumRuntime):

    def _apply_error_mitigation(self, results, instructions, error_mitigation):
        if error_mitigation is None:
            return results

        # TODO: add support for multiple registers
        # TODO: reconsider results length
        if len(results) > 1:
            raise ValueError(
                "Cannot have multiple registers in conjunction with readout error mitigation."
            )

        mapping = get_cl2qu_index_mapping(instructions)

        for mitigator in get_readout_mitigation(error_mitigation):
            new_result = mitigator.apply_error_mitigation(results, mapping, self.model)
            results[mitigator.name] = new_result

        return results  # TODO: new results object

    def execute(
        self,
        builder: QiskitBuilder,
        results_format: ResultsFormatting = None,
        repeats: int = None,
        error_mitigation: ErrorMitigationConfig = None,
    ):
        if not isinstance(builder, QiskitBuilder):
            raise ValueError("Wrong builder type passed to QASM runtime.")

        # TODO - add error_mitigation for Qasm sim
        results = self.engine.execute(builder)
        results = self._apply_error_mitigation(
            results, builder.instructions, error_mitigation
        )

        if all([is_generated_name(k) for k in results.keys()]):
            if len(results) == 1:
                return list(results.values())[0]
            else:
                squashed_results = list(results.values())
                if all(isinstance(val, np.ndarray) for val in squashed_results):
                    return np.array(squashed_results)
                return squashed_results
        else:
            return results
