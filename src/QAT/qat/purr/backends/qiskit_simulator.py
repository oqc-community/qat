# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from typing import Any, Dict, List, Union

import numpy as np
from qiskit import QiskitError, QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qat.purr.backends.echo import (
    Connectivity,
    apply_setup_to_hardware,
    generate_connectivity,
)
from qat.purr.compiler.builders import Axis, InstructionBuilder
from qat.purr.compiler.config import ResultsFormatting
from qat.purr.compiler.devices import PulseChannel, Qubit
from qat.purr.compiler.execution import InstructionExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Instruction
from qat.purr.compiler.runtime import QuantumRuntime
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def get_default_qiskit_hardware(qubit_count=20, noise_model=None) -> "QiskitHardwareModel":
    model = QiskitHardwareModel(qubit_count, noise_model)

    # Copy the echo model builders way of doing things.
    connectivity = generate_connectivity(Connectivity.Ring, qubit_count)
    return apply_setup_to_hardware(model, qubit_count, connectivity)


class QiskitBuilder(InstructionBuilder):
    """Builder around QASM circuits."""

    def __init__(
        self,
        hardware_model: QuantumHardwareModel,
        qubit_count: int,
    ):
        super().__init__(hardware_model=hardware_model)
        self.circuit = QuantumCircuit(qubit_count, qubit_count)
        self.shot_count = 1024
        self.bit_count = 0

    def measure(self, target: Qubit, *args, **kwargs) -> "InstructionBuilder":
        self.circuit.measure(target.index, self.bit_count)
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
        self.circuit.cnot(control.index, target_qubit.index)
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
        return self

    def clear(self):
        self.circuit.clear()
        self.bit_count = 0

    def merge_builder(self, other_builder: InstructionBuilder):
        """
        overloaded merge_builder Ensure that circuits are copied too
        As well as the rest of the base merging operations.
        """
        self.circuit.compose(other_builder.circuit, inplace=True)
        self.bit_count = other_builder.bit_count

        return super().merge_builder(other_builder)


class QiskitHardwareModel(QuantumHardwareModel):
    def __init__(self, qubit_count, noise_model=None):
        self.qubit_count = qubit_count
        self.noise_model = noise_model
        super().__init__()

    def create_runtime(self, existing_engine: InstructionExecutionEngine = None):
        # We aren't going to use the engine anyway.
        return QiskitRuntime(existing_engine or self.create_engine())

    def create_builder(self) -> InstructionBuilder:
        return QiskitBuilder(hardware_model=self, qubit_count=self.qubit_count)

    def create_engine(self) -> InstructionExecutionEngine:
        return QiskitEngine(hardware_model=self)


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

        qasm_sim = AerSimulator(noise_model=builder.model.noise_model)
        circuit = builder.circuit
        # TODO: Needs a more nuanced try/catch. Some exceptions we should catch, others we should re-throw.
        try:
            job = qasm_sim.run(transpile(circuit, qasm_sim), shots=builder.shot_count)
            results = job.result()
            distribution = results.get_counts(circuit)
        except QiskitError as e:
            distribution = dict()

        removals = self.model.qubit_count - builder.bit_count

        # Because qiskit needs all values up-front we just provide a maximal classical register then strim off
        # the values we aren't going to use.
        return {key[removals:]: value for key, value in distribution.items()}

    def optimize(self, instructions: List[Instruction]) -> List[Instruction]:
        log.info("No optimize implemented for QiskitEngine")
        return instructions

    def validate(self, instructions: List[Instruction]):
        pass


class QiskitRuntime(QuantumRuntime):
    model: QiskitHardwareModel

    def execute(
        self,
        builder: QiskitBuilder,
        results_format: ResultsFormatting = None,
        repeats: int = None,
    ):
        if not isinstance(builder, QiskitBuilder):
            raise ValueError("Wrong builder type passed to QASM runtime.")

        return self.engine.execute(builder)
