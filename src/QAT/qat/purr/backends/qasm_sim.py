# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from typing import List, Union

import numpy as np
from qat.purr.backends.echo import (
    Connectivity,
    apply_setup_to_hardware,
    generate_connectivity,
)
from qat.purr.compiler.builders import Axis, InstructionBuilder
from qat.purr.compiler.config import ResultsFormatting
from qat.purr.compiler.devices import PulseChannel, Qubit
from qat.purr.compiler.hardware_models import (
    QuantumHardwareModel,
    resolve_qb_pulse_channel,
)
from qat.purr.compiler.runtime import QuantumRuntime
from qiskit import Aer, QiskitError, QuantumCircuit, transpile


def get_default_qasm_hardware(qubit_count=20):
    model = QasmHardwareModel(qubit_count)

    # Copy the echo model builders way of doing things.
    connectivity = generate_connectivity(Connectivity.Ring, qubit_count)
    return apply_setup_to_hardware(model, qubit_count, connectivity)


class QasmBuilder(InstructionBuilder):
    """Builder around QASM circuits."""

    def __init__(self, qubit_count: int, hardware_model: QuantumHardwareModel):
        super().__init__(hardware_model)
        self.qubit_count = qubit_count
        self.circuit = QuantumCircuit(qubit_count, qubit_count)
        self.shot_count = 1024
        self.bit_count = 0

    def measure(self, target: Qubit, *args, **kwargs) -> "InstructionBuilder":
        self.circuit.measure(target.index, self.bit_count)
        self.bit_count = self.bit_count + 1
        return self

    def R(self, axis: Axis, target: Union[Qubit, PulseChannel], radii=None):
        qb, _ = resolve_qb_pulse_channel(target)
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
        qb, _ = resolve_qb_pulse_channel(target)
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

    def repeat(self, repeat_value: int, repetition_period=None):
        self.shot_count = repeat_value
        return self

    def clear(self):
        self.circuit = QuantumCircuit(self.qubit_count, self.qubit_count)
        self.bit_count = 0


class QasmHardwareModel(QuantumHardwareModel):
    def __init__(self, qubit_count):
        self.qubit_count = qubit_count
        super().__init__()

    def create_runtime(self, existing_engine=None):
        # We aren't going to use the engine anyway.
        return QasmRuntime(existing_engine or self.create_engine())

    def create_builder(self) -> "InstructionBuilder":
        return QasmBuilder(self.qubit_count, self)


class QasmRuntime(QuantumRuntime):
    model: QasmHardwareModel

    def execute(self, builder: QasmBuilder, results_format: ResultsFormatting = None):
        if not isinstance(builder, QasmBuilder):
            raise ValueError("Wrong builder type passed to QASM runtime.")

        qasm_sim = Aer.get_backend("qasm_simulator")
        circuit = builder.circuit

        # TODO: Needs a more nuanced try/catch. Some exceptions we should catch, others we should re-throw.
        try:
            job = qasm_sim.run(transpile(circuit, qasm_sim), builder.shot_count)
            results = job.result()
            distribution = results.get_counts(circuit)
        except QiskitError as e:
            distribution = dict()

        removals = self.model.qubit_count - builder.bit_count

        # Because qiskit needs all values up-front we just provide a maximal classical register then strim off
        # the values we aren't going to use.
        return {key[removals:]: value for key, value in distribution.items()}
