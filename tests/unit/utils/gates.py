# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import itertools
from inspect import isabstract

import numpy as np

from qat.ir.gates.base import GateBase
from qat.ir.gates.gates_1q import Gate1Q
from qat.ir.gates.gates_2q import Gate2Q
from qat.utils.state_tensors import StateOperator

test_angles = [0.0, np.pi / 2, -np.pi / 2, np.pi, -np.pi, 0.254, -25.4, 4.54]


def same_up_to_phase(gate1, gate2):
    r"""If gates :math:`U_{1}` and :math:`U_{2}` are equivalent upto a global phase,
    :math:`U_{2} = e^{i\alpha} U_{1}`, then

        :math:`U_{2}^{\dagger} U_{1} = e^{i\alpha}`.

    This fact can be used to test that this is true.
    """
    gate = np.conj(np.transpose(gate1)) @ gate2
    if np.isclose(gate[0, 0], 0.0 + 0.0j):
        return False
    return np.isclose(gate / gate[0, 0], np.eye(np.shape(gate1)[0])).all()


def get_non_abstract_subgates(cls: type):
    """Finds all non-abstract subclasses of a class."""
    classes = set()
    unchecked_classes = set([cls])
    while len(unchecked_classes) != 0:
        next_cls = next(iter(unchecked_classes))
        if not isabstract(next_cls):
            classes.add(next_cls)
        new_classes = next_cls.__subclasses__()
        for new_cls in new_classes:
            if not new_cls in classes:
                unchecked_classes.add(new_cls)
        unchecked_classes.remove(next_cls)
    return classes


def one_q_gate_tests():
    """Creates a list of 1Q gates to use for testings.

    Gates that are parameterised are generated for many different angles."""
    one_q_gates = get_non_abstract_subgates(Gate1Q)
    tests = []
    for gate in one_q_gates:
        params = {}
        fields = [key for key in gate.model_fields if key != "inst"]
        for field in fields:
            if field == "qubit":
                params[field] = [0]
            else:
                params[field] = test_angles
        keys, values = zip(*params.items())
        tests.extend([(gate, dict(zip(keys, v))) for v in itertools.product(*values)])
    return tests


def two_q_gate_tests():
    """Creates a list of 2Q gates to use for testings.

    Gates that are parameterised are generated for many different angles."""
    two_q_gates = get_non_abstract_subgates(Gate2Q)
    tests = []
    for gate in two_q_gates:
        params = {}
        fields = [key for key in gate.model_fields if key != "inst"]
        for field in fields:
            params["qubits"] = [(0, 1), (1, 0)]
            if field != "qubit1" and field != "qubit2":
                params[field] = test_angles
        keys, values = zip(*params.items())
        tests.extend([(gate, dict(zip(keys, v))) for v in itertools.product(*values)])
    return tests


def circuit_as_unitary(num_qubits, ir):
    """Calcualtes the unitary  for Qat IR composed of gates.
    Qubit operations are ignored, and control flow is not supported.
    """

    U = StateOperator(num_qubits)
    for inst in ir:
        if not isinstance(inst, GateBase):
            continue
        U.apply_gate(inst)
    return np.reshape(U.tensor, (2**num_qubits, 2**num_qubits))
