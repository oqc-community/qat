# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import qiskit_experiments.library.randomized_benchmarking.standard_rb as rb
from qiskit import qasm2

from qat.purr.compiler.runtime import get_builder
from qat.purr.integrations.qasm import Qasm2Parser
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def randomized_benchmarking(hardware, nseeds, *args, **kwargs):
    """
    Generate randomized benchmarks. Please see Qiskits ``randomized_benchmarking_seq``
    for argument documentation. Returns a dictionary holding the QuantumProcess that
    relates to the various benchmarking runs and a list of the sequence lengths.
    """
    # Due to the variable return value we can't directly unpack.
    lengths = kwargs.pop("lengths", [1, 2, 4])
    physical_qubits = kwargs.pop("physical_qubits", [0])

    results = dict()
    index = 0
    for seed in range(nseeds):
        circuit_list = []
        qiskit_results = rb.StandardRB(
            physical_qubits=physical_qubits,
            lengths=lengths,
            num_samples=1,
            seed=seed,
            *args,
            **kwargs,
        )
        circuits = qiskit_results.circuits()
        for circuit in circuits:
            qasm = qasm2.dumps(circuit)
            circuit_list.append(Qasm2Parser().parse(get_builder(hardware), qasm))
        results[index] = circuit_list
        index = index + 1

    return results, lengths
