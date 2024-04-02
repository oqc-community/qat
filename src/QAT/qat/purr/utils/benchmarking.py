# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import qiskit.ignis.verification.randomized_benchmarking as rb

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
    qiskit_results = rb.randomized_benchmarking_seq(nseeds, *args, **kwargs)
    if len(qiskit_results) > 2:
        raise NotImplementedError("Only standard rb currently supported.")

    rb_circs = qiskit_results[0]
    seq_lengths = qiskit_results[1]
    results = dict()
    index = 0
    for seed_iter in rb_circs:
        circuit_list = []
        for circuit in seed_iter:
            qasm = circuit.qasm()
            circuit_list.append(Qasm2Parser().parse(get_builder(hardware), qasm))
        results[index] = circuit_list
        index = index + 1

    return results, seq_lengths
