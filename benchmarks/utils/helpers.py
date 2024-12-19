# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from os import listdir, path
from pathlib import Path

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.frontends import QASMFrontend

from benchmarks.utils.models import get_mock_live_hardware

benchmarks_path = Path(__file__).parent.parent


# QASM2 Benchmarks
def load_qasm(qasm_file):
    path = Path(qasm_file)
    if not path.is_file() and not path.is_absolute():
        path = benchmarks_path.joinpath("qasm", path)
    with path.with_suffix(".qasm").open("r") as f:
        return f.read()


def all_qasm_files(num_qubits=2):
    """
    Returns a list of QASM file paths with a given number of qubits.
    """
    path = benchmarks_path.joinpath(Path(f"qasm/{num_qubits}qb/"))
    return [Path(f"{num_qubits}qb/" + f) for f in listdir(path)]


def load_experiments(
    num_qubits=[2, 10],
    return_circuit=True,
    return_builder=True,
    echo_hardware=True,
    mock_live_hardware=True,
    rtcs_hardware=True,
):
    """
    Fetch each QASM file for the specified qubit numbers, and returns a dict of experiments
    to use in benchmarking. Use the key word arguments to specify which hardware models to
    use, and to specify whether the circuit or builder should be stored in the experiement.
    """
    experiments = {}
    for qubits in num_qubits:
        # Prepare hardware models
        hw = {}
        if echo_hardware:
            hw["echo"] = get_default_echo_hardware(qubits)
        if mock_live_hardware:
            hw["mock_live"] = get_mock_live_hardware(qubits)
        if rtcs_hardware and qubits <= 2:
            # only use for two qubits due to its slow performance
            hw["rtcs"] = get_default_RTCS_hardware(qubits)

        # Create an experiment for each circuit
        qasm_files = all_qasm_files(qubits)
        for qasm_file in qasm_files:
            circuit = load_qasm(qasm_file)
            for hw_key, hw_val in hw.items():
                experiment = {"hardware": hw_val}
                if return_circuit:
                    experiment["circuit"] = qasm_file
                if return_builder:
                    experiment["builder"] = QASMFrontend().parse(circuit, hardware=hw_val)[
                        0
                    ]

                # use the name of the qasm file to specify the circuit key: delete .qasm
                circ_key = path.split(qasm_file)[1][:-5]
                experiments[f"{qubits}qb_{circ_key}[{hw_key}]"] = experiment

    return experiments
