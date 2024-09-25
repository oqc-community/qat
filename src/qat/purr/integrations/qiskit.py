# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import time
import uuid
from typing import List

from compiler_config.config import CompilerConfig
from qiskit import QuantumCircuit, assemble, qasm2, transpile
from qiskit.providers.basic_provider import BasicProviderJob, BasicSimulator
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.utils.logger import get_default_logger
from qat.qat import execute_qasm

logger = get_default_logger()


def run_qasm_circuit(circuit):
    backend_sim = BasicSimulator()
    transpiled_qc = transpile(circuit, backend_sim)
    result = backend_sim.run(transpiled_qc).result()

    def _results_to_binary(results_dictionary):
        """
        Key value from Qiskit is in hex, aka ``0x2: 242``.
        Transform to binary, sort by actual integer value.
        """
        binary = {bin(int(key, 16))[2:]: val for key, val in results_dictionary.items()}
        fill_value = max([len(val) for val in binary.keys()])
        return {
            key.zfill(fill_value): val
            for key, val in sorted(binary.items(), key=lambda kvp: int(kvp[0]))
        }

    # If we're a single value, just return, otherwise return list of results (or None).
    if not any(result.results):
        return None
    elif len(result.results) == 1:
        return _results_to_binary(result.results[0].data.counts)
    else:
        return [_results_to_binary(res.counts) for res in result.results]


def run_qasm_from_file(qasm_file: str):
    """Runs the passed-in QASM file against the QASM state simulator."""
    return run_qasm_circuit(QuantumCircuit.from_qasm_file(qasm_file))


def run_qasm(qasm_str: str):
    """Runs the passed-in QASM string against the QASM state simulator."""
    return run_qasm_circuit(QuantumCircuit.from_qasm_str(qasm_str))


class QatBackend(BasicSimulator):
    """
    Basic qiskit backend built to run QASM on our own backends.

    TODO: Expand this to become a proper back-end, as I don't believe we need to inherit
        off the QASM simulator as it stands.
    """

    def __init__(self, hardware=None, comp_config=None, **fields):
        super().__init__(**fields)
        self.hardware = hardware
        self.comp_config = comp_config

    def run(self, qobj, **backend_options):
        if not isinstance(qobj, List):
            qobj = [qobj]

        if not all(isinstance(val, QuantumCircuit) for val in qobj):
            raise ValueError(
                "Experiment list contains non QuantumCircuit objects which we can't "
                "process at this time."
            )

        options_keys = set(dir(self.options))
        assembled_data = assemble(
            qobj,
            self,
            **{key: val for key, val in backend_options.items() if key in options_keys},
        )

        if self.comp_config is None:
            self.comp_config = CompilerConfig()
        self.comp_config.repeats = assembled_data.config.shots
        self.comp_config.results_format.binary_count()

        # Merge the Qiskit header/config info together with the experiment as
        # represented as a QASM string
        experiment_info = [
            (
                qasm2.dumps(val),
                assembled_data.experiments[i].config,
                assembled_data.experiments[i].header,
            )
            for i, val in enumerate(qobj)
        ]

        results = []
        full_start = time.time()
        status = "COMPLETED"
        try:
            for circ, exp_config, header in experiment_info:
                target_hardware = self.hardware or get_default_RTCS_hardware()
                start = time.time()
                results_data = execute_qasm(circ, target_hardware, self.comp_config)
                end = time.time()

                # TODO: Massage results data into a form that should be returned. As
                #   we're count it should be hex binary count representation, x01: 5024
                #   etc.

                # Below, solution for a single classical register. Not sure how qat
                # deals with multiple cregs.
                if len(header.creg_sizes) == 1:
                    creg_label, _ = header.creg_sizes[0]
                    counts_qat = results_data[creg_label]
                    counts_qiskit = dict(
                        (hex(int(key[::-1], 2)), value)
                        for (key, value) in counts_qat.items()
                    )
                    results.append(
                        ExperimentResult(
                            self.comp_config.repeats,
                            True,
                            ExperimentResultData(counts=counts_qiskit),
                            status="DONE",
                            header=header,
                            name=header.name,
                            time_taken=end - start,
                        )
                    )
                else:
                    results.append(
                        ExperimentResult(
                            self.comp_config.repeats,
                            True,
                            ExperimentResultData(counts=results_data),
                            status="DONE",
                            header=header,
                            name=header.name,
                            time_taken=end - start,
                        )
                    )
        except Exception as ex:
            # Code further up infinitely recalls this method until it gets a result, so
            # catch and propagate message.
            status = str(ex)

        full_end = time.time()
        job_id = str(uuid.uuid4())
        return BasicProviderJob(
            self,
            job_id,
            Result(
                backend_name=self.name,
                backend_version=self._configuration.backend_version,
                qobj_id=assembled_data.qobj_id,
                job_id=job_id,
                results=results,
                status=status,
                success=any(results) and all(val.success for val in results),
                time_taken=full_end - full_start,
                header=assembled_data.header,
            ),
        )
