# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat import QAT
from qat.backend.waveform_v1 import WaveformV1Backend, WaveformV1Program 
from qat.executables import Executable
from qat.engines.waveform_v1 import EchoEngine
from qat.pipelines import EchoCompile, EchoExecute, EchoPostProcessing
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.frontends import QASMFrontend


def load_qasm(N):
    with open(f"benchmarks/qasm/qft/qb_{N}.qasm", "r") as f:
        qasm_file = f.read()
    return qasm_file


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=5, group="QFT")
@pytest.mark.parametrize("mode", ["Legacy", "Experimental"])
@pytest.mark.parametrize("qubits", [2, 3, 4, 5])
class TestQFT:

    hw = get_default_echo_hardware(32)

    def test_compile_qasm(self, benchmark, qubits, mode):
        circuit = load_qasm(qubits)

        # Wrapper functions for benchmarking
        if mode == "Legacy":

            def run():
                builder, _ = QASMFrontend().parse(circuit, self.hw)
                builder.serialize()

        elif mode == "Experimental":
            qat = QAT()
            qat.add_pipeline(
                "test",
                compile_pipeline=EchoCompile(self.hw),
                execute_pipeline=EchoExecute(),
                postprocess_pipeline=EchoPostProcessing(self.hw),
                engine=EchoEngine(),
                emitter=WaveformV1Backend(self.hw),
            )

            def run():
                executable, _ = qat.compile(circuit, pipeline="test")
                executable.serialize()

        benchmark(run)
        assert True

    def test_execute_qasm(self, benchmark, qubits, mode):
        circuit = load_qasm(qubits)

        # Wrapper functions for benchmarking
        if mode == "Legacy":
            builder, _ = QASMFrontend().parse(circuit, self.hw)
            blob = builder.serialize()

            def run():
                builder = QuantumInstructionBuilder.deserialize(blob)
                QASMFrontend().execute(builder, self.hw)

        elif mode == "Experimental":
            qat = QAT()
            qat.add_pipeline(
                "test",
                compile_pipeline=EchoCompile(self.hw),
                execute_pipeline=EchoExecute(),
                postprocess_pipeline=EchoPostProcessing(self.hw),
                engine=EchoEngine(),
                emitter=WaveformV1Backend(self.hw),
            )

            executable, _ = qat.compile(circuit, pipeline="test")
            blob = executable.serialize()

            def run():
                executable = Executable.deserialize(blob)
                qat.execute(executable, pipeline="test")

        benchmark(run)
        assert True
