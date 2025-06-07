# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import re
import shutil
from random import random
from typing import Dict

import numpy as np
import pytest
from compiler_config.config import (
    CompilerConfig,
    ErrorMitigationConfig,
    Languages,
    QuantumResultsFormat,
    Tket,
    TketOptimizations,
    get_optimizer_config,
)

import qat.pipelines.echo
from qat import QAT
from qat.backend.fallthrough import FallthroughBackend
from qat.backend.waveform_v1 import WaveformV1Backend
from qat.core.config.configure import get_config
from qat.core.config.descriptions import (
    HardwareLoaderDescription,
    PipelineFactoryDescription,
    PipelineInstanceDescription,
)
from qat.core.config.session import QatSessionConfig
from qat.core.config.validators import MismatchingHardwareModelException
from qat.core.pass_base import PassManager
from qat.core.pipeline import Pipeline
from qat.engines import NativeEngine
from qat.engines.waveform_v1 import EchoEngine
from qat.frontend import AutoFrontend, DefaultFrontend, FallthroughFrontend
from qat.middleend.middleends import FallthroughMiddleend
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.echo import get_pipeline as get_echo_pipeline
from qat.pipelines.legacy.echo import get_pipeline as get_legacy_echo_pipeline
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QIRFrontend
from qat.purr.compiler.hardware_models import ErrorMitigation, ReadoutMitigation
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    Assign,
    Instruction,
    PostProcessing,
    ResultsProcessing,
    Return,
    Variable,
)
from qat.purr.qat import _return_or_build, fetch_frontend
from qat.purr.qatconfig import QatConfig
from qat.runtime import LegacyRuntime, SimpleRuntime
from qat.runtime.executables import ChannelExecutable, Executable

from tests.unit.utils.qasm_qir import (
    ProgramFileType,
    get_all_qasm2_paths,
    get_all_qasm3_paths,
    get_all_qir_paths,
    get_test_file_path,
)

# Files that are not expected to pass tests are skipped.
skip_qasm2 = [
    "over_index_register.qasm",
    "invalid_w3x_couplings.qasm",
    "20qb.qasm",
    "huge.qasm",
    "mid_circuit_measure.qasm",
    "example_if.qasm",
]


skip_qasm3 = [
    "invalid_version.qasm",
    "no_header.qasm",
    "invalid_waveform.qasm",
    "invalid_port.qasm",
    "invalid_syntax.qasm",
    "invalid_frames.qasm",
    "invalid_couplings.qasm",
    "u_gate.qasm",
]


skip_qir = [
    "teleportchain.ll",
    "bell_qir_measure.bc",
    "cudaq-ghz.ll",  # test is designed to fail for no TKET optims
    "base64_bitcode_ghz",
]


skip_exec = [
    (ProgramFileType.QASM3, "lark_parsing_test.qasm"),
    (ProgramFileType.QASM3, "ecr_override_test.qasm"),
    (ProgramFileType.QASM3, "tmp.qasm"),
    (ProgramFileType.QASM3, "cx_override_test.qasm"),
    (ProgramFileType.QASM3, "cnot_override_test.qasm"),
    (ProgramFileType.QASM3, "invalid_pulse_length.qasm"),
    (ProgramFileType.QIR, "complicated.ll"),
    (ProgramFileType.QIR, "base_profile_ops.ll"),
]


multi_reg_files = [
    "qasm2-split_measure_assign.qasm",
    "qasm2-basic_results_formats.qasm",
    "qasm2-ordered_cregs.qasm",
    "qasm2-example.qasm",
]


no_acquire_files = [
    "qasm3-delay.qasm",
    "qasm3-ecr_test.qasm",
    "qasm3-redefine_defcal.qasm",
    "qasm3-arb_waveform.qasm",
    "qasm3-complex_gates_test.qasm",
]


skip_qasm2 = [
    get_test_file_path(ProgramFileType.QASM2, file_name) for file_name in skip_qasm2
]
qasm2_files = set(get_all_qasm2_paths()) - set(skip_qasm2)

skip_qasm3 = [
    get_test_file_path(ProgramFileType.QASM3, file_name) for file_name in skip_qasm3
]
qasm3_files = set(get_all_qasm3_paths()) - set(skip_qasm3)

skip_qir = [get_test_file_path(ProgramFileType.QIR, file_name) for file_name in skip_qir]
qir_files = set(get_all_qir_paths()) - set(skip_qir)


def equivalent_vars(self, other):
    """Used to overload '__eq__' so that a==b if they are equivalent."""
    return isinstance(self, type(other)) and (vars(self) == vars(other))


gen_name_pattern = re.compile(r"generated_name_[0-9]+")


def _strip_generated_rng(o):
    """Strip away the random number part of 'generated_name_4826354253464' allowing for
    simple equivalency check."""
    if isinstance(o, dict):
        for k, v in o.items():
            o[k] = _strip_generated_rng(v)
        return o
    elif isinstance(o, list):
        return [_strip_generated_rng(i) for i in o]
    elif isinstance(o, str):
        return gen_name_pattern.sub("generated_name", o)
    else:
        return o


def equivalent_generated(self, other):
    """Like 'equivalent_vars' but does not care about the random number part of
    generated names."""
    s_vars = _strip_generated_rng(vars(self))
    o_vars = _strip_generated_rng(vars(other))
    for d in [s_vars, o_vars]:
        for k, v in d.items():
            if isinstance(v, str):
                d[k] = gen_name_pattern.sub("generated_name", v)
    return isinstance(self, type(other)) and (s_vars == o_vars)


def short_file_name(path):
    return "-".join([path.parent.name, path.name])


skip_exec = [get_test_file_path(file_type, file_name) for file_type, file_name in skip_exec]
execution_files = (qasm2_files | qasm3_files | qir_files) - set(skip_exec)


def generate_random_linear(qubit_indices, random_data=True):
    output = {}
    for index in qubit_indices:
        random_0 = random() if random_data else 1
        random_1 = random() if random_data else 1
        output[str(index)] = {
            "0|0": random_0,
            "1|0": 1 - random_0,
            "1|1": random_1,
            "0|1": 1 - random_1,
        }
    return output


def apply_error_mitigation_setup(hw):
    lin_mit = generate_random_linear([q.index for q in hw.qubits])
    hw.error_mitigation = ErrorMitigation(ReadoutMitigation(linear=lin_mit))
    return hw


@pytest.mark.parametrize(
    "compiler_config",
    [
        pytest.param(None, id="No_config"),
        pytest.param(CompilerConfig(), id="Default_config"),
        pytest.param(
            CompilerConfig(optimizations=Tket(TketOptimizations.Two)),
            id="TketOptimizations.Two",
        ),
        pytest.param(
            CompilerConfig(
                results_format=QuantumResultsFormat().binary_count(),
                error_mitigation=ErrorMitigationConfig.LinearMitigation,
            ),
            id="ErrorMitigation",
        ),
    ],
)
@pytest.mark.parametrize(
    "hardware_model",
    [pytest.param(apply_error_mitigation_setup(EchoModelLoader(32).load()), id="Echo32Q")],
)
class TestQATParity:
    """Test that qat.core.QAT generates the same outputs as the qat.purr stack."""

    _purr_compiled = {}

    def purr_compile(self, request, file_string, hardware=None, compiler_config=None):
        key = request.node.callspec.id
        if key not in self._purr_compiled.keys():
            frontend = fetch_frontend(file_string)
            self._purr_compiled[key] = _return_or_build(
                file_string,
                frontend.parse,
                hardware=hardware,
                compiler_config=compiler_config,
            )
        return self._purr_compiled[key]

    @pytest.mark.parametrize(
        "qasm_file",
        qasm2_files,
        ids=short_file_name,
    )
    def test_qat_compile_qasm2(
        self, request, qasm_file: str, hardware_model, compiler_config, monkeypatch
    ):
        """Test that the same instructions are generated with both stacks."""

        monkeypatch.setattr(Instruction, "__eq__", equivalent_vars)
        monkeypatch.setattr(Variable, "__eq__", equivalent_vars)
        qasm_file_string = str(qasm_file)
        purr_ib, purr_metrics = self.purr_compile(
            request,
            qasm_file_string,
            hardware=hardware_model,
            compiler_config=compiler_config,
        )

        pipe = Pipeline(
            name="test",
            frontend=AutoFrontend(hardware_model),
            middleend=FallthroughMiddleend(),
            backend=FallthroughBackend(),
            runtime=LegacyRuntime(
                engine=hardware_model.create_engine(), results_pipeline=PassManager()
            ),
            model=hardware_model,
        )

        q = QAT()
        q.pipelines.add(pipe)

        qat_ib, qat_metrics = q.compile(
            qasm_file_string, compiler_config=compiler_config, pipeline="test"
        )

        assert purr_ib.instructions == qat_ib.instructions
        assert purr_metrics.optimized_circuit == qat_metrics.optimized_circuit

    @pytest.mark.parametrize(
        "qasm_file",
        qasm3_files,
        ids=short_file_name,
    )
    def test_qat_compile_qasm3(
        self, request, qasm_file: str, hardware_model, compiler_config, monkeypatch
    ):
        """Test that the same instructions are generated with both stacks."""

        monkeypatch.setattr(Instruction, "__eq__", equivalent_vars)
        monkeypatch.setattr(Variable, "__eq__", equivalent_vars)
        monkeypatch.setattr(Acquire, "__eq__", equivalent_generated)
        monkeypatch.setattr(PostProcessing, "__eq__", equivalent_generated)
        monkeypatch.setattr(ResultsProcessing, "__eq__", equivalent_generated)
        qasm_file_string = str(qasm_file)
        purr_ib, purr_metrics = self.purr_compile(
            request,
            qasm_file_string,
            hardware=hardware_model,
            compiler_config=compiler_config,
        )

        pipe = Pipeline(
            name="test",
            frontend=AutoFrontend(hardware_model),
            middleend=FallthroughMiddleend(),
            backend=FallthroughBackend(),
            runtime=LegacyRuntime(
                engine=hardware_model.create_engine(), results_pipeline=PassManager()
            ),
            model=hardware_model,
        )

        q = QAT()
        q.pipelines.add(pipe)
        qat_ib, qat_metrics = q.compile(
            qasm_file_string, compiler_config=compiler_config, pipeline="test"
        )
        assert purr_ib.instructions == qat_ib.instructions
        assert purr_metrics.optimized_circuit == qat_metrics.optimized_circuit

    @pytest.mark.parametrize(
        "qir_file",
        qir_files,
        ids=short_file_name,
    )
    def test_qat_compile_qir(
        self, request, qir_file: str, hardware_model, compiler_config, monkeypatch
    ):
        """Test that the same instructions are generated with both stacks."""

        # Tket placement is not yet supported for QIR in the experimental stack: we need to
        # infer from the compiler config which optimizations will be performed and skip if
        # placement is happening.
        if not compiler_config or not compiler_config.optimizations:
            optim_config = get_optimizer_config(Languages.QIR)
        else:
            optim_config = compiler_config.optimizations
        if isinstance(optim_config, Tket) and TketOptimizations.Empty not in optim_config:
            pytest.skip(
                "QIR with placement is not yet implemented in the experimental stack."
            )

        monkeypatch.setattr(Instruction, "__eq__", equivalent_vars)
        monkeypatch.setattr(Variable, "__eq__", equivalent_vars)
        monkeypatch.setattr(Acquire, "__eq__", equivalent_generated)
        monkeypatch.setattr(Assign, "__eq__", equivalent_generated)
        monkeypatch.setattr(PostProcessing, "__eq__", equivalent_generated)
        monkeypatch.setattr(ResultsProcessing, "__eq__", equivalent_generated)
        monkeypatch.setattr(Return, "__eq__", equivalent_generated)
        qir_file_string = str(qir_file)
        purr_ib, purr_metrics = self.purr_compile(
            request,
            qir_file_string,
            hardware=hardware_model,
            compiler_config=compiler_config,
        )

        pipe = Pipeline(
            name="test",
            frontend=AutoFrontend(hardware_model),
            middleend=FallthroughMiddleend(),
            backend=FallthroughBackend(),
            runtime=LegacyRuntime(
                engine=hardware_model.create_engine(), results_pipeline=PassManager()
            ),
            model=hardware_model,
        )

        q = QAT()
        q.pipelines.add(pipe)
        qat_ib, qat_metrics = q.compile(
            qir_file_string, compiler_config=compiler_config, pipeline="test"
        )
        assert purr_ib.instructions == qat_ib.instructions
        assert purr_metrics.optimized_circuit == qat_metrics.optimized_circuit

    @pytest.mark.parametrize("source_file", execution_files, ids=short_file_name)
    def test_qat_execute(
        self, request, source_file, hardware_model, compiler_config, monkeypatch
    ):
        if (
            compiler_config is not None
            and (em := compiler_config.error_mitigation) is not None
        ):
            if ErrorMitigationConfig.LinearMitigation in em:
                short_name = short_file_name(source_file)
                if short_name in multi_reg_files:
                    pytest.skip("ErrorMitigation not implemented for multiple registers.")
                elif short_name.startswith("qir-"):
                    pytest.skip("ErrorMitigation not implemented for QIR files.")
                elif short_name in no_acquire_files:
                    pytest.skip(
                        "ErrorMitigation not implemented for program with no acquire."
                    )
        instructions, _ = self.purr_compile(
            request,
            str(source_file),
            hardware=hardware_model,
            compiler_config=compiler_config,
        )
        purr_res, purr_metrics = QIRFrontend().execute(
            instructions, hardware=hardware_model, compiler_config=compiler_config
        )

        pipe = get_legacy_echo_pipeline(hardware_model, name="test")
        pipe = pipe.model_copy(update={"frontend": FallthroughFrontend()})

        q = QAT()
        q.pipelines.add(pipe)

        pkg, comp_metrics = q.compile(
            instructions, compiler_config=compiler_config, pipeline="test"
        )

        qat_res, _ = q.execute(pkg, compiler_config=compiler_config, pipeline="test")

        assert purr_res == qat_res
        assert purr_metrics.optimized_circuit == comp_metrics.optimized_circuit


class MockEngine(NativeEngine):
    def execute(self, package: Executable) -> Dict[str, np.ndarray]:
        return {}

    def startup(self): ...
    def shutdown(self): ...


@pytest.fixture
def echo_pipeline(qubit_count=32):
    model = EchoModelLoader(qubit_count=qubit_count).load()
    yield qat.pipelines.echo.get_pipeline(model)


@pytest.fixture
def fallthrough_pipeline(qubit_count=32):
    model = EchoModelLoader(qubit_count=qubit_count).load()
    yield Pipeline(
        name="fallthrough",
        frontend=FallthroughFrontend(),
        middleend=FallthroughMiddleend(),
        backend=FallthroughBackend(),
        runtime=SimpleRuntime(engine=MockEngine()),
        model=model,
    )


class TestQATPipelineSetup:
    def test_make_qat(self):
        q = QAT()
        assert set(q.pipelines.list()) == {"echo8", "echo16", "echo32"}
        assert q.pipelines.default == "echo32"
        assert q.pipelines.get("default").name == "echo32"

    def test_qat_session_respects_globalqat_config(self):
        qatconfig = get_config()
        OLD_LIMIT = qatconfig.MAX_REPEATS_LIMIT
        NEW_LIMIT = 324214
        assert OLD_LIMIT != NEW_LIMIT
        qatconfig.MAX_REPEATS_LIMIT = NEW_LIMIT

        q = QAT()
        assert set(q.pipelines.list()) == {"echo8", "echo16", "echo32"}
        assert q.pipelines.default == "echo32"
        assert q.pipelines.get("default").name == "echo32"
        assert q.config.MAX_REPEATS_LIMIT == NEW_LIMIT
        qatconfig.MAX_REPEATS_LIMIT = OLD_LIMIT

    def test_qat_session_extends_qatconfig_instance(self):
        qatconfig = QatConfig()
        qatconfig.MAX_REPEATS_LIMIT = 44325
        assert get_config().MAX_REPEATS_LIMIT != qatconfig.MAX_REPEATS_LIMIT

        q = QAT(qatconfig=qatconfig)
        assert set(q.pipelines.list()) == {"echo8", "echo16", "echo32"}
        assert q.pipelines.default == "echo32"
        assert q.pipelines.get("default").name == "echo32"
        assert q.config.MAX_REPEATS_LIMIT == qatconfig.MAX_REPEATS_LIMIT

    def test_make_qatconfig_list(self):
        pipelines = [
            PipelineInstanceDescription(
                name="echo8i", pipeline="qat.pipelines.echo.echo8", default=True
            ),
            PipelineInstanceDescription(
                name="echo16i", pipeline="qat.pipelines.echo.echo16"
            ),
            PipelineInstanceDescription(
                name="echo32i", pipeline="qat.pipelines.echo.echo32"
            ),
            PipelineFactoryDescription(
                name="echo6b",
                pipeline="qat.pipelines.echo.get_pipeline",
                hardware_loader="echo6loader",
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="echo6loader",
                type="qat.model.loaders.legacy.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]

        q = QAT(qatconfig=QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware))
        assert set(q.pipelines.list()) == {"echo8i", "echo16i", "echo32i", "echo6b"}

    def test_make_qatconfig_yaml(self, testpath):
        path = str(testpath / "files/qatconfig/pipelines.yaml")
        q = QAT(qatconfig=path)
        assert set(q.pipelines.list()) == {
            "echo8i",
            "echo16i",
            "echo6b",
            "echo-defaultfrontend",
            "echocustomconfig",
        }
        assert type(q.pipelines.get("echo-defaultfrontend").frontend) is DefaultFrontend
        assert q.pipelines.get("echocustomconfig").runtime.engine.x == 10

    def test_make_qatconfig_yaml_curdir(self, testpath, tmpdir, monkeypatch):
        config_file = testpath / "files/qatconfig/pipelines.yaml"
        q1 = QAT()
        monkeypatch.chdir(tmpdir)
        shutil.copy(config_file, tmpdir / "qatconfig.yaml")
        q2 = QAT()
        assert set(q2.pipelines.list()) != set(q1.pipelines.list())
        assert set(q2.pipelines.list()) == {
            "echo8i",
            "echo16i",
            "echo6b",
            "echo-defaultfrontend",
            "echocustomconfig",
        }

        assert type(q2.pipelines.get("echo-defaultfrontend").frontend) is DefaultFrontend
        assert q2.pipelines.get("echocustomconfig").runtime.engine.x == 10

    def test_FallthroughFrontend(self):
        frontend = FallthroughFrontend()
        assert frontend.model is None

    def test_FallthroughMiddleend(self):
        middleend = FallthroughMiddleend()
        assert middleend.model is None

    def test_FallthroughBackend(self):
        backend = FallthroughBackend()
        assert backend.model is None
        assert backend.emit(ir="blah") == "blah"

    def test_make_pipeline(self):
        qubit_count = 32
        model = EchoModelLoader(qubit_count=qubit_count).load()
        frontend = FallthroughFrontend()
        middleend = FallthroughMiddleend()
        backend = WaveformV1Backend(model)
        runtime = SimpleRuntime(engine=EchoEngine())

        Pipeline(
            name="default",
            frontend=frontend,
            middleend=middleend,
            backend=backend,
            runtime=runtime,
            model=model,
        )

    def test_add_pipeline(self, echo_pipeline):
        q = QAT()
        q.pipelines.add(echo_pipeline)
        assert echo_pipeline.name in q.pipelines.list()

    def test_remove_pipeline(self, echo_pipeline):
        q = QAT()
        q.pipelines.add(echo_pipeline)
        assert echo_pipeline.name in q.pipelines.list()
        q.pipelines.remove(echo_pipeline)
        assert echo_pipeline.name not in q.pipelines.list()

    def test_add_pipeline_set_default(self, fallthrough_pipeline):
        pipe1 = fallthrough_pipeline.model_copy(update={"name": "pipe1"})
        pipe2 = fallthrough_pipeline.model_copy(update={"name": "pipe2"})
        q = QAT()
        q.pipelines.add(pipe1, default=True)
        assert q.pipelines.default == pipe1.name
        q.pipelines.add(pipe2)
        assert q.pipelines.default == pipe1.name

    def test_set_default(self, fallthrough_pipeline):
        pipe1 = fallthrough_pipeline.model_copy(update={"name": "pipe1"})
        pipe2 = fallthrough_pipeline.model_copy(update={"name": "pipe2"})
        pipe3 = fallthrough_pipeline.model_copy(update={"name": "pipe3"})
        q = QAT()
        q.pipelines.add(pipe1)
        q.pipelines.add(pipe2)
        q.pipelines.add(pipe3)
        q.pipelines.set_default("pipe1")
        assert q.pipelines.default == "pipe1"
        q.pipelines.set_default("pipe2")
        assert q.pipelines.default == "pipe2"
        q.pipelines.set_default("pipe1")
        assert q.pipelines.default == "pipe1"

    def test_compile(self, fallthrough_pipeline):
        src = "test"
        q = QAT()
        q.pipelines.add(fallthrough_pipeline)
        pkg, _ = q.compile(src, pipeline="fallthrough")
        assert pkg == src

    def test_execute(self, fallthrough_pipeline):
        pkg = ChannelExecutable()
        q = QAT()
        q.pipelines.add(fallthrough_pipeline)
        res, _ = q.execute(pkg, pipeline="fallthrough")
        assert res == {}


def get_echo_model_without_acquire_delays():
    """Creates an echo model without acquire delays for comparison with legacy code."""
    model = EchoModelLoader(32).load()
    model.calibration_id = "default_id"
    model.default_repeat_count = TargetData.default().default_shots
    model.repeat_limit = TargetData.default().max_shots
    model.default_repetition_period = TargetData.default().QUBIT_DATA.passive_reset_time
    sample_time_qubit = TargetData.default().QUBIT_DATA.sample_time
    sample_time_resonator = TargetData.default().RESONATOR_DATA.sample_time
    for qubit in model.qubits:
        qubit.measure_acquire["delay"] = 0.0
        qubit.physical_channel.sample_time = sample_time_qubit
    for resonator in model.resonators:
        resonator.physical_channel.sample_time = sample_time_resonator
    return model


class TestQatEchoPipelines:
    """
    Parity tests specifically targeted at the echo engine in `purr` and new QAT. Aims to test:

        - New echo pipelines: Does compile and execute work in the way we expect using the new
          QAT infrastructure?
        - Code generation to a "waveform" executable: do the buffers we send to the target machine
          match those generated by legacy execution engines?
        - Execution and results processing: are the results outputted in consistent formats?
    """

    model = get_echo_model_without_acquire_delays()
    config = CompilerConfig(
        results_format=QuantumResultsFormat().binary_count(),
        repeats=TargetData.default().default_shots,
        repetition_period=TargetData.default().QUBIT_DATA.passive_reset_time,
    )
    core: QAT = None
    legacy_ir = {}
    executables = {}

    def setup_class(self):
        """Setup the QAT instance and pipelines for the tests."""
        self.core = QAT()
        pipe = get_legacy_echo_pipeline(self.model, "legacy")
        self.core.pipelines.add(pipe)
        pipe = get_echo_pipeline(self.model, "new")
        self.core.pipelines.add(pipe)

    def get_legacy_ir(self, request, qasm_file):
        key = request.node.callspec.id
        if key not in self.legacy_ir:
            legacy_ir, _ = self.core.compile(str(qasm_file), self.config, pipeline="legacy")
            self.legacy_ir[key] = legacy_ir

        return self.legacy_ir[key]

    def get_executable(self, request, qasm_file):
        key = request.node.callspec.id
        if key not in self.executables:
            executable, _ = self.core.compile(str(qasm_file), self.config, pipeline="new")
            self.executables[key] = executable

        return self.executables[key]

    @pytest.mark.parametrize(
        "qasm_file",
        qasm2_files,
        ids=short_file_name,
    )
    def test_code_generation(self, request, qasm_file):
        ir = self.get_legacy_ir(request, qasm_file)
        executable = self.get_executable(request, qasm_file)

        engine = self.model.create_engine()
        qatfile = InstructionEmitter().emit(ir.instructions, self.model)
        position_map = engine.create_duration_timeline(qatfile.instructions)
        buffers = engine.build_pulse_channel_buffers(position_map)
        buffers = engine.build_physical_channel_buffers(buffers)
        acquires = engine.build_acquire_list(position_map)

        assert buffers.keys() == executable.channel_data.keys()
        for key in buffers:
            purr_buffer = buffers[key]
            qat_buffer = executable.channel_data[key].buffer
            assert len(purr_buffer) <= len(qat_buffer)
            idx = min(len(purr_buffer), len(qat_buffer))
            excess = len(qat_buffer) - len(purr_buffer)
            if len(purr_buffer) > 0:
                assert np.allclose(purr_buffer[0:idx], qat_buffer[0:idx])
                assert np.allclose(qat_buffer[idx:], np.zeros(excess))

        # Acquires require more care: purr organizes acquires by pulse channel. We have
        # made a purposeful choice in the experimental code to organize by physical channel.
        # Lets make sure these match up
        for acquire_purr in acquires.values():
            if len(acquire_purr) == 0:
                continue
            assert len(acquire_purr) == 1
            acquire_purr = acquire_purr[0]
            acquire = executable.channel_data[
                acquire_purr.physical_channel.full_id()
            ].acquires
            assert len(acquire) == 1
            acquire = acquire[0]
            assert acquire_purr.samples == acquire.length
            assert acquire_purr.start == acquire.position

            # the API for acquire modes and echo engines have drifted: `purr` converts
            # INTEGRATOR -> RAW as it cannot deal with INTEGRATOR. new engines do not have
            # this restriction.
            assert acquire_purr.mode == AcquireMode.RAW
            assert acquire.mode == acquire.mode.INTEGRATOR

    @pytest.mark.parametrize(
        "qasm_file",
        qasm2_files,
        ids=short_file_name,
    )
    def test_results_formatting(self, request, qasm_file):
        ir = self.get_legacy_ir(request, qasm_file)
        package = self.get_executable(request, qasm_file)

        purr_results, _ = self.core.execute(ir, self.config, pipeline="legacy")
        echo_results, _ = self.core.execute(package, self.config, pipeline="new")
        assert purr_results.keys() == echo_results.keys()

        for key in purr_results.keys():
            assert purr_results[key] == echo_results[key]
        assert purr_results.keys() == echo_results.keys()

    @pytest.mark.parametrize(
        "qasm_file",
        qasm2_files,
        ids=short_file_name,
    )
    def test_qat_execute_rejects_different_hw_models(self, request, qasm_file):
        package = self.get_executable(request, qasm_file)
        package.calibration_id = "different_id"

        with pytest.raises(MismatchingHardwareModelException):
            self.core.execute(package, pipeline="new")

    @pytest.mark.parametrize(
        "qasm_file",
        qasm2_files,
        ids=short_file_name,
    )
    @pytest.mark.parametrize(
        "pipeline",
        ["legacy", "new"],
    )
    def test_serialised_executable(self, qasm_file, pipeline):
        """Test that the executable can be serialized and deserialized."""
        package, _ = self.core.compile(
            str(qasm_file), self.config, pipeline=pipeline, to_json=True
        )
        assert isinstance(package, str)
        results, _ = self.core.execute(package, self.config, pipeline=pipeline)
        assert isinstance(results, dict)
