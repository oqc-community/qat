import re
from pathlib import Path

import pytest
from compiler_config.config import CompilerConfig, Tket, TketOptimizations

from qat.core import QAT
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.frontends import QIRFrontend
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    Instruction,
    PostProcessing,
    ResultsProcessing,
    Return,
    Variable,
)
from qat.purr.compiler.metrics import CompilationMetrics
from qat.qat import _return_or_build, fetch_frontend

dir_path = Path(__file__).parent


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
]


qasm2_files = [
    p
    for p in Path(dir_path, "files", "qasm", "qasm2").glob("*.qasm")
    if p.name not in skip_qasm2
]

qasm3_files = [
    p
    for p in Path(dir_path, "files", "qasm", "qasm3").glob("*.qasm")
    if p.name not in skip_qasm3
]


qir_files = [
    p for p in Path(dir_path, "files", "qir").glob("*.ll") if p.name not in skip_qir
]
qir_files.extend(
    [p for p in Path(dir_path, "files", "qir").glob("*.bc") if p.name not in skip_qir]
)


def equivalent_vars(self, other):
    """Used to overload '__eq__' so that a==b if they are equivalent."""
    return isinstance(self, type(other)) and (vars(self) == vars(other))


gen_name_pattern = re.compile(r"generated_name_[0-9]+")


def _strip_generated_rng(o):
    """Strip away the randong number part of 'generated_name_4826354253464' allowing for
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


@pytest.mark.parametrize(
    "compiler_config",
    [
        pytest.param(None, id="No_config"),
        pytest.param(CompilerConfig(), id="Default_config"),
        pytest.param(
            CompilerConfig(optimizations=Tket(TketOptimizations.Two)),
            id="TketOptimizations.Two",
        ),
    ],
)
@pytest.mark.parametrize(
    "hardware_model", [pytest.param(get_default_echo_hardware(32), id="Echo32Q")]
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
        else:
            print(key)
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
        qat_ib, qat_metrics = QAT(hardware_model).compile(
            qasm_file_string, compiler_config=compiler_config
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
        qat_ib, qat_metrics = QAT(hardware_model).compile(
            qasm_file_string, compiler_config=compiler_config
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
        qat_ib, qat_metrics = QAT(hardware_model).compile(
            qir_file_string, compiler_config=compiler_config
        )
        assert purr_ib.instructions == qat_ib.instructions
        assert purr_metrics.optimized_circuit == qat_metrics.optimized_circuit

    @pytest.mark.parametrize("source_file", [*qir_files], ids=short_file_name)
    def test_qat_execute(
        self, request, source_file, hardware_model, compiler_config, monkeypatch
    ):
        instructions, _ = self.purr_compile(
            request,
            str(source_file),
            hardware=hardware_model,
            compiler_config=compiler_config,
        )
        try:
            purr_res, purr_metrics = QIRFrontend().execute(
                instructions, hardware=hardware_model, compiler_config=compiler_config
            )
        except ValueError:
            purr_res, purr_metrics = None, CompilationMetrics()
        if purr_res is None:
            qat_res, qat_metrics = None, CompilationMetrics()
        else:
            qat_res, qat_metrics = QAT(hardware_model).execute(
                instructions, compiler_config=compiler_config
            )
        assert purr_res == qat_res
        assert purr_metrics.as_dict() == qat_metrics.as_dict()
