# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from compiler_config.config import CompilerConfig, QuantumResultsFormat, Tket

from qat import QAT
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.measure import Discriminate, Equalise, PostSelect
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.post_processing import (
    LinearMapToRealMethod,
    MaxLikelihoodMethod,
    MLDiscriminateParams,
)
from qat.pipelines.waveform import EchoPipeline, PipelineConfig
from qat.runtime.passes.analysis import PostSelectionResult

_QASM_1Q = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    """

_QASM_2Q = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    h q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """


_NOISE = 0.05


def _three_state_ml_method() -> MaxLikelihoodMethod:
    """3-state MaxLikelihoodMethod: |0> and |1> allowed, |2> (key -2) disallowed."""
    return MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
            -2: MLDiscriminateParams(location=0 + 1j),
        },
    )


def _two_state_ml_method() -> MaxLikelihoodMethod:
    """2-state MaxLikelihoodMethod with both states allowed."""
    return MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
        },
    )


def _two_state_ml_method_state1_disallowed() -> MaxLikelihoodMethod:
    """2-state MaxLikelihoodMethod where state 1 (negative key) is disallowed."""
    return MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            -2: MLDiscriminateParams(location=-1 + 0j),
        }
    )


def _three_state_ml_method_all_allowed() -> MaxLikelihoodMethod:
    """3-state MaxLikelihoodMethod with every state allowed (positive keys)."""
    return MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
            2: MLDiscriminateParams(location=0 + 1j),
        },
    )


def _single_acquire_post_processing(executable):
    """Assert the executable has exactly one acquire and return its post_processing list."""
    assert len(executable.acquires) == 1
    acquire = next(iter(executable.acquires.values()))
    return acquire.post_processing


def _iq_by_qubit(executable, model, arrays_by_qubit: dict[int, np.ndarray]) -> dict:
    """Map each acquire output variable to the IQ array configured for its qubit index."""
    ch_to_qubit = {
        qubit.resonator.physical_channel.uuid: idx for idx, qubit in model.qubits.items()
    }
    return {
        var: arrays_by_qubit[ch_to_qubit[acq.physical_channel]]
        for var, acq in executable.acquires.items()
    }


def _lucy_model(qubit_count: int, method):
    """Build a Lucy model with ``mean_z_map_args`` cleared and post-processing applied.

    ``method`` may be a single post-processing method (applied to every qubit) or a
    ``dict`` keyed by qubit index for per-qubit methods.
    """
    model = LucyModelLoader(qubit_count=qubit_count).load()
    for idx, qubit in model.qubits.items():
        qubit.mean_z_map_args = None
        qubit.post_process_method = method[idx] if isinstance(method, dict) else method
    return model


def _compiler_config(
    repeats: int, *, post_selection: bool = True, results_format=None
) -> CompilerConfig:
    """Create a CompilerConfig with Tket disabled and a raw results format default."""
    return CompilerConfig(
        repeats=repeats,
        results_format=results_format or QuantumResultsFormat().raw(),
        optimizations=Tket().disable(),
        post_selection=post_selection,
    )


def _compile_pipeline(qasm: str, model, compiler_config):
    """Compile ``qasm`` and return ``(pipeline, executable)`` for the given model/config."""
    pipeline = EchoPipeline(config=PipelineConfig(name="post-selection-test"), model=model)
    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)
    return pipeline, executable


def _run_pipeline(pipeline, executable, compiler_config):
    """Execute a pipeline and return ``(results, res_mgr, met_mgr)``."""
    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    results = pipeline.runtime.execute(
        executable,
        res_mgr=res_mgr,
        met_mgr=met_mgr,
        compiler_config=compiler_config,
    )
    return results, res_mgr, met_mgr


def _gauss_iq(
    rng: np.random.Generator,
    real_mean: float,
    imag_mean: float,
    n: int,
    noise: float = _NOISE,
) -> np.ndarray:
    """Generate *n* IQ shots as Gaussian noise around ``real_mean + 1j*imag_mean``."""
    return (rng.normal(real_mean, noise, n) + 1j * rng.normal(imag_mean, noise, n)).astype(
        np.complex128
    )


def _patch_engine(pipeline, iq_by_var: dict[str, np.ndarray]) -> None:
    """Patch ``pipeline.engine.execute`` to inject controlled IQ data per acquire var."""
    original_execute = pipeline.engine.execute
    offsets: dict[str, int] = dict.fromkeys(iq_by_var, 0)

    def noisy_execute(program, **kwargs):
        result = original_execute(program, **kwargs)
        for var, iq in iq_by_var.items():
            n = program.shots
            start = offsets[var]
            assert start + n <= len(iq), (
                f"IQ array for {var!r} exhausted: requested {n} shots at offset "
                f"{start} but only {len(iq)} samples available"
            )
            result[var] = iq[start : start + n]
            offsets[var] += n
        return result

    pipeline.engine.execute = noisy_execute


class TestCompileTimeInstructionChain:
    """Compile-time assertions on the post-processing instruction chain."""

    def test_linear_map_emits_equalise_then_discriminate(self):
        """LinearMapToRealMethod (no disallowed states) emits Equalise → Discriminate."""
        model = _lucy_model(1, LinearMapToRealMethod())
        _, executable = _compile_pipeline(
            _QASM_1Q, model, _compiler_config(8, post_selection=False)
        )

        post_processing = _single_acquire_post_processing(executable)
        assert len(post_processing) == 2
        assert isinstance(post_processing[0], Equalise)
        assert isinstance(post_processing[1], Discriminate)
        assert post_processing[1].threshold == 0.0

    def test_disabled_flag_emits_only_discriminate(self):
        """post_selection=False suppresses PostSelect even with disallowed states."""
        model = _lucy_model(1, _three_state_ml_method())
        _, executable = _compile_pipeline(
            _QASM_1Q, model, _compiler_config(10, post_selection=False)
        )

        post_processing = _single_acquire_post_processing(executable)
        assert len(post_processing) == 1
        assert isinstance(post_processing[0], Discriminate)
        assert not any(isinstance(pp, PostSelect) for pp in post_processing)

    def test_enabled_flag_emits_discriminate_then_post_select(self):
        """post_selection=True emits Discriminate → PostSelect for disallowed states."""
        model = _lucy_model(1, _three_state_ml_method())
        _, executable = _compile_pipeline(
            _QASM_1Q, model, _compiler_config(10, post_selection=True)
        )

        post_processing = _single_acquire_post_processing(executable)
        assert len(post_processing) == 2
        assert isinstance(post_processing[0], Discriminate)
        assert isinstance(post_processing[1], PostSelect)

    @pytest.mark.parametrize("post_selection", [False, True])
    def test_flag_controls_post_select_emission(self, post_selection):
        """The post_selection flag consistently controls PostSelect emission."""
        model = _lucy_model(1, LinearMapToRealMethod(mean_z_map_args=[1 + 0j, 0j]))
        _, executable = _compile_pipeline(
            _QASM_1Q, model, _compiler_config(5, post_selection=post_selection)
        )

        post_processing = next(iter(executable.acquires.values())).post_processing
        assert any(isinstance(pp, PostSelect) for pp in post_processing) is post_selection


class TestRuntimePostSelection:
    """Runtime assertions on shot filtering recorded in the ResultManager."""

    def test_linear_map_records_no_post_selection_result(self):
        """No disallowed states means no PostSelectionResult is recorded."""
        model = _lucy_model(1, LinearMapToRealMethod())
        compiler_config = _compiler_config(8, post_selection=False)
        pipeline, executable = _compile_pipeline(_QASM_1Q, model, compiler_config)

        results, res_mgr, _ = _run_pipeline(pipeline, executable, compiler_config)

        assert results is not None
        assert not res_mgr.check_for_type(PostSelectionResult)

    def test_disabled_flag_records_no_post_selection_result(self):
        """With post_selection=False no PostSelectionResult is recorded at runtime."""
        model = _lucy_model(1, _three_state_ml_method())
        compiler_config = _compiler_config(10, post_selection=False)
        pipeline, executable = _compile_pipeline(_QASM_1Q, model, compiler_config)

        results, res_mgr, _ = _run_pipeline(pipeline, executable, compiler_config)

        assert results is not None
        assert not res_mgr.check_for_type(PostSelectionResult)

    def test_single_qubit_three_state_filters_disallowed_shots(self, function_seed):
        """Shots classified to the disallowed |2> state are filtered out.

        Injected shots: 5 near |0>, 3 near |1>, 2 near |2> → 8 retained, 2 discarded.
        """
        n_near_zero, n_near_one, n_near_two = 5, 3, 2
        shots = n_near_zero + n_near_one + n_near_two

        model = _lucy_model(1, _three_state_ml_method())
        compiler_config = _compiler_config(shots)
        pipeline, executable = _compile_pipeline(_QASM_1Q, model, compiler_config)

        rng = np.random.default_rng(seed=function_seed)
        iq_shots = np.concatenate(
            [
                _gauss_iq(rng, 1, 0, n_near_zero),
                _gauss_iq(rng, -1, 0, n_near_one),
                _gauss_iq(rng, 0, 1, n_near_two),
            ]
        )
        output_var = next(iter(executable.acquires))
        _patch_engine(pipeline, {output_var: iq_shots})

        results, res_mgr, _ = _run_pipeline(pipeline, executable, compiler_config)

        assert results is not None
        post_selection = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_selection.shots_requested == shots
        assert post_selection.shots_retained == n_near_zero + n_near_one

    def test_two_qubits_apply_global_and_mask(self, function_seed):
        """The global mask is the AND of both qubits' masks.

        q0 disallows {8, 9} and q1 disallows {7, 8, 9}; union {7, 8, 9} → 7 retained.
        """
        shots = 10
        model = _lucy_model(2, _three_state_ml_method())
        compiler_config = _compiler_config(shots)
        pipeline, executable = _compile_pipeline(_QASM_2Q, model, compiler_config)
        assert len(executable.acquires) == 2

        rng = np.random.default_rng(seed=function_seed)
        arrays_by_qubit = {
            0: np.concatenate(
                [_gauss_iq(rng, 1, 0, 5), _gauss_iq(rng, -1, 0, 3), _gauss_iq(rng, 0, 1, 2)]
            ),
            1: np.concatenate(
                [_gauss_iq(rng, 1, 0, 4), _gauss_iq(rng, -1, 0, 3), _gauss_iq(rng, 0, 1, 3)]
            ),
        }
        _patch_engine(pipeline, _iq_by_qubit(executable, model, arrays_by_qubit))

        results, res_mgr, _ = _run_pipeline(pipeline, executable, compiler_config)

        assert results is not None
        post_selection = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_selection.shots_requested == shots
        assert post_selection.shots_retained == shots - len({7, 8, 9})

    def test_mixed_methods_apply_global_and_mask(self, function_seed):
        """AND-mask holds when qubits use different post-processing methods.

        q0 (3-state ML) disallows {8, 9}; q1 (2-state ML) disallows {7, 8, 9};
        union {7, 8, 9} → 7 retained.
        """
        shots = 10
        methods = {0: _three_state_ml_method(), 1: _two_state_ml_method_state1_disallowed()}
        model = _lucy_model(2, methods)
        compiler_config = _compiler_config(shots)
        pipeline, executable = _compile_pipeline(_QASM_2Q, model, compiler_config)
        assert len(executable.acquires) == 2

        rng = np.random.default_rng(seed=function_seed)
        arrays_by_qubit = {
            0: np.concatenate(
                [_gauss_iq(rng, 1, 0, 5), _gauss_iq(rng, -1, 0, 3), _gauss_iq(rng, 0, 1, 2)]
            ),
            1: np.concatenate([_gauss_iq(rng, 1, 0, 7), _gauss_iq(rng, -1, 0, 3)]),
        }
        _patch_engine(pipeline, _iq_by_qubit(executable, model, arrays_by_qubit))

        results, res_mgr, _ = _run_pipeline(pipeline, executable, compiler_config)

        assert results is not None
        post_selection = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_selection.shots_requested == shots
        assert post_selection.shots_retained == shots - len({7, 8, 9})


class TestResultsFormat:
    """Results-format semantics of the compiled instruction chain."""

    def test_raw_format_contains_discriminate(self):
        """raw() with ML post-processing and post-selection disabled emits Discriminate."""
        model = _lucy_model(1, _two_state_ml_method())
        _, executable = _compile_pipeline(
            _QASM_1Q, model, _compiler_config(10, post_selection=False)
        )

        post_processing = _single_acquire_post_processing(executable)
        assert any(isinstance(pp, Discriminate) for pp in post_processing)

    @staticmethod
    def _binary_count_post_processing():
        """Post-processing chain compiled with binary_count() and post-selection on."""
        model = _lucy_model(1, _three_state_ml_method())
        compiler_config = _compiler_config(
            10,
            post_selection=True,
            results_format=QuantumResultsFormat().binary_count(),
        )
        _, executable = _compile_pipeline(_QASM_1Q, model, compiler_config)
        return _single_acquire_post_processing(executable)

    def test_binary_count_emits_discriminate_and_post_select(self):
        """binary_count() with post-selection emits both Discriminate and PostSelect."""
        post_processing = self._binary_count_post_processing()
        assert any(isinstance(pp, Discriminate) for pp in post_processing)
        assert any(isinstance(pp, PostSelect) for pp in post_processing)

    def test_binary_count_post_select_has_output_variable(self):
        """The emitted PostSelect instruction carries an output variable."""
        post_processing = self._binary_count_post_processing()
        post_select = next(pp for pp in post_processing if isinstance(pp, PostSelect))
        assert post_select.output_variable is not None

    def test_multistate_binary_compiles(self):
        """binary() with three all-allowed states compiles without error (advisory)."""
        model = _lucy_model(1, _three_state_ml_method_all_allowed())
        _, executable = _compile_pipeline(
            _QASM_1Q,
            model,
            _compiler_config(
                10,
                post_selection=False,
                results_format=QuantumResultsFormat().binary(),
            ),
        )

        assert executable is not None
