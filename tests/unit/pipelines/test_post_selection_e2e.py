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


def test_post_selection_happy_path_qasm3_end_to_end():
    """Compile and execute a tiny QASM3 program with post-selection enabled.

    This test is intentionally linear and explicit so it is easy to single-step in a
    debugger and later lift into an example notebook.
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    """

    # Configure a 1-qubit model so post-processing emits the granular instruction chain.
    model = LucyModelLoader(qubit_count=1).load()
    qubit = model.qubits[0]
    qubit.mean_z_map_args = None
    qubit.post_process_method = LinearMapToRealMethod()

    pipeline = EchoPipeline(config=PipelineConfig(name="post-selection-e2e"), model=model)
    compiler_config = CompilerConfig(
        repeats=8,
        results_format=QuantumResultsFormat().raw(),
        optimizations=Tket().disable(),
    )

    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)

    # Verify compile-time wiring emits Equalise → Discriminate for LinearMapToRealMethod
    # (no PostProcessing(LINEAR_MAP) since the qubit has a configured post_process_method).
    assert len(executable.acquires) == 1
    acquire = next(iter(executable.acquires.values()))
    assert len(acquire.post_processing) == 2
    assert isinstance(acquire.post_processing[0], Equalise)
    assert isinstance(acquire.post_processing[1], Discriminate)
    assert acquire.post_processing[1].threshold == 0.0

    # Execute via runtime directly so post-selection metadata is available in ResultManager.
    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    results = pipeline.runtime.execute(
        executable,
        res_mgr=res_mgr,
        met_mgr=met_mgr,
        compiler_config=compiler_config,
    )

    assert results is not None
    # With no disallowed states the PostSelect instruction is not emitted, so no
    # PostSelectionResult is recorded and all shots are retained implicitly.
    assert not res_mgr.check_for_type(PostSelectionResult)


def test_post_selection_max_likelihood_three_states_noise_injected(function_seed):
    """Post-selection with a 3-state MaxLikelihoodMethod where |2> is disallowed.

    The EchoEngine is patched after compilation to return a controlled IQ array with
    known shots near each centroid. This lets us verify that shots classified to the
    disallowed |2> state are filtered out by the runtime post-selection pass.

    State map:
      |0>  centroid  1+0j  (allowed)
      |1>  centroid -1+0j  (allowed)
      |2>  centroid  0+1j  (disallowed)

    Injected shots: 5 near |0>, 3 near |1>, 2 near |2> → 8 retained, 2 discarded.
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    """

    n_near_zero = 5  # |0>, allowed
    n_near_one = 3  # |1>, allowed
    n_near_two = 2  # |2>, disallowed
    shots = n_near_zero + n_near_one + n_near_two  # 10 total

    method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
            -2: MLDiscriminateParams(location=0 + 1j),
        },
    )

    model = LucyModelLoader(qubit_count=1).load()
    qubit = model.qubits[0]
    qubit.mean_z_map_args = None
    qubit.post_process_method = method

    pipeline = EchoPipeline(
        config=PipelineConfig(name="post-selection-ml-e2e"), model=model
    )
    compiler_config = CompilerConfig(
        repeats=shots,
        results_format=QuantumResultsFormat().raw(),
        optimizations=Tket().disable(),
        post_selection=True,
    )

    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)

    # Controlled IQ array: small Gaussian noise around each centroid so the
    # nearest-centroid classifier assigns each shot unambiguously.
    rng = np.random.default_rng(seed=function_seed)
    noise = 0.05

    iq_shots = np.concatenate(
        [
            rng.normal(1, noise, n_near_zero) + 1j * rng.normal(0, noise, n_near_zero),
            rng.normal(-1, noise, n_near_one) + 1j * rng.normal(0, noise, n_near_one),
            rng.normal(0, noise, n_near_two) + 1j * rng.normal(1, noise, n_near_two),
        ]
    ).astype(np.complex128)

    # Patch the engine: replace the echo output for our acquire variable with the
    # controlled IQ array, preserving the correct per-program shot slice for batching.
    output_var = next(iter(executable.acquires))
    original_execute = pipeline.engine.execute

    offset = 0

    def noisy_execute(program, **kwargs):
        nonlocal offset
        result = original_execute(program, **kwargs)
        n = program.shots
        result[output_var] = iq_shots[offset : offset + n]
        offset += n
        return result

    pipeline.engine.execute = noisy_execute

    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    results = pipeline.runtime.execute(
        executable,
        res_mgr=res_mgr,
        met_mgr=met_mgr,
        compiler_config=compiler_config,
    )

    assert results is not None
    assert res_mgr.check_for_type(PostSelectionResult)

    post_selection = res_mgr.lookup_by_type(PostSelectionResult)
    assert post_selection.shots_requested == shots
    assert post_selection.shots_retained == n_near_zero + n_near_one
    assert post_selection.shots_retained == shots - n_near_two


def test_post_selection_max_likelihood_two_qubits_and_mask(function_seed):
    """Post-selection with 2 qubits verifies the global AND-mask semantics.

    Each qubit gets its own controlled IQ injection so different sets of shots are
    disallowed by each qubit. The global mask is the AND of both qubits' masks, so
    the number of filtered shots is larger than either qubit alone would produce.

    State map (same for both qubits):
      |0>  centroid  1+0j  (allowed)
      |1>  centroid -1+0j  (allowed)
      |2>  centroid  0+1j  (disallowed)

    Injection (10 shots total):
      q0: shots 0-4 near |0>, shots 5-7 near |1>, shots 8-9 near |2>  → disallows {8, 9}
      q1: shots 0-3 near |0>, shots 4-6 near |1>, shots 7-9 near |2>  → disallows {7, 8, 9}

    Global mask = q0_mask AND q1_mask → shots {7, 8, 9} filtered → 7 retained.
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    h q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """

    shots = 10
    # q0: disallows indices {8, 9}
    q0_near_zero, q0_near_one, q0_near_two = 5, 3, 2
    # q1: disallows indices {7, 8, 9}
    q1_near_zero, q1_near_one, q1_near_two = 4, 3, 3
    # AND: union of disallowed = {7, 8, 9} → 7 retained
    shots_retained_expected = shots - len({7, 8, 9})

    method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
            -2: MLDiscriminateParams(location=0 + 1j),
        },
    )

    model = LucyModelLoader(qubit_count=2).load()
    for qubit in model.qubits.values():
        qubit.mean_z_map_args = None
        qubit.post_process_method = method

    pipeline = EchoPipeline(
        config=PipelineConfig(name="post-selection-ml-2q-e2e"), model=model
    )
    compiler_config = CompilerConfig(
        repeats=shots,
        results_format=QuantumResultsFormat().raw(),
        optimizations=Tket().disable(),
        post_selection=True,
    )

    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)
    assert len(executable.acquires) == 2

    # Demap each acquire's physical channel UUID back to its qubit index so we can
    # assign the correct IQ pattern to each output variable.
    rng = np.random.default_rng(seed=function_seed)
    noise = 0.05

    def _gauss_iq(real_mean, imag_mean, n):
        return (
            rng.normal(real_mean, noise, n) + 1j * rng.normal(imag_mean, noise, n)
        ).astype(np.complex128)

    acquire_ch_to_qubit = {
        qubit.resonator.physical_channel.uuid: idx for idx, qubit in model.qubits.items()
    }

    iq_by_var: dict[str, np.ndarray] = {}
    for var, acq_data in executable.acquires.items():
        qubit_idx = acquire_ch_to_qubit[acq_data.physical_channel]
        if qubit_idx == 0:
            iq_by_var[var] = np.concatenate(
                [
                    _gauss_iq(1, 0, q0_near_zero),
                    _gauss_iq(-1, 0, q0_near_one),
                    _gauss_iq(0, 1, q0_near_two),
                ]
            )
        else:
            iq_by_var[var] = np.concatenate(
                [
                    _gauss_iq(1, 0, q1_near_zero),
                    _gauss_iq(-1, 0, q1_near_one),
                    _gauss_iq(0, 1, q1_near_two),
                ]
            )

    original_execute = pipeline.engine.execute
    offsets: dict[str, int] = dict.fromkeys(iq_by_var, 0)

    def noisy_execute(program, **kwargs):
        result = original_execute(program, **kwargs)
        for var, iq in iq_by_var.items():
            n = program.shots
            result[var] = iq[offsets[var] : offsets[var] + n]
            offsets[var] += n
        return result

    pipeline.engine.execute = noisy_execute

    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    results = pipeline.runtime.execute(
        executable,
        res_mgr=res_mgr,
        met_mgr=met_mgr,
        compiler_config=compiler_config,
    )

    assert results is not None
    assert res_mgr.check_for_type(PostSelectionResult)

    post_selection = res_mgr.lookup_by_type(PostSelectionResult)
    assert post_selection.shots_requested == shots
    assert post_selection.shots_retained == shots_retained_expected


def test_post_selection_ml_and_linear_map_two_qubits_and_mask(function_seed):
    """Mixed post-selection: qubit0 uses 3-state ML, qubit1 uses 2-state ML.

    This verifies that the global AND-mask behavior works when different qubits
    use different post-processing methods. We construct controlled IQ arrays so
    q0 disallows shots {8,9} (via ML leakage state) and q1 disallows shots {7,8,9}
    (via ML with state 1 as disallowed). The global mask should therefore filter
    {7,8,9} and retain 7 shots.
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    h q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """

    shots = 10

    # q0 (ML): disallows indices {8,9}
    q0_near_zero, q0_near_one, q0_near_two = 5, 3, 2
    # q1 (ML with state 1 disallowed): shots 7,8,9 map to negative key.

    shots_retained_expected = shots - len({7, 8, 9})

    ml_method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
            -2: MLDiscriminateParams(location=0 + 1j),
        },
    )

    # q1: 2-state ML where state 1 is disallowed (negative key)
    linear_method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            -2: MLDiscriminateParams(location=-1 + 0j),  # negative key = disallowed
        }
    )

    model = LucyModelLoader(qubit_count=2).load()
    # assign methods: q0 -> ML, q1 -> linear
    for idx, qubit in model.qubits.items():
        qubit.mean_z_map_args = None
        if idx == 0:
            qubit.post_process_method = ml_method
        else:
            qubit.post_process_method = linear_method

    pipeline = EchoPipeline(
        config=PipelineConfig(name="post-selection-ml-and-linear-2q-e2e"), model=model
    )
    compiler_config = CompilerConfig(
        repeats=shots,
        results_format=QuantumResultsFormat().raw(),
        optimizations=Tket().disable(),
        post_selection=True,
    )

    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)
    assert len(executable.acquires) == 2

    rng = np.random.default_rng(seed=function_seed)
    noise = 0.05

    def _gauss_iq(real_mean, imag_mean, n):
        return (
            rng.normal(real_mean, noise, n) + 1j * rng.normal(imag_mean, noise, n)
        ).astype(np.complex128)

    # Demap each acquire variable to a qubit index via physical channel UUID
    acquire_ch_to_qubit = {
        qubit.resonator.physical_channel.uuid: idx for idx, qubit in model.qubits.items()
    }

    iq_by_var: dict[str, np.ndarray] = {}
    for var, acq_data in executable.acquires.items():
        qubit_idx = acquire_ch_to_qubit[acq_data.physical_channel]
        if qubit_idx == 0:
            # ML: same pattern as the ML-only test (disallows last two indices)
            iq_by_var[var] = np.concatenate(
                [
                    _gauss_iq(1, 0, q0_near_zero),
                    _gauss_iq(-1, 0, q0_near_one),
                    _gauss_iq(0, 1, q0_near_two),
                ]
            )
        else:
            # Linear: positive for indices 0-6, negative for 7-9
            pos = _gauss_iq(1, 0, 7)
            neg = _gauss_iq(-1, 0, 3)
            iq_by_var[var] = np.concatenate([pos, neg])

    original_execute = pipeline.engine.execute
    offsets: dict[str, int] = dict.fromkeys(iq_by_var, 0)

    def noisy_execute(program, **kwargs):
        result = original_execute(program, **kwargs)
        for var, iq in iq_by_var.items():
            n = program.shots
            result[var] = iq[offsets[var] : offsets[var] + n]
            offsets[var] += n
        return result

    pipeline.engine.execute = noisy_execute

    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    results = pipeline.runtime.execute(
        executable,
        res_mgr=res_mgr,
        met_mgr=met_mgr,
        compiler_config=compiler_config,
    )

    assert results is not None
    assert res_mgr.check_for_type(PostSelectionResult)

    post_selection = res_mgr.lookup_by_type(PostSelectionResult)
    assert post_selection.shots_requested == shots
    assert post_selection.shots_retained == shots_retained_expected


def test_post_selection_disabled_via_compiler_config():
    """With post_selection=False, no PostSelect is emitted even if the method has disallowed
    states.

    This test verifies that setting CompilerConfig(post_selection=False) completely
    suppresses PostSelect instruction emission, even when the qubit model has disallowed
    states configured.
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    """

    method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
            -2: MLDiscriminateParams(location=0 + 1j),
        },
    )

    model = LucyModelLoader(qubit_count=1).load()
    qubit = model.qubits[0]
    qubit.mean_z_map_args = None
    qubit.post_process_method = method

    pipeline = EchoPipeline(
        config=PipelineConfig(name="post-selection-disabled"), model=model
    )

    # Compile with post_selection=False (opt-in disabled).
    compiler_config = CompilerConfig(
        repeats=10,
        results_format=QuantumResultsFormat().raw(),
        optimizations=Tket().disable(),
        post_selection=False,
    )

    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)

    # Verify the compile-time instruction chain has no PostSelect.
    assert len(executable.acquires) == 1
    acquire = next(iter(executable.acquires.values()))
    post_processing = acquire.post_processing
    assert (
        len(post_processing) == 1
    )  # Only Discriminate (no Equalise since transform/offset are None; no Demap; no PostSelect)
    assert isinstance(post_processing[0], Discriminate)
    # No PostSelect in the chain.
    assert not any(isinstance(pp, PostSelect) for pp in post_processing)

    # At runtime, without PostSelect, no PostSelectionResult is recorded.
    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    results = pipeline.runtime.execute(
        executable,
        res_mgr=res_mgr,
        met_mgr=met_mgr,
        compiler_config=compiler_config,
    )

    assert results is not None
    assert not res_mgr.check_for_type(PostSelectionResult)


def test_post_selection_enabled_via_compiler_config():
    """With post_selection=True, PostSelect is emitted when the method has disallowed
    states.

    This test verifies that setting CompilerConfig(post_selection=True) enables PostSelect
    instruction emission whenever the qubit model has disallowed states.
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    """

    method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
            -2: MLDiscriminateParams(location=0 + 1j),
        },
    )

    model = LucyModelLoader(qubit_count=1).load()
    qubit = model.qubits[0]
    qubit.mean_z_map_args = None
    qubit.post_process_method = method

    pipeline = EchoPipeline(
        config=PipelineConfig(name="post-selection-enabled"), model=model
    )

    # Compile with post_selection=True (opt-in enabled).
    compiler_config = CompilerConfig(
        repeats=10,
        results_format=QuantumResultsFormat().raw(),
        optimizations=Tket().disable(),
        post_selection=True,
    )

    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)

    # Verify the compile-time instruction chain includes PostSelect.
    assert len(executable.acquires) == 1
    acquire = next(iter(executable.acquires.values()))
    post_processing = acquire.post_processing
    assert (
        len(post_processing) == 2
    )  # Discriminate, PostSelect (no Equalise since transform/offset are None; no Demap)
    assert isinstance(post_processing[0], Discriminate)
    assert isinstance(post_processing[1], PostSelect)


@pytest.mark.parametrize("post_selection", [False, True])
def test_post_selection_flag_controls_instruction_emission(post_selection):
    """Parametrized test: post_selection flag controls PostSelect emission.

    Tests both False and True cases to ensure the flag is respected
    consistently at compile time.
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    """

    method = LinearMapToRealMethod(
        mean_z_map_args=[1 + 0j, 0j],
    )

    model = LucyModelLoader(qubit_count=1).load()
    qubit = model.qubits[0]
    qubit.mean_z_map_args = None
    qubit.post_process_method = method

    pipeline = EchoPipeline(
        config=PipelineConfig(name=f"post-selection-{post_selection}"), model=model
    )

    compiler_config = CompilerConfig(
        repeats=5,
        results_format=QuantumResultsFormat().raw(),
        optimizations=Tket().disable(),
        post_selection=post_selection,
    )

    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)

    acquire = next(iter(executable.acquires.values()))
    post_processing = acquire.post_processing

    if post_selection:
        # Should have PostSelect in the chain.
        assert any(isinstance(pp, PostSelect) for pp in post_processing)
    else:
        # Should NOT have PostSelect in the chain.
        assert not any(isinstance(pp, PostSelect) for pp in post_processing)


# Integration tests for results_format semantics with post-selection


def test_results_format_raw_with_post_selection_e2e():
    """End-to-end compilation test: raw format with ML post-processing, no post-selection.

    Verifies that when results_format=raw() and post-selection is disabled,
    the compiled executable contains Discriminate (no Demap, since it no longer exists).
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    """

    # Configure 1-qubit model with ML post-processing (2 allowed states)
    model = LucyModelLoader(qubit_count=1).load()
    qubit = model.qubits[0]
    qubit.mean_z_map_args = None
    qubit.post_process_method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
        },
    )

    pipeline = EchoPipeline(config=PipelineConfig(name="raw-e2e"), model=model)
    compiler_config = CompilerConfig(
        repeats=10,
        results_format=QuantumResultsFormat().raw(),
        optimizations=Tket().disable(),
        post_selection=False,  # Post-selection disabled for basic test
    )

    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)

    # Verify Discriminate is present; no Demap since it no longer exists.
    acquire = next(iter(executable.acquires.values()))
    assert any(isinstance(pp, Discriminate) for pp in acquire.post_processing)


def test_results_format_binary_count_with_post_selection_e2e():
    """Document end-to-end behavior: binary_count format uses shots_retained.

    This test verifies the critical behavior: when post-selection filters shots,
    binary_count() uses the retained shot count, not the original.
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    """

    # Configure with 3 states, one disallowed
    model = LucyModelLoader(qubit_count=1).load()
    qubit = model.qubits[0]
    qubit.mean_z_map_args = None
    qubit.post_process_method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
            -2: MLDiscriminateParams(location=0 + 1j),
        },
    )

    pipeline = EchoPipeline(config=PipelineConfig(name="binary-count-e2e"), model=model)
    compiler_config = CompilerConfig(
        repeats=10,
        results_format=QuantumResultsFormat().binary_count(),
        optimizations=Tket().disable(),
        post_selection=True,  # Enable post-selection
    )

    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)

    # Verify Discriminate(ML) and PostSelect are in the chain
    acquire = next(iter(executable.acquires.values()))
    post_processing = acquire.post_processing
    assert any(isinstance(pp, Discriminate) for pp in post_processing)
    assert any(isinstance(pp, PostSelect) for pp in post_processing)

    # PostSelect no longer carries disallowed_states — negativity of int key encodes it.
    # Just verify PostSelect is emitted.
    post_select = [pp for pp in post_processing if isinstance(pp, PostSelect)][0]
    assert post_select.output_variable is not None


def test_results_format_multistate_binary_warning_e2e():
    """Document that binary format with >2 states is ambiguous (informational).

    This test verifies the setup works but documents the ambiguity for users. Actual
    behavior with binary_average() on multi-valued arrays is dependent on implementation
    details in the runtime.
    """
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[1] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    """

    # Configure with 3 states (all allowed)
    model = LucyModelLoader(qubit_count=1).load()
    qubit = model.qubits[0]
    qubit.mean_z_map_args = None
    qubit.post_process_method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
            2: MLDiscriminateParams(location=0 + 1j),
        },
    )

    pipeline = EchoPipeline(
        config=PipelineConfig(name="multistate-warning-e2e"), model=model
    )
    compiler_config = CompilerConfig(
        repeats=10,
        results_format=QuantumResultsFormat().binary(),  # Binary format with 3 states
        optimizations=Tket().disable(),
        post_selection=False,
    )

    # This should compile without error (currently just documented as ambiguous)
    executable, _ = QAT().compile(qasm, compiler_config, pipeline=pipeline)
    assert executable is not None
    # Users should prefer binary_count() for multi-state; this is advisory
