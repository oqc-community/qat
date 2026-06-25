# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests validating results_format semantics with the granular post-processing pipeline.

The granular pipeline (Equalise → Discriminate → PostSelect) stores each
intermediate stage as side-data in the ResultManager:

  - EqualiseResult  — complex IQ arrays after affine transform
  - DiscriminateResult — string state-label arrays

ResultTransform routes to the correct intermediate based on results_format:

  - raw()          → EqualiseResult  (complex IQ, post-mask)
  - binary()       → Discriminate output (per-shot ints, post-mask)
  - binary_count() → DiscriminateResult label counts ({label: count})

Post-selection applies the global mask to all intermediates, so counts and
arrays only reflect retained shots.
"""

import numpy as np
import pytest
from compiler_config.config import CompilerConfig, QuantumResultsFormat, Tket

from qat import QAT
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.executables import AcquireData, Executable
from qat.ir.instruction_basetypes import AcquireMode
from qat.ir.instructions import Assign
from qat.ir.measure import Discriminate, Equalise, PostSelect
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.post_processing import MaxLikelihoodMethod, MLDiscriminateParams
from qat.pipelines.waveform import EchoPipeline, PipelineConfig
from qat.runtime.passes.analysis import (
    DiscriminateResult,
    EqualiseResult,
    PostSelectionResult,
)
from qat.runtime.passes.transform import (
    AcquisitionPostprocessing,
    AssignResultsTransform,
    ResultTransform,
)
from qat.runtime.results_processing import label_count

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_3STATE_METHOD = MaxLikelihoodMethod(
    states={
        0: MLDiscriminateParams(location=1 + 0j),
        1: MLDiscriminateParams(location=-1 + 0j),
        -2: MLDiscriminateParams(location=0 + 1j),
    },
)

# All-allowed 2-state method for register-assembly tests (no negative keys).
_2STATE_METHOD = MaxLikelihoodMethod(
    states={
        0: MLDiscriminateParams(location=1 + 0j),
        1: MLDiscriminateParams(location=-1 + 0j),
    },
)

# IQ: 4 near |0>, 3 near |1>, 3 near |2>  (total 10 shots)
_IQ_10 = np.array(
    [
        0.9 + 0j,
        0.8 + 0j,
        1.0 + 0j,
        0.7 + 0j,  # → "0"   (allowed)
        -0.9 + 0j,
        -1.0 + 0j,
        -0.8 + 0j,  # → "1"   (allowed)
        0.05 + 0.9j,
        0.1 + 1.0j,
        0.0 + 0.8j,  # → "2"   (disallowed)
    ],
    dtype=np.complex128,
)


def _make_granular_acquire(with_post_select: bool) -> AcquireData:
    pp = [
        Equalise(output_variable="q0", transform=np.eye(2), offset=np.zeros(2)),
        Discriminate(output_variable="q0", method=_3STATE_METHOD),
    ]
    if with_post_select:
        # PostSelect filters all negative keys (disallowed states have negative keys).
        pp.append(PostSelect(output_variable="q0"))
    return AcquireData(
        mode=AcquireMode.INTEGRATOR,
        shape=(10,),
        post_processing=pp,
        physical_channel="ch0",
    )


def _run_acq_pp(iq: np.ndarray, with_post_select: bool) -> tuple[dict, ResultManager]:
    acquire = _make_granular_acquire(with_post_select)
    package = Executable(programs=[], acquires={"q0": acquire}, shots=10)
    res_mgr = ResultManager()
    result = AcquisitionPostprocessing().run({"q0": iq.copy()}, res_mgr, package=package)
    return result, res_mgr


# ---------------------------------------------------------------------------
# label_count unit tests
# ---------------------------------------------------------------------------


class TestLabelCount:
    def test_basic_count(self):
        labels = np.array(["0", "0", "1", "2", "0"])
        assert label_count(labels) == {"0": 3, "1": 1, "2": 1}

    def test_single_label(self):
        labels = np.array(["1"] * 5)
        assert label_count(labels) == {"1": 5}

    def test_preserves_string_keys(self):
        labels = np.array(["a", "b", "a"])
        result = label_count(labels)
        assert all(isinstance(k, str) for k in result)


# ---------------------------------------------------------------------------
# AcquisitionPostprocessing stores intermediates
# ---------------------------------------------------------------------------


class TestIntermediatesStoredInResManager:
    def test_equalise_result_stored(self):
        _, res_mgr = _run_acq_pp(_IQ_10, with_post_select=False)
        assert res_mgr.check_for_type(EqualiseResult)

    def test_discriminate_result_stored(self):
        _, res_mgr = _run_acq_pp(_IQ_10, with_post_select=False)
        assert res_mgr.check_for_type(DiscriminateResult)

    def test_equalise_output_is_complex(self):
        _, res_mgr = _run_acq_pp(_IQ_10, with_post_select=False)
        eq = res_mgr.lookup_by_type(EqualiseResult).outputs["q0"]
        assert np.iscomplexobj(eq)
        assert eq.shape == (10,)

    def test_discriminate_output_is_integer_keys(self):
        _, res_mgr = _run_acq_pp(_IQ_10, with_post_select=False)
        keys = res_mgr.lookup_by_type(DiscriminateResult).outputs["q0"]
        assert keys.dtype.kind in ("i", "u")  # integer dtype
        # Allowed keys: 0, 1; disallowed key: -2
        assert set(keys.tolist()).issubset({0, 1, -2})

    def test_post_selection_masks_intermediates(self):
        """After post-selection, EqualiseResult and DiscriminateResult are filtered."""
        _, res_mgr = _run_acq_pp(_IQ_10, with_post_select=True)
        post_sel = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_sel.shots_retained == 7  # 4 "0" + 3 "1"

        eq = res_mgr.lookup_by_type(EqualiseResult).outputs["q0"]
        assert eq.shape == (7,)

        labels = res_mgr.lookup_by_type(DiscriminateResult).outputs["q0"]
        assert labels.shape == (7,)
        assert -2 not in labels  # disallowed key (-2) filtered out


# ---------------------------------------------------------------------------
# ResultTransform routing — binary_count()
# ---------------------------------------------------------------------------


class TestBinaryCountFormat:
    def _run_result_transform(self, with_post_select: bool, shots: int = 10) -> dict:
        acq, res_mgr = _run_acq_pp(_IQ_10, with_post_select=with_post_select)
        package = Executable(programs=[], acquires={}, shots=shots)
        compiler_config = CompilerConfig(
            repeats=shots,
            results_format=QuantumResultsFormat().binary_count(),
        )
        result = ResultTransform().run(
            acq, res_mgr, compiler_config=compiler_config, package=package
        )
        # DynamicStructureReturn is active but "q0" is not a generated name so the
        # outer dict is preserved: {"q0": {"0": 4, ...}}.  Unwrap for assertions.
        assert isinstance(result, dict)
        return result["q0"]

    def test_returns_label_dict(self):
        result = self._run_result_transform(with_post_select=False)
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result)
        assert all(isinstance(v, int) for v in result.values())

    def test_counts_use_state_keys_not_float_converted(self):
        """Multi-state: integer dict keys (0, 1, -2) appear as string keys in count dict."""
        result = self._run_result_transform(with_post_select=False)
        # 4 classified as key 0, 3 as key 1, 3 as key -2
        assert result == {"0": 4, "1": 3, "-2": 3}

    def test_post_selection_filters_disallowed_state(self):
        """With post-selection the disallowed state "2" is absent from counts."""
        result = self._run_result_transform(with_post_select=True)
        assert "2" not in result
        assert result["0"] == 4
        assert result["1"] == 3

    def test_total_counts_match_retained_shots(self):
        result = self._run_result_transform(with_post_select=True)
        assert sum(result.values()) == 7

    def test_binary_count_via_register_assign_uses_label_count_register(self):
        """When output keys are register names assembled by AssignResultsTransform, ML
        acquire variables are routed through _label_count_register, not binary_count.

        Two single-qubit acquires (q0, q1_0) are assembled into a 2-bit register key "c" via
        an Assign.  Discriminate now emits integer keys directly; _label_count_register
        builds the count dict from the per-shot integer arrays explicitly.
        """
        # Build acquire for 2-state (no disallowed): 5"0" (key 0), 5"1" (key 1).
        _IQ_10_2STATE = np.concatenate(
            [
                np.full(5, 0.9 + 0j),  # → key 0
                np.full(5, -0.9 + 0j),  # → key 1
            ]
        )

        def _make_2state_acq(with_post_select):
            pp = [
                Equalise(output_variable="q0", transform=np.eye(2), offset=np.zeros(2)),
                Discriminate(output_variable="q0", method=_2STATE_METHOD),
            ]
            return AcquireData(
                mode=AcquireMode.INTEGRATOR,
                shape=(10,),
                post_processing=pp,
                physical_channel="ch0",
            )

        def _run_2state_acq_pp():
            acquire = _make_2state_acq(False)
            package = Executable(programs=[], acquires={"q0": acquire})
            acquisitions_raw = {"q0": _IQ_10_2STATE}
            res_mgr = ResultManager()
            result = AcquisitionPostprocessing().run(
                acquisitions_raw, res_mgr, package=package
            )
            return result, res_mgr

        acq_q0, res_mgr_q0 = _run_2state_acq_pp()
        q0_labels = res_mgr_q0.lookup_by_type(DiscriminateResult).outputs["q0"]

        acq_q1, res_mgr_q1 = _run_2state_acq_pp()
        q1_labels = res_mgr_q1.lookup_by_type(DiscriminateResult).outputs["q0"]

        # Combine into one res_mgr with two discriminate outputs.
        res_mgr = ResultManager()
        res_mgr.add(DiscriminateResult(outputs={"q0": q0_labels, "q1_0": q1_labels}))

        # Build executable: two ML acquires, one 2-bit register assign.
        acquire = _make_granular_acquire(with_post_select=False)
        package = Executable(
            programs=[],
            acquires={"q0": acquire, "q1_0": acquire},
            assigns=[Assign(name="c", value=["q0", "q1_0"])],
            returns={"c"},
            shots=10,
        )
        compiler_config = CompilerConfig(
            repeats=10,
            results_format=QuantumResultsFormat().binary_count(),
        )
        # Run AssignResultsTransform to replace per-qubit keys with the register key.
        acq_assigned = AssignResultsTransform().run(
            {**acq_q0, "q1_0": acq_q0["q0"].copy()},
            package=package,
        )
        result = ResultTransform().run(
            acq_assigned, res_mgr, compiler_config=compiler_config, package=package
        )
        counts = result.get("c", result) if isinstance(result, dict) else result
        # All keys must be 2-character bitstrings built from the per-qubit labels.
        assert isinstance(counts, dict)
        assert all(isinstance(k, str) and len(k) == 2 for k in counts), counts
        assert all(isinstance(v, int) for v in counts.values())
        assert sum(counts.values()) == 10


# ---------------------------------------------------------------------------
# ResultTransform routing — binary() (per-shot float output values)
# ---------------------------------------------------------------------------


class TestBinaryFormat:
    def _run_result_transform(self, with_post_select: bool, shots: int = 10) -> np.ndarray:
        acq, res_mgr = _run_acq_pp(_IQ_10, with_post_select=with_post_select)
        package = Executable(programs=[], acquires={}, shots=shots)
        compiler_config = CompilerConfig(
            repeats=shots,
            results_format=QuantumResultsFormat().binary(),
        )
        result = ResultTransform().run(
            acq, res_mgr, compiler_config=compiler_config, package=package
        )
        # Unwrap outer {"q0": array} dict.
        assert isinstance(result, dict)
        return np.asarray(result["q0"])

    def test_returns_float_array(self):
        """Binary() for granular pipeline returns per-shot output values, not a single
        int."""
        result = self._run_result_transform(with_post_select=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10,)

    def test_values_are_integer_state_keys(self):
        """Values are integer dict keys from Discriminate (0, 1, -2) not binary-averaged."""
        result = self._run_result_transform(with_post_select=False)
        assert set(result).issubset({0, 1, -2})

    def test_post_selection_reduces_array_length(self):
        result = self._run_result_transform(with_post_select=True)
        assert result.shape == (7,)
        assert set(result).issubset({0, 1})  # negative keys filtered out


# ---------------------------------------------------------------------------
# ResultTransform routing — raw() (equalised complex IQ)
# ---------------------------------------------------------------------------


class TestRawFormat:
    def _run_result_transform(self, with_post_select: bool, shots: int = 10) -> np.ndarray:
        acq, res_mgr = _run_acq_pp(_IQ_10, with_post_select=with_post_select)
        package = Executable(programs=[], acquires={}, shots=shots)
        compiler_config = CompilerConfig(
            repeats=shots,
            results_format=QuantumResultsFormat().raw(),
        )
        result = ResultTransform().run(
            acq, res_mgr, compiler_config=compiler_config, package=package
        )
        # Unwrap outer {"q0": array} dict.
        assert isinstance(result, dict)
        return np.asarray(result["q0"])

    def test_returns_complex_array(self):
        result = self._run_result_transform(with_post_select=False)
        assert np.iscomplexobj(result)

    def test_length_matches_shots(self):
        result = self._run_result_transform(with_post_select=False)
        assert len(result) == 10

    def test_values_are_equalised_iq_not_map_output(self):
        """Raw() returns IQ (complex), not the discriminated integer output values."""
        result = self._run_result_transform(with_post_select=False)
        # Values are IQ-domain complexes — real parts are not restricted to {0, 1, 2}
        assert not set(result.real).issubset({0.0, 1.0, 2.0})

    def test_post_selection_reduces_array_length(self):
        result = self._run_result_transform(with_post_select=True)
        assert len(result) == 7


# ---------------------------------------------------------------------------
# End-to-end via EchoPipeline
# ---------------------------------------------------------------------------

_QASM = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[1] c;
h q[0];
c[0] = measure q[0];
"""


def _make_pipeline_and_config(results_format, post_selection: bool):
    model = LucyModelLoader(qubit_count=1).load()
    qubit = model.qubits[0]
    qubit.mean_z_map_args = None
    qubit.post_process_method = _3STATE_METHOD
    pipeline = EchoPipeline(config=PipelineConfig(name="semantics-e2e"), model=model)
    compiler_config = CompilerConfig(
        repeats=10,
        results_format=results_format,
        optimizations=Tket().disable(),
        post_selection=post_selection,
    )
    return pipeline, compiler_config


def _inject_iq(pipeline, executable, iq_shots):
    """Patch engine to return controlled IQ data."""
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


@pytest.mark.parametrize("post_selection", [False, True])
def test_e2e_binary_count_uses_state_labels(post_selection):
    """binary_count e2e: result keys are string state labels, not float-converted 0/1."""

    pipeline, compiler_config = _make_pipeline_and_config(
        QuantumResultsFormat().binary_count(), post_selection
    )
    executable, _ = QAT().compile(_QASM, compiler_config, pipeline=pipeline)
    _inject_iq(pipeline, executable, _IQ_10)

    res_mgr = ResultManager()
    result = pipeline.runtime.execute(
        executable,
        res_mgr=res_mgr,
        met_mgr=MetricsManager(),
        compiler_config=compiler_config,
    )

    # Result is nested under the classical register name "c".
    assert isinstance(result, dict)
    counts = result.get("c", result)  # unwrap if present
    assert isinstance(counts, dict)
    # All keys must be string state labels/keys ("0", "1", "-2").
    assert all(isinstance(k, str) for k in counts)
    assert all(isinstance(v, int) for v in counts.values())
    if post_selection:
        # Disallowed state (key "-2") must be absent after post-selection.
        assert "-2" not in counts
    # Counts must sum to retained shots.
    expected_total = 7 if post_selection else 10
    assert sum(counts.values()) == expected_total


@pytest.mark.parametrize("post_selection", [False, True])
def test_e2e_binary_returns_per_shot_vals(post_selection):
    """Binary() e2e: returns per-shot int array from Discriminate, not a majority-vote
    int."""

    pipeline, compiler_config = _make_pipeline_and_config(
        QuantumResultsFormat().binary(), post_selection
    )
    executable, _ = QAT().compile(_QASM, compiler_config, pipeline=pipeline)
    _inject_iq(pipeline, executable, _IQ_10)

    res_mgr = ResultManager()
    result = pipeline.runtime.execute(
        executable,
        res_mgr=res_mgr,
        met_mgr=MetricsManager(),
        compiler_config=compiler_config,
    )

    # Unwrap classical register "c" and its single-bit dimension if present.
    raw = result.get("c", result) if isinstance(result, dict) else result
    arr = np.asarray(raw)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]  # remove outer register-bit dimension
    expected_len = 7 if post_selection else 10
    assert arr.shape == (expected_len,)
    # All values must be integer state keys from Discriminate, not arbitrary ints.
    assert arr.dtype.kind == "i"
    expected_values = {0, 1} if post_selection else {0, 1, -2}
    assert set(arr).issubset(expected_values)


@pytest.mark.parametrize("post_selection", [False, True])
def test_e2e_raw_returns_complex_iq(post_selection):
    """Raw() e2e: returns equalised complex IQ, not discriminated int values."""

    pipeline, compiler_config = _make_pipeline_and_config(
        QuantumResultsFormat().raw(), post_selection
    )
    executable, _ = QAT().compile(_QASM, compiler_config, pipeline=pipeline)
    _inject_iq(pipeline, executable, _IQ_10)

    res_mgr = ResultManager()
    result = pipeline.runtime.execute(
        executable,
        res_mgr=res_mgr,
        met_mgr=MetricsManager(),
        compiler_config=compiler_config,
    )

    # Unwrap classical register "c" and its single-bit dimension if present.
    raw = result.get("c", result) if isinstance(result, dict) else result
    arr = np.asarray(raw)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    expected_len = 7 if post_selection else 10
    assert np.iscomplexobj(arr), "raw() should return complex IQ values"
    assert arr.shape == (expected_len,)
