# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :class:`InsertPreSelectionMeasurement`."""

import pytest
from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.instruction_basetypes import AcquireMode, AcquirePurpose
from qat.ir.instruction_builder import PydQuantumInstructionBuilder
from qat.ir.instructions import Delay, Repeat
from qat.ir.measure import Acquire, Discriminate, Equalise, PostSelect
from qat.middleend.passes.analysis import ActivePulseChannelAnalysis
from qat.middleend.passes.transform import InsertPreSelectionMeasurement
from qat.middleend.passes.validation import NoMidCircuitMeasurementValidation
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.post_processing import (
    LinearMapToRealMethod,
    MaxLikelihoodMethod,
    MLDiscriminateParams,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_qubit_model():
    """A single-qubit hardware model."""
    return LucyModelLoader(qubit_count=1).load()


@pytest.fixture()
def two_qubit_model():
    """A two-qubit hardware model."""
    return LucyModelLoader(qubit_count=2).load()


@pytest.fixture()
def ml_method():
    """A two-state MaxLikelihoodMethod."""
    return MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1 + 0j),
            1: MLDiscriminateParams(location=-1 + 0j),
        },
    )


@pytest.fixture()
def presel_config():
    """A CompilerConfig with pre_selection enabled."""
    return CompilerConfig(pre_selection=True)


@pytest.fixture()
def no_presel_config():
    """A CompilerConfig with pre_selection disabled."""
    return CompilerConfig(pre_selection=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_qubit_for_preselection(
    qubit,
    disallowed_states=None,
    method=None,
):
    """Set up a qubit with preselect_disallowed_states and a post_process_method."""
    qubit.mean_z_map_args = None
    qubit.post_process_method = method or LinearMapToRealMethod()
    qubit.preselect_disallowed_states = (
        disallowed_states if disallowed_states is not None else {1}
    )


def _build_simple_ir(model):
    """Build a minimal IR with one X gate and one measurement per qubit."""
    builder = PydQuantumInstructionBuilder(hardware_model=model)
    builder.repeat(100)
    for qubit in model.qubits.values():
        builder.X(target=qubit)
        builder.measure_with_granular_post_processing(qubit)
    return builder


def _build_drive_only_ir(model):
    """Build a minimal IR with only drive-channel gates — no measurements.

    This simulates a circuit where the measure/acquire channels for each qubit
    are **not** present in ``ActivePulseChannelResults`` when the analysis pass
    runs, because no ``Pulse``/``Acquire`` targets those channels before
    ``InsertPreSelectionMeasurement`` executes.
    """
    builder = PydQuantumInstructionBuilder(hardware_model=model)
    builder.repeat(100)
    for qubit in model.qubits.values():
        builder.X(target=qubit)
    return builder


def _build_res_mgr(model, builder):
    """Run ActivePulseChannelAnalysis and return the populated ResultManager."""
    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    ActivePulseChannelAnalysis(model).run(builder, res_mgr, met_mgr)
    return res_mgr


def _run_preselection(model, builder, res_mgr, config):
    """Run the InsertPreSelectionMeasurement pass and return the result."""
    return InsertPreSelectionMeasurement(model).run(
        builder, res_mgr, compiler_config=config
    )


# ---------------------------------------------------------------------------
# Tests — disabled / no-op cases
# ---------------------------------------------------------------------------


class TestPreSelectionDisabled:
    """When pre_selection is off, the pass should be a no-op."""

    def test_no_op_when_disabled(self, single_qubit_model, no_presel_config):
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)
        original_count = len(builder.instructions)

        result = _run_preselection(single_qubit_model, builder, res_mgr, no_presel_config)
        assert len(result.instructions) == original_count

    def test_no_op_when_disallowed_states_empty(self, single_qubit_model, presel_config):
        qubit = single_qubit_model.qubits[0]
        qubit.mean_z_map_args = None
        qubit.post_process_method = LinearMapToRealMethod()
        qubit.preselect_disallowed_states = set()
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)
        original_count = len(builder.instructions)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)
        assert len(result.instructions) == original_count


# ---------------------------------------------------------------------------
# Tests — linear map method
# ---------------------------------------------------------------------------


class TestPreSelectionLinearMap:
    """Pre-selection with LinearMapToRealMethod qubits."""

    def test_injects_presel_instructions(self, single_qubit_model, presel_config):
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)
        original_count = len(builder.instructions)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)
        # Should have injected extra instructions
        assert len(result.instructions) > original_count

    def test_presel_acquire_has_correct_purpose(self, single_qubit_model, presel_config):
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        # Find pre-selection acquires
        presel_acquires = [
            instr
            for instr in result.instructions
            if isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
        ]
        assert len(presel_acquires) == 1

    def test_presel_output_variable_prefix(self, single_qubit_model, presel_config):
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        presel_post_selects = [
            instr
            for instr in result.instructions
            if isinstance(instr, PostSelect) and instr.output_variable.startswith("presel_")
        ]
        assert len(presel_post_selects) == 1
        # Disallowed states are now in PostSelect.additional_disallowed.
        ps_instr = presel_post_selects[0]
        assert 1 in ps_instr.additional_disallowed

    def test_presel_emits_full_chain(self, single_qubit_model, presel_config):
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        # Check that the pre-selection chain is lowered primitives + granular steps.
        presel_var = "presel_0"
        chain_types = [
            type(instr)
            for instr in result.instructions
            if (
                isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
            )
            or (hasattr(instr, "output_variable") and instr.output_variable == presel_var)
        ]
        # Synchronize is only emitted when there are >= 2 active channels;
        # the single-qubit fixture has 1 active channel so no sync is expected.
        assert Acquire in chain_types
        assert Equalise in chain_types
        assert Discriminate in chain_types
        assert PostSelect in chain_types

    def test_presel_inserted_after_first_repeat(self, single_qubit_model, presel_config):
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        repeat_index = next(
            i for i, instr in enumerate(result.instructions) if isinstance(instr, Repeat)
        )
        presel_acquire_index = next(
            i
            for i, instr in enumerate(result.instructions)
            if isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
        )
        assert repeat_index < presel_acquire_index


# ---------------------------------------------------------------------------
# Tests — max likelihood method
# ---------------------------------------------------------------------------


class TestPreSelectionMaxLikelihood:
    """Pre-selection with MaxLikelihoodMethod (multi-state) qubits."""

    def test_disallowed_states_derived_correctly(
        self, single_qubit_model, ml_method, presel_config
    ):
        _configure_qubit_for_preselection(
            single_qubit_model.qubits[0],
            disallowed_states={1},
            method=ml_method,
        )
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        ps = [
            instr
            for instr in result.instructions
            if isinstance(instr, PostSelect) and instr.output_variable == "presel_0"
        ]
        assert len(ps) == 1
        assert 1 in ps[0].additional_disallowed

    def test_four_state_dimon(self, single_qubit_model, presel_config):
        """Dimon-style 4-state qubit: states at indices 1, 2, 3 are disallowed."""
        method = MaxLikelihoodMethod(
            states={
                0: MLDiscriminateParams(location=1 + 0j),
                1: MLDiscriminateParams(location=-1 + 0j),
                -2: MLDiscriminateParams(location=0 + 1j),
                -3: MLDiscriminateParams(location=0 - 1j),
            },
        )
        _configure_qubit_for_preselection(
            single_qubit_model.qubits[0],
            disallowed_states={1},
            method=method,
        )
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        ps = [
            instr
            for instr in result.instructions
            if isinstance(instr, PostSelect) and instr.output_variable == "presel_0"
        ]
        assert len(ps) == 1
        # Disallowed states are in PostSelect.additional_disallowed.
        assert 1 in ps[0].additional_disallowed


# ---------------------------------------------------------------------------
# Tests — multi-qubit
# ---------------------------------------------------------------------------


class TestPreSelectionMultiQubit:
    """Pre-selection on multi-qubit circuits with mixed flags."""

    def test_only_flagged_qubits_get_preselection(self, two_qubit_model, presel_config):
        # Only qubit 0 needs pre-selection; explicitly disable for qubit 1.
        _configure_qubit_for_preselection(two_qubit_model.qubits[0])
        two_qubit_model.qubits[1].preselect_disallowed_states = set()
        builder = _build_simple_ir(two_qubit_model)
        res_mgr = _build_res_mgr(two_qubit_model, builder)

        result = _run_preselection(two_qubit_model, builder, res_mgr, presel_config)

        presel_acquires = [
            instr
            for instr in result.instructions
            if isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
        ]
        # Only one pre-selection acquire (for qubit 0).
        assert len(presel_acquires) == 1

    def test_both_qubits_preselected(self, two_qubit_model, presel_config):
        for qubit in two_qubit_model.qubits.values():
            _configure_qubit_for_preselection(qubit)
        builder = _build_simple_ir(two_qubit_model)
        res_mgr = _build_res_mgr(two_qubit_model, builder)

        result = _run_preselection(two_qubit_model, builder, res_mgr, presel_config)

        presel_acquires = [
            instr
            for instr in result.instructions
            if isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
        ]
        assert len(presel_acquires) == 2


# ---------------------------------------------------------------------------
# Tests — legacy qubit (mean_z_map_args, post_process_method=None)
# ---------------------------------------------------------------------------


class TestPreSelectionLegacyQubit:
    """Pre-selection with legacy qubits (mean_z_map_args set, post_process_method=None)."""

    @staticmethod
    def _configure_legacy_qubit(qubit, disallowed_states=None):
        """Set up a qubit in legacy mode: mean_z_map_args set, post_process_method=None.

        The Qubit mutex validator prevents simultaneous assignment of both fields, so we
        bypass it via ``__dict__`` after model construction.
        """
        qubit.__dict__["post_process_method"] = None
        qubit.__dict__["mean_z_map_args"] = [1 + 0j, 0j]
        qubit.__dict__["preselect_disallowed_states"] = (
            disallowed_states if disallowed_states is not None else {1}
        )

    def test_legacy_qubit_injects_presel_instructions(
        self, single_qubit_model, presel_config
    ):
        self._configure_legacy_qubit(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)
        original_count = len(builder.instructions)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)
        assert len(result.instructions) > original_count

    def test_legacy_qubit_emits_full_chain(self, single_qubit_model, presel_config):
        """Legacy mean_z_map_args qubit must produce Equalise → Discriminate →
        PostSelect."""
        self._configure_legacy_qubit(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        presel_var = "presel_0"
        chain_types = [
            type(instr)
            for instr in result.instructions
            if (
                isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
            )
            or (hasattr(instr, "output_variable") and instr.output_variable == presel_var)
        ]
        assert Acquire in chain_types
        assert Equalise in chain_types
        assert Discriminate in chain_types
        assert PostSelect in chain_types

    def test_legacy_qubit_disallowed_states_in_post_select(
        self, single_qubit_model, presel_config
    ):
        self._configure_legacy_qubit(single_qubit_model.qubits[0], disallowed_states={1})
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        ps = [
            instr
            for instr in result.instructions
            if isinstance(instr, PostSelect) and instr.output_variable == "presel_0"
        ]
        assert len(ps) == 1
        assert 1 in ps[0].additional_disallowed

    def test_legacy_qubit_equalise_transform_matches_build_legacy_equalise_args(
        self, single_qubit_model, presel_config
    ):
        """Equalise transform/offset must match
        QuantumInstructionBuilder.build_legacy_equalise_args."""
        import numpy as np

        from qat.ir.instruction_builder import PydQuantumInstructionBuilder

        mean_z_map_args = [2 + 1j, 0.5 + 0.25j]
        self._configure_legacy_qubit(single_qubit_model.qubits[0])
        single_qubit_model.qubits[0].__dict__["mean_z_map_args"] = mean_z_map_args
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        eq = next(
            instr
            for instr in result.instructions
            if isinstance(instr, Equalise) and instr.output_variable == "presel_0"
        )
        expected_transform, expected_offset, _ = (
            PydQuantumInstructionBuilder.build_legacy_equalise_args(mean_z_map_args)
        )
        np.testing.assert_array_almost_equal(eq.transform, expected_transform)
        np.testing.assert_array_almost_equal(eq.offset, expected_offset)

    def test_out_of_the_box_no_explicit_configuration(
        self, single_qubit_model, presel_config
    ):
        """A fresh Lucy model qubit has mean_z_map_args and preselect_disallowed_states={1}
        by default, so pre-selection should activate with no per-qubit configuration."""
        # Do NOT call _configure_qubit_for_preselection — use factory defaults.
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)
        original_count = len(builder.instructions)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        assert len(result.instructions) > original_count
        presel_acquires = [
            instr
            for instr in result.instructions
            if isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
        ]
        assert len(presel_acquires) == 1


# ---------------------------------------------------------------------------
# Tests — mid-circuit validation relaxation
# ---------------------------------------------------------------------------


class TestMidCircuitValidationRelaxation:
    """NoMidCircuitMeasurementValidation should skip pre-selection acquires."""

    def test_preselection_does_not_trigger_mid_circuit_error(
        self, single_qubit_model, presel_config
    ):
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        # This should NOT raise, because the pre-selection acquire
        # has purpose=PRE_SELECTION.
        NoMidCircuitMeasurementValidation(single_qubit_model).run(builder)


# ---------------------------------------------------------------------------
# Tests — inactive qubit filtering
# ---------------------------------------------------------------------------


class TestPreSelectionInactiveQubitFiltering:
    """Pre-selection must not emit blocks for inactive qubits."""

    def test_inactive_qubit_excluded(self, two_qubit_model, presel_config):
        """A 2-qubit model where only qubit 0 is used in the circuit.

        Both qubits are configured for pre-selection, but only qubit 0 should receive a pre-
        selection block because qubit 1 has no active pulse channels.
        """
        for qubit in two_qubit_model.qubits.values():
            _configure_qubit_for_preselection(qubit)

        # Build IR that only touches qubit 0.
        builder = PydQuantumInstructionBuilder(hardware_model=two_qubit_model)
        builder.repeat(100)
        builder.X(target=two_qubit_model.qubits[0])
        builder.measure_with_granular_post_processing(two_qubit_model.qubits[0])

        res_mgr = _build_res_mgr(two_qubit_model, builder)

        result = _run_preselection(two_qubit_model, builder, res_mgr, presel_config)

        presel_acquires = [
            instr
            for instr in result.instructions
            if isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
        ]
        # Only qubit 0 should have a pre-selection acquire.
        assert len(presel_acquires) == 1

        presel_post_selects = [
            instr
            for instr in result.instructions
            if isinstance(instr, PostSelect) and instr.output_variable.startswith("presel_")
        ]
        assert len(presel_post_selects) == 1
        assert presel_post_selects[0].output_variable == "presel_0"


# ---------------------------------------------------------------------------
# Tests — disallowed states passed via PostSelect.additional_disallowed
# ---------------------------------------------------------------------------


class TestPreSelectionAlwaysDisallowsBackground:
    """Disallowed states are passed through PostSelect.additional_disallowed.

    The Discriminate method is left unchanged; the PostSelect instruction carries the
    preselect_disallowed_states indices directly.
    """

    def test_disallowed_states_linear_map(self, single_qubit_model, presel_config):
        """PostSelect.additional_disallowed contains the configured index for
        LinearMapToRealMethod."""
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        ps = [
            instr
            for instr in result.instructions
            if isinstance(instr, PostSelect) and instr.output_variable.startswith("presel_")
        ]
        assert len(ps) == 1
        # preselect_disallowed_states={1} should appear in additional_disallowed.
        assert 1 in ps[0].additional_disallowed

    def test_disallowed_states_max_likelihood(
        self, single_qubit_model, ml_method, presel_config
    ):
        """PostSelect.additional_disallowed contains the configured index for
        MaxLikelihoodMethod."""
        _configure_qubit_for_preselection(
            single_qubit_model.qubits[0],
            disallowed_states={1},
            method=ml_method,
        )
        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        ps = [
            instr
            for instr in result.instructions
            if isinstance(instr, PostSelect) and instr.output_variable.startswith("presel_")
        ]
        assert len(ps) == 1
        assert 1 in ps[0].additional_disallowed

    def test_presel_measure_delay_uses_generated_acquire_delay(
        self, single_qubit_model, presel_config, mocker
    ):
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_simple_ir(single_qubit_model)
        qubit = single_qubit_model.qubits[0]

        meas_pulse, acquire = builder._generate_measure_acquire(
            qubit,
            mode=AcquireMode.INTEGRATOR,
            output_variable="presel_0",
        )
        acquire.delay = 7.5e-7
        mocker.patch.object(
            builder,
            "_generate_measure_acquire",
            return_value=[meas_pulse, acquire],
        )

        res_mgr = _build_res_mgr(single_qubit_model, builder)
        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        presel_acquire_idx = next(
            i
            for i, instr in enumerate(result.instructions)
            if isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
        )
        pre_delay = result.instructions[presel_acquire_idx - 1]

        assert isinstance(pre_delay, Delay)
        assert pre_delay.target == acquire.target
        assert pre_delay.duration == pytest.approx(7.5e-7)

    def test_presel_post_acquire_delay_uses_resonator_relaxation_delay(
        self, single_qubit_model, presel_config
    ):
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        qubit = single_qubit_model.qubits[0]
        qubit.resonator.relaxation_delay = 3.2e-6

        builder = _build_simple_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)
        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        presel_acquire_idx = next(
            i
            for i, instr in enumerate(result.instructions)
            if isinstance(instr, Acquire) and instr.purpose == AcquirePurpose.PRE_SELECTION
        )
        post_delay = result.instructions[presel_acquire_idx + 1]

        assert isinstance(post_delay, Delay)
        assert post_delay.target == result.instructions[presel_acquire_idx].target
        assert post_delay.duration == pytest.approx(3.2e-6)


# ---------------------------------------------------------------------------
# Tests — ActivePulseChannelResults update
# ---------------------------------------------------------------------------


class TestActiveChannelResultsUpdate:
    """InsertPreSelectionMeasurement must register the injected measure/acquire channels in
    ActivePulseChannelResults so that InactivePulseChannelSanitisation (which runs
    afterwards in the pipeline) does not strip the injected instructions."""

    def test_measure_channel_added_to_active_results(
        self, single_qubit_model, presel_config
    ):
        """Measure pulse channel is added to ActivePulseChannelResults after injection."""
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        qubit = single_qubit_model.qubits[0]
        # Drive-only IR: measure/acquire channels are NOT active before the pass.
        builder = _build_drive_only_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        from qat.middleend.passes.analysis import ActivePulseChannelResults

        active_before = res_mgr.lookup_by_type(ActivePulseChannelResults).targets
        assert qubit.measure_pulse_channel.uuid not in active_before

        _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        active_after = res_mgr.lookup_by_type(ActivePulseChannelResults).targets
        assert qubit.measure_pulse_channel.uuid in active_after

    def test_acquire_channel_added_to_active_results(
        self, single_qubit_model, presel_config
    ):
        """Acquire pulse channel is added to ActivePulseChannelResults after injection."""
        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        qubit = single_qubit_model.qubits[0]
        builder = _build_drive_only_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        from qat.middleend.passes.analysis import ActivePulseChannelResults

        active_before = res_mgr.lookup_by_type(ActivePulseChannelResults).targets
        assert qubit.resonator.acquire_pulse_channel.uuid not in active_before

        _run_preselection(single_qubit_model, builder, res_mgr, presel_config)

        active_after = res_mgr.lookup_by_type(ActivePulseChannelResults).targets
        assert qubit.resonator.acquire_pulse_channel.uuid in active_after

    def test_injected_instructions_survive_inactive_channel_sanitisation(
        self, single_qubit_model, presel_config
    ):
        """Injected pre-selection Pulse/Acquire are not stripped by
        InactivePulseChannelSanitisation when the circuit has no prior measurements."""
        from qat.ir.measure import Acquire
        from qat.middleend.passes.transform import InactivePulseChannelSanitisation

        _configure_qubit_for_preselection(single_qubit_model.qubits[0])
        builder = _build_drive_only_ir(single_qubit_model)
        res_mgr = _build_res_mgr(single_qubit_model, builder)

        result = _run_preselection(single_qubit_model, builder, res_mgr, presel_config)
        result = InactivePulseChannelSanitisation().run(result, res_mgr)

        presel_acquires = [
            i
            for i in result.instructions
            if isinstance(i, Acquire) and i.purpose == AcquirePurpose.PRE_SELECTION
        ]
        assert len(presel_acquires) == 1, (
            "Pre-selection Acquire was stripped by InactivePulseChannelSanitisation"
        )
