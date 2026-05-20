# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
"""Unit tests for runtime transform passes that post-process acquisition results and format
runtime outputs.

Includes tests for AcquisitionPostprocessing, QBlox-specific handling and the result
formatting transforms used by the runtime.
"""

import numpy as np
import pytest
from compiler_config.config import InlineResultsProcessing

from qat.backend.qblox.acquisition import (
    Acquisition,
    BinnedAcqData,
    BinnedAndScopeAcqData,
    IntegData,
)
from qat.core.result_base import ResultManager
from qat.executables import AcquireData, Executable
from qat.ir.instruction_basetypes import AcquireMode, PostProcessType, ProcessAxis
from qat.ir.instructions import Assign as PydAssign, Variable as PydVariable
from qat.ir.measure import Discriminate, Equalise, PostProcessing, PostSelect
from qat.model.post_processing import MaxLikelihoodMethod, MLStateMap
from qat.model.target_data import TargetData
from qat.runtime.passes.analysis import PostSelectionResult
from qat.runtime.passes.transform import (
    AcquisitionPostprocessing,
    AssignResultsTransform,
    InlineResultsProcessingTransform,
    QBloxAcquisitionPostProcessing,
)


class TestPostProcessingTransform:
    """Tests for AcquisitionPostprocessing using legacy PostProcessing instructions.

    Verifies that MEAN, LINEAR_MAP_COMPLEX_TO_REAL and DISCRIMINATE post-processing steps
    produce correctly shaped and valued outputs for both RAW and INTEGRATOR acquire modes.
    """

    target_data = TargetData()

    def test_raw_to_bits(self):
        """RAW acquisition data is correctly reduced to discriminated bits.

        Applies MEAN(TIME) → LINEAR_MAP_COMPLEX_TO_REAL → DISCRIMINATE to a (1000, 254) RAW
        array and verifies the result is a (1000,) array of -1.0.
        """
        mock_readout = {"test": np.ones((1000, 254))}
        pp_instructions = [
            PostProcessing(
                output_variable="test",
                process_type="mean",
                axes=[ProcessAxis.TIME],
            ),
            PostProcessing(
                output_variable="test",
                process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                axes=[ProcessAxis.SEQUENCE],
                args=[-2.54, 1.1],
            ),
            PostProcessing(
                output_variable="test",
                process_type=PostProcessType.DISCRIMINATE,
                axes=[ProcessAxis.SEQUENCE],
                args=[1.6],
            ),
        ]
        acquire = AcquireData(
            mode=AcquireMode.RAW,
            shape=(1000, 254),
            post_processing=pp_instructions,
            results_processing=InlineResultsProcessing.Experiment,
            physical_channel="ch1",
        )
        package = Executable(programs=[], acquires={"test": acquire})
        result = AcquisitionPostprocessing(self.target_data).run(
            mock_readout, package=package
        )
        assert len(result) == 1
        assert "test" in result
        assert np.shape(result["test"]) == (1000,)
        assert np.allclose(result["test"], -1.0)

    def test_integrator_to_bits(self):
        """INTEGRATOR acquisition data is correctly reduced to discriminated bits.

        Applies LINEAR_MAP_COMPLEX_TO_REAL → DISCRIMINATE to a (1000,) array and verifies
        the result is a (1000,) array of -1.0.
        """
        mock_readout = {"test": np.ones(1000)}
        pp_instructions = [
            PostProcessing(
                output_variable="test",
                process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                axes=[ProcessAxis.SEQUENCE],
                args=[-2.54, 1.1],
            ),
            PostProcessing(
                output_variable="test",
                process_type=PostProcessType.DISCRIMINATE,
                axes=[ProcessAxis.SEQUENCE],
                args=[1.6],
            ),
        ]
        acquire = AcquireData(
            mode=AcquireMode.RAW,
            shape=(1000, 254),
            post_processing=pp_instructions,
            results_processing=InlineResultsProcessing.Experiment,
            physical_channel="ch1",
        )
        package = Executable(programs=[], acquires={"test": acquire})
        result = AcquisitionPostprocessing(self.target_data).run(
            mock_readout, package=package
        )
        assert len(result) == 1
        assert "test" in result
        assert np.shape(result["test"]) == (1000,)
        assert np.allclose(result["test"], -1.0)


class TestInlineResultsProcessingTransform:
    """Tests for InlineResultsProcessingTransform result-format handling.

    Verifies that Program-mode processing reduces an array to a scalar and that Experiment-
    mode processing leaves an ndarray unchanged.
    """

    def test_run_results_processing_with_program(self):
        """Program-mode processing reduces the acquisition array to a single integer."""
        results = {"test": np.random.rand(254, 100)}
        acquire = AcquireData(
            mode=AcquireMode.RAW,
            shape=(254, 100),
            results_processing=InlineResultsProcessing.Program,
            physical_channel="ch1",
        )
        package = Executable(programs=[], acquires={"test": acquire})
        results = InlineResultsProcessingTransform().run(results, package=package)
        assert isinstance(results["test"], int)

    def test_run_results_processing_with_experiment(self):
        """Experiment-mode processing leaves the acquisition array as an ndarray."""
        results = {"test": np.random.rand(254)}
        acquire = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(254,),
            results_processing=InlineResultsProcessing.Experiment,
            physical_channel="ch1",
        )
        package = Executable(programs=[], acquires={"test": acquire})
        InlineResultsProcessingTransform().run(results, package=package)
        assert isinstance(results["test"], np.ndarray)
        assert len(results["test"]) == 254


class TestAssignResultsTransform:
    """Tests for AssignResultsTransform classical variable assignment and output
    filtering."""

    def test_only_returns_what_is_asked(self):
        """Only variable names listed in the package returns set appear in the output."""
        results = {"q0": np.random.rand(100), "q1": np.random.rand(100)}
        payload = Executable(programs=[], acquires={}, returns=set(["q0"]))
        results = AssignResultsTransform().run(results, package=payload)
        assert "q0" in results
        assert "q1" not in results

    def test_assigns_with_variables(self):
        """Assign instructions referencing Variable objects are resolved from the results
        dict and the nested list structure is preserved in the output."""
        results = {
            "q0": np.asarray([1] * 100),
            "q1": np.asarray([2] * 100),
            "q2": np.asarray([3] * 100),
        }
        package = Executable(
            programs=[],
            acquires={},
            returns=set(["c"]),
            assigns=[
                PydAssign(
                    name="c",
                    value=[
                        PydVariable(name="q0"),
                        PydVariable(name="q1"),
                        [PydVariable(name="q2")],
                    ],
                )
            ],
        )

        results = AssignResultsTransform().run(results, package=package)
        assert len(results) == 1
        assert "c" in results
        results = results["c"]
        assert len(results) == 3
        assert isinstance(results[0], np.ndarray)
        assert np.allclose(results[0], 1)
        assert len(results[0]) == 100
        assert isinstance(results[1], np.ndarray)
        assert np.allclose(results[1], 2)
        assert len(results[1]) == 100
        assert isinstance(results[2], list)
        assert isinstance(results[2][0], np.ndarray)
        assert np.allclose(results[2][0], 3)
        assert len(results[2][0]) == 100


class TestQBloxAcquisitionPostProcessing:
    """Tests for QBloxAcquisitionPostProcessing.

    Each test builds a minimal playback dict with one pulse channel and one named
    acquisition, constructs the matching Executable, and verifies the post-processed result.
    """

    _shots = 10
    _multiplier = 2.0
    _constant = 0.5

    def _make_integrator_acquisition(self, iq_values: np.ndarray) -> Acquisition:
        """Build an Acquisition whose integrator bins contain the supplied IQ values."""
        return Acquisition(
            acquisition=BinnedAndScopeAcqData(
                bins=BinnedAcqData(
                    integration=IntegData(
                        path0=iq_values.real.tolist(),
                        path1=iq_values.imag.tolist(),
                    ),
                    threshold=np.zeros(len(iq_values)).tolist(),
                )
            )
        )

    def _make_discriminator_acquisition(self, threshold_values: np.ndarray) -> Acquisition:
        """Build an Acquisition whose threshold bins contain the supplied values."""
        return Acquisition(
            acquisition=BinnedAndScopeAcqData(
                bins=BinnedAcqData(
                    integration=IntegData(
                        path0=np.zeros(len(threshold_values)).tolist(),
                        path1=np.zeros(len(threshold_values)).tolist(),
                    ),
                    threshold=threshold_values.tolist(),
                )
            )
        )

    def _make_package(self, pp_instructions: list, shape: tuple) -> Executable:
        acquire = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=shape,
            post_processing=pp_instructions,
            physical_channel="ch0",
        )
        return Executable(programs=[], acquires={"v": acquire})

    def test_linear_map_complex_to_real(self):
        """LINEAR_MAP_COMPLEX_TO_REAL is correctly handled via apply_post_processing."""
        iq = np.ones(self._shots, dtype=complex)
        playback = {"ch0": {"v": self._make_integrator_acquisition(iq)}}
        pp = [
            PostProcessing(
                output_variable="v",
                process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                args=[self._multiplier, self._constant],
            )
        ]
        package = self._make_package(pp, (self._shots,))
        result = QBloxAcquisitionPostProcessing().run(playback, package=package)
        expected = np.real(self._multiplier * iq + self._constant)
        assert np.allclose(result["v"], expected)

    def test_discriminate_uses_hardware_threshold(self):
        """DISCRIMINATE reads thrld_data directly (hardware path), not raw IQ."""
        threshold_vals = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0], dtype=float)
        playback = {"ch0": {"v": self._make_discriminator_acquisition(threshold_vals)}}
        pp = [
            PostProcessing(
                output_variable="v",
                process_type=PostProcessType.DISCRIMINATE,
            )
        ]
        package = self._make_package(pp, (self._shots,))
        result = QBloxAcquisitionPostProcessing().run(playback, package=package)
        expected = 1 - 2 * threshold_vals
        assert np.allclose(result["v"], expected)

    def test_equalise_discriminate_with_linear_map_method(self):
        """Equalise with a rotation-and-shift matrix produces the same real projection as
        LINEAR_MAP_COMPLEX_TO_REAL applied to all-ones IQ data."""
        iq = np.ones(self._shots, dtype=complex)
        playback = {"ch0": {"v": self._make_integrator_acquisition(iq)}}
        m = complex(self._multiplier)
        a, b = m.real, m.imag
        pp = [
            Equalise(
                output_variable="v",
                transform=np.array([[a, -b], [b, a]], dtype=float),
                offset=np.array([self._constant, 0.0], dtype=float),
            ),
        ]
        package = self._make_package(pp, (self._shots,))
        result = QBloxAcquisitionPostProcessing().run(playback, package=package)
        expected = np.real(self._multiplier * iq + self._constant)
        assert np.allclose(result["v"], expected)

    def test_unknown_post_process_type_raises(self):
        """An unrecognised PostProcessType raises NotImplementedError."""
        iq = np.ones(self._shots, dtype=complex)
        playback = {"ch0": {"v": self._make_integrator_acquisition(iq)}}
        pp = [PostProcessing(output_variable="v", process_type=PostProcessType.MUL)]
        package = self._make_package(pp, (self._shots,))
        with pytest.raises(NotImplementedError):
            QBloxAcquisitionPostProcessing().run(playback, package=package)


class TestQBloxAcquisitionPostProcessingPostSelection:
    """Tests for post-selection via QBloxAcquisitionPostProcessing.

    Mirrors TestAcquisitionPostprocessingPostSelection but exercises the QBlox playback path
    so that per-output masks are accumulated, ANDed into a global mask, applied to the
    output arrays and recorded in the ResultManager.
    """

    _n_shots = 6

    def _make_integrator_acquisition(self, iq_values: np.ndarray) -> Acquisition:
        """Build an Acquisition whose integrator bins contain the supplied IQ values."""
        return Acquisition(
            acquisition=BinnedAndScopeAcqData(
                bins=BinnedAcqData(
                    integration=IntegData(
                        path0=iq_values.real.tolist(),
                        path1=iq_values.imag.tolist(),
                    ),
                    threshold=np.zeros(len(iq_values)).tolist(),
                )
            )
        )

    def _make_package(self, name: str, pp_instructions: list, n_shots: int) -> Executable:
        """Build a minimal Executable with a single INTEGRATOR acquire keyed by ``name``."""
        acquire = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(n_shots,),
            post_processing=pp_instructions,
            physical_channel="ch0",
        )
        return Executable(programs=[], acquires={name: acquire})

    def _linear_pp(self, name: str, disallowed_states: list) -> list:
        """Equalise(identity) → Discriminate(threshold=0) → PostSelect."""
        instructions = [
            Equalise(
                output_variable=name,
                transform=np.eye(2, dtype=float),
                offset=np.zeros(2),
            ),
            Discriminate(output_variable=name, threshold=0.0),
        ]
        if disallowed_states:
            instructions.append(
                PostSelect(output_variable=name, disallowed_states=disallowed_states)
            )
        return instructions

    def test_post_select_filters_shots_and_records_metadata(self):
        """Shots classified to the disallowed state are removed and PostSelectionResult is
        stored in res_mgr."""
        # positive (label "0", valid), negative (label "1", invalid), alternating
        iq = np.array([1.0, -1.0, 0.5, -0.5, 0.8, -0.2], dtype=complex)
        pp = self._linear_pp("v", disallowed_states=["1"])
        playback = {"ch0": {"v": self._make_integrator_acquisition(iq)}}
        package = self._make_package("v", pp, self._n_shots)
        res_mgr = ResultManager()

        result = QBloxAcquisitionPostProcessing().run(playback, res_mgr, package=package)

        assert result["v"].shape == (3,)
        post_sel = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_sel.shots_requested == self._n_shots
        assert post_sel.shots_retained == 3

    def test_all_shots_filtered_returns_empty_array(self):
        """When every shot is disallowed the output array is empty."""
        iq = np.array([1.0, -1.0, 0.5, -0.5], dtype=complex)
        pp = self._linear_pp("v", disallowed_states=["0", "1"])
        playback = {"ch0": {"v": self._make_integrator_acquisition(iq)}}
        package = self._make_package("v", pp, 4)
        res_mgr = ResultManager()

        result = QBloxAcquisitionPostProcessing().run(playback, res_mgr, package=package)

        assert result["v"].shape == (0,)
        assert res_mgr.lookup_by_type(PostSelectionResult).shots_retained == 0

    def test_no_post_select_does_not_record_metadata(self):
        """Without a PostSelect instruction no PostSelectionResult is added to res_mgr."""
        iq = np.ones(4, dtype=complex)
        pp = [Discriminate(output_variable="v", threshold=0.0)]
        playback = {"ch0": {"v": self._make_integrator_acquisition(iq)}}
        package = self._make_package("v", pp, 4)
        res_mgr = ResultManager()

        QBloxAcquisitionPostProcessing().run(playback, res_mgr, package=package)

        assert not res_mgr.check_for_type(PostSelectionResult)

    def test_global_mask_anded_across_two_channels(self):
        """When two named acquires both post-select, the global AND mask is applied to both
        and a single PostSelectionResult is stored."""
        # ch0/q0: valid, invalid, valid, valid
        # ch1/q1: valid, valid, invalid, valid
        # global:  valid, invalid, invalid, valid → 2 retained
        iq0 = np.array([1.0, -1.0, 0.5, 0.8], dtype=complex)
        iq1 = np.array([0.5, 0.9, -1.0, 0.3], dtype=complex)
        pp0 = self._linear_pp("q0", disallowed_states=["1"])
        pp1 = self._linear_pp("q1", disallowed_states=["1"])
        acquire0 = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(4,),
            post_processing=pp0,
            physical_channel="ch0",
        )
        acquire1 = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(4,),
            post_processing=pp1,
            physical_channel="ch1",
        )
        package = Executable(programs=[], acquires={"q0": acquire0, "q1": acquire1})
        playback = {
            "ch0": {"q0": self._make_integrator_acquisition(iq0)},
            "ch1": {"q1": self._make_integrator_acquisition(iq1)},
        }
        res_mgr = ResultManager()

        result = QBloxAcquisitionPostProcessing().run(playback, res_mgr, package=package)

        assert result["q0"].shape == (2,)
        assert result["q1"].shape == (2,)
        post_sel = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_sel.shots_requested == 4
        assert post_sel.shots_retained == 2

    def test_res_mgr_none_does_not_raise(self):
        """Omitting res_mgr (default None) is safe; the pass returns normally."""
        iq = np.ones(4, dtype=complex)
        pp = [Discriminate(output_variable="v", threshold=0.0)]
        playback = {"ch0": {"v": self._make_integrator_acquisition(iq)}}
        package = self._make_package("v", pp, 4)

        # Should not raise even without a ResultManager.
        result = QBloxAcquisitionPostProcessing().run(playback, package=package)
        assert "v" in result


class TestAcquisitionPostprocessingPostSelection:
    """Tests for post-selection via the granular Equalise → Discriminate → PostSelect
    chain."""

    _n_shots = 6

    def _make_linear_chain(self, output_variable, disallowed_states):
        """Emit Equalise(identity) → Discriminate(threshold=0) → PostSelect(disallowed)."""
        instructions = [
            Equalise(
                output_variable=output_variable,
                transform=np.eye(2, dtype=float),
                offset=np.zeros(2),
            ),
            Discriminate(output_variable=output_variable, threshold=0.0),
        ]
        if disallowed_states:
            instructions.append(
                PostSelect(
                    output_variable=output_variable, disallowed_states=disallowed_states
                )
            )
        return instructions

    def _make_ml_chain(self, output_variable, states):
        """Build a Discriminate(MaxLikelihood) → PostSelect chain for the given states.

        PostSelect is only appended when at least one state is marked disallowed.
        """
        method = MaxLikelihoodMethod(states=states)
        disallowed = [s.label for s in states if s.disallowed]
        instructions = [Discriminate(output_variable=output_variable, method=method)]
        if disallowed:
            instructions.append(
                PostSelect(output_variable=output_variable, disallowed_states=disallowed)
            )
        return instructions

    def _package_and_acquisitions(self, pp, response):
        """Build a single-qubit Executable and matching acquisitions dict from a response
        list."""
        acquire = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(len(response),),
            post_processing=pp,
            physical_channel="ch0",
        )
        package = Executable(programs=[], acquires={"q0": acquire})
        acquisitions = {"q0": np.asarray(response, dtype=float)}
        return package, acquisitions

    def test_linear_disallowed_state_1_reduces_shots(self):
        """Shots classified to label "1" (negative projection) are removed."""
        pp = self._make_linear_chain("q0", disallowed_states=["1"])
        # positive(valid), negative(invalid), positive, negative, positive, negative
        response = [1.0, -1.0, 0.5, -0.5, 0.8, -0.2]
        package, acquisitions = self._package_and_acquisitions(pp, response)
        res_mgr = ResultManager()
        result = AcquisitionPostprocessing().run(acquisitions, res_mgr, package=package)
        assert result["q0"].shape == (3,)
        post_sel = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_sel.shots_requested == 6
        assert post_sel.shots_retained == 3

    def test_ml_three_states_one_disallowed_reduces_shots(self):
        """Shots classified to a disallowed ML state are removed."""
        states = [
            MLStateMap(label="ten", output_value=0, location=1.0 + 0j, disallowed=False),
            MLStateMap(
                label="thirty", output_value=1, location=-1.0 + 0j, disallowed=False
            ),
            MLStateMap(label="twenty", output_value=2, location=0.0 + 1j, disallowed=True),
        ]
        pp = self._make_ml_chain("q0", states)
        # near state0, near state1, near state2 (disallowed), state0, state0, state1
        response = np.array(
            [0.9 + 0j, -0.9 + 0j, 0.05 + 0.9j, 1.1 + 0j, 0.8 + 0j, -0.7 + 0j]
        )
        acquire = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(6,),
            post_processing=pp,
            physical_channel="ch0",
        )
        package = Executable(programs=[], acquires={"q0": acquire})
        acquisitions = {"q0": response}
        res_mgr = ResultManager()
        result = AcquisitionPostprocessing().run(acquisitions, res_mgr, package=package)
        assert result["q0"].shape == (5,)
        post_sel = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_sel.shots_requested == 6
        assert post_sel.shots_retained == 5

    def test_global_mask_applied_across_two_acquires(self):
        """When two acquires both post-select, the global mask is ANDed."""
        pp0 = self._make_linear_chain("q0", disallowed_states=["1"])
        pp1 = self._make_linear_chain("q1", disallowed_states=["1"])
        n = 4
        # q0: valid, invalid, valid, valid
        # q1: valid, valid, invalid, valid
        # global: valid, invalid, invalid, valid → 2 shots retained
        acquire0 = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(n,),
            post_processing=pp0,
            physical_channel="ch0",
        )
        acquire1 = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(n,),
            post_processing=pp1,
            physical_channel="ch1",
        )
        package = Executable(programs=[], acquires={"q0": acquire0, "q1": acquire1})
        acquisitions = {
            "q0": np.array([1.0, -1.0, 0.5, 0.8]),
            "q1": np.array([0.5, 0.9, -1.0, 0.3]),
        }
        res_mgr = ResultManager()
        result = AcquisitionPostprocessing().run(acquisitions, res_mgr, package=package)
        assert result["q0"].shape == (2,)
        assert result["q1"].shape == (2,)
        post_sel = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_sel.shots_requested == 4
        assert post_sel.shots_retained == 2

    def test_all_shots_filtered_returns_empty_arrays(self):
        """When every shot is disallowed, all output arrays are empty."""
        pp = self._make_linear_chain("q0", disallowed_states=["0", "1"])
        response = [1.0, -1.0, 0.5, -0.5]
        package, acquisitions = self._package_and_acquisitions(pp, response)
        res_mgr = ResultManager()
        result = AcquisitionPostprocessing().run(acquisitions, res_mgr, package=package)
        assert result["q0"].shape == (0,)
        post_sel = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_sel.shots_retained == 0

    def test_non_post_select_passes_unaffected(self):
        """A pipeline without PostSelect does not trigger post-selection.

        Applies only LINEAR_MAP_COMPLEX_TO_REAL; verifies the output shape is unchanged and
        no PostSelectionResult is recorded.
        """
        pp = PostProcessing(
            output_variable="q0",
            process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
            args=[1.0, 0.0],
        )
        acquire = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(4,),
            post_processing=[pp],
            physical_channel="ch0",
        )
        package = Executable(programs=[], acquires={"q0": acquire})
        acquisitions = {"q0": np.array([1.0 + 0j, -1.0 + 0j, 0.5 + 0j, -0.5 + 0j])}
        res_mgr = ResultManager()
        result = AcquisitionPostprocessing().run(acquisitions, res_mgr, package=package)
        assert result["q0"].shape == (4,)
        assert not res_mgr.check_for_type(PostSelectionResult)

    def test_raw_mode_shots_filtered_on_axis_0(self):
        """For RAW acquisitions (n_shots, n_time), the mask must be applied to axis 0.

        The test uses n_shots=4, n_time=8 so the two dimensions are distinct and the wrong-
        axis bug would produce an incorrect shape.
        """
        n_shots = 4
        n_time = 8
        # MEAN(TIME) first so classifier receives a 1-D (n_shots,) array.
        pp_mean = PostProcessing(
            output_variable="q0",
            process_type=PostProcessType.MEAN,
            axes=[ProcessAxis.TIME],
        )
        pp_disc = Discriminate(output_variable="q0", threshold=0.0)
        pp_select = PostSelect(output_variable="q0", disallowed_states=["1"])
        acquire = AcquireData(
            mode=AcquireMode.RAW,
            shape=(n_shots, n_time),
            post_processing=[pp_mean, pp_disc, pp_select],
            physical_channel="ch0",
        )
        package = Executable(programs=[], acquires={"q0": acquire})
        # shots 0,2 → mean > 0 → state 0 (allowed); shots 1,3 → mean < 0 → state 1 (disallowed)
        row_pos = np.ones(n_time, dtype=float)
        row_neg = -np.ones(n_time, dtype=float)
        raw = np.array([row_pos, row_neg, row_pos, row_neg])  # shape (4, 8)
        acquisitions = {"q0": raw}
        res_mgr = ResultManager()
        result = AcquisitionPostprocessing().run(acquisitions, res_mgr, package=package)
        # 2 valid shots retained; result is 1-D after the MEAN step
        assert result["q0"].shape == (2,)
        post_sel = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_sel.shots_requested == n_shots
        assert post_sel.shots_retained == 2

    def test_global_mask_skips_outputs_without_sequence_axis(self):
        """Masking only applies where final axes include ProcessAxis.SEQUENCE.

        q0 carries the post-selection chain and is filtered. q1 is SCOPE data with no
        sequence axis; even if its trailing dimension equals n_shots, it must not be masked.
        """
        n_shots = 4
        pp_q0 = [
            Discriminate(output_variable="q0", threshold=0.0),
            PostSelect(output_variable="q0", disallowed_states=["1"]),
        ]
        acquire_q0 = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(n_shots,),
            post_processing=pp_q0,
            physical_channel="ch0",
        )
        acquire_q1 = AcquireData(
            mode=AcquireMode.SCOPE,
            shape=(n_shots,),
            post_processing=[],
            physical_channel="ch1",
        )
        package = Executable(programs=[], acquires={"q0": acquire_q0, "q1": acquire_q1})
        acquisitions = {
            "q0": np.array([1.0, -1.0, 0.5, -0.5]),
            "q1": np.array([10.0, 20.0, 30.0, 40.0]),
        }
        res_mgr = ResultManager()

        result = AcquisitionPostprocessing().run(acquisitions, res_mgr, package=package)

        assert result["q0"].shape == (2,)
        assert np.allclose(result["q1"], np.array([10.0, 20.0, 30.0, 40.0]))
        post_sel = res_mgr.lookup_by_type(PostSelectionResult)
        assert post_sel.shots_requested == n_shots
        assert post_sel.shots_retained == 2
