# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Oxford Quantum Circuits Ltd

from copy import deepcopy

import numpy as np
import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.qblox.analysis_passes import (
    BindingPass,
    BindingResult,
    IterBound,
    QbloxLegalisationPass,
    TILegalisationPass,
    TriagePass,
    TriageResult,
)
from qat.purr.backends.qblox.codegen import PreCodegenPass, PreCodegenResult
from qat.purr.backends.qblox.transform_passes import (
    DesugaringPass,
    RepeatSanitisation,
    ScopeSanitisation,
)
from qat.purr.compiler.instructions import DeviceUpdate
from qat.purr.core.metrics_base import MetricsManager
from qat.purr.core.pass_base import PassManager
from qat.purr.core.result_base import ResultManager

from tests.unit.utils.builder_nuggets import resonator_spect


class TestBindingPassUnevenSweeps:
    @pytest.mark.parametrize(
        "value",
        [
            [1, 2, 4, 10],
            np.array([1, 2, 4, 10]),
            # The first step matches the average step, but the spacing is uneven.
            [1, 2, 4, 4],
            # Boolean differences are a logical xor, which hides the alternating sign.
            [False, True, False],
            np.array([False, True, False]),
        ],
    )
    def test_extract_iter_bound_rejects_uneven_by_default(self, value):
        with pytest.raises(ValueError, match="Not a regularly partitioned space"):
            BindingPass.extract_iter_bound(value)

    @pytest.mark.parametrize("value", [[1, 2, 4, 10], np.array([1, 2, 4, 10])])
    def test_extract_iter_bound_allow_uneven(self, value):
        bound = BindingPass.extract_iter_bound(value, allow_uneven=True)
        assert bound == IterBound(start=1, step=None, end=10, count=4)

    @pytest.mark.parametrize(
        "value", [[False, True, False], np.array([False, True, False])]
    )
    def test_extract_iter_bound_allow_uneven_boolean(self, value):
        bound = BindingPass.extract_iter_bound(value, allow_uneven=True)
        assert bound == IterBound(start=False, step=None, end=False, count=3)

    def test_extract_iter_bound_allow_uneven_drops_step_of_even_values(self):
        bound = BindingPass.extract_iter_bound([-0.1, 0, 0.1, 0.2, 0.3], allow_uneven=True)
        assert bound == IterBound(-0.1, None, 0.3, 5)

    @staticmethod
    def _bind(builder, **pass_kwargs):
        res_mgr = ResultManager()
        (
            PassManager()
            | ScopeSanitisation()
            | DesugaringPass()
            | TriagePass()
            | BindingPass(**pass_kwargs)
        ).run(builder, res_mgr, MetricsManager())
        return res_mgr.lookup_by_type(BindingResult)

    def test_binding_pass_rejects_uneven_sweep_by_default(self):
        builder = resonator_spect(get_default_echo_hardware(), uneven=True)
        with pytest.raises(ValueError, match="Not a regularly partitioned space"):
            self._bind(builder)

    @pytest.mark.parametrize("num_points", [4, 10, 100])
    def test_binding_pass_allows_uneven_sweep(self, num_points):
        model = get_default_echo_hardware()
        uneven = self._bind(
            resonator_spect(model, num_points=num_points, uneven=True),
            allow_uneven_sweeps=True,
        )
        even = self._bind(resonator_spect(model, num_points=num_points))

        def sweep_bounds(result, has_step):
            return {
                (bound.start, bound.end, bound.count)
                for bounds in result.iter_bound_results.values()
                for bound in bounds.values()
                if (bound.step is not None) == has_step and bound.count == num_points
            }

        uneven_bounds = sweep_bounds(uneven, has_step=False)
        assert uneven_bounds
        assert uneven_bounds <= sweep_bounds(even, has_step=True)


def test_precodegen_pass():
    model = get_default_echo_hardware()
    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    builder = resonator_spect(model)

    pipeline = (
        PassManager()
        | RepeatSanitisation(model)
        | ScopeSanitisation()
        | DesugaringPass()
        | TriagePass()
        | BindingPass()
        | TILegalisationPass()
        | QbloxLegalisationPass()
        | PreCodegenPass()
    )

    pipeline.run(builder, res_mgr, met_mgr)

    triage_result = res_mgr.lookup_by_type(TriageResult)
    target_map = triage_result.target_map
    precodegen_result = res_mgr.lookup_by_type(PreCodegenResult)

    assert precodegen_result.alloc_mgrs.keys() == target_map.keys()
    for target, alloc_mgr in precodegen_result.alloc_mgrs.items():
        assert len(alloc_mgr.registers) >= 2
        assert len(alloc_mgr.labels) >= 2


def test_qblox_legalisation_pass():
    model = get_default_echo_hardware()
    builder = resonator_spect(model)
    res_mgr = ResultManager()
    met_mgr = MetricsManager()

    (
        PassManager()
        | ScopeSanitisation()
        | DesugaringPass()
        | TriagePass()
        | BindingPass()
        | TILegalisationPass()
    ).run(builder, res_mgr, met_mgr)

    triage_result = res_mgr.lookup_by_type(TriageResult)
    binding_result = deepcopy(res_mgr.lookup_by_type(BindingResult))

    QbloxLegalisationPass().run(builder, res_mgr)

    legal_binding_result = res_mgr.lookup_by_type(BindingResult)

    for target, instructions in triage_result.target_map.items():
        scoping_result = binding_result.scoping_results[target]
        rw_result = binding_result.rw_results[target]

        bounds = binding_result.iter_bound_results[target]
        legal_bounds = legal_binding_result.iter_bound_results[target]

        assert set(legal_bounds.keys()) == set(bounds.keys())

        for name in scoping_result.symbol2scopes:
            bound = bounds[name]
            legal_bound = legal_bounds[name]
            if name in rw_result.reads:
                device_updates = [
                    inst for inst in rw_result.reads[name] if isinstance(inst, DeviceUpdate)
                ]
                for du in device_updates:
                    assert du.target == target
                    if du.attribute == "frequency":
                        assert legal_bound != bound
                        assert legal_bound == IterBound(
                            start=QbloxLegalisationPass.freq_as_steps(bound.start),
                            step=QbloxLegalisationPass.freq_as_steps(bound.step),
                            end=QbloxLegalisationPass.freq_as_steps(bound.end),
                            count=bound.count,
                        ).astype(np.uint32)
            else:
                assert legal_bound == bound
