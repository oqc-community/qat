# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

from copy import deepcopy

import numpy as np
import pytest

from qat.backend.passes.purr.analysis import (
    BindingPass,
    BindingResult,
    IterBound,
    TILegalisationPass,
    TriagePass,
    TriageResult,
)
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.backend.qblox.passes.analysis import (
    PreCodegenPass,
    PreCodegenResult,
    QbloxLegalisationPass,
)
from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.middleend.passes.purr.transform import RepeatSanitisation, ScopeSanitisation
from qat.purr.compiler.instructions import DeviceUpdate

from tests.unit.utils.builder_nuggets import resonator_spect


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
class TestAnalysisPasses:
    def test_precodegen_pass(self, qblox_model):
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        builder = resonator_spect(qblox_model)

        target_data = QbloxTargetData.default()
        pipeline = (
            PassManager()
            | RepeatSanitisation(qblox_model, target_data)
            | ScopeSanitisation()
            | TriagePass()
            | BindingPass()
            | TILegalisationPass()
            | QbloxLegalisationPass()
            | PreCodegenPass()
        )

        pipeline.run(builder, res_mgr, met_mgr)

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        target_map = triage_result.target_map
        precodegen_result: PreCodegenResult = res_mgr.lookup_by_type(PreCodegenResult)

        assert precodegen_result.alloc_mgrs.keys() == target_map.keys()
        for target, alloc_mgr in precodegen_result.alloc_mgrs.items():
            assert len(alloc_mgr.registers) >= 2
            assert len(alloc_mgr.labels) >= 2

    def test_qblox_legalisation_pass(self, qblox_model):
        builder = resonator_spect(qblox_model)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()

        (
            PassManager()
            | ScopeSanitisation()
            | TriagePass()
            | BindingPass()
            | TILegalisationPass()
        ).run(builder, res_mgr, met_mgr)

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        binding_result: BindingResult = deepcopy(res_mgr.lookup_by_type(BindingResult))

        QbloxLegalisationPass().run(builder, res_mgr)

        legal_binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)

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
                        inst
                        for inst in rw_result.reads[name]
                        if isinstance(inst, DeviceUpdate)
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
