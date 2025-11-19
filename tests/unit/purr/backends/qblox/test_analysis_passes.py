# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

from copy import deepcopy

import numpy as np

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
