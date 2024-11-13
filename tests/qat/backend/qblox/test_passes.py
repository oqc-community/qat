from copy import deepcopy

import numpy as np

from qat.backend.analysis_passes import (
    BindingPass,
    BindingResult,
    IterBound,
    TILegalisationPass,
    TriagePass,
    TriageResult,
)
from qat.backend.transform_passes import RepeatSanitisation, ScopeSanitisation
from qat.ir.pass_base import PassManager
from qat.ir.result_base import ResultManager
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.qblox.analysis_passes import QbloxLegalisationPass
from qat.purr.backends.qblox.codegen import PreCodegenPass, PreCodegenResult
from qat.purr.compiler.instructions import DeviceUpdate

from tests.qat.utils.builder_nuggets import resonator_spect


class TestAnalysisPasses:
    def test_precodegen_pass(self):
        model = get_default_echo_hardware()
        res_mgr = ResultManager()
        builder = resonator_spect(model)

        pipeline = (
            PassManager()
            | RepeatSanitisation()
            | ScopeSanitisation()
            | TriagePass()
            | BindingPass()
            | TILegalisationPass()
            | QbloxLegalisationPass()
            | PreCodegenPass()
        )

        pipeline.run(builder, res_mgr, model)

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        target_map = triage_result.target_map
        precodegen_result: PreCodegenResult = res_mgr.lookup_by_type(PreCodegenResult)

        assert precodegen_result.alloc_mgrs.keys() == target_map.keys()
        for target, alloc_mgr in precodegen_result.alloc_mgrs.items():
            assert len(alloc_mgr.registers) >= 2
            assert len(alloc_mgr.labels) >= 2

    def test_qblox_legalisation_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        (
            PassManager()
            | ScopeSanitisation()
            | TriagePass()
            | BindingPass()
            | TILegalisationPass()
        ).run(builder, res_mgr)

        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)
        bounds = deepcopy(binding_result.iter_bounds)
        QbloxLegalisationPass().run(builder, res_mgr)
        legal_iter_bounds = binding_result.iter_bounds

        assert set(legal_iter_bounds.keys()) == set(bounds.keys())
        for target, symbol2bound in legal_iter_bounds.items():
            for name in binding_result.symbol2scopes:
                assert set(symbol2bound.keys()) == set(bounds[target].keys())
                bound = bounds[target][name]
                legal_bound = symbol2bound[name]
                if name in binding_result.reads:
                    device_updates = [
                        inst
                        for inst in binding_result.reads[name]
                        if isinstance(inst, DeviceUpdate) and inst.target == target
                    ]
                    for du in device_updates:
                        if du.attribute == "frequency":
                            assert legal_bound != bound
                            assert legal_bound == IterBound(
                                start=np.array(
                                    [QbloxLegalisationPass.freq_as_steps(bound.start)],
                                    dtype=int,
                                ).view(np.uint32)[0],
                                step=np.array(
                                    [QbloxLegalisationPass.freq_as_steps(bound.step)],
                                    dtype=int,
                                ).view(np.uint32)[0],
                                end=np.array(
                                    [QbloxLegalisationPass.freq_as_steps(bound.end)],
                                    dtype=int,
                                ).view(np.uint32)[0],
                                count=bound.count,
                            )
                else:
                    assert legal_bound == bound
