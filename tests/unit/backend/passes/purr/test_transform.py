# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025-2025 Oxford Quantum Circuits Ltd


from qat.backend.passes.purr.analysis import TriagePass, TriageResult
from qat.backend.passes.purr.transform import DesugaringPass
from qat.core.result_base import ResultManager
from qat.model.loaders.purr import EchoModelLoader

from tests.unit.utils.builder_nuggets import resonator_spect


class TestTransformPasses:
    def test_desugaring_pass(self):
        model = EchoModelLoader().load()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        TriagePass().run(builder, res_mgr)
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)

        assert len(triage_result.sweeps) == 1
        sweep = next(iter(triage_result.sweeps))
        assert len(sweep.variables) == 1

        DesugaringPass().run(builder, res_mgr)
        assert len(sweep.variables) == 2
        assert f"sweep_{hash(sweep)}" in sweep.variables
