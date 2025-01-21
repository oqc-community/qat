# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

import pytest

from qat.backend.analysis_passes import TriagePass, TriageResult
from qat.backend.transform_passes import DesugaringPass, ReturnSanitisation
from qat.backend.validation_passes import ReturnSanitisationValidation
from qat.ir.pass_base import QatIR
from qat.ir.result_base import ResultManager
from qat.purr.backends.echo import get_default_echo_hardware

from tests.qat.utils.builder_nuggets import resonator_spect


class TestTransformPasses:
    def test_return_sanitisation_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()
        ir = QatIR(builder)

        with pytest.raises(ValueError):
            ReturnSanitisationValidation().run(ir, res_mgr)

        ReturnSanitisation().run(ir, res_mgr)
        ReturnSanitisationValidation().run(ir, res_mgr)

    def test_desugaring_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()
        ir = QatIR(builder)

        TriagePass().run(ir, res_mgr)
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)

        assert len(triage_result.sweeps) == 1
        sweep = next(iter(triage_result.sweeps))
        assert len(sweep.variables) == 1

        DesugaringPass().run(ir, res_mgr)
        assert len(sweep.variables) == 2
        assert f"sweep_{hash(sweep)}" in sweep.variables
