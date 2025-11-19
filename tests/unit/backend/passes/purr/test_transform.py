# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025-2025 Oxford Quantum Circuits Ltd

import pytest

from qat.backend.passes.purr.analysis import (
    BindingPass,
    BindingResult,
    TriagePass,
    TriageResult,
)
from qat.backend.passes.purr.transform import DesugaringPass, ScopePeeling
from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.middleend.passes.purr.transform import (
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import EndRepeat, EndSweep, Repeat, Sweep

from tests.unit.utils.builder_nuggets import resonator_spect, time_and_phase_iteration


def test_desugaring_pass():
    model = EchoModelLoader().load()
    builder = resonator_spect(model)
    res_mgr = ResultManager()

    TriagePass().run(builder, res_mgr)
    triage_result = res_mgr.lookup_by_type(TriageResult)

    assert len(triage_result.sweeps) == 1
    sweep = next(iter(triage_result.sweeps))
    assert len(sweep.variables) == 1

    DesugaringPass().run(builder, res_mgr)
    assert len(sweep.variables) == 2
    assert f"{hash(sweep)}" in sweep.variables


@pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
def test_scope_peeling_pass(qubit_indices):
    model = EchoModelLoader().load()
    builder = time_and_phase_iteration(model, qubit_indices)

    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    pipeline = (
        PassManager()
        | RepeatSanitisation(model, TargetData.default())
        | ScopeSanitisation()
        | ReturnSanitisation()
        | DesugaringPass()
        | TriagePass()
        | BindingPass()
    )

    pipeline.run(builder, res_mgr, met_mgr)

    sweep = next((inst for inst in builder.instructions if isinstance(inst, Sweep)))
    end_sweep = next((inst for inst in builder.instructions if isinstance(inst, EndSweep)))
    repeat = next((inst for inst in builder.instructions if isinstance(inst, Repeat)))
    end_repeat = next(
        (inst for inst in builder.instructions if isinstance(inst, EndRepeat))
    )
    valid_scopes = [(sweep, end_sweep), (repeat, end_repeat)]
    invalid_scopes = [(sweep, end_repeat), (repeat, end_sweep), (end_repeat, sweep)]

    triage_result = res_mgr.lookup_by_type(TriageResult)
    binding_result = res_mgr.lookup_by_type(BindingResult)
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()
        assert (sweep, end_sweep) in binding_result.scoping_results[
            drive_channel
        ].scope2symbols
        assert (repeat, end_repeat) in binding_result.scoping_results[
            drive_channel
        ].scope2symbols

    len_before = len(builder.instructions)
    ScopePeeling().run(builder, res_mgr, met_mgr, scopes=[])
    assert len(builder.instructions) == len_before

    with pytest.raises(ValueError):
        ScopePeeling().run(builder, res_mgr, met_mgr, scopes=invalid_scopes)

    ScopePeeling().run(builder, res_mgr, met_mgr, scopes=valid_scopes)
    assert len(builder.instructions) == len_before - 2 * len(valid_scopes)

    # Mark and destroy corrupt results
    res_mgr.mark_as_dirty(triage_result, binding_result)
    res_mgr.cleanup()

    # TriageResult has been destroyed and is not expected to be found
    with pytest.raises(ValueError):
        res_mgr.lookup_by_type(TriageResult)

    # BindingResult has been destroyed and is not expected to be found
    with pytest.raises(ValueError):
        res_mgr.lookup_by_type(BindingResult)

    TriagePass().run(builder, res_mgr, met_mgr)
    # Variable t's usage becomes illegal because the sweep got remove
    with pytest.raises(ValueError):
        BindingPass().run(builder, res_mgr, met_mgr)
