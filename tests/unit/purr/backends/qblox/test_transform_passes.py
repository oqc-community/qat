# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.qblox.analysis_passes import (
    BindingPass,
    BindingResult,
    TriagePass,
    TriageResult,
)
from qat.purr.backends.qblox.transform_passes import (
    RepeatSanitisation,
    ReturnSanitisation,
    ScopePeeling,
    ScopeSanitisation,
)
from qat.purr.compiler.instructions import EndRepeat, EndSweep, Repeat, Sweep
from qat.purr.core.metrics_base import MetricsManager
from qat.purr.core.pass_base import PassManager
from qat.purr.core.result_base import ResultManager

from tests.unit.utils.builder_nuggets import time_and_phase_iteration


@pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
def test_scope_peeling_pass(qubit_indices):
    model = get_default_echo_hardware()
    builder = time_and_phase_iteration(model, qubit_indices)

    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    pipeline = (
        PassManager()
        | RepeatSanitisation(model)
        | ScopeSanitisation()
        | ReturnSanitisation()
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
