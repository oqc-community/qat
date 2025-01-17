# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import itertools
from copy import deepcopy
from typing import Dict

import numpy as np
import pytest

from qat.backend.analysis_passes import (
    BindingPass,
    BindingResult,
    CFGPass,
    CFGResult,
    IterBound,
    TILegalisationPass,
    TriagePass,
    TriageResult,
)
from qat.backend.graph import ControlFlowGraph
from qat.backend.transform_passes import (
    DesugaringPass,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.backend.validation_passes import (
    NCOFrequencyVariability,
    ReturnSanitisationValidation,
)
from qat.ir.pass_base import QatIR
from qat.ir.result_base import ResultManager
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import (
    Acquire,
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    Repeat,
    Return,
    Sweep,
    Variable,
)

from tests.qat.utils.builder_nuggets import resonator_spect


class TestAnalysisPasses:
    def test_triage_pass(self):
        index = 0
        model = get_default_echo_hardware()
        qubit = model.get_qubit(index)
        builder = resonator_spect(model, [index])

        res_mgr = ResultManager()
        ir = QatIR(builder)
        ReturnSanitisation().run(ir, res_mgr)
        ReturnSanitisationValidation().run(ir, res_mgr)
        TriagePass().run(ir, res_mgr)
        result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        assert result
        assert result.sweeps
        assert result.returns
        assert not result.assigns
        assert result.target_map
        assert result.acquire_map
        assert result.rp_map
        assert result.pp_map
        assert result is not TriageResult()

        target_map = result.target_map
        assert isinstance(target_map, Dict)
        assert len(target_map) == 3 + 2 * 3
        assert qubit.get_drive_channel() in target_map
        assert qubit.get_measure_channel() in target_map
        assert qubit.get_acquire_channel() in target_map

        acquire_map = result.acquire_map
        assert isinstance(acquire_map, Dict)
        assert len(acquire_map) == 1
        assert qubit.get_acquire_channel() in acquire_map
        assert len(acquire_map[qubit.get_acquire_channel()]) == 1

        assert len(result.returns) == 1
        ret_inst = result.returns[0]
        assert isinstance(ret_inst, Return)
        assert all(
            [
                var.output_variable in ret_inst.variables
                for var in itertools.chain(*acquire_map.values())
            ]
        )

    @pytest.mark.parametrize(
        "value, bound",
        [
            ([1, 2, 4, 10], None),
            ([-0.1, 0, 0.1, 0.2, 0.3], IterBound(-0.1, 0.1, 0.3, 5)),
        ]
        + [
            (np.linspace(b[0], b[1], 100), IterBound(b[0], 1, b[1], 100))
            for b in [(1, 100), (-50, 49)]
        ],
    )
    def test_extract_iter_bound(self, value, bound):
        if bound is None:
            with pytest.raises(ValueError):
                BindingPass.extract_iter_bound(value)
        else:
            assert BindingPass.extract_iter_bound(value) == bound

    @pytest.mark.parametrize("num_points", [1, 10, 100])
    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1], [0, 1, 2]])
    def test_binding_pass(self, num_points, qubit_indices):
        model = get_default_echo_hardware()
        builder = resonator_spect(model, qubit_indices=qubit_indices, num_points=num_points)
        res_mgr = ResultManager()
        ir = QatIR(builder)

        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 0
        assert len(repeats) == 1
        assert len(end_repeats) == 0

        with pytest.raises(ValueError):
            BindingPass().run(ir, res_mgr)

        ScopeSanitisation().run(ir, res_mgr)
        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 1
        assert len(repeats) == 1
        assert len(end_repeats) == 1

        TriagePass().run(ir, res_mgr)
        BindingPass().run(ir, res_mgr)
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)

        for target, instructions in triage_result.target_map.items():
            scoping_result = binding_result.scoping_results[target]
            rw_result = binding_result.rw_results[target]
            iter_bound_result = binding_result.iter_bound_results[target]

            assert len(scoping_result.scope2symbols) == len(sweeps) + len(repeats)

            sweeps = [inst for inst in instructions if isinstance(inst, Sweep)]
            device_updates = set(
                inst
                for inst in instructions
                if isinstance(inst, DeviceUpdate) and isinstance(inst.value, Variable)
            )
            acquires = [inst for inst in instructions if isinstance(inst, Acquire)]
            end_sweeps = [inst for inst in instructions if isinstance(inst, EndSweep)]

            iter_bounds = {
                name: BindingPass.extract_iter_bound(value)
                for inst in sweeps
                for name, value in inst.variables.items()
            }

            for inst in sweeps:
                iter_name = f"sweep_{hash(inst)}"
                count = len(next(iter(inst.variables.values())))
                iter_bounds[iter_name] = IterBound(start=1, step=1, end=count, count=count)

            for inst in repeats:
                name = f"repeat_{hash(inst)}"
                iter_bounds[name] = IterBound(
                    start=1, step=1, end=inst.repeat_count, count=inst.repeat_count
                )

            for scope, symbols in scoping_result.scope2symbols.items():
                for name in symbols:
                    assert name in iter_bounds
                    assert name in scoping_result.symbol2scopes
                    assert scope in scoping_result.symbol2scopes[name]

                    for t in triage_result.target_map:
                        assert iter_bound_result[name] == iter_bounds[name]

            for symbol, scopes in scoping_result.symbol2scopes.items():
                assert scopes
                for scope in scopes:
                    assert scope in scoping_result.scope2symbols
                    assert scope[0] in sweeps + repeats
                    assert scope[1] in end_sweeps + end_repeats
                assert symbol in iter_bounds

            if device_updates:
                assert {inst.value.name for inst in device_updates} <= set(
                    rw_result.reads.keys()
                )

            if acquires:
                assert {f"acquire_{hash(inst)}" for inst in acquires} <= set(
                    rw_result.reads.keys()
                )

            for name in iter_bounds:
                # name is read => name is writen to
                assert (name not in set(rw_result.reads.keys())) or (
                    name in set(rw_result.writes.keys())
                )

    def test_ti_legalisation_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()
        ir = QatIR(builder)

        ScopeSanitisation().run(ir, res_mgr)
        TriagePass().run(ir, res_mgr)
        BindingPass().run(ir, res_mgr)

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        binding_result: BindingResult = deepcopy(res_mgr.lookup_by_type(BindingResult))

        TILegalisationPass().run(ir, res_mgr)

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
                        assert legal_bound != bound
                        assert legal_bound == IterBound(
                            start=TILegalisationPass.decompose_freq(bound.start, target)[1],
                            step=bound.step,
                            end=TILegalisationPass.decompose_freq(bound.end, target)[1],
                            count=bound.count,
                        )
                else:
                    assert legal_bound == bound

    def test_cfg_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()
        ir = QatIR(builder)

        ScopeSanitisation().run(ir, res_mgr)
        CFGPass().run(ir, res_mgr)
        result: CFGResult = res_mgr.lookup_by_type(CFGResult)
        assert result.cfg
        assert result.cfg is not ControlFlowGraph()
        assert len(result.cfg.nodes) == 5
        assert len(result.cfg.edges) == 6


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


class TestValidationPasses:
    def test_nco_freq_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()
        ir = QatIR(builder)

        NCOFrequencyVariability().run(ir, res_mgr, model)

        channel = next(iter(model.pulse_channels.values()))
        channel.fixed_if = True

        with pytest.raises(ValueError):
            NCOFrequencyVariability().run(ir, res_mgr, model)

        channel.fixed_if = False
        NCOFrequencyVariability().run(ir, res_mgr, model)
