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
from qat.backend.transform_passes import ReturnSanitisation, ScopeSanitisation
from qat.backend.validation_passes import (
    NCOFrequencyVariability,
    ReturnSanitisationValidation,
)
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
        ReturnSanitisation().run(builder, res_mgr)
        ReturnSanitisationValidation().run(builder, res_mgr)
        TriagePass().run(builder, res_mgr)
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
    @pytest.mark.parametrize("qubit_indices", [[0, 1, 2, 3]])
    def test_binding_pass_with_resonator_spect(self, num_points, qubit_indices):
        model = get_default_echo_hardware()
        builder = resonator_spect(model, qubit_indices=qubit_indices, num_points=num_points)
        res_mgr = ResultManager()

        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 0
        assert len(repeats) == 1
        assert len(end_repeats) == 0

        TriagePass().run(builder, res_mgr)
        with pytest.raises(ValueError):
            BindingPass().run(builder, res_mgr)

        ScopeSanitisation().run(builder, res_mgr)
        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 1
        assert len(repeats) == 1
        assert len(end_repeats) == 1

        BindingPass().run(builder, res_mgr)
        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)

        assert len(binding_result.scope2symbols) == len(sweeps) + len(repeats)
        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]

        iter_bounds = {
            name: BindingPass.extract_iter_bound(value)
            for inst in sweeps
            for name, value in inst.variables.items()
        }
        for inst in repeats:
            name = f"repeat_{inst.repeat_count}_{hash(inst)}"
            iter_bounds[name] = IterBound(
                start=1, step=1, end=inst.repeat_count, count=inst.repeat_count
            )

        for scope, symbols in binding_result.scope2symbols.items():
            for name in symbols:
                assert name in iter_bounds
                assert name in binding_result.symbol2scopes
                assert scope in binding_result.symbol2scopes[name]

                for t in triage_result.target_map:
                    assert binding_result.iter_bounds[t][name] == iter_bounds[name]

        for symbol, scopes in binding_result.symbol2scopes.items():
            assert scopes
            for scope in scopes:
                assert scope in binding_result.scope2symbols
                assert scope[0] in sweeps + repeats
                assert scope[1] in end_sweeps + end_repeats
            assert symbol in iter_bounds

        device_updates = set(
            inst
            for inst in builder.instructions
            if isinstance(inst, DeviceUpdate) and isinstance(inst.value, Variable)
        )
        acquires = set(inst for inst in builder.instructions if isinstance(inst, Acquire))
        assert set(binding_result.reads.keys()) == {
            inst.value.name for inst in device_updates
        }
        for inst in acquires:
            assert inst.output_variable in set(binding_result.writes.keys())
        for name in iter_bounds:
            assert name in set(binding_result.writes.keys())

    def test_ti_legalisation_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        ScopeSanitisation().run(builder, res_mgr)
        TriagePass().run(builder, res_mgr)
        BindingPass().run(builder, res_mgr)

        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)
        bounds = deepcopy(binding_result.iter_bounds)
        TILegalisationPass().run(builder, res_mgr)
        legal_iter_bounds = binding_result.iter_bounds

        assert set(legal_iter_bounds.keys()) == set(bounds.keys())
        for target, symbol2bound in legal_iter_bounds.items():
            assert set(symbol2bound.keys()) == set(bounds[target].keys())
            for name in binding_result.symbol2scopes:
                bound = bounds[target][name]
                legal_bound = symbol2bound[name]
                if name in binding_result.reads:
                    device_updates = [
                        inst
                        for inst in binding_result.reads[name]
                        if isinstance(inst, DeviceUpdate) and inst.target == target
                    ]
                    for du in device_updates:
                        assert legal_bound != bound
                        assert legal_bound == IterBound(
                            start=TILegalisationPass.decompose_freq(bound.start, du.target)[
                                1
                            ],
                            step=bound.step,
                            end=TILegalisationPass.decompose_freq(bound.end, du.target)[1],
                            count=bound.count,
                        )
                else:
                    assert legal_bound == bound

    def test_cfg_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        ScopeSanitisation().run(builder, res_mgr)
        CFGPass().run(builder, res_mgr)
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

        with pytest.raises(ValueError):
            ReturnSanitisationValidation().run(builder, res_mgr)

        ReturnSanitisation().run(builder, res_mgr)
        ReturnSanitisationValidation().run(builder, res_mgr)


class TestValidationPasses:
    def test_nco_freq_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        NCOFrequencyVariability().run(builder, res_mgr, model)

        channel = next(iter(model.pulse_channels.values()))
        channel.fixed_if = True

        with pytest.raises(ValueError):
            NCOFrequencyVariability().run(builder, res_mgr, model)

        channel.fixed_if = False
        NCOFrequencyVariability().run(builder, res_mgr, model)