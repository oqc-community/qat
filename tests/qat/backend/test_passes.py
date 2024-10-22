import itertools
from typing import Dict

import numpy as np
import pytest

from qat.backend.analysis_passes import (
    BindingPass,
    BindingResult,
    CFGPass,
    CFGResult,
    IterBound,
    TriagePass,
    TriageResult,
)
from qat.backend.graph import ControlFlowGraph
from qat.backend.transform_passes import (
    ReturnSanitisation,
    ScopeSanitisation,
    SweepDecomposition,
)
from qat.backend.validation_passes import (
    NCOFrequencyVariability,
    ReturnSanitisationValidation,
    ScopeSanitisationValidation,
)
from qat.ir.result_base import ResultManager
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import (
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    Repeat,
    Return,
    Sweep,
    Variable,
)
from tests.qat.utils.builder_nuggets import multidim_sweep, resonator_spect, singledim_sweep


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

        instructions_by_target = result.target_map
        assert isinstance(instructions_by_target, Dict)
        assert len(instructions_by_target) == 3 + 2 * 3
        assert qubit.get_drive_channel() in instructions_by_target
        assert qubit.get_measure_channel() in instructions_by_target
        assert qubit.get_acquire_channel() in instructions_by_target

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

    def test_cfg_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        ScopeSanitisation().run(builder, res_mgr)
        ScopeSanitisationValidation().run(builder, res_mgr)
        CFGPass().run(builder, res_mgr)
        result: CFGResult = res_mgr.lookup_by_type(CFGResult)
        assert result.cfg
        assert result.cfg is not ControlFlowGraph()
        assert len(result.cfg.nodes) == 5
        assert len(result.cfg.edges) == 6

    @pytest.mark.parametrize("num_points", [1, 10, 100])
    def test_binding_pass(self, num_points):
        model = get_default_echo_hardware()
        builder = resonator_spect(model, num_points=num_points)
        res_mgr = ResultManager()

        ReturnSanitisation().run(builder, res_mgr)
        TriagePass().run(builder, res_mgr)
        BindingPass().run(builder, res_mgr)
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        var_bounds_result: BindingResult = res_mgr.lookup_by_type(BindingResult)

        instructions_by_target = triage_result.target_map
        variable_bounds = var_bounds_result.iter_bounds_map
        assert variable_bounds.keys() == instructions_by_target.keys()

        device_updates = [
            inst
            for inst in builder.instructions
            if isinstance(inst, DeviceUpdate)
            and isinstance(inst.value, Variable)
            and inst.attribute == "frequency"
        ]

        raw_bounds = {
            next(iter(inst.variables.keys())): BindingPass.extract_iter_bound(
                next(iter(inst.variables.values()))
            )
            for inst in builder.instructions
            if isinstance(inst, Sweep)
        }
        for t in instructions_by_target:
            for name, bounds in variable_bounds[t].items():
                du = next(
                    (
                        inst
                        for inst in device_updates
                        if inst.target == t and inst.value.name == name
                    ),
                    None,
                )

                if du:
                    bb_freq = BindingPass.get_baseband_freq(du.target)
                    assert bounds == IterBound(
                        BindingPass.freq_as_steps(raw_bounds[name].start - bb_freq),
                        BindingPass.freq_as_steps(raw_bounds[name].step),
                        BindingPass.freq_as_steps(raw_bounds[name].end - bb_freq),
                        raw_bounds[name].count,
                    )
                else:
                    assert variable_bounds[t][name] == raw_bounds[name]

    @pytest.mark.parametrize(
        "value, bound",
        [([1, 2, 4, 10], None), ([-0.1, 0, 0.1, 0.2, 0.3], IterBound(-0.1, 0.1, 0.3, 5))]
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


class TestTransformPasses:
    def test_sweep_decomposition_pass(self):
        model = get_default_echo_hardware()
        builder = singledim_sweep(model)
        res_mgr = ResultManager()

        sweeps = [s for s in builder.instructions if isinstance(s, Sweep)]
        expected = sum([len(s.variables) for s in sweeps], 0)
        SweepDecomposition().run(builder, res_mgr)
        sweeps = [s for s in builder.instructions if isinstance(s, Sweep)]
        assert len(sweeps) == expected

        builder = multidim_sweep(model)
        sweeps = [s for s in builder.instructions if isinstance(s, Sweep)]
        expected = sum([len(s.variables) for s in sweeps], 0)
        SweepDecomposition().run(builder, res_mgr)
        sweeps = [s for s in builder.instructions if isinstance(s, Sweep)]
        assert len(sweeps) == expected

    def test_scope_sanitisation_pass(self):
        model = get_default_echo_hardware()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 0
        assert len(repeats) == 1
        assert len(end_repeats) == 0

        with pytest.raises(ValueError):
            ScopeSanitisationValidation().run(builder, res_mgr)

        ScopeSanitisation().run(builder, res_mgr)
        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 1
        assert len(repeats) == 1
        assert len(end_repeats) == 1

        ReturnSanitisation().run(builder, res_mgr)
        ScopeSanitisationValidation().run(builder, res_mgr)

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
        builder = singledim_sweep(model)
        res_mgr = ResultManager()

        NCOFrequencyVariability().run(builder, res_mgr, model)

        channel = next(iter(model.pulse_channels.values()))
        channel.fixed_if = True

        with pytest.raises(ValueError):
            NCOFrequencyVariability().run(builder, res_mgr, model)

        channel.fixed_if = False
        NCOFrequencyVariability().run(builder, res_mgr, model)
