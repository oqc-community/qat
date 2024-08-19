import itertools
from typing import Dict

import numpy as np
import pytest

from qat.purr.backends.analysis_passes import CFGPass, TriagePass, VariableBoundsPass
from qat.purr.backends.codegen import CodegenResultType
from qat.purr.backends.optimisation_passes import (
    ReturnSanitisation,
    ScopeSanitisation,
    SweepDecomposition,
)
from qat.purr.backends.qblox.fast.codegen import PreCodegenPass
from qat.purr.backends.qblox.instructions import EndRepeat, EndSweep
from qat.purr.backends.verification_passes import (
    ReturnSanitisationValidation,
    ScopeSanitisationValidation,
)
from qat.purr.compiler.instructions import DeviceUpdate, Repeat, Return, Sweep, Variable
from tests.qat.qblox.builder_nuggets import multidim_sweep, resonator_spect
from tests.qat.qblox.utils import ClusterInfo


@pytest.mark.parametrize(
    "value, bounds",
    [([1, 2, 4, 10], None), ([-0.1, 0, 0.1, 0.2, 0.3], (-0.1, 0.1, 0.3, 5))]
    + [(np.linspace(b[0], b[1], 100), (b[0], 1, b[1], 100)) for b in [(1, 100), (-50, 49)]],
)
def test_extract_sweep_bounds(value, bounds):
    if bounds is None:
        with pytest.raises(ValueError):
            VariableBoundsPass.extract_variable_bounds(value)
    else:
        assert VariableBoundsPass.extract_variable_bounds(value) == bounds


@pytest.mark.parametrize("model", [ClusterInfo], indirect=True)
class TestPassPipeline:
    def test_triage_pass(self, model):
        qubit_index = 0
        qubit = model.get_qubit(qubit_index)
        builder = resonator_spect(model, qubit_index)

        ReturnSanitisation().run(builder)
        ReturnSanitisationValidation().run(builder)
        analyses = (triage_pass := TriagePass()).run(builder)
        assert len(analyses.data) == 7  # number of results under TriagePass pass
        infos = analyses.data.keys()

        for type in [
            CodegenResultType.SWEEPS,
            CodegenResultType.RETURN,
            CodegenResultType.ASSIGNS,
            CodegenResultType.TARGET_VIEW,
            CodegenResultType.ACQUIRE_MAP,
            CodegenResultType.PP_MAP,
            CodegenResultType.RP_MAP,
        ]:
            info = next((i for i in infos if i.result_type == type))
            assert info.ir_id == hash(builder)
            assert info.pass_id == triage_pass.id()
            assert info.result_type == type

        target_view = analyses.get_result(CodegenResultType.TARGET_VIEW)
        assert isinstance(target_view, Dict)
        assert len(target_view) == 3
        assert qubit.get_drive_channel() in target_view
        assert qubit.get_measure_channel() in target_view
        assert qubit.get_acquire_channel() in target_view

        acquire_map = analyses.get_result(CodegenResultType.ACQUIRE_MAP)
        assert isinstance(acquire_map, Dict)
        assert len(acquire_map) == 1
        assert qubit.get_acquire_channel() in acquire_map
        assert qubit.get_measure_channel() in acquire_map
        assert len(acquire_map[qubit.get_acquire_channel()]) == 1

        # TODO - test other result types

        ret_inst = analyses.get_result(CodegenResultType.RETURN)
        assert isinstance(ret_inst, Return)
        assert all(
            [
                var.output_variable in ret_inst.variables
                for var in itertools.chain(*acquire_map.values())
            ]
        )

    def test_sweep_decomposition_pass(self, model):
        builder = resonator_spect(model)
        sweeps = [s for s in builder.instructions if isinstance(s, Sweep)]
        expected = sum([len(s.variables) for s in sweeps], 0)
        SweepDecomposition().run(builder)
        assert len([s for s in builder.instructions if isinstance(s, Sweep)]) == expected

        builder = multidim_sweep(model)
        sweeps = [s for s in builder.instructions if isinstance(s, Sweep)]
        expected = sum([len(s.variables) for s in sweeps], 0)
        SweepDecomposition().run(builder)
        assert len([s for s in builder.instructions if isinstance(s, Sweep)]) == expected

    def test_scope_sanitisation_pass(self, model):
        builder = resonator_spect(model)

        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 0
        assert len(repeats) == 1
        assert len(end_repeats) == 0

        with pytest.raises(ValueError):
            ScopeSanitisationValidation().run(builder)

        ScopeSanitisation().run(builder)
        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 1
        assert len(repeats) == 1
        assert len(end_repeats) == 1

        ReturnSanitisation().run(builder)
        ScopeSanitisationValidation().run(builder)

    def test_return_sanitisation_pass(self, model):
        qubit_index = 0
        builder = resonator_spect(model, qubit_index)

        with pytest.raises(ValueError):
            ReturnSanitisationValidation().run(builder)

        ReturnSanitisation().run(builder)
        ReturnSanitisationValidation().run(builder)

    def test_cfg_pass(self, model):
        builder = resonator_spect(model)

        ScopeSanitisation().run(builder)
        ScopeSanitisationValidation().run(builder)
        analyses = (cfg_pass := CFGPass()).run(builder)
        assert len(analyses.data) == 1
        info, value = next(iter(analyses.data.items()))
        assert info.ir_id == hash(builder)
        assert info.pass_id == cfg_pass.id()
        assert info.result_type == CodegenResultType.CFG
        assert len(value.nodes) == 5
        assert len(value.edges) == 6

    def test_var_bounds_pass(self, model):
        builder = resonator_spect(model)

        ReturnSanitisation().run(builder)
        analyses = TriagePass().run(builder)
        VariableBoundsPass().run(builder, analyses)
        target_view = analyses.get_result(CodegenResultType.TARGET_VIEW)
        var_bounds = analyses.get_result(CodegenResultType.VARIABLE_BOUNDS)
        assert var_bounds.keys() == target_view.keys()

        device_updates = [
            inst
            for inst in builder.instructions
            if isinstance(inst, DeviceUpdate)
            and isinstance(inst.value, Variable)
            and inst.attribute == "frequency"
        ]

        raw_bounds = {
            next(iter(inst.variables.keys())): VariableBoundsPass.extract_variable_bounds(
                next(iter(inst.variables.values()))
            )
            for inst in builder.instructions
            if isinstance(inst, Sweep)
        }
        for t in target_view:
            for name, bounds in var_bounds[t].items():
                du = next(
                    (
                        inst
                        for inst in device_updates
                        if inst.target == t and inst.value.name == name
                    ),
                    None,
                )

                if du:
                    bb_freq = VariableBoundsPass.get_baseband_freq(du.target)
                    assert bounds == (
                        VariableBoundsPass.freq_as_steps(raw_bounds[name][0] - bb_freq),
                        VariableBoundsPass.freq_as_steps(raw_bounds[name][1]),
                        VariableBoundsPass.freq_as_steps(raw_bounds[name][2] - bb_freq),
                        raw_bounds[name][3],
                    )
                else:
                    assert var_bounds[t][name] == raw_bounds[name]

    def test_precodegen_pass(self, model):
        builder = resonator_spect(model)

        ReturnSanitisation().run(builder)
        analyses = TriagePass().run(builder)
        VariableBoundsPass().run(builder, analyses)
        PreCodegenPass().run(builder, analyses)
        target_view = analyses.get_result(CodegenResultType.TARGET_VIEW)
        contexts = analyses.get_result(CodegenResultType.CONTEXTS)
        assert contexts.keys() == target_view.keys()

        # TODO - test more once symbol table visibility is clear
