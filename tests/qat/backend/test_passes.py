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
    IntermediateFrequencyAnalysis,
    IntermediateFrequencyResult,
    IterBound,
    TILegalisationPass,
    TimelineAnalysis,
    TimelineAnalysisResult,
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
    FrequencyValidation,
    NCOFrequencyVariability,
    NoAcquireWeightsValidation,
    NoMultipleAcquiresValidation,
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
    Pulse,
    PulseShapeType,
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

    class TestTimelineAnalysis:

        def test_times_are_rounded(self):
            # construct a builder with non-rounded times
            model = get_default_echo_hardware()
            pulse_channel = next(iter(model.pulse_channels.values()))
            builder = model.create_builder()
            block_time = pulse_channel.block_time
            block_size = pulse_channel.block_size
            builder.delay(pulse_channel, 0.9 * block_time)
            builder.delay(pulse_channel, 1.1 * block_time)

            # execute the timeline pass
            res_mgr = ResultManager()
            ir = QatIR(builder)
            TriagePass().run(ir, res_mgr)
            TimelineAnalysis().run(ir, res_mgr)
            res = res_mgr.lookup_by_type(TimelineAnalysisResult)

            # check the results are as expected
            assert len(res.durations) == 1
            assert pulse_channel in res.durations
            durations = res.durations[pulse_channel]
            assert len(durations) == 2
            assert np.all(durations == [block_size, 2 * block_size])

        def test_sync_gives_correct_delays(self):
            model = get_default_echo_hardware()
            pulse_channels = iter(model.pulse_channels.values())
            pulse_channels = [next(pulse_channels) for _ in range(3)]

            # set block size and sample time to be fixed
            sample_time = pulse_channels[0].physical_channel.sample_time
            for pulse_channel in pulse_channels:
                pulse_channel.physical_channel.block_size = 1
                pulse_channel.physical_channel.sample_time = sample_time

            # create some mock data to test sync
            builder = model.create_builder()
            builder.delay(pulse_channels[0], 2 * sample_time)
            builder.delay(pulse_channels[1], 3 * sample_time)
            builder.delay(pulse_channels[2], sample_time)
            builder.synchronize(pulse_channels)

            # execute the timeline pass
            res_mgr = ResultManager()
            ir = QatIR(builder)
            TriagePass().run(ir, res_mgr)
            TimelineAnalysis().run(ir, res_mgr)
            res = res_mgr.lookup_by_type(TimelineAnalysisResult)

            # check the results are as expected
            assert len(res.durations) == 3
            assert all([pulse_channel in res.durations for pulse_channel in pulse_channels])
            assert all(len(durations) == 2 for durations in res.durations.values())
            assert np.all(res.durations[pulse_channels[0]] == [2, 1])
            assert np.all(res.durations[pulse_channels[1]] == [3, 0])
            assert np.all(res.durations[pulse_channels[2]] == [1, 2])

    class TestIntermediateFrequencyAnalysis:

        def test_different_frequencies_with_fixed_if_yields_error(self):
            model = get_default_echo_hardware()
            physical_channel = next(iter(model.physical_channels.values()))
            pulse_channels = iter(
                model.get_pulse_channels_from_physical_channel(physical_channel)
            )
            pulse_channel_1 = next(pulse_channels)
            pulse_channel_2 = next(pulse_channels)
            pulse_channel_2.frequency = pulse_channel_1.frequency + 1e8

            # some dummy data just to test the IFs
            builder = model.create_builder()
            builder.delay(pulse_channel_1, 8e-8)
            builder.delay(pulse_channel_2, 1.6e-7)

            # run the pass: should pass
            res_mgr = ResultManager()
            ir = QatIR(builder)
            TriagePass().run(ir, res_mgr)
            IntermediateFrequencyAnalysis(model).run(ir, res_mgr)

            # fix IF for one channel: should pass
            pulse_channel_1.fixed_if = True
            IntermediateFrequencyAnalysis(model).run(ir, res_mgr)

            # fix IF for one channel: should pass
            pulse_channel_1.fixed_if = False
            pulse_channel_2.fixed_if = True
            IntermediateFrequencyAnalysis(model).run(ir, res_mgr)

            # fix IF for both channels: should fail
            pulse_channel_1.fixed_if = True
            with pytest.raises(ValueError):
                IntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        def test_same_frequencies_with_fixed_if_passes(self):
            model = get_default_echo_hardware()
            physical_channel = next(iter(model.physical_channels.values()))
            pulse_channels = iter(
                model.get_pulse_channels_from_physical_channel(physical_channel)
            )
            pulse_channel_1 = next(pulse_channels)
            pulse_channel_2 = next(pulse_channels)
            pulse_channel_1.fixed_if = True
            pulse_channel_2.fixed_if = True
            pulse_channel_2.frequency = pulse_channel_1.frequency

            # some dummy data just to test the IFs
            builder = model.create_builder()
            builder.delay(pulse_channel_1, 8e-8)
            builder.delay(pulse_channel_2, 1.6e-7)

            # run the pass
            res_mgr = ResultManager()
            ir = QatIR(builder)
            TriagePass().run(ir, res_mgr)
            IntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        def test_fixed_if_returns_frequencies(self):
            model = get_default_echo_hardware()

            # Find two pulse channels with different physical channels
            pulse_channels = iter(model.pulse_channels.values())
            pulse_channel_1 = next(pulse_channels)
            pulse_channel_2 = next(pulse_channels)
            while pulse_channel_1.physical_channel == pulse_channel_2.physical_channel:
                pulse_channel_2 = next(pulse_channels)

            pulse_channel_1.fixed_if = True
            pulse_channel_2.fixed_if = False  # test this isn't saved

            # some dummy data just to test the IF
            builder = model.create_builder()
            builder.delay(pulse_channel_1, 8e-8)
            builder.delay(pulse_channel_2, 1.6e-7)

            # run the pass
            res_mgr = ResultManager()
            ir = QatIR(builder)
            TriagePass().run(ir, res_mgr)
            IntermediateFrequencyAnalysis(model).run(ir, res_mgr)
            res = res_mgr.lookup_by_type(IntermediateFrequencyResult)
            assert pulse_channel_1.physical_channel in res.frequencies
            assert pulse_channel_2.physical_channel not in res.frequencies


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

    class TestFrequencyValidation:
        res_mgr = ResultManager()
        model = get_default_echo_hardware()

        def get_single_pulse_channel(self):
            return next(iter(self.model.pulse_channels.values()))

        def get_two_pulse_channels_from_single_physical_channel(self):
            physical_channel = next(iter(self.model.physical_channels.values()))
            pulse_channels = iter(
                self.model.get_pulse_channels_from_physical_channel(physical_channel)
            )
            return next(pulse_channels), next(pulse_channels)

        def get_two_pulse_channels_from_different_physical_channels(self):
            physical_channels = iter(self.model.physical_channels.values())
            pulse_channel_1 = next(
                iter(
                    self.model.get_pulse_channels_from_physical_channel(
                        next(physical_channels)
                    )
                )
            )
            pulse_channel_2 = next(
                iter(
                    self.model.get_pulse_channels_from_physical_channel(
                        next(physical_channels)
                    )
                )
            )
            return pulse_channel_1, pulse_channel_2

        def set_frequency_range(self, pulse_channel, lower_tol, upper_tol):
            phys_chan = pulse_channel.physical_channel
            phys_chan.pulse_channel_max_frequency = pulse_channel.frequency + upper_tol
            phys_chan.pulse_channel_min_frequency = pulse_channel.frequency - lower_tol

        @pytest.mark.parametrize("freq", [-1e-9, -1e-8, 0, 1e8, 1e9])
        def test_raises_no_error_when_freq_shift_in_range(self, freq):
            channel = self.get_single_pulse_channel()
            self.set_frequency_range(channel, 1e9, 1e9)
            builder = self.model.create_builder()
            builder.frequency_shift(channel, freq)
            FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

        @pytest.mark.parametrize("freq", [-1e9, 1e9])
        def test_raises_value_error_when_freq_shift_out_of_range(self, freq):
            channel = self.get_single_pulse_channel()
            self.set_frequency_range(channel, 1e8, 1e8)
            builder = self.model.create_builder()
            builder.frequency_shift(channel, freq)
            with pytest.raises(ValueError):
                FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

        @pytest.mark.parametrize("freq", [-2e8, 2e8])
        def test_moves_out_and_in_raises_value_error(self, freq):
            channel = self.get_single_pulse_channel()
            self.set_frequency_range(channel, 1e8, 1e8)
            builder = self.model.create_builder()
            builder.frequency_shift(channel, freq)
            builder.frequency_shift(channel, -freq)
            with pytest.raises(ValueError):
                FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

        @pytest.mark.parametrize("sign", [+1, -1])
        def test_no_value_error_for_two_channels_same_physical_channel(self, sign):
            channels = self.get_two_pulse_channels_from_single_physical_channel()
            self.set_frequency_range(channels[0], 1e9, 1e9)
            builder = self.model.create_builder()

            # interweave with random instructions
            builder.phase_shift(channels[0], np.pi)
            builder.frequency_shift(channels[0], sign * 1e8)
            builder.phase_shift(channels[1], -np.pi)
            builder.frequency_shift(channels[1], sign * 2e8)
            builder.frequency_shift(channels[0], sign * 3e8)
            builder.phase_shift(channels[1], -2.54)
            builder.frequency_shift(channels[1], sign * 4e8)
            FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

        @pytest.mark.parametrize("sign", [+1, -1])
        def test_value_error_for_two_channels_same_physical_channel(self, sign):
            channels = self.get_two_pulse_channels_from_single_physical_channel()
            self.set_frequency_range(channels[0], 1e9, 1e9)
            builder = self.model.create_builder()

            # interweave with random instructions
            builder.phase_shift(channels[0], np.pi)
            builder.frequency_shift(channels[0], sign * 5e8)
            builder.phase_shift(channels[1], -np.pi)
            builder.frequency_shift(channels[1], sign * 6e8)
            builder.frequency_shift(channels[0], sign * 1e8)
            builder.phase_shift(channels[1], -2.54)
            builder.frequency_shift(channels[1], sign * 5e8)
            with pytest.raises(ValueError):
                FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

        @pytest.mark.parametrize("sign", [+1, -1])
        def test_no_value_error_for_two_channels_different_physical_channel(self, sign):
            channels = self.get_two_pulse_channels_from_different_physical_channels()
            self.set_frequency_range(channels[0], 1e9, 1e9)
            self.set_frequency_range(channels[1], 5e8, 5e8)
            builder = self.model.create_builder()
            builder.frequency_shift(channels[0], sign * 4e8)
            builder.frequency_shift(channels[1], sign * 1e8)
            builder.frequency_shift(channels[0], sign * 3e8)
            builder.frequency_shift(channels[1], sign * 4e8)
            FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

        @pytest.mark.parametrize("sign", [+1, -1])
        def test_value_error_for_two_channels_different_physical_channel(self, sign):
            channels = self.get_two_pulse_channels_from_different_physical_channels()
            self.set_frequency_range(channels[0], 1e9, 1e9)
            self.set_frequency_range(channels[1], 5e8, 5e8)
            builder = self.model.create_builder()
            builder.frequency_shift(channels[0], sign * 4e8)
            builder.frequency_shift(channels[1], sign * 1e8)
            builder.frequency_shift(channels[0], sign * 7e8)
            builder.frequency_shift(channels[1], sign * 4e8)
            with pytest.raises(ValueError):
                FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

        def test_fixed_if_raises_not_implemented_error(self):
            channels = self.get_two_pulse_channels_from_different_physical_channels()
            builder = self.model.create_builder()
            channels[0].fixed_if = True
            builder.frequency_shift(channels[0], 1e8)
            builder.frequency_shift(channels[1], 1e8)
            with pytest.raises(NotImplementedError):
                FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

        def test_fixed_if_does_not_affect_other_channel(self):
            channels = self.get_two_pulse_channels_from_different_physical_channels()
            builder = self.model.create_builder()
            channels[0].fixed_if = True
            builder.frequency_shift(channels[1], 1e8)
            FrequencyValidation(self.model).run(QatIR(builder), self.res_mgr)

    class TestNoAcquireWeightsValidation:

        def test_acquire_with_filter_raises_error(self):
            model = get_default_echo_hardware()
            res_mgr = ResultManager()
            qubit = model.get_qubit(0)
            channel = qubit.get_acquire_channel()
            builder = model.create_builder()
            builder.acquire(
                channel, delay=0.0, filter=Pulse(channel, PulseShapeType.SQUARE, 1e-6)
            )
            with pytest.raises(NotImplementedError):
                NoAcquireWeightsValidation().run(QatIR(builder), res_mgr)

    class TestNoMultipleAcquiresValidation:

        def test_multiple_acquires_on_same_pulse_channel_raises_error(self):
            model = get_default_echo_hardware()
            res_mgr = ResultManager()
            qubit = model.get_qubit(0)
            channel = qubit.get_acquire_channel()
            builder = model.create_builder()
            builder.acquire(channel, delay=0.0)

            # Test should run as there is only one acquire
            NoMultipleAcquiresValidation().run(QatIR(builder), res_mgr)

            # Add another acquire and test it breaks it
            builder.acquire(channel, delay=0.0)
            with pytest.raises(NotImplementedError):
                NoMultipleAcquiresValidation().run(QatIR(builder), res_mgr)

        def test_multiple_acquires_that_share_physical_channel_raises_error(self):
            model = get_default_echo_hardware()
            res_mgr = ResultManager()
            qubit = model.get_qubit(0)
            acquire_channel = qubit.get_acquire_channel()
            # in practice, we shouldn't do this with a measure channel, but convinient for
            # testing
            measure_channel = qubit.get_measure_channel()
            builder = model.create_builder()
            builder.acquire(acquire_channel, delay=0.0)
            builder.acquire(measure_channel, delay=0.0)
            with pytest.raises(NotImplementedError):
                NoMultipleAcquiresValidation().run(QatIR(builder), res_mgr)
