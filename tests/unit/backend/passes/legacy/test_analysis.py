# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import itertools
from copy import deepcopy
from typing import Dict

import numpy as np
import pytest

from qat.backend.graph import ControlFlowGraph
from qat.backend.passes.legacy.analysis import (
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
from qat.backend.passes.legacy.lowering import PartitionByPulseChannel
from qat.core.result_base import ResultManager
from qat.ir.lowered import PartitionedIR
from qat.middleend.passes.legacy.transform import (
    InstructionGranularitySanitisation,
    LowerSyncsToDelays,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.middleend.passes.legacy.validation import ReturnSanitisationValidation
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import (
    Acquire,
    Delay,
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    Pulse,
    Repeat,
    Return,
    Sweep,
    Variable,
)

from tests.unit.utils.builder_nuggets import resonator_spect


class TestAnalysisPasses:
    def test_triage_pass(self):
        index = 0
        model = EchoModelLoader().load()
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
        assert len(target_map) == 7
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

        active_targets = result.active_targets
        assert len(active_targets) == 2
        assert qubit.get_measure_channel() in active_targets
        assert qubit.get_acquire_channel() in active_targets

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
        model = EchoModelLoader().load()
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

        TriagePass().run(builder, res_mgr)
        BindingPass().run(builder, res_mgr)
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
        model = EchoModelLoader().load()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        ScopeSanitisation().run(builder, res_mgr)
        TriagePass().run(builder, res_mgr)
        BindingPass().run(builder, res_mgr)

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        binding_result: BindingResult = deepcopy(res_mgr.lookup_by_type(BindingResult))

        TILegalisationPass().run(builder, res_mgr)

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
        model = EchoModelLoader().load()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        ScopeSanitisation().run(builder, res_mgr)
        CFGPass().run(builder, res_mgr)
        result: CFGResult = res_mgr.lookup_by_type(CFGResult)
        assert result.cfg
        assert result.cfg is not ControlFlowGraph()
        assert len(result.cfg.nodes) == 5
        assert len(result.cfg.edges) == 6


class TestTimelineAnalysis:
    def test_timelines_match(self):
        """Test the results of the timeline analyis match with the expectation."""
        model = EchoModelLoader().load()
        target_data = TargetData.random()

        qubits = model.qubits[0:2]
        drive_channels = [qubit.get_drive_channel() for qubit in qubits]
        measure_channels = [qubit.get_measure_channel() for qubit in qubits]
        acquire_channels = [qubit.get_acquire_channel() for qubit in qubits]
        cr_channel = qubits[1].get_cross_resonance_channel(qubits[0])
        crc_channel = qubits[0].get_cross_resonance_cancellation_channel(qubits[1])

        builder = model.create_builder()
        builder.delay(drive_channels[0], drive_channels[0].sample_time * 8 * 10)
        builder.delay(drive_channels[1], drive_channels[1].sample_time * 8 * 5)
        builder.synchronize([*drive_channels, cr_channel, crc_channel])
        builder.ECR(*qubits)
        builder.X(qubits[0])
        builder.delay(qubits[1], drive_channels[1].sample_time * 8 * 50)
        builder.synchronize(
            [drive_channels[0], measure_channels[0], acquire_channels[0], crc_channel]
        )
        builder.synchronize(
            [drive_channels[1], measure_channels[1], acquire_channels[1], cr_channel]
        )
        builder.measure(qubits[0])
        builder.measure(qubits[1])

        builder = InstructionGranularitySanitisation(model, target_data).run(builder)
        builder = LowerSyncsToDelays().run(builder)

        res_mgr = ResultManager()
        ir = PartitionByPulseChannel().run(builder, res_mgr)
        TimelineAnalysis().run(ir, res_mgr)
        res = res_mgr.lookup_by_type(TimelineAnalysisResult)

        # Do some manual analysis of times
        durations = {}
        for inst in builder.instructions:
            for qt in inst.quantum_targets:
                durations.setdefault(qt, []).append(inst.duration)

        assert durations.keys() == res.target_map.keys()
        for key, val in durations.items():
            val = np.asarray(val)
            ends = np.cumsum(val)
            starts = ends - val
            assert np.all(
                np.isclose(val * 1e9, res.target_map[key].samples * key.sample_time * 1e9)
            )
            assert np.all(
                np.isclose(
                    ends * 1e9,
                    (res.target_map[key].end_positions + 1) * key.sample_time * 1e9,
                )
            )
            assert np.all(
                np.isclose(
                    starts * 1e9,
                    res.target_map[key].start_positions * key.sample_time * 1e9,
                )
            )

    @pytest.mark.parametrize("drive_width", [45e-9, 48e-8, 52e-9])
    @pytest.mark.parametrize("cr_width", [134e-9, 136e-9, 139e-9])
    @pytest.mark.parametrize("control_delay", [None, 103e-9, 104e-9, 106e-9])
    @pytest.mark.parametrize("target_delay", [None, 74e-9, 80e-9, 85e-9])
    def test_ECR_timings_integrated_with_granularity_sanitisation(
        self, drive_width, cr_width, control_delay, target_delay
    ):
        """Checks that the timings of an ECR gate are still as expected if the timings do
        not exactly match the granularity.

        The test checks the following circuit

                    |‾‾‾‾‾‾‾‾‾‾‾|           |‾‾‾‾‾|
        Q1  --------|   Delay   |-----------|     |-------
                    |___________|           |     |
                                            | ECR |
                    |‾‾‾‾‾‾‾|               |     |
        Q2  --------| Delay |---------------|     |-------
                    |_______|               |_____|

        with varying drive and CR pulse widths. It varys the widths slightly so they either

        #. they are integer multiples of the granularity of the channels,
        #. they are slightly lower than an integer multiple of the granularity of the
           channels,
        #. they are slightly higher than an integer multiple of the channels.

        All three situations for each pulse type are matrix tested.

        When the timeline analysis is used alongside the InstructionGranularitySanitisation
        pass, the times will be rounded to be integer multiples of the granularity. This
        tests that the pulses within the ECR happen when expected relative to one-another.

        It also runs the same tests with delays occuring on the two qubits before the ECR.
        This means that the qubits start out-of-sync, and checks that this works as expected
        still. It checks the delays with the above criteria, with the addition to not having
        the pulse.
        """

        model = EchoModelLoader().load()
        target_data = TargetData.random()

        qubit1 = model.qubits[0]
        qubit2 = model.qubits[1]
        qubit1.pulse_hw_x_pi_2["width"] = drive_width
        qubit1.pulse_hw_zx_pi_4[qubit2.full_id()]["width"] = cr_width
        qubit2.pulse_hw_zx_pi_4[qubit1.full_id()]["width"] = cr_width

        builder = model.create_builder()
        if control_delay:
            builder.delay(qubit1.get_drive_channel(), time=control_delay)
        if target_delay:
            builder.delay(qubit2.get_drive_channel(), time=target_delay)
        builder.ECR(qubit1, qubit2)

        res_mgr = ResultManager()
        InstructionGranularitySanitisation(model, target_data).run(builder, res_mgr)
        LowerSyncsToDelays().run(builder, res_mgr)
        ir = PartitionByPulseChannel().run(builder, res_mgr)
        TimelineAnalysis().run(ir, res_mgr)

        target_map = ir.target_map
        timeline_res: TimelineAnalysisResult = res_mgr.lookup_by_type(
            TimelineAnalysisResult
        )

        # inspect the drive channel: make sure there are two pulses from the ECR gate
        drive_chan = qubit1.get_drive_channel()
        drive_chan_target = qubit2.get_drive_channel()
        drive_pulses = [isinstance(inst, Pulse) for inst in target_map[drive_chan]]
        assert sum(drive_pulses) == 2
        drive_pulse_idxs = [i for i, val in enumerate(drive_pulses) if val]

        # inspect the CRC channel: make sure there are two pulses from the ECR gate
        crc_chan = qubit2.get_cross_resonance_cancellation_channel(qubit1)
        crc_pulses = [isinstance(inst, Pulse) for inst in target_map[crc_chan]]
        assert sum(crc_pulses) == 2
        crc_pulse_idxs = [i for i, val in enumerate(crc_pulses) if val]

        # inspect the CR channel: make sure there are two pulses from the ECR gate
        cr_chan = qubit1.get_cross_resonance_channel(qubit2)
        cr_pulses = [isinstance(inst, Pulse) for inst in target_map[cr_chan]]
        assert sum(cr_pulses) == 2
        cr_pulse_idxs = [i for i, val in enumerate(cr_pulses) if val]

        # assertion to check the the sample times match up so we don't have to worry about
        # converting samples -> times :)
        assert drive_chan.sample_time == crc_chan.sample_time
        assert drive_chan.sample_time == cr_chan.sample_time
        assert drive_chan.sample_time == drive_chan_target.sample_time

        # The CR and CRC pulses should always be in sync; let's check that!
        for i in range(2):
            assert (
                timeline_res.target_map[cr_chan].start_positions[cr_pulse_idxs[i]]
                == timeline_res.target_map[crc_chan].start_positions[crc_pulse_idxs[i]]
            )
            assert (
                timeline_res.target_map[cr_chan].end_positions[cr_pulse_idxs[i]]
                == timeline_res.target_map[crc_chan].end_positions[crc_pulse_idxs[i]]
            )

        # The drive pulses should follow immediately after one-another
        assert (
            timeline_res.target_map[drive_chan].end_positions[drive_pulse_idxs[0]] + 1
            == timeline_res.target_map[drive_chan].start_positions[drive_pulse_idxs[1]]
        )

        # The drive pulses happen inbetween the CR pulses. Check pulses follow immediately
        assert (
            timeline_res.target_map[cr_chan].end_positions[cr_pulse_idxs[0]] + 1
            == timeline_res.target_map[drive_chan].start_positions[drive_pulse_idxs[0]]
        )
        assert (
            timeline_res.target_map[drive_chan].end_positions[drive_pulse_idxs[1]] + 1
            == timeline_res.target_map[cr_chan].start_positions[cr_pulse_idxs[1]]
        )


class TestIntermediateFrequencyAnalysis:
    def test_different_frequencies_with_fixed_if_yields_error(self):
        model = EchoModelLoader().load()
        physical_channel = next(iter(model.physical_channels.values()))
        pulse_channels = iter(
            model.get_pulse_channels_from_physical_channel(physical_channel)
        )
        pulse_channel_1 = next(pulse_channels)
        pulse_channel_2 = next(pulse_channels)
        pulse_channel_2.frequency = pulse_channel_1.frequency + 1e8

        # some dummy data just to test the IFs
        res_mgr = ResultManager()
        ir = PartitionedIR(
            target_map={
                pulse_channel_1: [Delay(pulse_channel_1, 8e-8)],
                pulse_channel_2: [Delay(pulse_channel_2, 1.6e-7)],
            }
        )

        # run the pass: should pass
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
        model = EchoModelLoader().load()
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
        res_mgr = ResultManager()
        ir = PartitionedIR(
            target_map={
                pulse_channel_1: [Delay(pulse_channel_1, 8e-8)],
                pulse_channel_2: [Delay(pulse_channel_2, 1.6e-7)],
            }
        )

        # run the pass
        IntermediateFrequencyAnalysis(model).run(ir, res_mgr)

    def test_fixed_if_returns_frequencies(self):
        model = EchoModelLoader().load()

        # Find two pulse channels with different physical channels
        pulse_channels = iter(model.pulse_channels.values())
        pulse_channel_1 = next(pulse_channels)
        pulse_channel_2 = next(pulse_channels)
        while pulse_channel_1.physical_channel == pulse_channel_2.physical_channel:
            pulse_channel_2 = next(pulse_channels)

        pulse_channel_1.fixed_if = True
        pulse_channel_2.fixed_if = False  # test this isn't saved

        # some dummy data just to test the IF
        res_mgr = ResultManager()
        ir = PartitionedIR(
            target_map={
                pulse_channel_1: [Delay(pulse_channel_1, 8e-8)],
                pulse_channel_2: [Delay(pulse_channel_2, 1.6e-7)],
            }
        )

        # run the pass
        IntermediateFrequencyAnalysis(model).run(ir, res_mgr)
        res = res_mgr.lookup_by_type(IntermediateFrequencyResult)
        assert pulse_channel_1.physical_channel in res.frequencies
        assert pulse_channel_2.physical_channel not in res.frequencies
