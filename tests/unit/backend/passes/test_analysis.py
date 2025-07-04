import numpy as np
import pytest

from qat.backend.passes.analysis import (
    IntermediateFrequencyResult,
    PydIntermediateFrequencyAnalysis,
    PydTimelineAnalysis,
    PydTimelineAnalysisResult,
)
from qat.backend.passes.lowering import PydPartitionByPulseChannel
from qat.backend.passes.purr.analysis import (
    IntermediateFrequencyAnalysis,
    TimelineAnalysis,
    TimelineAnalysisResult,
)
from qat.backend.passes.purr.lowering import PartitionByPulseChannel
from qat.core.result_base import ResultManager
from qat.ir.conversion import ConvertToPydanticIR
from qat.ir.instructions import Delay
from qat.ir.lowered import PartitionedIR
from qat.ir.waveforms import Pulse
from qat.middleend.passes.purr.transform import (
    InstructionGranularitySanitisation as LegInstructionGranularitySanitisation,
)
from qat.middleend.passes.purr.transform import (
    LowerSyncsToDelays as LegLowerSyncsToDelays,
)
from qat.model.loaders.converted import PydEchoModelLoader
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import Delay as LegDelay


class TestIntermediateFrequencyAnalysis:
    def test_different_frequencies_with_fixed_if_yields_error(self):
        model = PydEchoModelLoader().load()
        pulse_channels = iter(model.qubits[0].all_pulse_channels)

        pulse_channel_1 = next(pulse_channels)
        pulse_channel_2 = next(pulse_channels)
        pulse_channel_2.frequency = pulse_channel_1.frequency + 1e8

        # some dummy data just to test the IFs
        res_mgr = ResultManager()
        ir = PartitionedIR(
            target_map={
                pulse_channel_1.uuid: [Delay(target=pulse_channel_1.uuid, duration=8e-8)],
                pulse_channel_2.uuid: [Delay(target=pulse_channel_2.uuid, duration=1.6e-7)],
            }
        )

        # run the pass: should pass
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        # fix IF for one channel: should pass
        pulse_channel_1.fixed_if = True
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        # fix IF for one channel: should pass
        pulse_channel_1.fixed_if = False
        pulse_channel_2.fixed_if = True
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        # fix IF for both channels: should fail
        pulse_channel_1.fixed_if = True
        with pytest.raises(ValueError):
            PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

    def test_same_frequencies_with_fixed_if_passes(self):
        model = PydEchoModelLoader().load()
        pulse_channels = iter(model.qubits[0].all_pulse_channels)

        pulse_channel_1 = next(pulse_channels)
        pulse_channel_2 = next(pulse_channels)
        pulse_channel_1.fixed_if = True
        pulse_channel_2.fixed_if = True
        pulse_channel_2.frequency = pulse_channel_1.frequency

        # some dummy data just to test the IFs
        res_mgr = ResultManager()
        ir = PartitionedIR(
            target_map={
                pulse_channel_1.uuid: [Delay(target=pulse_channel_1.uuid, duration=8e-8)],
                pulse_channel_2.uuid: [Delay(target=pulse_channel_2.uuid, duration=1.6e-7)],
            }
        )

        # run the pass
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

    #
    def test_fixed_if_returns_frequencies(self):
        model = PydEchoModelLoader().load()

        # Find two pulse channels with different physical channels

        physical_channel_1 = model.qubits[0].physical_channel
        physical_channel_2 = model.qubits[1].physical_channel
        pulse_channel_1 = model.qubits[0].drive_pulse_channel
        pulse_channel_2 = model.qubits[1].drive_pulse_channel

        pulse_channel_1.fixed_if = True
        pulse_channel_2.fixed_if = False  # test this isn't saved

        # some dummy data just to test the IF
        res_mgr = ResultManager()
        ir = PartitionedIR(
            target_map={
                pulse_channel_1.uuid: [Delay(target=pulse_channel_1.uuid, duration=8e-8)],
                pulse_channel_2.uuid: [Delay(target=pulse_channel_2.uuid, duration=1.6e-7)],
            }
        )

        # run the pass
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)
        res = res_mgr.lookup_by_type(IntermediateFrequencyResult)
        assert physical_channel_1.uuid in res.frequencies
        assert physical_channel_2.uuid not in res.frequencies


class TestIntermediateFrequencyAnalysisParity:
    def test_different_frequencies_with_fixed_if_yields_error(self):
        leg_model = EchoModelLoader().load()
        model = PydEchoModelLoader().load()
        physical_channel = next(iter(leg_model.physical_channels.values()))
        pulse_channels = iter(
            leg_model.get_pulse_channels_from_physical_channel(physical_channel)
        )
        pulse_channel_1 = next(pulse_channels)
        pulse_channel_2 = next(pulse_channels)
        pulse_channel_2.frequency = pulse_channel_1.frequency + 1e8

        # some dummy data just to test the IFs
        res_mgr = ResultManager()
        leg_ir = PartitionedIR(
            target_map={
                pulse_channel_1: [LegDelay(pulse_channel_1, 8e-8)],
                pulse_channel_2: [LegDelay(pulse_channel_2, 1.6e-7)],
            }
        )

        # run the pass: should pass
        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir, res_mgr)
        IntermediateFrequencyAnalysis(leg_model).run(leg_ir, res_mgr)
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        # fix IF for one channel: should pass
        pulse_channel_1.fixed_if = True
        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir, res_mgr)
        IntermediateFrequencyAnalysis(leg_model).run(leg_ir, res_mgr)
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        # fix IF for one channel: should pass
        pulse_channel_1.fixed_if = False
        pulse_channel_2.fixed_if = True
        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir, res_mgr)
        IntermediateFrequencyAnalysis(leg_model).run(leg_ir, res_mgr)
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        # fix IF for both channels: should fail
        pulse_channel_1.fixed_if = True
        with pytest.raises(ValueError):
            ir = ConvertToPydanticIR(leg_model, model).run(leg_ir, res_mgr)
            IntermediateFrequencyAnalysis(leg_model).run(leg_ir, res_mgr)
            PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

    def test_same_frequencies_with_fixed_if_passes(self):
        leg_model = EchoModelLoader().load()
        model = PydEchoModelLoader().load()
        physical_channel = next(iter(leg_model.physical_channels.values()))
        pulse_channels = iter(
            leg_model.get_pulse_channels_from_physical_channel(physical_channel)
        )
        pulse_channel_1 = next(pulse_channels)
        pulse_channel_2 = next(pulse_channels)
        pulse_channel_1.fixed_if = True
        pulse_channel_2.fixed_if = True
        pulse_channel_2.frequency = pulse_channel_1.frequency

        # some dummy data just to test the IFs
        res_mgr = ResultManager()
        leg_ir = PartitionedIR(
            target_map={
                pulse_channel_1: [LegDelay(pulse_channel_1, 8e-8)],
                pulse_channel_2: [LegDelay(pulse_channel_2, 1.6e-7)],
            }
        )

        # run the pass
        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir, res_mgr)
        IntermediateFrequencyAnalysis(leg_model).run(leg_ir, res_mgr)
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

    def test_fixed_if_returns_frequencies(self):
        leg_model = EchoModelLoader().load()
        model = PydEchoModelLoader().load()

        pulse_channel_1 = leg_model.qubits[0].get_default_pulse_channel()
        pulse_channel_2 = leg_model.qubits[1].get_default_pulse_channel()

        pulse_channel_1.fixed_if = True
        pulse_channel_2.fixed_if = False  # test this isn't saved

        # some dummy data just to test the IF
        leg_res_mgr = ResultManager()
        res_mgr = ResultManager()
        leg_ir = PartitionedIR(
            target_map={
                pulse_channel_1: [LegDelay(pulse_channel_1, 8e-8)],
                pulse_channel_2: [LegDelay(pulse_channel_2, 1.6e-7)],
            }
        )

        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir, res_mgr)
        model.qubits[0].drive_pulse_channel.fixed_if = True
        model.qubits[1].drive_pulse_channel.fixed_if = False

        # run the pass
        IntermediateFrequencyAnalysis(leg_model).run(leg_ir, leg_res_mgr)
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)
        leg_res = leg_res_mgr.lookup_by_type(IntermediateFrequencyResult)
        res = res_mgr.lookup_by_type(IntermediateFrequencyResult)
        assert pulse_channel_1.physical_channel in leg_res.frequencies
        assert pulse_channel_2.physical_channel not in leg_res.frequencies
        assert model.qubits[0].physical_channel.uuid in res.frequencies
        assert model.qubits[1].physical_channel.uuid not in res.frequencies


class TestTimelineAnalysis:
    def test_timelines_match(self):
        """Test the results of the timeline analyis match with the expectation."""
        loader = PydEchoModelLoader()
        leg_model = loader._legacy.load()
        pyd_model = loader.load()
        target_data = TargetData.random()
        sample_time = target_data.QUBIT_DATA.sample_time

        qubits = leg_model.qubits[0:2]
        drive_channels = [qubit.get_drive_channel() for qubit in qubits]
        measure_channels = [qubit.get_measure_channel() for qubit in qubits]
        acquire_channels = [qubit.get_acquire_channel() for qubit in qubits]
        cr_channel = qubits[1].get_cross_resonance_channel(qubits[0])
        crc_channel = qubits[0].get_cross_resonance_cancellation_channel(qubits[1])

        builder = leg_model.create_builder()
        builder.delay(drive_channels[0], sample_time * 8 * 10)
        builder.delay(drive_channels[1], sample_time * 8 * 5)
        builder.synchronize([*drive_channels, cr_channel, crc_channel])
        builder.ECR(*qubits)
        builder.X(qubits[0])
        builder.delay(qubits[1], sample_time * 8 * 50)
        builder.synchronize(
            [drive_channels[0], measure_channels[0], acquire_channels[0], crc_channel]
        )
        builder.synchronize(
            [drive_channels[1], measure_channels[1], acquire_channels[1], cr_channel]
        )
        builder.measure(qubits[0])
        builder.measure(qubits[1])

        builder = LegInstructionGranularitySanitisation(leg_model, target_data).run(builder)
        builder = LegLowerSyncsToDelays().run(builder)

        res_mgr = ResultManager()
        pyd_builder = ConvertToPydanticIR(leg_model, pyd_model).run(builder, res_mgr)

        ir = PydPartitionByPulseChannel().run(pyd_builder, res_mgr)
        PydTimelineAnalysis(pyd_model, target_data).run(ir, res_mgr)
        res = res_mgr.lookup_by_type(PydTimelineAnalysisResult)

        # Do some manual analysis of times
        durations = {}
        for inst in pyd_builder:
            for qt in inst.targets:
                durations.setdefault(qt, []).append(inst.duration)

        assert durations.keys() == res.target_map.keys()
        for key, val in durations.items():
            val = np.asarray(val)
            ends = np.cumsum(val)
            starts = ends - val
            assert np.all(
                np.isclose(val * 1e9, res.target_map[key].samples * sample_time * 1e9)
            )
            assert np.all(
                np.isclose(
                    ends * 1e9,
                    (res.target_map[key].end_positions + 1) * sample_time * 1e9,
                )
            )
            assert np.all(
                np.isclose(
                    starts * 1e9,
                    res.target_map[key].start_positions * sample_time * 1e9,
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

        loader = PydEchoModelLoader()
        leg_model = loader._legacy.load()
        pyd_model = loader.load()
        target_data = TargetData.random()

        qubit1 = leg_model.qubits[0]
        qubit2 = leg_model.qubits[1]
        qubit1.pulse_hw_x_pi_2["width"] = drive_width
        qubit1.pulse_hw_zx_pi_4[qubit2.full_id()]["width"] = cr_width
        qubit2.pulse_hw_zx_pi_4[qubit1.full_id()]["width"] = cr_width

        builder = leg_model.create_builder()
        if control_delay:
            builder.delay(qubit1.get_drive_channel(), time=control_delay)
        if target_delay:
            builder.delay(qubit2.get_drive_channel(), time=target_delay)
        builder.ECR(qubit1, qubit2)

        res_mgr = ResultManager()
        LegInstructionGranularitySanitisation(leg_model, target_data).run(builder, res_mgr)
        LegLowerSyncsToDelays().run(builder, res_mgr)

        pyd_builder = ConvertToPydanticIR(leg_model, pyd_model).run(builder, res_mgr)

        ir = PydPartitionByPulseChannel().run(pyd_builder, res_mgr)
        PydTimelineAnalysis(pyd_model, target_data).run(ir, res_mgr)

        target_map = ir.target_map
        timeline_res: PydTimelineAnalysisResult = res_mgr.lookup_by_type(
            PydTimelineAnalysisResult
        )

        # Inspect the drive channel: make sure there are two pulses from the ECR gate.
        pyd_qubit1 = pyd_model.qubit_with_index(0)
        pyd_qubit2 = pyd_model.qubit_with_index(1)
        drive_pulse_ch = pyd_qubit1.drive_pulse_channel
        drive_pulses = [isinstance(inst, Pulse) for inst in target_map[drive_pulse_ch.uuid]]
        assert sum(drive_pulses) == 2
        drive_pulse_idxs = [i for i, val in enumerate(drive_pulses) if val]

        # Inspect the CRC channel: make sure there are two pulses from the ECR gate.
        crc_pulse_ch = pyd_qubit2.cross_resonance_cancellation_pulse_channels[0]
        crc_pulses = [isinstance(inst, Pulse) for inst in target_map[crc_pulse_ch.uuid]]
        assert sum(crc_pulses) == 2
        crc_pulse_idxs = [i for i, val in enumerate(crc_pulses) if val]

        # Inspect the CR channel: make sure there are two pulses from the ECR gate.
        cr_pulse_ch = pyd_qubit1.cross_resonance_pulse_channels[1]
        cr_pulses = [isinstance(inst, Pulse) for inst in target_map[cr_pulse_ch.uuid]]
        assert sum(cr_pulses) == 2
        cr_pulse_idxs = [i for i, val in enumerate(cr_pulses) if val]

        # The CR and CRC pulses should always be in sync; let's check that!
        for i in range(2):
            assert (
                timeline_res.target_map[cr_pulse_ch.uuid].start_positions[cr_pulse_idxs[i]]
                == timeline_res.target_map[crc_pulse_ch.uuid].start_positions[
                    crc_pulse_idxs[i]
                ]
            )
            assert (
                timeline_res.target_map[cr_pulse_ch.uuid].end_positions[cr_pulse_idxs[i]]
                == timeline_res.target_map[crc_pulse_ch.uuid].end_positions[
                    crc_pulse_idxs[i]
                ]
            )

        # The drive pulses should follow immediately after one-another
        assert (
            timeline_res.target_map[drive_pulse_ch.uuid].end_positions[drive_pulse_idxs[0]]
            + 1
            == timeline_res.target_map[drive_pulse_ch.uuid].start_positions[
                drive_pulse_idxs[1]
            ]
        )

        # The drive pulses happen inbetween the CR pulses. Check pulses follow immediately
        assert (
            timeline_res.target_map[cr_pulse_ch.uuid].end_positions[cr_pulse_idxs[0]] + 1
            == timeline_res.target_map[drive_pulse_ch.uuid].start_positions[
                drive_pulse_idxs[0]
            ]
        )
        assert (
            timeline_res.target_map[drive_pulse_ch.uuid].end_positions[drive_pulse_idxs[1]]
            + 1
            == timeline_res.target_map[cr_pulse_ch.uuid].start_positions[cr_pulse_idxs[1]]
        )


class TestTimelineAnalysisParity:
    def test_timelines_match(self):
        """Test the results of the timeline analyis match with the expectation."""
        loader = PydEchoModelLoader()
        leg_model = loader._legacy.load()
        pyd_model = loader.load()
        target_data = TargetData.default()

        qubits = leg_model.qubits[0:2]
        drive_channels = [qubit.get_drive_channel() for qubit in qubits]
        measure_channels = [qubit.get_measure_channel() for qubit in qubits]
        acquire_channels = [qubit.get_acquire_channel() for qubit in qubits]
        cr_channel = qubits[1].get_cross_resonance_channel(qubits[0])
        crc_channel = qubits[0].get_cross_resonance_cancellation_channel(qubits[1])

        builder = leg_model.create_builder()
        builder.delay(drive_channels[0], target_data.QUBIT_DATA.sample_time * 8 * 10)
        builder.delay(drive_channels[1], target_data.QUBIT_DATA.sample_time * 8 * 5)
        builder.synchronize([*drive_channels, cr_channel, crc_channel])
        builder.ECR(*qubits)
        builder.X(qubits[0])
        builder.delay(qubits[1], target_data.QUBIT_DATA.sample_time * 8 * 50)
        builder.synchronize(
            [drive_channels[0], measure_channels[0], acquire_channels[0], crc_channel]
        )
        builder.synchronize(
            [drive_channels[1], measure_channels[1], acquire_channels[1], cr_channel]
        )
        builder.measure(qubits[0])
        builder.measure(qubits[1])

        builder = LegInstructionGranularitySanitisation(leg_model, target_data).run(builder)
        builder = LegLowerSyncsToDelays().run(builder)

        pyd_res_mgr = ResultManager()
        pyd_builder = ConvertToPydanticIR(leg_model, pyd_model).run(builder, pyd_res_mgr)

        pyd_ir = PydPartitionByPulseChannel().run(pyd_builder, pyd_res_mgr)
        PydIntermediateFrequencyAnalysis(pyd_model).run(pyd_ir, pyd_res_mgr)
        PydTimelineAnalysis(pyd_model, target_data).run(pyd_ir, pyd_res_mgr)
        pyd_res = pyd_res_mgr.lookup_by_type(PydTimelineAnalysisResult)

        leg_res_mgr = ResultManager()
        leg_ir = PartitionByPulseChannel().run(builder, leg_res_mgr)
        IntermediateFrequencyAnalysis(leg_model).run(leg_ir, leg_res_mgr)
        TimelineAnalysis().run(leg_ir, leg_res_mgr)
        leg_res = leg_res_mgr.lookup_by_type(TimelineAnalysisResult)

        assert pyd_res.total_duration == leg_res.total_duration
        assert len(pyd_res.target_map.values()) == len(leg_res.target_map.values())
        for pyd_timeline, res_timeline in zip(
            pyd_res.target_map.values(), leg_res.target_map.values()
        ):
            assert np.isclose(pyd_timeline.samples, res_timeline.samples).all()
            assert np.isclose(
                pyd_timeline.start_positions, res_timeline.start_positions
            ).all()
            assert np.isclose(pyd_timeline.end_positions, res_timeline.end_positions).all()
