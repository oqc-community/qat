import pytest

from qat.backend.passes.analysis import (
    IntermediateFrequencyResult,
    PydIntermediateFrequencyAnalysis,
)
from qat.backend.passes.legacy.analysis import IntermediateFrequencyAnalysis
from qat.core.result_base import ResultManager
from qat.ir.conversion import ConvertToPydanticIR
from qat.ir.instructions import Delay
from qat.ir.lowered import PartitionedIR
from qat.model.loaders.converted import PydEchoModelLoader
from qat.model.loaders.legacy import EchoModelLoader
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
        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir)
        IntermediateFrequencyAnalysis(leg_model).run(leg_ir, res_mgr)
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        # fix IF for one channel: should pass
        pulse_channel_1.fixed_if = True
        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir)
        IntermediateFrequencyAnalysis(leg_model).run(leg_ir, res_mgr)
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        # fix IF for one channel: should pass
        pulse_channel_1.fixed_if = False
        pulse_channel_2.fixed_if = True
        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir)
        IntermediateFrequencyAnalysis(leg_model).run(leg_ir, res_mgr)
        PydIntermediateFrequencyAnalysis(model).run(ir, res_mgr)

        # fix IF for both channels: should fail
        pulse_channel_1.fixed_if = True
        with pytest.raises(ValueError):
            ir = ConvertToPydanticIR(leg_model, model).run(leg_ir)
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
        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir)
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

        ir = ConvertToPydanticIR(leg_model, model).run(leg_ir)
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
