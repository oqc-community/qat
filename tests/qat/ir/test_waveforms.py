import numpy as np
import pytest

from qat.ir.waveforms import CustomWaveform, Pulse, PulseShapeType, PulseType, Waveform
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import (
    CrossResonanceCancelPulse,
    CrossResonancePulse,
    CustomPulse,
    DrivePulse,
    MeasurePulse,
)
from qat.purr.compiler.instructions import Pulse as LegacyPulse
from qat.purr.compiler.instructions import SecondStatePulse
from qat.utils.ir_converter import IRConverter


class TestPulseConversion:

    model = get_default_echo_hardware(4)

    pulse_to_type_map = {
        DrivePulse: PulseType.DRIVE,
        CrossResonancePulse: PulseType.CROSS_RESONANCE,
        CrossResonanceCancelPulse: PulseType.CROSS_RESONANCE_CANCEL,
        SecondStatePulse: PulseType.SECOND_STATE,
        MeasurePulse: PulseType.MEASURE,
        LegacyPulse: PulseType.OTHER,
    }

    type_to_pulse_map = {
        PulseType.DRIVE: DrivePulse,
        PulseType.CROSS_RESONANCE: CrossResonancePulse,
        PulseType.CROSS_RESONANCE_CANCEL: CrossResonanceCancelPulse,
        PulseType.SECOND_STATE: SecondStatePulse,
        PulseType.MEASURE: MeasurePulse,
        PulseType.OTHER: LegacyPulse,
    }

    @pytest.mark.parametrize("LegacyPulseType", list(pulse_to_type_map.keys()))
    def test_legacy_to_pydantic(self, LegacyPulseType):
        pulse = LegacyPulseType(
            self.model.get_qubit(0).get_drive_channel(),
            PulseShapeType.GAUSSIAN,
            8e-8,
            drag=0.5,
            ignore_channel_scale=True,
        )
        converter = IRConverter(self.model)
        pydantic_pulse = converter._legacy_to_pydantic_pulse(pulse)
        assert pulse.shape == pydantic_pulse.waveform.shape
        assert pulse.width == pydantic_pulse.waveform.width
        assert pulse.quantum_targets[0].full_id() == pydantic_pulse.targets
        assert pulse.ignore_channel_scale == pydantic_pulse.ignore_channel_scale
        assert pulse.drag == pydantic_pulse.waveform.drag
        assert pydantic_pulse.type == self.pulse_to_type_map[LegacyPulseType]

    def test_legacy_to_pydantic_custom(self):
        pulse = CustomPulse(
            self.model.get_qubit(0).get_drive_channel(), samples=np.ones(101).tolist()
        )
        converter = IRConverter(self.model)
        pydantic_pulse = converter._legacy_to_pydantic_pulse(pulse)
        assert pulse.quantum_targets[0].full_id() == pydantic_pulse.targets
        assert pydantic_pulse.type == PulseType.OTHER
        assert pulse.samples == pydantic_pulse.waveform.samples

    @pytest.mark.parametrize("pydantic_type", PulseType)
    def test_pydantic_to_legacy(self, pydantic_type):
        waveform = Waveform(shape=PulseShapeType.SQUARE, width=8e-8, amp=1.0, drag=0.5)
        pydantic_pulse = Pulse(
            self.model.get_qubit(0).get_drive_channel(),
            ignore_channel_scale=True,
            type=pydantic_type,
            waveform=waveform,
        )
        converter = IRConverter(self.model)
        legacy_pulse = converter._pydantic_to_legacy_pulse(pydantic_pulse)
        assert legacy_pulse.shape == pydantic_pulse.waveform.shape
        assert legacy_pulse.width == pydantic_pulse.waveform.width
        assert legacy_pulse.quantum_targets[0].full_id() == pydantic_pulse.targets
        assert legacy_pulse.ignore_channel_scale == pydantic_pulse.ignore_channel_scale
        assert legacy_pulse.drag == pydantic_pulse.waveform.drag
        assert isinstance(legacy_pulse, self.type_to_pulse_map[pydantic_type])

    def test_pydantic_to_legacy_custom(self):
        waveform = CustomWaveform(samples=np.ones(101).tolist())
        pydantic_pulse = Pulse(
            self.model.get_qubit(0).get_drive_channel(),
            ignore_channel_scale=True,
            type=PulseType.OTHER,
            waveform=waveform,
        )
        converter = IRConverter(self.model)
        legacy_pulse = converter._pydantic_to_legacy_pulse(pydantic_pulse)
        assert isinstance(legacy_pulse, CustomPulse)
        assert legacy_pulse.samples == pydantic_pulse.waveform.samples

    @pytest.mark.parametrize("LegacyPulseType", list(pulse_to_type_map.keys()))
    def test_legacy_to_pydantic_to_legacy(self, LegacyPulseType):
        pulse = LegacyPulseType(
            self.model.get_qubit(0).get_drive_channel(),
            PulseShapeType.GAUSSIAN,
            8e-8,
            drag=0.5,
            ignore_channel_scale=True,
        )
        converter = IRConverter(self.model)
        pydantic_pulse = converter._legacy_to_pydantic_pulse(pulse)
        legacy_pulse = converter._pydantic_to_legacy_pulse(pydantic_pulse)
        assert vars(legacy_pulse) == vars(pulse)
        assert type(legacy_pulse) == type(pulse)
