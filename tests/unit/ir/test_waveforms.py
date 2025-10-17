import numpy as np

from qat.ir.waveforms import Pulse, SampledWaveform, SquareWaveform, waveform_classes


class TestWaveformHashing:
    class_names = set(waveform_classes) - set([SampledWaveform])

    def test_different_clashes(self):
        for cls in self.class_names:
            waveform = cls(width=80e-9, amp=1.0)
            for other_cls in self.class_names:
                other_waveform = other_cls(width=80e-9, amp=1.0)
                if cls is other_cls:
                    assert hash(waveform) == hash(other_waveform)
                else:
                    assert hash(waveform) != hash(other_waveform)

    def test_different_parameters(self):
        for cls in self.class_names:
            waveform = cls(width=80e-9, amp=1.0)
            other_waveform = cls(width=100e-9, amp=1.0)
            assert hash(waveform) != hash(other_waveform)

            other_waveform = cls(width=80e-9, amp=0.5)
            assert hash(waveform) != hash(other_waveform)

            other_waveform = cls(width=100e-9, amp=0.5)
            assert hash(waveform) != hash(other_waveform)


class TestPulse:
    def test_update_duration_for_sampled(self):
        waveform = SampledWaveform(
            samples=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            sample_time=1e-9,
        )
        assert np.isclose(waveform.duration, 10e-9)
        pulse = Pulse(waveform=waveform, target="test")
        assert np.isclose(pulse.duration, 10e-9)

        pulse.update_duration(20e-9)
        assert np.isclose(pulse.duration, 20e-9)
        assert np.isclose(waveform.duration, 20e-9)
        assert len(waveform.samples) == 20
        assert np.allclose(waveform.samples[10:], 0.0)

    def test_update_duration_for_square(self):
        waveform = SquareWaveform(width=10e-9, amp=1.0)
        pulse = Pulse(waveform=waveform, target="test")
        assert np.isclose(pulse.duration, 10e-9)

        pulse.update_duration(20e-9)
        assert np.isclose(pulse.duration, 20e-9)
        assert np.isclose(waveform.width, 20e-9)
