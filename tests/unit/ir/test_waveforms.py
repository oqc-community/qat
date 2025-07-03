from qat.ir.waveforms import SampledWaveform, waveform_classes


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
