# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.ir.waveforms import (
    ExtraSoftSquareWaveform,
    Pulse,
    SampledWaveform,
    SofterSquareWaveform,
    SquareWaveform,
    waveform_classes,
)


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


@pytest.mark.parametrize("waveform", [SofterSquareWaveform, ExtraSoftSquareWaveform])
class TestSoftSquares:
    def test_std_dev_is_used_for_shape_evaluation(self, waveform):
        """Regression test to ensure std_dev is used in SofterSquareWaveform and
        ExtraSoftSquareWaveform.

        Tests that changing the std_dev parameter changes the shape of the waveform.
        """

        waveform1 = waveform(amp=1.0, width=400e-9, rise=50e-9, std_dev=100e-9)
        waveform2 = waveform(amp=1.0, width=400e-9, rise=50e-9, std_dev=200e-9)

        times = np.linspace(-200e-9, 200e-9, 41)
        samples1 = waveform1.sample(times).samples
        samples2 = waveform2.sample(times).samples

        assert not np.allclose(samples1, samples2)
