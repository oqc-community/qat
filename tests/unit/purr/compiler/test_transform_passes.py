# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import Variable
from qat.purr.compiler.runtime import get_builder
from qat.purr.compiler.transform_passes import DeviceUpdateSanitisation
from qat.purr.core.metrics_base import MetricsManager
from qat.purr.core.result_base import ResultManager


class TestDeviceUpdateSanitisation:
    def test_device_update_semantics(self):
        model = get_default_echo_hardware()

        qubit = model.get_qubit(0)
        freq_var = f"freq{qubit.index}"
        drive_channel = qubit.get_drive_channel()

        # Valid DeviceUpdate
        builder = get_builder(model)
        builder.device_assign(drive_channel, "frequency", Variable(freq_var))
        DeviceUpdateSanitisation().run(builder, ResultManager(), MetricsManager())

        # Non valid DeviceUpdate
        builder = get_builder(model)
        builder.device_assign(drive_channel, "freakuency", Variable(freq_var))
        with pytest.raises(ValueError):
            DeviceUpdateSanitisation().run(builder, ResultManager(), MetricsManager())

        builder = get_builder(model)
        builder.device_assign(drive_channel, "bias", Variable("b1"))
        builder.device_assign(drive_channel, "bias", Variable("b2"))
        with pytest.raises(ValueError):
            DeviceUpdateSanitisation().run(builder, ResultManager(), MetricsManager())

    def test_device_update_sanitisation(self):
        model = get_default_echo_hardware()

        qubit = model.get_qubit(0)
        scale_var = f"scale{qubit.index}"
        freq_var = f"freq{qubit.index}"
        drive_channel = qubit.get_drive_channel()
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        builder = get_builder(model)
        builder.device_assign(drive_channel, "scale", Variable(scale_var))
        builder.device_assign(measure_channel, "scale", Variable(scale_var))
        builder.device_assign(acquire_channel, "scale", Variable(scale_var))

        length_before = len(builder.instructions)
        DeviceUpdateSanitisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == length_before

        builder.device_assign(drive_channel, "frequency", Variable(freq_var))
        builder.device_assign(measure_channel, "frequency", Variable(freq_var))
        builder.device_assign(acquire_channel, "frequency", Variable(freq_var))

        length_before = len(builder.instructions)
        DeviceUpdateSanitisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == length_before

        builder.device_assign(drive_channel, "scale", Variable(scale_var))
        builder.device_assign(measure_channel, "scale", Variable(scale_var))
        builder.device_assign(acquire_channel, "scale", Variable(scale_var))

        length_before = len(builder.instructions)
        DeviceUpdateSanitisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == length_before - 3
