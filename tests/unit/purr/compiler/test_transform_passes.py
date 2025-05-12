import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.qblox.metrics_base import MetricsManager
from qat.purr.backends.qblox.pass_base import QatIR
from qat.purr.backends.qblox.result_base import ResultManager
from qat.purr.compiler.instructions import Variable
from qat.purr.compiler.runtime import get_builder
from qat.purr.compiler.transform_passes import DeviceUpdateSanitisation


class TestDeviceUpdateSanitisation:
    def test_device_update_semantics(self):
        model = get_default_echo_hardware()

        qubit = model.get_qubit(0)
        freq_var = f"freq{qubit.index}"
        drive_channel = qubit.get_drive_channel()

        # Valid DeviceUpdate
        builder = get_builder(model)
        builder.device_assign(drive_channel, "frequency", Variable(freq_var))
        DeviceUpdateSanitisation().run(QatIR(builder), ResultManager(), MetricsManager())

        # Non valid DeviceUpdate
        builder = get_builder(model)
        builder.device_assign(drive_channel, "freakuency", Variable(freq_var))
        with pytest.raises(ValueError):
            DeviceUpdateSanitisation().run(
                QatIR(builder), ResultManager(), MetricsManager()
            )

        builder = get_builder(model)
        builder.device_assign(drive_channel, "bias", Variable("b1"))
        builder.device_assign(drive_channel, "bias", Variable("b2"))
        with pytest.raises(ValueError):
            DeviceUpdateSanitisation().run(
                QatIR(builder), ResultManager(), MetricsManager()
            )

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
        DeviceUpdateSanitisation().run(QatIR(builder), ResultManager(), MetricsManager())
        assert len(builder.instructions) == length_before

        builder.device_assign(drive_channel, "frequency", Variable(freq_var))
        builder.device_assign(measure_channel, "frequency", Variable(freq_var))
        builder.device_assign(acquire_channel, "frequency", Variable(freq_var))

        length_before = len(builder.instructions)
        DeviceUpdateSanitisation().run(QatIR(builder), ResultManager(), MetricsManager())
        assert len(builder.instructions) == length_before

        builder.device_assign(drive_channel, "scale", Variable(scale_var))
        builder.device_assign(measure_channel, "scale", Variable(scale_var))
        builder.device_assign(acquire_channel, "scale", Variable(scale_var))

        length_before = len(builder.instructions)
        DeviceUpdateSanitisation().run(QatIR(builder), ResultManager(), MetricsManager())
        assert len(builder.instructions) == length_before - 3
