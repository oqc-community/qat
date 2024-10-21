import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.devices import Calibratable, PulseChannelView
from qat.purr.compiler.hardware_models import QuantumHardwareModel

from tests.qat.utils.models import get_jagged_echo_hardware


def deserialize_hw_model():
    # Loads in an old HW model without cached properties in physical channels
    return Calibratable().load_calibration_from_file(
        "tests/qat/files/serialization/echo.json"
    )


def test_deserialization():
    # Checks that an old HW model serializes correctly
    model = deserialize_hw_model()
    assert isinstance(model, QuantumHardwareModel)
    for pc in model.pulse_channels.values():
        # A view of a pulse channel cannot "see" private members of a pulse channel
        # when deserialized due to that attribute not being saved previously.
        # Shouldn't be a problem since in practice one shouldn't try to access
        # _id and _physical_channel directly
        if isinstance(pc, PulseChannelView):
            pc = pc.pulse_channel
        assert pc._id == pc.id
        assert pc._physical_channel == pc.physical_channel
        assert pc._full_id == pc.full_id()


@pytest.mark.parametrize(
    "hw",
    [
        get_default_echo_hardware,
        get_default_RTCS_hardware,
        get_jagged_echo_hardware,
        deserialize_hw_model,
    ],
)
class TestCachedId:
    def test_new_partial_ids(self, hw):
        # Tests that changing the IDs of pulse channels generates new
        # hashes.
        model = hw()
        for key, target in model.pulse_channels.items():
            hash_val = hash(target)
            target.id = f"new_id_{key}"
            new_hash_val = hash(target)
            assert hash_val != new_hash_val

    def test_swap_partial_ids(self, hw):
        # Tests that swapping two pulse channel IDs keeps the association
        # between hashes and full IDs.
        model = hw()
        keys = list(model.pulse_channels.keys())
        pc1 = model.pulse_channels[keys[0]]
        pc2 = model.pulse_channels[keys[1]]
        hashs_before = [hash(pc1), hash(pc2)]
        old_id = pc1.id
        pc1.id = pc2.id
        pc2.id = old_id
        hashs_after = [hash(pc2), hash(pc1)]
        assert hashs_before == hashs_after

    def test_change_physical_channels(self, hw):
        # Tests that changing physical channels generates a new hash
        model = hw()
        qubits = model.qubits
        pulse_channel_0 = qubits[0].get_drive_channel()
        pulse_channel_1 = qubits[1].get_drive_channel()
        hashs_before = [hash(pulse_channel_0), hash(pulse_channel_1)]
        tmp = pulse_channel_1.physical_channel
        pulse_channel_1.physical_channel = pulse_channel_0.physical_channel
        pulse_channel_0.physical_channel = tmp
        hashs_after = [hash(pulse_channel_0), hash(pulse_channel_1)]
        assert hashs_before != hashs_after
