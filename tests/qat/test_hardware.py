import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from tests.qat.utils import get_jagged_echo_hardware


@pytest.mark.parametrize(
    "hw", [get_default_echo_hardware, get_default_RTCS_hardware, get_jagged_echo_hardware]
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
