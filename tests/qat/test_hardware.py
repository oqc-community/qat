import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.devices import _get_uuid


@pytest.mark.parametrize("hw", [get_default_echo_hardware, get_default_RTCS_hardware])
class TestCachedId:
    def test_new_partial_ids(self, hw):
        # Tests that changing the IDs of pulse channels generates new
        # hashes.
        model = hw()
        for key, target in model.pulse_channels.items():
            hash_val = target.__hash__()
            target.id = f"new_id_{key}"
            new_hash_val = target.__hash__()
            assert hash_val != new_hash_val

    def test_swap_partial_ids(self, hw):
        # Tests that swapping two pulse channel IDs keeps the association
        # between UUIDs and full IDs.
        # _get_uuid.cache_clear()
        model = hw()
        print(_get_uuid.cache_info())
        keys = list(model.pulse_channels.keys())
        pc1 = model.pulse_channels[keys[0]]
        pc2 = model.pulse_channels[keys[1]]
        print(_get_uuid(pc1.full_id()))
        print(_get_uuid(pc2.full_id()))
        print(hash(_get_uuid(pc1.full_id())))
        print(hash(_get_uuid(pc2.full_id())))
        hashs_before = [hash(pc1), hash(pc2)]
        print(hashs_before)
        old_id = pc1.id
        pc1.id = pc2.id
        pc2.id = old_id
        print(_get_uuid(pc1.full_id()))
        print(_get_uuid(pc2.full_id()))
        print(hash(_get_uuid(pc1.full_id())))
        print(hash(_get_uuid(pc2.full_id())))
        hashs_after = [hash(pc2), hash(pc1)]
        assert hashs_before == hashs_after

    def test_change_physical_channels(self, hw):
        # Tests that changing physical channels generates a new hash
        model = hw()
        pulse_channel_0 = model.get_qubit(0).get_drive_channel()
        pulse_channel_1 = model.get_qubit(1).get_drive_channel()
        hashs_before = [pulse_channel_0.__hash__(), pulse_channel_1.__hash__()]
        tmp = pulse_channel_1.physical_channel
        pulse_channel_1.physical_channel = pulse_channel_0.physical_channel
        pulse_channel_0.physical_channel = tmp
        hashs_after = [pulse_channel_0.__hash__(), pulse_channel_1.__hash__()]
        assert hashs_before != hashs_after
