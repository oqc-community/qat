import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware


@pytest.mark.parametrize(
    "model", [get_default_echo_hardware(2), get_default_RTCS_hardware()]
)
class TestCachedId:
    def test_new_partial_ids(self, model):
        # Tests that changing the IDs of pulse channels generates new
        # hashes.
        for key, target in model.pulse_channels.items():
            hash_val = target.__hash__()
            target.id = f"new_id_{key}"
            new_hash_val = target.__hash__()
            assert hash_val != new_hash_val

    def test_swap_partial_ids(self, model):
        # Tests that swapping two pulse channel IDs keeps the association
        # between UUIDs and full IDs.
        keys = list(model.pulse_channels.keys())
        hashs_before = [model.pulse_channels[keys[0]], model.pulse_channels[keys[1]]]
        old_id = model.pulse_channels[keys[0]]
        model.pulse_channels[keys[0]] = model.pulse_channels[keys[1]]
        model.pulse_channels[keys[1]] = old_id
        hashs_after = [model.pulse_channels[keys[1]], model.pulse_channels[keys[0]]]
        assert hashs_before == hashs_after
