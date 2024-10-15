import pytest

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.integrations.features import OpenPulseFeatures


@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
        get_default_qiskit_hardware,
        get_default_RTCS_hardware,
    ],
)
def test_openpulsefeatures(get_hardware):
    opf = OpenPulseFeatures()
    opf.for_hardware(get_hardware())
    blob = opf.to_json_dict()
    assert ["open_pulse"] == list(blob.keys())
    assert list(blob["open_pulse"].keys()) == [
        "version",
        "ports",
        "frames",
        "waveforms",
        "constraints",
    ]
