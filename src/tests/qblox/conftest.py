import pytest

from qat.purr.backends.qblox.live import QbloxLiveHardwareModel

from .utils import ClusterInfo, setup_qblox_hardware_model


@pytest.fixture
def model(request):
    info: ClusterInfo = request.param
    if not info.name:
        info.name = request.node.originalname

    model = QbloxLiveHardwareModel()
    setup_qblox_hardware_model(model, info)
    model.control_hardware.connect()
    yield model
    model.control_hardware.disconnect()
