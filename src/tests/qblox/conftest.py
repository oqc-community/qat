import pytest

from qat.purr.backends.qblox.live import QbloxLiveHardwareModel

from .utils import setup_qblox_hardware_model


@pytest.fixture
def model(request):
    cluster_kit = request.param
    name = request.node.originalname
    model = QbloxLiveHardwareModel()
    setup_qblox_hardware_model(model, cluster_kit, name)
    model.control_hardware.connect()
    yield model
    model.control_hardware.disconnect()
