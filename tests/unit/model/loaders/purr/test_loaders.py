import tempfile
from pathlib import Path

import pytest

from qat.model.loaders.purr import (
    EchoModelLoader,
    FileModelLoader,
    LucyModelLoader,
    QiskitModelLoader,
    RTCSModelLoader,
)
from qat.purr.compiler.hardware_models import HardwareModel as LegacyHardwareModel


class TestLoaders:
    @pytest.fixture
    def lucy_legacy_calibration_file_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cal.json"
            model = LucyModelLoader().load()
            model.save_calibration_to_file(path)
            yield path

    @pytest.mark.parametrize(
        "Loader",
        [EchoModelLoader, LucyModelLoader, QiskitModelLoader, RTCSModelLoader],
    )
    def test_default_load(self, Loader):
        loader = Loader()
        model = loader.load()
        assert isinstance(model, LegacyHardwareModel)

    def test_file_load(self, lucy_legacy_calibration_file_path):
        path = lucy_legacy_calibration_file_path
        loader = FileModelLoader(path)
        model = loader.load()
        assert isinstance(model, LegacyHardwareModel)
        # Checks whether the hash is generated correctly.
        assert len(model.calibration_id) == 32
