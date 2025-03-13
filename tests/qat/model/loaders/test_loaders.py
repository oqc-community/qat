import tempfile
from pathlib import Path

import pytest

from qat.model.builder import PhysicalHardwareModelBuilder
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.converted import EchoModelLoader
from qat.model.loaders.file import FileModelLoader
from qat.purr.backends.echo import Connectivity
from qat.utils.hardware_model import generate_connectivity_data


class TestConversionLoaders:
    @pytest.fixture
    def random_model(self, n_qubits=4, seed=42):
        physical_connectivity, logical_connectivity, logical_connectivity_quality = (
            generate_connectivity_data(n_qubits, n_qubits, seed=seed)
        )

        return PhysicalHardwareModelBuilder(
            physical_connectivity=physical_connectivity,
            logical_connectivity=logical_connectivity,
            logical_connectivity_quality=logical_connectivity_quality,
        ).model

    @pytest.fixture
    def random_model_json_path(self, random_model):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cal.json"
            path.write_text(random_model.model_dump_json())
            yield path

    @pytest.mark.parametrize(
        "Loader",
        [EchoModelLoader],
    )
    def test_default_load(self, Loader):
        loader = Loader()
        model = loader.load()
        assert isinstance(model, PhysicalHardwareModel)

    def test_echo_connectivity(self):
        loader = EchoModelLoader(qubit_count=4, connectivity=Connectivity.Ring)
        model = loader.load()
        assert isinstance(model, PhysicalHardwareModel)
        assert model.logical_connectivity == {
            0: {1, 3},
            1: {0, 2},
            2: {1, 3},
            3: {0, 2},
        }

    def test_file_load(self, random_model_json_path):
        path = random_model_json_path
        loader = FileModelLoader(path)
        model = loader.load()
        assert isinstance(model, PhysicalHardwareModel)
