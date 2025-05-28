import pytest

from qat.core.pass_base import PassManager
from qat.frontend.custom import CustomFrontend
from qat.model.loaders.legacy.echo import EchoModelLoader
from qat.purr.compiler.builders import QuantumInstructionBuilder


class TestCustomFrontend:
    @pytest.fixture(scope="class")
    def legacy_model(self):
        return EchoModelLoader(32).load()

    @pytest.mark.parametrize("src", ["test_source", bytes("test_source", "utf-8")])
    def test_emit(self, legacy_model, src):
        frontend = CustomFrontend(legacy_model, pipeline=PassManager())
        result = frontend.emit(src)
        assert result == src

    def test_emit_with_builder(self, legacy_model):
        frontend = CustomFrontend(legacy_model, pipeline=PassManager())
        builder = QuantumInstructionBuilder(legacy_model)
        result = frontend.emit(builder)
        assert result == builder

    def test_check_and_return_source(self, legacy_model):
        frontend = CustomFrontend(legacy_model)
        src = "test_source"
        result = frontend.check_and_return_source(src)
        assert result == src
