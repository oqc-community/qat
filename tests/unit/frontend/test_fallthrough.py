from qat.frontend.fallthrough import FallthroughFrontend
from qat.model.loaders.purr.echo import Connectivity, EchoModelLoader
from qat.purr.compiler.builders import QuantumInstructionBuilder


class TestFallthroughFrontend:
    def test_emit(self):
        frontend = FallthroughFrontend()
        model = EchoModelLoader(32, connectivity=Connectivity.Ring).load()
        builder = QuantumInstructionBuilder(model)
        result = frontend.emit(builder)
        assert result == builder

    def test_check_and_return_source(self):
        frontend = FallthroughFrontend()
        src = "test_source"  # This can be any object, as the frontend does not modify it
        result = frontend.check_and_return_source(src)
        assert result == src
