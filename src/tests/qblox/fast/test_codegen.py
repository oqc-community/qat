import pytest

from qat.purr.backends.qblox.fast.codegen import EmitterMixin, FastQbloxEmitter
from qat.purr.backends.qblox.fast.live import FastQbloxLiveEngine
from qat.purr.utils.logger import get_default_logger
from src.tests.qblox.builder_nuggets import resonator_spect

log = get_default_logger()


@pytest.mark.parametrize("model", [None], indirect=True)
class TestFastQbloxEmitter:
    def test_emit_cfg(self, model):
        builder = resonator_spect(model)
        engine = FastQbloxLiveEngine(model)
        instructions = engine.optimize(builder.instructions)
        cfg = EmitterMixin(instructions).emit_cfg()
        assert cfg is not None
        assert len(cfg.nodes) == 5
        assert len(cfg.edges) == 6

    def test_emit_packages(self, model):
        builder = resonator_spect(model)
        engine = FastQbloxLiveEngine(model)
        instructions = engine.optimize(builder.instructions)
        packages = FastQbloxEmitter(instructions).emit_packages()
        assert packages is not None
