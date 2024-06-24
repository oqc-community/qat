import pytest

from qat.purr.backends.qblox.fast.live import FastQbloxLiveEngine
from src.tests.qblox.builder_nuggets import resonator_spect


@pytest.mark.parametrize("model", [None], indirect=True)
class TestFastQbloxLiveEngine:
    def test_execute(self, model):
        builder = resonator_spect(model)
        engine = FastQbloxLiveEngine(model)
        results = engine.execute(builder.instructions)
        assert results is not None
