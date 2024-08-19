import pytest

from qat.ir.pass_base import InvokerMixin, PassManager
from qat.purr.backends.analysis_passes import TriagePass
from qat.purr.backends.optimisation_passes import (
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
    SweepDecomposition,
)
from qat.purr.backends.qblox.fast.codegen import FastQbloxEmitter
from qat.purr.backends.verification_passes import (
    RepeatSanitisationValidation,
    ReturnSanitisationValidation,
    ScopeSanitisationValidation,
)
from qat.purr.utils.logger import get_default_logger
from tests.qat.qblox.builder_nuggets import resonator_spect
from tests.qat.qblox.utils import ClusterInfo

log = get_default_logger()


@pytest.mark.parametrize("model", [ClusterInfo()], indirect=True)
class TestFastQbloxEmitter(InvokerMixin):
    def build_pass_pipeline(self, *args, **kwargs):
        model = kwargs["model"]
        pipeline = PassManager()
        pipeline.add(SweepDecomposition())
        pipeline.add(RepeatSanitisation(model))
        pipeline.add(ScopeSanitisation())
        pipeline.add(ReturnSanitisation())
        pipeline.add(ScopeSanitisationValidation())
        pipeline.add(ReturnSanitisationValidation())
        pipeline.add(RepeatSanitisationValidation())
        pipeline.add(TriagePass())
        return pipeline

    def test_emit_packages(self, model):
        qubit_index = 0
        qubit = model.get_qubit(qubit_index)
        builder = resonator_spect(model, qubit_index=qubit_index)
        model.create_runtime().run_pass_pipeline(builder)
        analyses = self.run_pass_pipeline(builder, model=model)
        packages = FastQbloxEmitter(analyses).emit_packages(builder)
        assert len(packages) == 1
        pkg = packages[0]
        assert pkg.target == qubit.get_measure_channel()
        assert pkg.sequence.acquisitions
        assert pkg.sequence.program
        assert pkg.sequence.waveforms
        assert not pkg.sequence.weights
