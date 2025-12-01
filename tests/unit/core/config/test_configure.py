from dataclasses import dataclass

import pytest

from qat.backend.fallthrough import FallthroughBackend
from qat.core.config.configure import get_config, override_config
from qat.core.config.session import QatSessionConfig
from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import AnalysisPass, PassManager, TransformPass, ValidationPass
from qat.core.qat import QAT
from qat.core.result_base import ResultInfoMixin, ResultManager
from qat.engines.waveform.echo import EchoEngine
from qat.frontend.fallthrough import FallthroughFrontend
from qat.middleend import CustomMiddleend
from qat.model.loaders.purr.echo import EchoModelLoader
from qat.pipelines.pipeline import Pipeline
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import Instruction
from qat.runtime.simple import SimpleRuntime


@dataclass
class DummyResult(ResultInfoMixin):
    max_repeats: int = 0


class DummyAnalysis(AnalysisPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        result = DummyResult()
        result.max_repeats = get_config().MAX_REPEATS_LIMIT
        res_mgr.add(result)
        return builder


class DummyValidation(ValidationPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        for inst in builder.instructions:
            if not isinstance(inst, Instruction):
                raise ValueError(f"{inst} is not an valid instruction")
        return builder


class DummyTransform(TransformPass):
    def run(
        self,
        builder: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        builder.instructions = builder.instructions[::-1]
        return builder


class DummyTransformReturnRepeats(TransformPass):
    def run(
        self,
        builder: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        return [res_mgr.lookup_by_type(DummyResult).max_repeats]


def test_get_config_global_pipe():
    hw = EchoModelLoader(qubit_count=4).load()
    b = InstructionBuilder(hw)

    res_mgr = ResultManager()
    met_mgr = MetricsManager()

    Pm = PassManager() | DummyValidation() | DummyTransform() | DummyAnalysis()

    MAX_REPEATS_LIMIT = 253423
    OLD_LIMIT = get_config().MAX_REPEATS_LIMIT
    assert OLD_LIMIT != MAX_REPEATS_LIMIT
    try:
        get_config().MAX_REPEATS_LIMIT = MAX_REPEATS_LIMIT

        Pm.run(b, res_mgr=res_mgr, met_mgr=met_mgr)

        dummy_result = res_mgr.lookup_by_type(DummyResult)
        assert dummy_result.max_repeats == MAX_REPEATS_LIMIT
    finally:
        get_config().MAX_REPEATS_LIMIT = OLD_LIMIT


def test_get_config_session_pipe():
    hw = EchoModelLoader(qubit_count=4).load()
    b = InstructionBuilder(hw)

    res_mgr = ResultManager()
    met_mgr = MetricsManager()

    Pm = PassManager() | DummyValidation() | DummyTransform() | DummyAnalysis()
    MAX_REPEATS_LIMIT = 253423

    assert get_config().MAX_REPEATS_LIMIT != MAX_REPEATS_LIMIT
    config = QatSessionConfig(MAX_REPEATS_LIMIT=253423)

    with override_config(config):
        Pm.run(b, res_mgr=res_mgr, met_mgr=met_mgr)
        dummy_result = res_mgr.lookup_by_type(DummyResult)
        assert dummy_result.max_repeats == MAX_REPEATS_LIMIT


class TestQatSession:
    MAX_REPEATS_LIMIT_1 = 25342_1
    MAX_REPEATS_LIMIT_2 = 25342_2

    @pytest.fixture
    def sessions(self, REPEAT_LIMIT_1=25342_1, REPEAT_LIMIT_2=25342_2):
        q1, q2 = QAT(), QAT()
        GLOBAL_LIMIT = get_config().MAX_REPEATS_LIMIT
        assert GLOBAL_LIMIT not in {
            REPEAT_LIMIT_1,
            REPEAT_LIMIT_2,
        }
        q1.config.MAX_REPEATS_LIMIT = REPEAT_LIMIT_1
        q2.config.MAX_REPEATS_LIMIT = REPEAT_LIMIT_2

        assert q1.config.MAX_REPEATS_LIMIT == REPEAT_LIMIT_1
        assert q2.config.MAX_REPEATS_LIMIT == REPEAT_LIMIT_2
        assert get_config().MAX_REPEATS_LIMIT == GLOBAL_LIMIT

        yield q1, q2

    @pytest.fixture
    def model(self):
        yield EchoModelLoader(qubit_count=4).load()

    @pytest.fixture
    def pipeline(self, model):
        Pm = (
            PassManager()
            | DummyValidation()
            | DummyTransform()
            | DummyAnalysis()
            | DummyTransformReturnRepeats()
        )

        frontend = FallthroughFrontend()
        middleend = CustomMiddleend(model=model, pipeline=Pm)
        backend = FallthroughBackend()
        runtime = SimpleRuntime(engine=EchoEngine())

        P = Pipeline(
            name="dummy",
            frontend=frontend,
            middleend=middleend,
            backend=backend,
            runtime=runtime,
            model=model,
        )

        yield P

    @pytest.fixture
    def builder(self, model):
        yield InstructionBuilder(model)

    def test_get_config(self, sessions, pipeline, builder):
        q1, q2 = sessions

        q1.pipelines.add(pipeline, default=True)
        q2.pipelines.add(pipeline, default=True)

        pkg1, _ = q1.compile(builder)
        pkg2, _ = q2.compile(builder)
        assert pkg1[0] == q1.config.MAX_REPEATS_LIMIT
        assert pkg2[0] == q2.config.MAX_REPEATS_LIMIT
        assert (
            len(
                {
                    get_config().MAX_REPEATS_LIMIT,
                    q1.config.MAX_REPEATS_LIMIT,
                    q2.config.MAX_REPEATS_LIMIT,
                }
            )
            == 3
        )

    def test_set_global_after_init(self, sessions, pipeline, builder):
        q = QAT()
        q.pipelines.add(pipeline, default=True)

        MAX_REPEATS_LIMIT = 253423
        OLD_LIMIT = get_config().MAX_REPEATS_LIMIT
        assert OLD_LIMIT != MAX_REPEATS_LIMIT
        assert q.config.MAX_REPEATS_LIMIT == OLD_LIMIT
        try:
            get_config().MAX_REPEATS_LIMIT = MAX_REPEATS_LIMIT
            pkg1, _ = q.compile(builder)
            assert pkg1[0] == q.config.MAX_REPEATS_LIMIT
        finally:
            get_config().MAX_REPEATS_LIMIT = OLD_LIMIT

    def test_set_global_before_init(self, sessions, pipeline, builder):
        MAX_REPEATS_LIMIT = 253423
        OLD_LIMIT = get_config().MAX_REPEATS_LIMIT
        assert OLD_LIMIT != MAX_REPEATS_LIMIT
        try:
            get_config().MAX_REPEATS_LIMIT = MAX_REPEATS_LIMIT
            # creates a new config item with the current set globals
            q = QAT()
            q.pipelines.add(pipeline, default=True)
            assert q.config.MAX_REPEATS_LIMIT == MAX_REPEATS_LIMIT

            pkg1, _ = q.compile(builder)
            assert pkg1[0] == MAX_REPEATS_LIMIT
        finally:
            get_config().MAX_REPEATS_LIMIT = OLD_LIMIT
