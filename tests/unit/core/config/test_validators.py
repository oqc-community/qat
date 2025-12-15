# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.backend.base import BaseBackend
from qat.core.config.validators import (
    is_backend,
    is_engine,
    is_frontend,
    is_hardwareloader,
    is_instrument_builder,
    is_middleend,
    is_passmanager,
    is_passmanager_factory,
    is_pipeline_factory,
    is_pipeline_instance,
    is_runtime,
    is_target_data,
    is_updateable_pipeline,
)
from qat.core.pass_base import PassManager
from qat.frontend.base import BaseFrontend
from qat.instrument.base import InstrumentBuilder
from qat.middleend.base import BaseMiddleend
from qat.model.loaders.base import BaseModelLoader
from qat.model.target_data import AbstractTargetData
from qat.purr.backends.echo import EchoEngine
from qat.runtime import BaseRuntime

from tests.unit.utils.engines import MockConnectedEngine
from tests.unit.utils.loaders import MockModelLoader
from tests.unit.utils.pipelines import (
    MockCompileUpdateablePipeline,
    MockExecuteUpdateablePipeline,
    MockUpdateablePipeline,
    get_mock_pipeline,
)

[BaseFrontend, BaseMiddleend, BaseBackend, BaseRuntime, InstrumentBuilder]


@pytest.mark.parametrize(
    "object, validator",
    [
        (BaseFrontend, is_frontend),
        (BaseMiddleend, is_middleend),
        (BaseBackend, is_backend),
        (BaseRuntime, is_runtime),
        (InstrumentBuilder, is_instrument_builder),
        (PassManager, is_passmanager),
        (BaseModelLoader, is_hardwareloader),
    ],
)
class TestBaseUnits:
    @pytest.mark.parametrize("value", [5, int, "hey", None, lambda: None])
    def test_with_other_value(self, object, validator, value):
        with pytest.raises(ValueError, match="is not a valid"):
            validator(value)

    def test_with_base_class(self, object, validator):
        class MockObject(object):
            pass

        assert validator(MockObject) is MockObject


@pytest.mark.filterwarnings("ignor:WaveformV1 support:DeprecationWarning")
class TestIsPipelineInstance:
    @pytest.mark.parametrize("value", [5, int, "hey", None, lambda: None])
    def test_with_non_pipeline_instance(self, value):
        with pytest.raises(ValueError, match="is not a valid Pipeline instance"):
            is_pipeline_instance(value)

    @pytest.mark.parametrize(
        "pipeline_loader",
        [
            MockCompileUpdateablePipeline,
            MockExecuteUpdateablePipeline,
            MockExecuteUpdateablePipeline,
        ],
    )
    def test_with_pipeline_instance(self, pipeline_loader):
        model = MockModelLoader().load()
        pipeline = pipeline_loader(config=dict(name="test_pipeline"), model=model)
        assert is_pipeline_instance(pipeline.pipeline) is pipeline.pipeline
        with pytest.raises(ValueError, match="is not a valid Pipeline instance"):
            is_pipeline_instance(pipeline)


class TestIsEngine:
    @pytest.mark.parametrize("value", [5, int, "hey", None, lambda: None])
    def test_with_non_engine(self, value):
        with pytest.raises(ValueError, match="is not a valid Engine"):
            is_engine(value)

    def test_with_legacy_engine(self):
        assert is_engine(EchoEngine) is EchoEngine

    def test_with_engine(self):
        assert is_engine(MockConnectedEngine) is MockConnectedEngine


class TestIsTargetData:
    @pytest.mark.parametrize("value", [5, "hey", None])
    def test_with_non_targetdata(self, value):
        with pytest.raises(ValueError, match="is not a valid TargetData"):
            is_target_data(value)

    def test_with_targetdata(self):
        class MyTargetData(AbstractTargetData):
            pass

        assert is_target_data(MyTargetData) is MyTargetData

    def test_with_targetdata_factory(self):
        class MyTargetData(AbstractTargetData):
            pass

        def my_factory() -> MyTargetData:
            return MyTargetData()

        assert is_target_data(my_factory) is my_factory

    def test_callable_with_invalid_return(self):
        def my_factory() -> int:
            return 42

        with pytest.raises(ValueError, match="is not a valid TargetData"):
            is_target_data(my_factory)

    def test_callable_with_no_return_annotation(self):
        def my_factory():
            return AbstractTargetData()

        with pytest.raises(ValueError, match="is not a valid TargetData"):
            is_target_data(my_factory)

    def test_callable_with_none_return(self):
        def my_factory() -> None:
            return None

        with pytest.raises(ValueError, match="is not a valid TargetData"):
            is_target_data(my_factory)


class TestIsPassManagerFactory:
    def test_callable_with_none_return(self):
        def my_factory() -> None:
            return None

        with pytest.raises(ValueError, match="does not have a valid return annotation"):
            is_passmanager_factory(my_factory)

    def test_non_callable(self):
        not_a_function = 42

        with pytest.raises(ValueError, match="is not callable"):
            is_passmanager_factory(not_a_function)

    def test_callable_with_invalid_return(self):
        def my_factory() -> int:
            return 42

        with pytest.raises(ValueError, match="does not return a PassManager"):
            is_passmanager_factory(my_factory)

    def test_valid_passmanager_factory(self):
        def my_factory() -> PassManager:
            return PassManager()

        assert is_passmanager_factory(my_factory) is my_factory


class TestIsPipelineFactory:
    def test_callable_with_none_return(self):
        def my_factory() -> None:
            return None

        with pytest.raises(ValueError, match="does not have a valid return annotation"):
            is_pipeline_factory(my_factory)

    def test_non_callable(self):
        not_a_function = 42

        with pytest.raises(ValueError, match="is not callable"):
            is_pipeline_factory(not_a_function)

    def test_callable_with_invalid_return(self):
        def my_factory() -> int:
            return 42

        with pytest.raises(ValueError, match="does not return a BasePipeline"):
            is_pipeline_factory(my_factory)

    def test_valid_pipeline_factory(self):
        assert is_pipeline_factory(get_mock_pipeline) is get_mock_pipeline


@pytest.mark.filterwarnings("ignore:WaveformV1 support:DeprecationWarning")
class TestIsUpdateablePipeline:
    @pytest.mark.parametrize("value", [5, int, "hey", get_mock_pipeline, None])
    def test_with_other_value(self, value):
        with pytest.raises(ValueError, match="is not a valid UpdateablePipeline"):
            is_updateable_pipeline(value)

    def test_with_updateable_pipelines(self):
        assert is_updateable_pipeline(MockUpdateablePipeline) is MockUpdateablePipeline
