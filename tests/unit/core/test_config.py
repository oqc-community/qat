import pytest
from pydantic import ValidationError

import qat
import qat.pipelines
from qat.core.config import (
    HardwareLoaderDescription,
    PipelineBuilderDescription,
    PipelineInstanceDescription,
)
from qat.model.loaders.base import BaseModelLoader
from qat.purr.qatconfig import QatConfig


class TestPipelineInstanceDescription:
    def test_valid_description(self):
        desc = PipelineInstanceDescription(
            name="echo8a", pipeline="qat.pipelines.echo.echo8", default=True
        )

        assert desc.name == "echo8a"

    def test_invalid_pipeline_class_raises(self):
        with pytest.raises(ValidationError):
            PipelineInstanceDescription(
                name="invalid",
                pipeline="qat.backend.fallthrough.FallthroughBackend",
                default=True,
            )


class TestPipelineBuilderDescription:
    def test_valid_description(self):
        desc = PipelineBuilderDescription(
            name="echo8a",
            pipeline="qat.pipelines.echo.get_pipeline",
            default=True,
            hardware_loader="echo8a",
        )

        assert desc.name == "echo8a"

    def test_invalid_pipeline_class_raises(self):
        with pytest.raises(ValidationError):
            PipelineBuilderDescription(
                name="invalid",
                pipeline="qat.backend.fallthrough.FallthroughBackend",
                default=True,
            )


class TestHardwareLoaderDescription:
    def test_valid_description(self):
        desc = HardwareLoaderDescription(
            name="somehardware", loader="qat.model.loaders.legacy.EchoModelLoader"
        )
        assert isinstance(desc.construct(), BaseModelLoader)

    def test_custom_init(self):
        desc = HardwareLoaderDescription(
            name="somehardware",
            loader="qat.model.loaders.legacy.EchoModelLoader",
            init={"qubit_count": 8},
        )

        loader = desc.construct()
        assert isinstance(loader, BaseModelLoader)
        assert loader.qubit_count == 8


class TestQATConfig:
    """These are tests of QAT config features only used by QAT.pipelines"""

    def test_make_qatconfig_list(self):
        pipelines = [
            PipelineInstanceDescription(
                name="echo8a", pipeline="qat.pipelines.echo.echo8", default=True
            ),
            PipelineInstanceDescription(
                name="echo16a", pipeline="qat.pipelines.echo.echo16"
            ),
            PipelineInstanceDescription(
                name="echo32a", pipeline="qat.pipelines.echo.echo32"
            ),
            PipelineBuilderDescription(
                name="echo6",
                pipeline="qat.pipelines.echo.get_pipeline",
                hardware_loader="echo6loader",
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="echo6loader",
                loader="qat.model.loaders.legacy.EchoModelLoader",
                init={"qubit_count": 6},
            )
        ]

        qc = QatConfig(PIPELINES=pipelines, HARDWARE=hardware)

        pipes = {P.name: P.pipeline for P in qc.PIPELINES}
        assert pipes == {
            "echo8a": qat.pipelines.echo.echo8,
            "echo16a": qat.pipelines.echo.echo16,
            "echo32a": qat.pipelines.echo.echo32,
            "echo6": qat.pipelines.echo.get_pipeline,
        }

        assert len(qc.HARDWARE) == 1
        assert qc.HARDWARE[0].name == qc.PIPELINES[-1].hardware_loader

    def test_mismatching_hardware_loader_raises(self):
        pipelines = [
            PipelineBuilderDescription(
                name="echo6",
                pipeline="qat.pipelines.echo.get_pipeline",
                hardware_loader="echo6loader",
                default=True,
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="notwhatwearelookingfor",
                loader="qat.model.loaders.legacy.EchoModelLoader",
                init={"qubit_count": 6},
            )
        ]

        with pytest.raises(ValidationError):
            QatConfig(PIPELINES=pipelines, HARDWARE=hardware)
