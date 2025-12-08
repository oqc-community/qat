# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import pytest
from pydantic import ValidationError

import qat
import qat.pipelines
from qat.core.config.descriptions import (
    EngineDescription,
    HardwareLoaderDescription,
    InstrumentBuilderDescription,
    PipelineClassDescription,
    PipelineInstanceDescription,
    UpdateablePipelineDescription,
)
from qat.core.config.session import QatSessionConfig
from qat.core.pipelines.configurable import ConfigurablePipeline
from qat.engines.qblox.execution import QbloxEngine
from qat.engines.qblox.live import QbloxCompositeInstrument, QbloxLeafInstrument
from qat.instrument.base import ConfigInstrumentBuilder
from qat.model.loaders.cache import CacheAccessLoader
from qat.model.loaders.purr import EchoModelLoader, QbloxDummyModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.updateable import UpdateablePipeline
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel

from tests.unit.utils.engines import InitableEngine
from tests.unit.utils.pipelines import MockPipeline


@pytest.fixture
def qatconfig_testfiles(testpath):
    return testpath / "files" / "qatconfig"


class TestQatSessionConfigForPipelines:
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
            UpdateablePipelineDescription(
                name="echo6",
                pipeline="qat.pipelines.echo.EchoPipeline",
                hardware_loader="echo6loader",
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="echo6loader",
                type="qat.model.loaders.purr.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]

        qc = QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware)

        pipes = {P.name: P.pipeline for P in qc.PIPELINES}
        assert pipes == {
            "echo8a": qat.pipelines.echo.echo8,
            "echo16a": qat.pipelines.echo.echo16,
            "echo32a": qat.pipelines.echo.echo32,
            "echo6": qat.pipelines.echo.EchoPipeline,
        }

        assert len(qc.HARDWARE) == 1
        assert qc.HARDWARE[0].name == qc.PIPELINES[-1].hardware_loader

    def test_make_qatconfig_class_desc(self):
        pipelines = [
            PipelineClassDescription(
                name="echocustom",
                hardware_loader="echo6loader",
                engine="echo",
                frontend="qat.frontend.FallthroughFrontend",
                middleend="qat.middleend.FallthroughMiddleend",
                backend="qat.backend.WaveformV1Backend",
                runtime="qat.runtime.SimpleRuntime",
                default=True,
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="echo6loader",
                type="qat.model.loaders.purr.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]
        engines = [
            EngineDescription(name="echo", type="qat.engines.waveform_v1.EchoEngine"),
            EngineDescription(
                name="mock",
                type="tests.unit.utils.engines.MockEngineWithModel",
                hardware_loader="echo6loader",
            ),
        ]

        assert pipelines[0].hardware_loader == "echo6loader"
        assert pipelines[0].engine == "echo"
        qc = QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware, ENGINES=engines)

        pipe_names = {P.name for P in qc.PIPELINES}
        assert pipe_names == {"echocustom"}
        assert len(qc.HARDWARE) == 1
        assert qc.HARDWARE[0].name == qc.PIPELINES[-1].hardware_loader == "echo6loader"
        assert len(qc.ENGINES) == 2
        assert qc.ENGINES[0].name == qc.PIPELINES[-1].engine == "echo"
        assert qc.ENGINES[1].name == "mock"
        assert (
            qc.ENGINES[1].hardware_loader
            == qc.PIPELINES[-1].hardware_loader
            == "echo6loader"
        )

    def test_mismatching_hardware_loader_raises(self):
        pipelines = [
            UpdateablePipelineDescription(
                name="echo6",
                pipeline="qat.pipelines.echo.EchoPipeline",
                hardware_loader="echo6loader",
                default=True,
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="notwhatwearelookingfor",
                type="qat.model.loaders.purr.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]

        with pytest.raises(ValidationError):
            QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware)

    def test_invalid_hardware_on_engine_raises(self):
        pipelines = []

        hardware = [
            HardwareLoaderDescription(
                name="echo6loader",
                type="qat.model.loaders.purr.EchoModelLoader",
                config={"qubit_count": 6},
            )
        ]

        engines = [
            EngineDescription(
                name="mock",
                type="tests.unit.utils.engines.MockEngineWithModel",
                hardware_loader="notwhatwearelookingfor",
            )
        ]

        with pytest.raises(ValidationError):
            QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware, ENGINES=engines)

    def test_mismatching_hardware_on_engine_and_pipeline_raises(self):
        pipelines = [
            UpdateablePipelineDescription(
                name="echo6",
                pipeline="qat.pipelines.echo.EchoPipeline",
                hardware_loader="echo6loader",
                default=True,
                engine="mock",
            ),
        ]

        hardware = [
            HardwareLoaderDescription(
                name="echo6loader",
                type="qat.model.loaders.purr.EchoModelLoader",
                config={"qubit_count": 6},
            ),
            HardwareLoaderDescription(
                name="echo8loader",
                type="qat.model.loaders.purr.EchoModelLoader",
                config={"qubit_count": 8},
            ),
        ]

        engines = [
            EngineDescription(
                name="mock",
                type="tests.unit.utils.engines.MockEngineWithModel",
                hardware_loader="echo8loader",
            )
        ]

        with pytest.raises(ValidationError):
            QatSessionConfig(PIPELINES=pipelines, HARDWARE=hardware, ENGINES=engines)

    def test_yaml_custom_config(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "customconfig.yaml")
        assert len(qatconfig.PIPELINES) == 1
        assert len(qatconfig.HARDWARE) == 1
        assert len(qatconfig.ENGINES) == 1

        loader = qatconfig.HARDWARE[0].construct()
        engine = qatconfig.ENGINES[0].construct()
        with pytest.warns(DeprecationWarning):
            P = qatconfig.PIPELINES[0].construct(loader=loader, engine=engine)

        assert isinstance(P, ConfigurablePipeline)
        assert P.runtime.engine.x == 10
        assert P.runtime.engine.cblam.host == "someurl.com"
        assert P.runtime.engine.cblam.timeout == 60

    def test_yaml_legacy_pipeline(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "legacyengine.yaml")

        assert len(qatconfig.PIPELINES) == 1
        assert len(qatconfig.HARDWARE) == 1
        assert len(qatconfig.ENGINES) == 1

        loader = qatconfig.HARDWARE[0].construct()
        model_cache = {"test": loader.load()}
        engine = qatconfig.ENGINES[0].construct(model=model_cache["test"])
        cache_loader = CacheAccessLoader(model_cache, "test")
        with pytest.warns(DeprecationWarning):
            P = qatconfig.PIPELINES[0].construct(loader=cache_loader, engine=engine)

        assert isinstance(engine, QuantumExecutionEngine)
        assert isinstance(engine.model, QuantumHardwareModel)

        assert isinstance(P, ConfigurablePipeline)
        assert isinstance(P.model, QuantumHardwareModel)
        assert engine == P.engine
        assert engine.model == P.model == model_cache["test"]

    def test_yaml_factory_with_echo_engine(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(
            qatconfig_testfiles / "pipelinefactorywithengine.yaml"
        )

        assert len(qatconfig.PIPELINES) == 1
        assert len(qatconfig.HARDWARE) == 1
        assert len(qatconfig.ENGINES) == 1

        P = qatconfig.PIPELINES[0]
        assert isinstance(P, UpdateablePipelineDescription)
        loader = qatconfig.HARDWARE[0]
        assert isinstance(loader, HardwareLoaderDescription)
        engine = qatconfig.ENGINES[0]
        assert isinstance(engine, EngineDescription)

        loader = loader.construct()
        engine = engine.construct()
        with pytest.warns(DeprecationWarning):
            P = P.construct(loader=loader, engine=engine)

        assert isinstance(loader, EchoModelLoader)
        assert isinstance(engine, InitableEngine)
        assert engine.x == 10
        assert engine.cblam.host == "someurl.com"
        assert engine.cblam.timeout == 60

        assert isinstance(P, MockPipeline)
        assert P.engine == engine

    def test_yaml_factory_with_qblox_engine(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "qblox_engines.yaml")

        assert not qatconfig.PIPELINES
        assert len(qatconfig.HARDWARE) == 1
        assert len(qatconfig.ENGINES) == 1

        hardware_loader_desc = qatconfig.HARDWARE[0]
        assert isinstance(hardware_loader_desc, HardwareLoaderDescription)
        hardware_loader = hardware_loader_desc.construct()
        assert isinstance(hardware_loader, QbloxDummyModelLoader)
        model = hardware_loader.load()
        assert isinstance(model, QbloxLiveHardwareModel)

        engine_desc = qatconfig.ENGINES[0]
        assert isinstance(engine_desc, EngineDescription)

        instrument_builder_desc = engine_desc.instrument_builder
        assert isinstance(instrument_builder_desc, InstrumentBuilderDescription)
        assert len(instrument_builder_desc.configs) == 3
        instrument_builder = instrument_builder_desc.construct()
        assert isinstance(instrument_builder, ConfigInstrumentBuilder)

        instrument = instrument_builder.build()
        assert isinstance(instrument, QbloxCompositeInstrument)
        for comp in instrument.components.values():
            assert isinstance(comp, QbloxLeafInstrument)

        engine = engine_desc.construct()
        assert isinstance(engine, QbloxEngine)

    def test_yaml_qblox_pipeline(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "qblox_pipelines.yaml")

        assert len(qatconfig.PIPELINES) == 1
        assert len(qatconfig.HARDWARE) == 1
        assert len(qatconfig.ENGINES) == 1

        hardware_loader_desc = qatconfig.HARDWARE[0]
        assert isinstance(hardware_loader_desc, HardwareLoaderDescription)
        hardware_loader = hardware_loader_desc.construct()
        assert isinstance(hardware_loader, QbloxDummyModelLoader)
        model = hardware_loader.load()
        assert isinstance(model, QbloxLiveHardwareModel)

        engine_desc = qatconfig.ENGINES[0]
        assert isinstance(engine_desc, EngineDescription)
        engine = engine_desc.construct()
        assert isinstance(engine, QbloxEngine)

        pipeline_desc = qatconfig.PIPELINES[0]
        assert isinstance(pipeline_desc, UpdateablePipelineDescription)

        pipeline = pipeline_desc.construct(loader=hardware_loader, engine=engine)
        assert isinstance(pipeline, UpdateablePipeline)
        assert isinstance(pipeline.model, QuantumHardwareModel)
        assert pipeline.engine is engine

    def test_yaml_custom_result_pipeline(self, qatconfig_testfiles):
        qatconfig = QatSessionConfig.from_yaml(
            qatconfig_testfiles / "customresultspipeline.yaml"
        )

        from qat.model.loaders.purr import EchoModelLoader

        loader = EchoModelLoader()

        desc = qatconfig.PIPELINES[0]
        with pytest.warns(DeprecationWarning):
            P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        results_pipeline = P.runtime.results_pipeline
        assert len(results_pipeline.passes) == 1
        dummypass = results_pipeline.passes[0]._pass
        assert isinstance(dummypass.model, QuantumHardwareModel)
        assert dummypass.some_int == 12

    def test_yaml_custom_config_from_env(self, monkeypatch, qatconfig_testfiles):
        from qat.model.loaders.purr import EchoModelLoader

        monkeypatch.setenv("SOME_ENV_VAR", "A_VALUE")
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "envvar.yaml")

        loader = EchoModelLoader()

        desc = qatconfig.PIPELINES[0]
        engine = qatconfig.ENGINES[0].construct()
        with pytest.warns(DeprecationWarning):
            P = desc.construct(loader=loader, engine=engine)
        assert isinstance(P, ConfigurablePipeline)
        assert P.runtime.engine.x == "A_VALUE"

    def test_yaml_defaults(self, qatconfig_testfiles):
        from qat.backend import DefaultBackend
        from qat.engines import DefaultEngine
        from qat.frontend import DefaultFrontend
        from qat.middleend import DefaultMiddleend
        from qat.model.loaders.purr import EchoModelLoader
        from qat.runtime import DefaultRuntime

        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "defaults.yaml")

        loader = EchoModelLoader()

        desc = qatconfig.PIPELINES[0]
        with pytest.warns(DeprecationWarning):
            P = desc.construct(loader=loader)
        assert isinstance(P, ConfigurablePipeline)
        assert type(P.frontend) is DefaultFrontend
        assert type(P.middleend) is DefaultMiddleend
        assert type(P.backend) is DefaultBackend
        assert type(P.runtime) is DefaultRuntime
        assert type(P.runtime.engine) is DefaultEngine

    def test_yaml_custom_config_missing_from_env(self, monkeypatch, qatconfig_testfiles):
        monkeypatch.delenv("SOME_ENV_VAR", raising=False)
        with pytest.raises(ValueError):
            QatSessionConfig.from_yaml(qatconfig_testfiles / "envvar.yaml")

    def test_yaml_custom_config_invalid_type(self, qatconfig_testfiles):
        """Check that invalid config (types) raise exceptions.

        Error is raised with config instantiation as import type is not found."""

        with pytest.raises(ValidationError, match="No module"):
            QatSessionConfig.from_yaml(qatconfig_testfiles / "invalid" / "type.yaml")

    def test_yaml_custom_config_invalid_arg(self, qatconfig_testfiles):
        """Check that invalid config (types) raise exceptions

        Config is only validated on engine construction"""

        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "invalid" / "arg.yaml")
        desc = qatconfig.ENGINES[0]

        with pytest.raises(ValueError):
            desc.construct()

    def test_multiple_defaults_on_full_and_separate_pipelines_raises(
        self, qatconfig_testfiles
    ):
        with pytest.raises(ValidationError, match="Expected exactly one default"):
            QatSessionConfig.from_yaml(
                qatconfig_testfiles / "invalid" / "duplicate_defaults.yaml"
            )

    def test_duplicate_names_raises(self, qatconfig_testfiles):
        """Check that duplicate names in hardware, engines or pipelines raise exceptions."""
        with pytest.raises(ValidationError, match="Duplicate name"):
            QatSessionConfig.from_yaml(
                qatconfig_testfiles / "invalid" / "duplicate_names.yaml"
            )

    def test_default_on_full(self, qatconfig_testfiles):
        """Checks that only a single full pipeline is marked as default."""
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "default_full.yaml")
        assert len(qatconfig.PIPELINES) == 1
        assert qatconfig.COMPILE is None
        assert qatconfig.EXECUTE is None
        assert qatconfig.PIPELINES[0].default is True

    def test_default_on_seperate(self, qatconfig_testfiles):
        """Checks that only a single compile or execute pipeline is marked as default."""
        qatconfig = QatSessionConfig.from_yaml(
            qatconfig_testfiles / "default_separate.yaml"
        )
        assert len(qatconfig.COMPILE) == 1
        assert len(qatconfig.EXECUTE) == 1
        assert qatconfig.COMPILE[0].default is True
        assert qatconfig.EXECUTE[0].default is True

    def test_no_default_on_all(self, qatconfig_testfiles):
        """Checks that no default is set when no full or separate pipelines are given."""
        with pytest.raises(ValidationError, match="Expected exactly one default"):
            QatSessionConfig.from_yaml(
                qatconfig_testfiles / "invalid" / "no_default_on_all.yaml"
            )

    def test_multiple_defaults(self, qatconfig_testfiles):
        """Checks that multiple defaults on compile and execute pipelines raise exceptions."""
        with pytest.raises(ValidationError, match="Expected exactly one default"):
            QatSessionConfig.from_yaml(
                qatconfig_testfiles / "invalid" / "multiple_defaults.yaml"
            )

    def test_passes_with_exactly_one_default(self, qatconfig_testfiles):
        """Checks that exactly one default compile and execute pipeline is set."""
        qatconfig = QatSessionConfig.from_yaml(
            qatconfig_testfiles / "choose_correct_default.yaml"
        )

        assert len(qatconfig.COMPILE) == 3
        assert len(qatconfig.EXECUTE) == 3

        assert qatconfig.COMPILE[1].default is True
        assert qatconfig.EXECUTE[1].default is True

    @pytest.mark.parametrize(
        "qatconfig_file",
        [
            "wrong_hardware_model_on_compile.yaml",
            "wrong_hardware_model_on_engine.yaml",
            "wrong_hardware_model_on_execute.yaml",
            "wrong_hardware_model_on_pipelines.yaml",
        ],
    )
    def test_engine_with_incorrect_hardware(self, qatconfig_testfiles, qatconfig_file):
        """Checks that an engine with incorrect hardware raises an error."""
        with pytest.raises(ValidationError, match="Hardware Loader"):
            QatSessionConfig.from_yaml(qatconfig_testfiles / "invalid" / qatconfig_file)

    @pytest.mark.parametrize(
        "qatconfig_file",
        [
            "wrong_engine_in_pipelines.yaml",
            "wrong_engine_in_execute.yaml",
        ],
    )
    def test_wrong_engine_in_execute(self, qatconfig_testfiles, qatconfig_file):
        """Checks that an engine not found in the config raises an error."""
        with pytest.raises(ValidationError, match="requires engine"):
            QatSessionConfig.from_yaml(qatconfig_testfiles / "invalid" / qatconfig_file)

    @pytest.mark.parametrize(
        "qatconfig_file",
        [
            "duplicate_engine_names.yaml",
            "duplicate_hardware_names.yaml",
        ],
    )
    def test_duplicate_names_in_engines_or_hardware(
        self, qatconfig_testfiles, qatconfig_file
    ):
        """Checks that duplicate names in engines or hardware raise an error."""
        with pytest.raises(ValidationError, match="Duplicate name"):
            QatSessionConfig.from_yaml(qatconfig_testfiles / "invalid" / qatconfig_file)

    def test_target_data(self, qatconfig_testfiles):
        """Checks that target data is correctly loaded from the config."""
        qatconfig = QatSessionConfig.from_yaml(qatconfig_testfiles / "target_data.yaml")

        assert len(qatconfig.COMPILE) == 2
        assert qatconfig.COMPILE[0].name == "legacy-compile"
        assert qatconfig.COMPILE[1].name == "compile"

        for partial in [qatconfig.COMPILE[i].target_data for i in range(2)]:
            target_data = partial()
            assert isinstance(target_data, TargetData)
            assert target_data.default_shots == 254
            assert target_data.max_shots == 2540
            assert target_data.QUBIT_DATA.passive_reset_time == 1e-2
