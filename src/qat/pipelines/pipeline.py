# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig

from qat.backend.base import BaseBackend
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.engines import NativeEngine
from qat.executables import Executable
from qat.frontend import BaseFrontend
from qat.middleend.middleends import BaseMiddleend
from qat.model.target_data import TargetData
from qat.model.validators import MismatchingHardwareModelException
from qat.pipelines.base import AbstractPipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.base import BaseRuntime


class Pipeline(AbstractPipeline):
    """An immutable pipeline that can be constructed to compile and execute quantum
    programs.

    It is designed to be immutable to allow compilation passes to calculate important
    quantities at instantiation using the given components, such as calibration information
    and target data. Thus it assumes information about the device is non-changing and is
    not suitable for calibrations.
    """

    def __init__(
        self,
        name: str,
        model: QuantumHardwareModel,
        frontend: BaseFrontend,
        middleend: BaseMiddleend,
        backend: BaseBackend,
        runtime: BaseRuntime,
        target_data: TargetData = TargetData.default(),
        disable_model_validation: bool = False,
    ):
        self._name = name
        self._model = model
        self._target_data = target_data
        self._frontend = frontend
        self._middleend = middleend
        self._backend = backend
        self._runtime = runtime

        if not disable_model_validation:
            self._validate_consistent_model(
                model,
                frontend,
                middleend,
                backend,
                runtime,
                runtime.engine,
            )

    @property
    def name(self) -> str:
        """Returns the name of the pipeline."""
        return self._name

    @property
    def model(self) -> QuantumHardwareModel:
        """Returns the quantum hardware model used by the pipeline."""
        return self._model

    @property
    def target_data(self) -> TargetData:
        """Returns the target data used by the pipeline."""
        return self._target_data

    @property
    def frontend(self) -> BaseFrontend:
        """Returns the compilation frontend."""
        return self._frontend

    @property
    def middleend(self) -> BaseMiddleend:
        """Returns the middleend of the pipeline."""
        return self._middleend

    @property
    def backend(self) -> BaseBackend:
        """Returns the backend of the pipeline."""
        return self._backend

    @property
    def runtime(self) -> BaseRuntime:
        """Returns the runtime of the pipeline."""
        return self._runtime

    @property
    def engine(self) -> NativeEngine:
        """Returns the engine of the pipeline."""
        return self._runtime.engine

    @staticmethod
    def _validate_consistent_model(
        model: QuantumHardwareModel,
        *args,
    ):
        """Validates that the hardware model supplied to the Pipeline matches the hardware
        model embedded in other fields."""
        for component in args:
            if hasattr(component, "model") and component.model not in {model, None}:
                raise ValueError(f"{model} hardware does not match supplied hardware")

    def copy(self) -> "Pipeline":
        """Returns a new instance of the pipeline with the same components."""

        return Pipeline(
            model=self._model,
            target_data=self._target_data,
            frontend=self._frontend,
            middleend=self._middleend,
            backend=self._backend,
            runtime=self._runtime,
            name=self._name,
        )

    def copy_with_name(self, name: str) -> "Pipeline":
        """Returns a new instance of the pipeline with a different name."""

        return Pipeline(
            model=self._model,
            target_data=self._target_data,
            frontend=self._frontend,
            middleend=self._middleend,
            backend=self._backend,
            runtime=self._runtime,
            name=name,
        )

    def compile(
        self, program, compiler_config: CompilerConfig | None = None
    ) -> tuple[Executable, MetricsManager]:
        """Compiles a source program into an executable using the pipeline's components.

        :param program: The source program to compile.
        :param compiler_config: Configuration options for the compiler, such as optimization
            and results formatting.
        :return: An executable of the compiled program for the target device and the metrics
            manager containing metrics collected during compilation.
        """

        # TODO: Improve metrics and config handling
        compiler_config = compiler_config or CompilerConfig()
        metrics_manager = MetricsManager(compiler_config.metrics)
        compilation_results = ResultManager()

        ir = self.frontend.emit(
            program,
            compilation_results,
            metrics_manager,
            compiler_config=compiler_config,
        )

        ir = self.middleend.emit(
            ir,
            compilation_results,
            metrics_manager,
            compiler_config=compiler_config,
        )

        package = self.backend.emit(
            ir,
            compilation_results,
            metrics_manager,
            compiler_config=compiler_config,
        )

        return package, metrics_manager

    def execute(
        self, package: Executable, compiler_config: CompilerConfig | None = None
    ) -> tuple[dict, MetricsManager]:
        """Uses the runtime and engine in the pipeline to execute a compiled program and
        process the results.

        Checks that the hardware model in the pipeline matches the hardware model used
        during compilation.

        :param package: The compiled program to execute.
        :param compiler_config: Configuration options for the compiler, such as optimization
            and results formatting.
        :return: A dictionary of results from the execution and the metrics manager
            containing metrics collected during execution.
        """

        compiler_config = CompilerConfig() if compiler_config is None else compiler_config

        if self.model.calibration_id != package.calibration_id:
            raise MismatchingHardwareModelException(
                f"Hardware id in the executable package '{self.model.calibration_id}'' "
                f"does not match the hardware id '{package.calibration_id}' used "
                "during compilation."
            )

        pp_results = ResultManager()
        metrics_manager = MetricsManager(compiler_config.metrics)
        results = self.runtime.execute(
            package,
            res_mgr=pp_results,
            met_mgr=metrics_manager,
            compiler_config=compiler_config,
        )
        return results, metrics_manager
