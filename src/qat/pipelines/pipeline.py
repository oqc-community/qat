# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig, MetricsType

from qat.backend.base import BaseBackend
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.engines import NativeEngine
from qat.executables import Executable
from qat.frontend import BaseFrontend
from qat.middleend.base import BaseMiddleend
from qat.model.target_data import TargetData
from qat.model.validators import MismatchingHardwareModelException
from qat.pipelines.base import BasePipeline
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.base import BaseRuntime


class CompilePipeline(BasePipeline):
    """Implements a pipeline that compiles quantum programs.

    In addition to the hardware model and target data, a compilation pipeline consists of
    the following components:

    #. Frontend: Compiles a high-level language-specific, but target-agnostic,
       input (e.g., QASM, QIR, ...) to QAT's intermediate representation (IR), QatIR.
    #. Middleend: Takes the QatIR and performs a sequences of  passes that validate and
       optimise the IR, and prepare it for codegen.
    #. Backend: Handles code generation to allow the program to be executed on the target.

    In the future, this might be relaxed so compilation pipelines can be defined more
    abstractly, only requiring the :meth:`compile` to be implemented.
    """

    def __init__(
        self,
        name: str,
        model: QuantumHardwareModel,
        frontend: BaseFrontend,
        middleend: BaseMiddleend,
        backend: BaseBackend,
        target_data: TargetData | None = None,
        disable_model_validation: bool = False,
    ):
        if not disable_model_validation:
            self._validate_consistent_model(model, frontend, middleend, backend)
        self._name = name
        self._model = model
        self._target_data = target_data if target_data is not None else TargetData.default()
        self._frontend = frontend
        self._middleend = middleend
        self._backend = backend
        self.disable_model_validation = disable_model_validation

    def compile(
        self,
        program,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ) -> tuple[Executable, MetricsManager]:
        """Compiles a source program into an executable using the pipeline's components.

        :param program: The source program to compile.
        :param compiler_config: Configuration options for the compiler, such as optimization
            and results formatting.
        :return: An executable of the compiled program for the target device and the metrics
            manager containing metrics collected during compilation.
        """

        # TODO: Improve metrics and config handling
        compiler_config = compiler_config or CompilerConfig(
            metrics=MetricsType.Experimental
        )
        metrics_manager = MetricsManager(compiler_config.metrics)
        compilation_results = ResultManager()

        ir = self.frontend.emit(
            program,
            compilation_results,
            metrics_manager,
            compiler_config=compiler_config,
            **kwargs,
        )

        ir = self.middleend.emit(
            ir,
            compilation_results,
            metrics_manager,
            compiler_config=compiler_config,
            **kwargs,
        )

        package = self.backend.emit(
            ir,
            compilation_results,
            metrics_manager,
            compiler_config=compiler_config,
            **kwargs,
        )

        return package, metrics_manager

    @property
    def frontend(self) -> BaseFrontend:
        return self._frontend

    @property
    def middleend(self) -> BaseMiddleend:
        return self._middleend

    @property
    def backend(self) -> BaseBackend:
        return self._backend


class ExecutePipeline(BasePipeline):
    """Implements a pipeline that can execute quantum programs.

    In addition to the hardware model and target data, an execution pipeline consists of
    the following components:

    #. Runtime: Manages the execution of the program, including the engine and the post-
       processing of the results.
    #. Engine: Communicates the compiled program with the target devices, and returns the
       results.

    In the future, this might be relaxed so execution pipelines can be defined more
    abstractly, only requiring the :meth:`execute` to be implemented.
    """

    def __init__(
        self,
        name: str,
        model: QuantumHardwareModel,
        runtime: BaseRuntime,
        target_data: TargetData | None = None,
        disable_model_validation: bool = False,
    ):
        if not disable_model_validation:
            self._validate_consistent_model(model, runtime, runtime.engine)
        super().__init__(name, model, target_data)
        self._runtime = runtime
        self.disable_model_validation = disable_model_validation

    def execute(
        self,
        executable: Executable,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ) -> tuple[dict, MetricsManager]:
        """Uses the runtime and engine in the pipeline to execute a compiled program and
        process the results.

        Checks that the hardware model in the pipeline matches the hardware model used
        during compilation.

        :param executable: The compiled program to execute.
        :param compiler_config: Configuration options for the compiler, such as optimization
            and results formatting.
        :return: A dictionary of results from the execution and the metrics manager
            containing metrics collected during execution.
        """

        compiler_config = CompilerConfig() if compiler_config is None else compiler_config

        if (
            self.model is not None
            and self.model.calibration_id != executable.calibration_id
        ):
            raise MismatchingHardwareModelException(
                f"Hardware id '{executable.calibration_id}' in the executable "
                f"does not match the hardware model id '{self.model.calibration_id}' used "
                "for execution in this pipeline."
            )

        pp_results = ResultManager()
        metrics_manager = MetricsManager(compiler_config.metrics)
        results = self.runtime.execute(
            executable,
            res_mgr=pp_results,
            met_mgr=metrics_manager,
            compiler_config=compiler_config,
            **kwargs,
        )
        return results, metrics_manager

    @property
    def runtime(self) -> BaseRuntime:
        return self._runtime

    @property
    def engine(self) -> NativeEngine:
        """Returns the engine within the runtime."""
        return self._runtime.engine


class Pipeline(CompilePipeline, ExecutePipeline):
    """A composite pipeline that can both compile and execute quantum programs.

    This pipeline combines the functionality of both compilation and execution pipelines,
    allowing for a complete workflow from source program to execution. See
    :class:`CompilePipeline` and :class:`ExecutePipeline` for more details on the
    components involved in compilation and execution.
    """

    def __init__(
        self,
        name: str,
        model: QuantumHardwareModel,
        frontend: BaseFrontend,
        middleend: BaseMiddleend,
        backend: BaseBackend,
        runtime: BaseRuntime,
        target_data: TargetData | None = None,
        disable_model_validation: bool = False,
    ):
        if not disable_model_validation:
            self._validate_consistent_model(
                model, frontend, middleend, backend, runtime, runtime.engine
            )
        self._name = name
        self._model = model
        self._target_data = target_data if target_data is not None else TargetData.default()
        self._frontend = frontend
        self._middleend = middleend
        self._backend = backend
        self._runtime = runtime
        self.disable_model_validation = disable_model_validation
