# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


from qat.backend.base import BaseBackend
from qat.engines import NativeEngine
from qat.frontend import BaseFrontend
from qat.middleend.middleends import BaseMiddleend
from qat.model.target_data import TargetData
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
        allow_type_mismatch=False,
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
