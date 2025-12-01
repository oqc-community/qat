# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from inspect import signature

from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import TargetData
from qat.model.validators import MismatchingHardwareModelException
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class AbstractPipeline(ABC):
    """An abstraction of pipelines used in QAT to compile and execute quantum programs.

    Abstractly a pipeline needs to be given a hardware model and some target data that
    contains the parameters of the target device needed to correctly compile and execute.
    In the future, this might be relaxed so that the model and/or target data is not
    concretely needed to define a pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> QuantumHardwareModel | PhysicalHardwareModel: ...

    @property
    @abstractmethod
    def target_data(self) -> TargetData: ...

    def is_subtype_of(self, cls):
        """Matches a given type against the pipeline."""
        return isinstance(self, cls)


class BasePipeline(AbstractPipeline, ABC):
    """A base implementation of the abstract pipeline that provides access to the
    components of the pipeline.

    Subclasses should implement the `copy` and `copy_with_name` methods.
    """

    def __init__(self, name: str, model: QuantumHardwareModel, target_data: TargetData):
        self._name = name
        self._model = model
        self._target_data = target_data if target_data is not None else TargetData.default()

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> QuantumHardwareModel:
        return self._model

    @property
    def target_data(self) -> TargetData:
        return self._target_data

    def copy(self) -> "BasePipeline":
        """Returns a new instance of the pipeline with the same components."""

        cls = self.__class__
        arg_names = signature(cls.__init__).parameters.keys()
        args = {k: getattr(self, k) for k in arg_names if k != "self"}
        return cls(**args)

    def copy_with_name(self, name: str) -> "BasePipeline":
        """Returns a new instance of the pipeline with the same components, but with a
        different name."""

        cls = self.__class__
        arg_names = signature(cls.__init__).parameters.keys()
        args = {k: getattr(self, k) for k in arg_names if k != "self"}
        args["name"] = name
        return cls(**args)

    def _validate_consistent_model(self, model: QuantumHardwareModel, *args):
        """Validates that the model is consistent across all components of the pipeline.

        :param model: The hardware model to validate against.
        :param args: The components of the pipeline to validate.
        """
        for component in args:
            if hasattr(component, "model") and component.model not in (model, None):
                raise MismatchingHardwareModelException(
                    f"{model} hardware does not match supplied hardware"
                )
