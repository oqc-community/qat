# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import inspect
from typing import Callable

from qat.instrument.base import InstrumentBuilder, InstrumentConcept


def is_pipeline_instance(value: type):
    """A validator which raises when the input not a BasePipeline instance."""
    from qat.pipelines.pipeline import BasePipeline

    if not isinstance(value, BasePipeline):
        raise ValueError(f"{value} is not a valid Pipeline instance")
    return value


def is_frontend(value: type):
    """A validator which raises when the input not a Frontend."""
    from qat.frontend.base import BaseFrontend

    if not (inspect.isclass(value) and issubclass(value, BaseFrontend)):
        raise ValueError(f"{value} is not a valid Frontend")
    return value


def is_middleend(value: type):
    """A validator which raises when the input not a Middleend."""

    from qat.middleend.base import BaseMiddleend

    if not (inspect.isclass(value) and issubclass(value, BaseMiddleend)):
        raise ValueError(f"{value} is not a valid Middleend")
    return value


def is_backend(value: type):
    """A validator which raises when the input not a Backend."""

    from qat.backend.base import BaseBackend

    if not (inspect.isclass(value) and issubclass(value, BaseBackend)):
        raise ValueError(f"{value} is not a valid Backend")
    return value


def is_runtime(value: type):
    """A validator which raises when the input not a Runtime."""
    from qat.runtime import BaseRuntime

    if not (inspect.isclass(value) and issubclass(value, BaseRuntime)):
        raise ValueError(f"{value} is not a valid Runtime")
    return value


def is_instrument(value: type):
    """
    A validator which raises when the input not an Instrument.
    """

    if not (inspect.isclass(value) and issubclass(value, InstrumentConcept)):
        raise ValueError(f"{value} is not a valid Instrument builder")
    return value


def is_instrument_builder(value: type):
    """
    A validator which raises when the input not an Instrument builder.
    """

    if not (inspect.isclass(value) and issubclass(value, InstrumentBuilder)):
        raise ValueError(f"{value} is not a valid Instrument builder")
    return value


def is_engine(value: type):
    """A validator which raises when the input not an Engine.

    This permits NativeEngines and (legacy qat.purr) InstructionExecutionEngine.
    In future we intend to only accept NativeEngines (once qat.purr is deprecated)
    """
    from qat.engines import NativeEngine
    from qat.purr.compiler.execution import InstructionExecutionEngine

    if not (
        inspect.isclass(value)
        and issubclass(value, (NativeEngine, InstructionExecutionEngine))
    ):
        raise ValueError(
            f"{value} is not a valid Engine (either NativeEngine or legacy "
            "InstructionExecutionEngine)"
        )
    return value


def is_target_data(value: type):
    """A validator which raises when the input not a TargetData. Allows both TargetData
    classes and functions which return TargetData instances."""

    from qat.model.target_data import AbstractTargetData

    if inspect.isclass(value) and issubclass(value, AbstractTargetData):
        return value

    if callable(value):
        return_annotation = inspect.signature(value).return_annotation
        if isinstance(return_annotation, type) and issubclass(
            return_annotation, AbstractTargetData
        ):
            return value
    raise ValueError(f"{value} is not a valid TargetData")


def is_passmanager(value: type):
    """A validator which raises when the input not an PassManager."""
    from qat.core.pass_base import PassManager

    if not (inspect.isclass(value) and issubclass(value, PassManager)):
        raise ValueError(f"{value} is not a valid PassManager")
    return value


def is_passmanager_factory(value: type):
    """A validator which raises when the input is not function that returns a
    PassManager."""
    from qat.core.pass_base import PassManager

    check_callable(value, PassManager)
    return value


def is_hardwareloader(value: type):
    """A validator which raises when the input not a model loader."""
    from qat.model.loaders.base import BaseModelLoader

    if not (inspect.isclass(value) and issubclass(value, BaseModelLoader)):
        raise ValueError(f"{value} is not a valid Hardware Model Loader")
    return value


def is_pipeline_factory(value: type):
    """A validator which raises when the input not function which returns a Pipeline."""
    from qat.pipelines.base import BasePipeline

    check_callable(value, BasePipeline)
    return value


def is_updateable_pipeline(value: type):
    """A validator which raises when the input not an UpdateablePipeline instance."""
    from qat.pipelines.updateable import UpdateablePipeline

    if not (inspect.isclass(value) and issubclass(value, UpdateablePipeline)):
        raise ValueError(f"{value} is not a valid UpdateablePipeline.")
    return value


def requires_model(value: type) -> bool:
    """Determines whether a class or function expects a model

    In future we will also inspect the type is a hardware model but currently
    we are not consistent enough with type hinting for this.
    """
    return "model" in inspect.signature(value).parameters


def requires_target_data(value: type) -> bool:
    """Determines whether a class or function expects target data.

    In future we will also inspect the type is a target data but currently we are not
    consistent enough with type hinting for this.
    """
    return "target_data" in inspect.signature(value).parameters


def check_callable(value: Callable, return_type: type):
    """Raises if the value is not a callable, and the return type is not the expected
    type."""

    if not callable(value):
        raise ValueError(f"{value} is not callable")

    return_annotation = inspect.signature(value).return_annotation
    if not isinstance(return_annotation, type):
        raise ValueError(f"{value} does not have a valid return annotation")

    if not issubclass(return_annotation, return_type):
        raise ValueError(f"{value} does not return a {return_type.__name__}")
