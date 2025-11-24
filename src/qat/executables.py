# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from compiler_config.config import InlineResultsProcessing
from pydantic import BaseModel, PositiveInt, field_validator
from pydantic_core import from_json

from qat.ir.instructions import Assign
from qat.ir.measure import PostProcessing
from qat.purr.compiler.instructions import AcquireMode
from qat.utils.pydantic import RehydratableModel


class AcquireData(BaseModel):
    """Holds the information required at runtime surrounding an acquisition.

    :param mode: The type of acquisition performed on the hardware.
    :param shape: The shape the data is expected to be returned in. Reflects the type of
        acquisition (e.g. scope or integrator mode), as well as the iteration structure
        (e.g. shots, loops). A shape of (1000,) would indicate 1000 shots of a single
        integrator acquisition.
    :param physical_channel: The physical channel the acquisition is performed on, needed
        for post-processing reasons (e.g. error mitigation).
    :param post_processing: A list of post-processing steps to be applied to the execution
        data.
    :param results_processing: Contains information surrounding how the results should be
        formatted after post-processing.
    """

    mode: AcquireMode
    shape: tuple[int, ...]
    physical_channel: str
    post_processing: list[PostProcessing] = []
    results_processing: InlineResultsProcessing | None = None


class AbstractProgram(RehydratableModel, ABC):
    """Abstract base class for programs to be executed by an engine.

    Child classes must implement the :attr:`acquire_shapes` property, which returns a
    mapping of acquisition output variables to the shape of the acquisition for this
    program. This specification enforces programs to be explicit about what acquisition
    outputs they will produce.
    """

    @property
    @abstractmethod
    def acquire_shapes(self) -> dict[str, tuple[int, ...]]: ...


Program = TypeVar("Program", bound=AbstractProgram)


class Executable(BaseModel, Generic[Program]):
    """:class:`Executable`s are packages of instructions that will be executed by the
    runtime. They contain a :class:`Program`, or a number of :class:`Program`s to be
    executed by the engine. They also contain the acquire information, such as
    post-processing and result assignments.

    :param programs: The program(s) to be executed against the engine.
    :param assigns: Assigns results to given variables.
    :param returns: Which acqusitions/variables should be returned.
    :param calibration_id: The (unique) cabliration ID of the underlying hardware model.
    :param shots: The total number of shots performed in the :class:`Executable`.
    """

    programs: list[Program] = []
    acquires: dict[str, AcquireData] = dict()
    assigns: list[Assign] = []
    returns: set[str] = set()
    calibration_id: str = ""
    shots: PositiveInt | None = None

    def serialize(self, indent: int = 4) -> str:
        """Serializes the executable as a JSON blob."""
        return self.model_dump_json(indent=indent, exclude_none=True)

    @classmethod
    def deserialize(cls, blob: str):
        """Instantiates a executable from a JSON blob."""
        return cls(**from_json(blob))

    @field_validator("programs", mode="before")
    @classmethod
    def _rehydrate_programs(cls, data) -> list[Program]:
        """This validator determines the correct type of program to rehydrate based on the
        input data structure."""
        if isinstance(data, list):
            return [
                AbstractProgram._rehydrate_object(d) if isinstance(d, dict) else d
                for d in data
            ]
        else:
            return cls._rehydrate_programs([data])

    @field_validator("programs", mode="after")
    @classmethod
    def _validate_programs_are_the_same_type(cls, programs):
        """Ensures that if multiple programs are provided, they are all of the same type."""
        if isinstance(programs, list):
            types = set(type(p) for p in programs)
            if len(types) > 1:
                raise ValueError(
                    f"All programs in the executable must be of the same type. Saw {types}."
                )
        return programs
