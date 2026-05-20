# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
"""IR models for measurement and post-processing instructions.

This module defines :class:`Acquire`, :class:`PostProcessing` and
``MeasureBlock`` representations used in the intermediate representation to
describe measurement-related operations and software post-processing steps.
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import BeforeValidator, Field, PrivateAttr, field_validator, model_validator

# The following things from legacy instructions are unchanged, so just import for now.
from qat.ir.instruction_basetypes import AcquireMode, PostProcessType, ProcessAxis
from qat.ir.instructions import (
    Delay,
    Instruction,
    QuantumInstruction,
    QuantumInstructionBlock,
    Synchronize,
)
from qat.ir.waveforms import Pulse
from qat.model.post_processing import MaxLikelihoodMethod
from qat.utils.pydantic import (
    FloatNDArray,
    QubitId,
    ValidatedList,
    ValidatedSet,
    _validate_set,
)


class Acquire(QuantumInstruction):
    """Instruction representing an acquisition of a resonator/qubit signal.

    :param duration: Acquisition duration in seconds.
    :param mode: Acquisition mode describing how the hardware returns data.
    :param delay: Optional delay inserted before the acquisition value is valid.
    :param filter: Optional pulse defining an integration/filter kernel applied on the
        readout channel by the hardware or software.
    :param output_variable: Name of the variable where the acquisition result will be
        stored.
    :param rotation: optional angle in radians applied to the acquired complex value,
        equivalent to a complex multiplier of ``exp(-j * rotation)``.  Ignored if
        `filter` is provided.
    :param threshold: Optional threshold value for discriminating the acquired value
      into a binary state label. Ignored if `filter` is provided or if `mode` is not
      `AcquireMode.INTEGRATOR`.
    """

    duration: float = 1e-6
    mode: AcquireMode = AcquireMode.INTEGRATOR
    delay: float | None = 0.0
    filter: Pulse | None = Field(default=None)
    output_variable: str = Field(min_length=1)
    rotation: float | None = Field(
        default=0.0,
        deprecated="rotation is deprecated and will be removed in a future release.",
    )
    threshold: float | None = Field(
        default=0.0,
        deprecated="threshold is deprecated and will be removed in a future release.",
    )

    @property
    def pulse_channel(self):
        return next(iter(self.targets))

    @property
    def target(self):
        return next(iter(self.targets))

    @model_validator(mode="before")
    def _validate_filter(cls, data: dict):
        if isinstance(data, dict):
            duration = data.get("duration")
            filter = data.get("filter")

            if filter:
                filter_duration = (
                    filter.duration if isinstance(filter, Pulse) else filter["duration"]
                )
                # TODO: COMPILER-722 -- with this in place, it required to add a duration to both the pulse
                #  and the Acquire instruction, would be better to figure out how to share information in some cases.
                if filter_duration == 0:  # < 0 condition already tested in `Waveform`
                    raise ValueError("Filter duration cannot be equal to zero.")

                if not np.isclose(filter_duration, duration, atol=1e-9):
                    raise ValueError(
                        f"Filter duration '{filter_duration}' must be equal to Acquire "
                        f"duration '{duration}'."
                    )

        return data


class PostProcessing(Instruction):
    """States what post-processing should happen after data has been acquired.

    This can happen in the FPGA's or a software post-process.
    """

    output_variable: str | None = None
    process_type: PostProcessType
    axes: list[ProcessAxis] = []
    args: list[float | complex] = []
    result_needed: bool = False

    @classmethod
    def _from_legacy(cls, legacy_pp):
        # private as we dont want to support this in the long-term
        return cls(
            output_variable=legacy_pp.output_variable,
            process_type=legacy_pp.process.value,
            axes=legacy_pp.axes,
            args=legacy_pp.args,
            result_needed=legacy_pp.result_needed,
        )

    @field_validator("axes", mode="before")
    @classmethod
    def _axes_as_list(cls, axes=None):
        if axes:
            return axes if isinstance(axes, list) else [axes]
        return []

    @field_validator("args", mode="before")
    def _validate_args(cls, args=None):
        """Ensures that the args are not numpy arrays or numpy numbers."""

        if args is None:
            args = []
        args = [args] if not isinstance(args, list | np.ndarray) else args
        return np.asarray(args).tolist()


class Equalise(Instruction):
    """Apply an affine transform in the IQ (complex) plane to readout data.

    This is the first stage of the granular post-processing pipeline.

    In superconducting qubit readout the downconverted IQ signal is distorted
    by three hardware imperfections: **phase imbalance** (LO quadrature paths
    not exactly 90° apart), **gain imbalance** (unequal I/Q amplifier chains),
    and **DC offsets** (mixer leakage and biases).  As a result, raw ``(I, Q)``
    samples cluster on a distorted, offset ellipse rather than a compact point
    cloud, degrading any downstream discriminator.

    The ``Equalise`` instruction corrects all three imperfections in a single
    real affine transform:

    .. math::

        \\begin{pmatrix} I' \\\\ Q' \\end{pmatrix}
        = A \\begin{pmatrix} I \\\\ Q \\end{pmatrix} + \\begin{pmatrix} b_I \\\\ b_Q \\end{pmatrix}

    where ``A`` is a **real** 2×2 matrix (``transform``) and ``[b_I, b_Q]``
    is the real offset vector (``offset``).  The output is returned as a
    complex value ``I' + j Q'``.

    Each :class:`Equalise` instruction operates on a single readout channel.
    To equalise multiple channels, emit one instruction per channel with its
    own ``output_variable``.

    The default ``transform`` (2×2 identity) and default ``offset`` (zero
    vector) are a no-op pass-through for already-calibrated hardware.

    Runtime implementation: :func:`qat.runtime.post_processing.apply_equalise`.

    :param output_variable: Variable name whose data should be transformed.
    :param transform: Real ``(2, 2)`` matrix ``A``.
    :param offset: Real offset vector ``[b_I, b_Q]``, shape ``(2,)``.
        Defaults to the zero vector.
    """

    output_variable: str
    transform: FloatNDArray = Field(default_factory=lambda: np.eye(2))
    offset: FloatNDArray = Field(default_factory=lambda: np.zeros(2))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Equalise):
            return False
        return (
            self.output_variable == other.output_variable
            and np.array_equal(self.transform, other.transform)
            and np.array_equal(self.offset, other.offset)
        )

    def __hash__(self):
        return hash((self.output_variable, self.transform.tobytes(), self.offset.tobytes()))


class Discriminate(Instruction):
    """Discriminate equalised values to string state labels.

    For the linear-map path a sign-based threshold comparison is used:
    values above ``threshold`` → label ``"0"``, values at or below → label
    ``"1"``. For the maximum-likelihood path the nearest centroid in the
    complex plane determines the state's configured string label
    (``MLStateMap.label``).

    Exactly one of ``threshold`` or ``method`` must be provided.

    Runtime implementation:
    :func:`qat.runtime.post_processing.apply_discriminate_instruction`.

    :param output_variable: Variable name whose data should be discriminated.
    :param threshold: Scalar threshold for the linear-map discrimination path.
        ``None`` when ``method`` is provided.
    :param method: Post-process method object for the ML path. ``None`` when
        ``threshold`` is provided.
    """

    output_variable: str
    threshold: float | None = None
    method: MaxLikelihoodMethod | None = None

    @model_validator(mode="after")
    def _validate_exactly_one_of_threshold_or_method(self):
        has_threshold = self.threshold is not None
        has_method = self.method is not None
        if has_threshold == has_method:
            raise ValueError(
                "Exactly one of 'threshold' or 'method' must be provided in Discriminate."
            )
        return self


class PostSelect(Instruction):
    """Remove shots whose state label appears in ``disallowed_states``.

    Emitting this instruction with an empty ``disallowed_states`` list is a
    no-op (all shots are considered valid) but is safe to emit unconditionally
    so that the pipeline structure is uniform.

    Runtime implementation: :func:`qat.runtime.post_processing.apply_post_select`.

    :param output_variable: Variable name whose state labels should be
        screened.
    :param disallowed_states: String state labels that should be marked
        invalid. Shots mapped to these labels will have their validity mask
        entry set to ``False``.
    """

    output_variable: str
    disallowed_states: list[str] = Field(default_factory=list)


class Demap(Instruction):
    """De-map string state labels to final integer output values.

    This is the final stage of the granular post-processing pipeline. Each
    shot is mapped from a string state label (produced by :class:`Discriminate`)
    to the configured integer output value written to the classical register.

    Runtime implementation: :func:`qat.runtime.post_processing.apply_demap_instruction`.

    :param output_variable: Variable name whose state labels should be mapped.
    :param state_map: Mapping from string state label to integer output value, e.g.
        ``{"0": 0, "1": 1}`` for the standard binary convention.
    """

    output_variable: str
    state_map: dict[str, int]


# Union type for all granular post-processing instructions
GranularPostProcessInstruction = Equalise | Discriminate | PostSelect | Demap

VALID_MEASURE_INSTR = Synchronize | Acquire | Pulse | Delay


class MeasureBlock(QuantumInstructionBlock):
    """Encapsulates a measurement of a single (or multiple) qubit(s).

    It should only contain instructions that are associated with a measurement such as a
    measure pulse, an acquire or a synchronize.
    """

    instructions: ValidatedList[VALID_MEASURE_INSTR] = Field(
        default_factory=lambda: ValidatedList[VALID_MEASURE_INSTR]()
    )

    qubit_targets: Annotated[ValidatedSet[QubitId], BeforeValidator(_validate_set)] = Field(
        default_factory=lambda: ValidatedSet()
    )
    _output_variables: list[str] = PrivateAttr(default=[])

    def add(self, *instructions: QuantumInstruction):
        for instruction in instructions:
            if isinstance(instruction, Acquire) and (delay := instruction.delay) > 0.0:
                self._output_variables.append(instruction.output_variable)
                super().add(Delay(target=instruction.target, duration=delay))
                instruction.delay = 0.0
            super().add(instruction)
        return self

    @property
    def output_variables(self):
        return self._output_variables


acq_mode_process_axis = {
    ProcessAxis.SEQUENCE: AcquireMode.INTEGRATOR,
    ProcessAxis.TIME: AcquireMode.SCOPE,
    "sequence": AcquireMode.INTEGRATOR,
    "time": AcquireMode.SCOPE,
}
