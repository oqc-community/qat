# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd
"""Post-processing method models for qubit measurement results.

This module defines the post-processing methods available for qubit measurement
discrimination. The main interface is the :class:`PostProcessMethod` union, which is a
discriminated union over all supported post-processing method models.

The discriminator field is ``method``, which is present in the base class
:class:`MethodBase` and is used by Pydantic to determine which concrete model to use
when parsing or serializing data.

Supported methods:

- ``linear_map_complex_to_real``: See :class:`LinearMapToRealMethod`
- ``max_likelihood``: See :class:`MaxLikelihoodMethod`

Key encoding convention for :class:`MaxLikelihoodMethod`
---------------------------------------------------------
The ``states`` dict maps an **integer output key** to an :class:`MLDiscriminateParams`.
Non-negative keys (``≥ 0``) represent **allowed** states whose value will be written
to the classical register.  Negative keys (``< 0``) represent **disallowed** states;
shots classified to those states are filtered out by
:class:`~qat.ir.measure.PostSelect`.

The reserved sentinel :data:`BG_KEY` is used by the runtime when ``p_min > 0``
rejects a shot.

Post-selection is only supported for :class:`MaxLikelihoodMethod`.
:class:`LinearMapToRealMethod` always emits states ``0`` and ``1`` via a threshold
discriminator and does not support post-selection.
"""

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from qat.utils.pydantic import FloatNDArray, NoExtraFieldsModel

BG_KEY = -99
"""Reserved integer key used by the runtime when ``p_min > 0`` rejects a shot.

The runtime stores this value in the integer state array produced by
:func:`~qat.runtime.post_processing.apply_discriminate_instruction` for any shot
whose normalised likelihood falls below :attr:`MaxLikelihoodMethod.p_min`.
The subsequent :class:`~qat.ir.measure.PostSelect` step filters all negative keys
(including this one) from the results.
"""


class MethodIndicator(str, Enum):
    """Enum for discriminating post-processing methods applied to qubit measurement
    results."""

    LINEAR_MAP_COMPLEX_TO_REAL = "linear_map_complex_to_real"
    MAX_LIKELIHOOD = "max_likelihood"


class MethodBase(NoExtraFieldsModel):
    """Base class for all post-processing methods.

    Contains the discriminator field.
    """

    method: MethodIndicator = Field(
        ..., description="Discriminator for the post-processing method type."
    )


class LinearMapToRealMethod(MethodBase):
    """Threshold-based discriminator using a calibrated complex-to-real projection.

    Projects each complex IQ readout onto a real value via:

    .. math::

        v = \\mathrm{Re}(a \\cdot \\mathrm{IQ} + b)

    where ``a`` and ``b`` are the two entries of ``mean_z_map_args``.  The result is
    threshold-classified: above threshold → state ``0``, at-or-below → state ``1``.

    Post-selection is not supported for this method; state outputs are always the
    non-negative integers ``0`` and ``1``.  Use :class:`MaxLikelihoodMethod` with
    negative keys to configure post-selection on measurement outcomes.
    """

    method: Literal[MethodIndicator.LINEAR_MAP_COMPLEX_TO_REAL] = Field(
        MethodIndicator.LINEAR_MAP_COMPLEX_TO_REAL,
        description="Discriminator for linear map complex to real method.",
    )
    mean_z_map_args: list[complex] = Field(
        max_length=2,
        min_length=2,
        default_factory=lambda: [1 + 0j, 0j],
        description="Arguments for the linear map.",
    )


class MLDiscriminateParams(NoExtraFieldsModel):
    """IQ-plane centroid for a single measurement outcome.

    Each :class:`MLDiscriminateParams` entry represents one possible measurement outcome.
    The integer output value and disallowed flag are encoded by the **dict key**
    in the parent :class:`MaxLikelihoodMethod`'s ``states`` mapping:

    - Non-negative key (``≥ 0``) → allowed state; the key is the integer written to
      the classical register.
    - Negative key (``< 0``) → disallowed state; shots assigned here are filtered by
      :class:`~qat.ir.measure.PostSelect`.

    The optional ``label`` field carries a human-readable name for the state (e.g.
    ``"|0⟩"`` or ``"|10⟩"``).  It is purely informational — the runtime uses the dict
    key for discrimination and the ``location`` for distance calculations.

    Example — a standard qubit with states |0⟩ and |1⟩::

        MaxLikelihoodMethod(states={
            0: MLDiscriminateParams(location=1+0j, label="|0⟩"),
            1: MLDiscriminateParams(location=-1+0j, label="|1⟩"),
        })

    Example — qutrit classifier that rejects the leakage state (key -2)::

        MaxLikelihoodMethod(states={
            0: MLDiscriminateParams(location=1+0j, label="|0⟩"),
            1: MLDiscriminateParams(location=-1+0j, label="|1⟩"),
            -2: MLDiscriminateParams(location=0+1j, label="|2⟩ (leakage)"),
        })
    """

    location: complex = Field(..., description="Complex IQ-plane centroid for this state.")
    label: str | None = Field(
        default=None,
        description="Optional human-readable label for this state, e.g. '|0⟩' or '|10⟩'. "
        "Purely informational; not used by the runtime discriminator.",
    )


class MaxLikelihoodMethod(MethodBase):
    """Maximum likelihood discriminator for qubit measurement results.

    Each shot is assigned to the state ``k`` with the highest normalised likelihood:

    .. math::

        \\tilde{p}_k(z) = \\frac{L_k(z)}{\\sum_j L_j(z)},
        \\quad L_k(z) = \\exp\\!\\left(-\\frac{|z - \\mathrm{loc}_k|^2}{2\\,\\nu}\\right)

    where :math:`\\nu` is the noise power (variance) ``noise_est``.

    Likelihoods are computed in log-domain with log-sum-exp stabilisation to avoid
    underflow on extreme outliers.

    **Outlier rejection** — set ``p_min > 0`` to automatically reject shots whose
    winning normalised likelihood falls below the threshold. Those shots are assigned
    :data:`BG_KEY` and discarded by :class:`~qat.ir.measure.PostSelect`.
    Default ``p_min=0.0`` disables the check entirely (zero overhead).

    **Key encoding convention** — the ``states`` dict maps an integer output key to an
    :class:`MLDiscriminateParams`:

    - Non-negative key (``≥ 0``) → allowed state; the key value is written to the
      classical register.
    - Negative key (``< 0``) → disallowed state; shots assigned to these states are
      removed by :class:`~qat.ir.measure.PostSelect`.

    Runtime implementation:
    :func:`qat.runtime.post_processing.apply_discriminate_instruction`.

    :param states: Per-state IQ-plane centroids, keyed by integer output value
        (non-negative) or disallowed sentinel (negative).
    :param noise_est: Global Gaussian noise power (variance, :math:`\\sigma^2`). Must be
        strictly positive. Default ``1.0``.
    :param p_min: Minimum normalised likelihood for acceptance. Must be in ``[0, 1]``.
    :param transform: Real ``(2, 2)`` IQ affine pre-transform matrix ``A``.
        If ``None`` (default), no :class:`~qat.ir.measure.Equalise` step is emitted.
    :param offset: Real offset vector ``[b_I, b_Q]`` for the affine pre-transform.
        If ``None`` (default), no :class:`~qat.ir.measure.Equalise` step is emitted.
    """

    method: Literal[MethodIndicator.MAX_LIKELIHOOD] = Field(
        MethodIndicator.MAX_LIKELIHOOD,
        description="Discriminator for maximum likelihood method.",
    )
    states: dict[int, MLDiscriminateParams] = Field(
        ...,
        description="Mapping from integer output key to MLDiscriminateParams. Non-negative keys "
        "are allowed states (key = classical register value). Negative keys are "
        "disallowed states filtered by PostSelect.",
    )
    noise_est: float = Field(
        1.0,
        gt=0,
        description="Global Gaussian noise power (variance, σ²) used in the normalised "
        "likelihood discriminator (log-likelihood = −|z−loc|²/(2·noise_est)). "
        "Must be strictly positive. Default is 1.0.",
    )
    p_min: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum normalised likelihood for the winning state. Shots whose "
        "best-state normalised likelihood is below this threshold are assigned "
        "BG_KEY and discarded by post-selection. Must be in [0, 1]. "
        "Default 0.0 disables the check entirely.",
    )
    transform: FloatNDArray | None = Field(
        default=None,
        description="Real (2, 2) affine transform matrix applied to IQ data before "
        "discrimination. If None, no equalisation step is emitted.",
    )
    offset: FloatNDArray | None = Field(
        default=None,
        description="Real offset vector [b_I, b_Q] applied to IQ data before "
        "discrimination. If None, no equalisation step is emitted.",
    )

    @model_validator(mode="after")
    def _validate_states_non_empty(self):
        """Ensure at least one state is defined for the max-likelihood method."""
        if not self.states:
            raise ValueError("MaxLikelihoodMethod must define at least one state.")
        return self

    @model_validator(mode="after")
    def _validate_states_do_not_use_bg_key(self):
        """Ensure user-defined states do not collide with the reserved BG_KEY."""
        if BG_KEY in self.states:
            raise ValueError(
                "MaxLikelihoodMethod 'states' must not contain BG_KEY; "
                "BG_KEY is reserved for p_min rejections."
            )
        return self

    @model_validator(mode="after")
    def _validate_transform_and_offset_both_or_neither(self):
        """Ensure transform and offset are either both set or both None."""
        has_transform = self.transform is not None
        has_offset = self.offset is not None
        if has_transform != has_offset:
            raise ValueError(
                "MaxLikelihoodMethod requires both 'transform' and 'offset' to be set "
                "together, or both left as None."
            )
        return self

    @model_validator(mode="after")
    def _validate_transform_is_2x2(self):
        """Ensure transform is a 2x2 matrix if it is set."""
        if self.transform is not None and self.transform.shape != (2, 2):
            raise ValueError("MaxLikelihoodMethod 'transform' must be a 2x2 matrix.")
        return self

    @model_validator(mode="after")
    def _validate_offset_is_shape_2(self):
        """Ensure offset is a length-2 vector if it is set."""
        if self.offset is not None and self.offset.shape != (2,):
            raise ValueError(
                "MaxLikelihoodMethod 'offset' must be a 1-D vector of length 2; "
                f"got shape {self.offset.shape}."
            )
        return self


# Union type for use in Qubit model
PostProcessMethod = LinearMapToRealMethod | MaxLikelihoodMethod
