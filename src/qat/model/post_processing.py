"""Post-processing method models for qubit measurement results.

This module defines the post-processing methods available for qubit measurement discrimination. The main interface is the
:class:`PostProcessMethod` union, which is a discriminated union over all supported post-processing method models.

The discriminator field is ``method``, which is present in the base class :class:`MethodBase` and is used by Pydantic
to determine which concrete model to use when parsing or serializing data. Each post-processing method (such as
:class:`LinearMapToRealMethod` or :class:`MaxLikelihoodMethod`) inherits from :class:`MethodBase` and sets a unique value
for the ``method`` field, as defined in the :class:`MethodIndicator` enum.

Usage in the Qubit model (see ``device.py``)::

    class Qubit(Component):
        ...
        post_process_method: PostProcessMethod | None = Field(
            discriminator="method", default=None
        )
        ...

Here, the ``discriminator='method'`` argument is required for Pydantic to correctly (de)serialize the union type.

Supported methods:

- ``linear_map_complex_to_real``: See :class:`LinearMapToRealMethod`
- ``max_likelihood``: See :class:`MaxLikelihoodMethod`
"""

from collections import Counter
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from qat.utils.pydantic import FloatNDArray, NoExtraFieldsModel

BG_LABEL = "|?>"
"""Reserved state label emitted by the ML discriminator for shots whose normalised
likelihood for the winning state falls below :attr:`MaxLikelihoodMethod.p_min`.

Users must not use this string as an :attr:`MLStateMap.label` (to prevent collision).
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

    where ``a`` and ``b`` are the two entries of ``mean_z_map_args``.  The
    result is sign-classified: positive → state ``"0"``, non-positive →
    state ``"1"``.

    Set ``disallowed_states`` to filter out shots by their classified state
    label (e.g. ``{"1"}`` to discard shots **not** in the ground state).
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
    disallowed_states: set[str] = Field(
        default_factory=set,
        description="Set of disallowed state labels (matching the string labels used by "
        "the granular pipeline). If a measurement is classified to a state whose label "
        "appears in this set, the shot should be discarded.",
    )

    @property
    def declared_states(self) -> set[str]:
        """All state labels known to this method.

        :returns: ``{"0", "1"}`` for a standard threshold discriminator.
        """
        return {"0", "1"}


class MLStateMap(NoExtraFieldsModel):
    """Demapping from a quantum state to its IQ-plane location and C-register output value.

    Each :class:`MLStateMap` entry represents one possible measurement outcome.  The
    ``label`` is string routing key that threads through the discrimination pipeline
    (``Discriminate`` → ``PostSelect`` → ``Demap``). The ``output_value`` is the final
    integer written to the classical register after ``Demap`` has been applied.

    The ``disallowed`` field marks a state as invalid for use-cases such as erasure
    checks and pre-selection.  Shots assigned to a ``disallowed=True`` state are
    subsequently removed by :class:`~qat.ir.measure.PostSelect`.

    Example — a standard qubit with states |0⟩ and |1⟩::

        MLStateMap(label="0", output_value=0, location=1+0j)
        MLStateMap(label="1", output_value=1, location=-1+0j)

    Example — qutrit classifier that rejects the leakage state |2⟩::

        MLStateMap(label="0", output_value=0, location=1+0j)
        MLStateMap(label="1", output_value=1, location=-1+0j)
        MLStateMap(label="2", output_value=2, location=0+1j, disallowed=True)
    """

    label: str = Field(
        ...,
        description="String label that identifies this state through the discrimination "
        "pipeline (Discriminate → PostSelect → Demap). Used as the key in Demap.state_map.",
    )
    output_value: int = Field(
        ...,
        description="Final integer value written to the classical register for this state, "
        "e.g. 0 for |0⟩ and 1 for |1⟩.",
    )
    location: complex = Field(..., description="Complex location in the measurement space.")
    disallowed: bool = Field(
        False,
        description="Whether the state is allowed or not. If a measurement is classified "
        "to a state with disallowed=True, then the shot should be discarded.",
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
    winning normalised likelihood falls below the threshold. Those shots are labelled
    :data:`BG_LABEL` and discarded by :class:`~qat.ir.measure.PostSelect`.
    Default ``p_min=0.0`` disables the check entirely (zero overhead).

    **Disallowed states** — individual states can be marked ``disallowed=True`` on
    :class:`MLStateMap` for erasure checks and pre-selection; these are independent of
    ``p_min`` and handled by separate :class:`~qat.ir.measure.PostSelect` instructions.

    Runtime implementation:
    :func:`qat.runtime.post_processing.apply_discriminate_instruction`.

    :param states: Per-state IQ-plane centroids, labels and output values.
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
    states: list[MLStateMap] = Field(
        ..., description="List of MLStateMap objects representing possible states."
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
        "best-state normalised likelihood is below this threshold are labelled "
        f'"{BG_LABEL}" and discarded by post-selection. Must be in [0, 1]. '
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
    def _validate_no_reserved_bg_label(self):
        """Ensure no MLStateMap uses the reserved BG_LABEL ('|?>')."""
        if any(s.label == BG_LABEL for s in self.states):
            raise ValueError(
                f"The label {BG_LABEL!r} is reserved for the p_min background state "
                "and cannot be used as an MLStateMap label."
            )
        return self

    @model_validator(mode="after")
    def _validate_no_duplicate_labels(self):
        """Ensure all MLStateMap labels are unique.

        Duplicate labels would cause ``emit_granular_post_processing`` to silently
        overwrite states in the ``Demap.state_map`` dict comprehension.
        """
        counts = Counter(s.label for s in self.states)
        duplicates = sorted(lbl for lbl, n in counts.items() if n > 1)
        if duplicates:
            raise ValueError(
                f"MLStateMap labels must be unique; duplicates found: {duplicates}"
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

    @property
    def declared_states(self) -> set[str]:
        """All state labels known to this method.

        Includes labels from :attr:`states` and :data:`BG_LABEL` when
        ``p_min > 0``.

        :returns: Set of all recognised state label strings.
        """
        labels = {s.label for s in self.states}
        if self.p_min > 0.0:
            labels.add(BG_LABEL)
        return labels

    @property
    def disallowed_states(self) -> set[str]:
        """State labels marked ``disallowed=True``, plus ``BG_LABEL`` if ``p_min > 0``.

        :returns: Set of disallowed state label strings.
        """
        labels = {s.label for s in self.states if s.disallowed}
        if self.p_min > 0.0:
            labels.add(BG_LABEL)
        return labels


# Union type for use in Qubit model
PostProcessMethod = LinearMapToRealMethod | MaxLikelihoodMethod
