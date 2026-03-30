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

from enum import Enum
from typing import Literal

from pydantic import Field

from qat.utils.pydantic import NoExtraFieldsModel


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
    """Model for linear mapping post-processing method."""

    method: Literal[MethodIndicator.LINEAR_MAP_COMPLEX_TO_REAL] = Field(
        MethodIndicator.LINEAR_MAP_COMPLEX_TO_REAL,
        description="Discriminator for linear map complex to real method.",
    )
    mean_z_map_args: list[complex] = Field(
        max_length=2,
        min_length=2,
        default=[1 + 0j, 0j],
        description="Arguments for the linear map.",
    )


class MLStateMap(NoExtraFieldsModel):
    """Model representing a mapping from a quantum state to a value and location."""

    state: str = Field(..., description="State label (e.g., '0', '1').")
    val: int = Field(
        ..., description="Integer value associated with the state, or -1 if invalid"
    )
    location: complex = Field(..., description="Complex location in the measurement space.")


class MaxLikelihoodMethod(MethodBase):
    """Model for maximum likelihood post-processing method."""

    method: Literal[MethodIndicator.MAX_LIKELIHOOD] = Field(
        MethodIndicator.MAX_LIKELIHOOD,
        description="Discriminator for maximum likelihood method.",
    )
    noise_est: float = Field(
        ..., description="Estimated noise parameter for likelihood calculation."
    )
    states: list[MLStateMap] = Field(
        ..., description="List of MLStateMap objects representing possible states."
    )


# Union type for use in Qubit model
PostProcessMethod = LinearMapToRealMethod | MaxLikelihoodMethod
