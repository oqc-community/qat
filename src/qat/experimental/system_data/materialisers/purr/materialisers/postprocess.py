# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Post-process method builders for PuRR materialisation."""

from typing import Any

from qat.experimental.system_data.canonical.schema import (
    LinearMapToRealMethodData,
    MaxLikelihoodDiscriminateParams,
    MaxLikelihoodMethodData,
    PostProcessMethodData,
)
from qat.experimental.system_data.materialisers.errors import MaterialisationIntegrityError
from qat.experimental.system_data.materialisers.purr.materialisers.common import _as_complex
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)

_POST_PROCESS_METHOD_LINEAR = "linear_map_complex_to_real"
_POST_PROCESS_METHOD_MAX_LIKELIHOOD = "max_likelihood"


def _raise_post_process_integrity_error(
    message: str,
    *,
    path: str,
    details: dict[str, Any],
) -> None:
    """Raise a materialiser-stage integrity error for post-process payload issues."""

    raise MaterialisationIntegrityError(
        message,
        source_type="purr",
        path=path,
        details=details,
    )


def _parse_linear_method_from_mean_z_map_args(
    mean_z_map_args: Any,
) -> LinearMapToRealMethodData | None:
    """Parse legacy linear discriminator payload into canonical data."""

    if mean_z_map_args is None:
        return None

    if not isinstance(mean_z_map_args, list | tuple) or len(mean_z_map_args) != 2:
        _raise_post_process_integrity_error(
            "Invalid mean_z_map_args payload shape.",
            path="$.mean_z_map_args",
            details={"value": mean_z_map_args},
        )

    if not all(isinstance(value, int | float | complex) for value in mean_z_map_args):
        _raise_post_process_integrity_error(
            "mean_z_map_args values must be numeric or complex.",
            path="$.mean_z_map_args",
            details={"value": mean_z_map_args},
        )

    return LinearMapToRealMethodData(
        mean_z_map_args=(
            _as_complex(mean_z_map_args[0], default=1.0 + 0.0j),
            _as_complex(mean_z_map_args[1], default=0.0 + 0.0j),
        )
    )


def _normalise_ml_states(
    raw_states: Any,
) -> tuple[tuple[int, MaxLikelihoodDiscriminateParams], ...] | None:
    """Normalise max-likelihood state-centroid map into canonical tuples."""

    if raw_states is None:
        return None
    if not isinstance(raw_states, dict) or not raw_states:
        _raise_post_process_integrity_error(
            "Max-likelihood states must be a non-empty mapping.",
            path="$.post_process_method.states",
            details={"value": raw_states},
        )

    states: list[tuple[int, MaxLikelihoodDiscriminateParams]] = []
    for key, value in raw_states.items():
        try:
            state_key = int(key)
        except (TypeError, ValueError):
            _raise_post_process_integrity_error(
                "Max-likelihood state keys must be integer-like.",
                path="$.post_process_method.states",
                details={"key": key},
            )

        if not isinstance(value, dict):
            _raise_post_process_integrity_error(
                "Max-likelihood state entries must be mappings.",
                path=f"$.post_process_method.states.{key}",
                details={"value": value},
            )

        location = value.get("location")
        if not isinstance(location, int | float | complex):
            _raise_post_process_integrity_error(
                "Max-likelihood state location must be numeric or complex.",
                path=f"$.post_process_method.states.{key}.location",
                details={"value": location},
            )

        label = value.get("label")
        if label is not None and not isinstance(label, str):
            _raise_post_process_integrity_error(
                "Max-likelihood state label must be a string when provided.",
                path=f"$.post_process_method.states.{key}.label",
                details={"value": label},
            )

        states.append(
            (
                state_key,
                MaxLikelihoodDiscriminateParams(
                    location=_as_complex(location, default=0.0 + 0.0j),
                    label=label,
                ),
            )
        )

    return tuple(sorted(states, key=lambda item: item[0]))


def _normalise_ml_transform(
    raw_transform: Any,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Normalise an optional 2x2 affine transform matrix."""

    if raw_transform is None:
        return None
    if (
        not isinstance(raw_transform, list | tuple)
        or len(raw_transform) != 2
        or not all(isinstance(row, list | tuple) and len(row) == 2 for row in raw_transform)
        or not all(isinstance(value, int | float) for row in raw_transform for value in row)
    ):
        _raise_post_process_integrity_error(
            "Max-likelihood transform must be a 2x2 numeric matrix.",
            path="$.post_process_method.transform",
            details={"value": raw_transform},
        )
    return (
        (float(raw_transform[0][0]), float(raw_transform[0][1])),
        (float(raw_transform[1][0]), float(raw_transform[1][1])),
    )


def _normalise_ml_offset(raw_offset: Any) -> tuple[float, float] | None:
    """Normalise an optional 2-element IQ offset vector."""

    if raw_offset is None:
        return None
    if (
        not isinstance(raw_offset, list | tuple)
        or len(raw_offset) != 2
        or not all(isinstance(value, int | float) for value in raw_offset)
    ):
        _raise_post_process_integrity_error(
            "Max-likelihood offset must be a 2-element numeric vector.",
            path="$.post_process_method.offset",
            details={"value": raw_offset},
        )
    return (float(raw_offset[0]), float(raw_offset[1]))


def _parse_max_likelihood_method(
    raw_method: dict[str, Any],
) -> MaxLikelihoodMethodData | None:
    """Parse max-likelihood discriminator payload into canonical data.

    :raises MaterialisationIntegrityError: If explicit max-likelihood payload fields are
        present but malformed.
    """

    states = _normalise_ml_states(raw_method.get("states"))
    transform = _normalise_ml_transform(raw_method.get("transform"))
    offset = _normalise_ml_offset(raw_method.get("offset"))

    if states is None:
        _raise_post_process_integrity_error(
            "Max-likelihood method requires a non-empty states mapping.",
            path="$.post_process_method.states",
            details={"value": raw_method.get("states")},
        )

    noise_est = raw_method.get("noise_est", 1.0)
    p_min = raw_method.get("p_min", 0.0)
    if not isinstance(noise_est, int | float) or not isinstance(p_min, int | float):
        _raise_post_process_integrity_error(
            "Max-likelihood noise_est and p_min must be numeric.",
            path="$.post_process_method",
            details={"noise_est": noise_est, "p_min": p_min},
        )

    return MaxLikelihoodMethodData(
        states=states,
        noise_est=float(noise_est),
        p_min=float(p_min),
        transform=transform,
        offset=offset,
    )


def _parse_post_process_method_payload(raw_method: Any) -> PostProcessMethodData | None:
    """Parse modern post_process_method payload into canonical data."""

    if raw_method is None:
        return None
    if not isinstance(raw_method, dict):
        _raise_post_process_integrity_error(
            "post_process_method must be a mapping when provided.",
            path="$.post_process_method",
            details={"value": raw_method},
        )

    method = raw_method.get("method")
    if not isinstance(method, str):
        _raise_post_process_integrity_error(
            "post_process_method.method must be a string.",
            path="$.post_process_method.method",
            details={"value": method},
        )
    if method == _POST_PROCESS_METHOD_LINEAR:
        return _parse_linear_method_from_mean_z_map_args(raw_method.get("mean_z_map_args"))
    if method == _POST_PROCESS_METHOD_MAX_LIKELIHOOD:
        return _parse_max_likelihood_method(raw_method)
    _raise_post_process_integrity_error(
        "Unsupported post_process_method method value.",
        path="$.post_process_method.method",
        details={"value": method},
    )


def _build_post_process_method(qubit_payload: dict[str, Any], pulse_key: str):
    """Build canonical post-processing metadata when the PuRR payload exposes it.

    Returns ``None`` only when no post-process method is applicable for the mode.
    If a modern ``post_process_method`` payload is explicitly provided but malformed,
    a materialiser integrity error is raised rather than silently falling back.

    :raises MaterialisationIntegrityError: If explicit post-process payload fields are
        malformed.
    """

    if pulse_key != "acquire":
        return None

    qubit_id = qubit_payload.get("id", "<unknown>")

    has_legacy_mean_z = qubit_payload.get("mean_z_map_args") is not None
    has_new_method = qubit_payload.get("post_process_method") is not None

    if has_new_method:
        new_method = _parse_post_process_method_payload(
            qubit_payload.get("post_process_method")
        )
        if new_method is not None:
            if has_legacy_mean_z:
                logger.warning(
                    "Qubit %s defines both post_process_method and mean_z_map_args. "
                    "Using post_process_method and ignoring legacy mean_z_map_args.",
                    qubit_id,
                )
            return new_method

    legacy_method = _parse_linear_method_from_mean_z_map_args(
        qubit_payload.get("mean_z_map_args")
    )
    if legacy_method is not None:
        return legacy_method

    return None
