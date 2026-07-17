# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Exception hierarchy for the experimental materialisation boundary."""

from __future__ import annotations

from typing import Any


class MaterialisationError(RuntimeError):
    """Base class for materialisation boundary errors."""

    code = "materialisation_error"

    def __init__(
        self,
        message: str,
        *,
        source_type: str | None = None,
        source_version: str | None = None,
        path: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.source_type = source_type
        self.source_version = source_version
        self.path = path
        self.details = details or {}
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Return a structured representation of the error context."""

        return {
            "code": self.code,
            "message": self.message,
            "source_type": self.source_type,
            "source_version": self.source_version,
            "path": self.path,
            "details": self.details,
        }


class UnsupportedSourceError(MaterialisationError):
    """Raised when a source type is unsupported."""

    code = "unsupported_source"

    @classmethod
    def for_source(
        cls,
        *,
        source_type: str,
        source_version: str | None,
        supported_sources: tuple[str, ...],
        cause: Exception | None = None,
    ) -> UnsupportedSourceError:
        """Create a standard unsupported-source error payload."""

        return cls(
            f"Unsupported source type: {source_type}",
            source_type=source_type,
            source_version=source_version,
            details={"supported_sources": supported_sources},
            cause=cause,
        )


class UnsupportedSourceVersionError(MaterialisationError):
    """Raised when a source version is unsupported."""

    code = "unsupported_source_version"

    @classmethod
    def for_version(
        cls,
        *,
        source_type: str,
        source_version: str,
        supported_versions: tuple[str, ...],
    ) -> UnsupportedSourceVersionError:
        """Create a standard unsupported-source-version error."""

        return cls(
            (f"Unsupported source version '{source_version}' for source '{source_type}'."),
            source_type=source_type,
            source_version=source_version,
            details={"supported_versions": supported_versions},
        )


class SourceValidationError(MaterialisationError):
    """Raised when ingress DTO validation fails."""

    code = "source_validation_error"


class SourceConsistencyError(MaterialisationError):
    """Raised when cross-entity consistency validation fails."""

    code = "source_consistency_error"


class SourceIntegrityError(MaterialisationError):
    """Raised when authenticity or integrity checks fail."""

    code = "source_integrity_error"

    @classmethod
    def for_check_failure(
        cls,
        *,
        source_type: str,
        source_version: str,
        check: str,
        cause: Exception | None = None,
    ) -> SourceIntegrityError:
        """Create a standard integrity-check failure error payload."""

        return cls(
            (
                f"Source integrity verification failed during '{check}' "
                f"for source '{source_type}' version '{source_version}'."
            ),
            source_type=source_type,
            source_version=source_version,
            details={"check": check},
            cause=cause,
        )


class MaterialisationValidationError(MaterialisationError):
    """Raised when materialiser-stage adaptation or validation fails."""

    code = "materialisation_validation_error"


class MaterialisationConsistencyError(MaterialisationError):
    """Raised when materialiser-stage state is contradictory or impossible."""

    code = "materialisation_consistency_error"


class MaterialisationIntegrityError(MaterialisationError):
    """Raised when materialiser-stage payload data is malformed."""

    code = "materialisation_integrity_error"
