# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import pytest

from qat.exceptions.compilation import CompilationError, UnsupportedProgramError


class TestCompilationErrors:
    def test_compilation_error_is_value_error(self):
        """Callers that catch ValueError from compilation must still catch it."""
        with pytest.raises(ValueError, match="bad program"):
            raise CompilationError("bad program")

    def test_unsupported_program_error_is_compilation_error(self):
        with pytest.raises(CompilationError, match="too long"):
            raise UnsupportedProgramError("too long")
