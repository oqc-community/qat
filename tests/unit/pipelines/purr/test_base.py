# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest

from qat.pipelines.legacy.base import LegacyPipelineConfig
from qat.runtime.connection import ConnectionMode


class TestLegacyPipelineConfig:
    @pytest.mark.parametrize(
        "connection_mode", ["MANUAL", "DEFAULT", "ALWAYS", "ALWAYS_ON_EXECUTE"]
    )
    def test_with_string_connection_mode(self, connection_mode):
        config = LegacyPipelineConfig(name="test_pipeline", connection_mode=connection_mode)
        assert isinstance(config.connection_mode, ConnectionMode)
        assert config.connection_mode.name == connection_mode
