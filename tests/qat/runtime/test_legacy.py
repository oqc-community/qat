# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.purr.backends.live import LiveDeviceEngine
from qat.runtime import LegacyRuntime
from qat.runtime.connection import ConnectionMode


class TestLegacyRuntime:

    class MockConnectedEngine(LiveDeviceEngine):
        is_connected: bool = False

        def __init__(self):
            pass

        def execute(self, *args):
            pass

        def startup(self):
            self.is_connected = True

        def shutdown(self):
            self.is_connected = False

    @pytest.mark.parametrize(
        "mode", [ConnectionMode.CONNECT_AT_BEGINNING, ConnectionMode.ALWAYS]
    )
    def test_connect_with_connect_at_beginning(self, mode):
        engine = self.MockConnectedEngine()
        assert engine.is_connected == False
        runtime = LegacyRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True

    @pytest.mark.parametrize(
        "mode",
        [ConnectionMode.ALWAYS_ON_EXECUTE, ConnectionMode.MANUAL, ConnectionMode.DEFAULT],
    )
    def test_connect_without_connect_at_beginning(self, mode):
        engine = self.MockConnectedEngine()
        assert engine.is_connected == False
        runtime = LegacyRuntime(engine, connection_mode=mode)
        assert engine.is_connected == False

    @pytest.mark.parametrize(
        "mode",
        [
            ConnectionMode.DISCONNECT_AT_END,
            ConnectionMode.ALWAYS,
            ConnectionMode.ALWAYS_ON_EXECUTE,
        ],
    )
    def test_disconnect_with_disconnect_at_end(self, mode):
        engine = self.MockConnectedEngine()
        engine.startup()
        runtime = LegacyRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True
        del runtime
        assert engine.is_connected == False

    @pytest.mark.parametrize("mode", [ConnectionMode.MANUAL, ConnectionMode.DEFAULT])
    def test_disconnect_without_disconnect_at_end(self, mode):
        engine = self.MockConnectedEngine()
        engine.startup()
        runtime = LegacyRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True
        del runtime
        assert engine.is_connected == True

    @pytest.mark.parametrize(
        "mode",
        [
            ConnectionMode.CONNECT_BEFORE_EXECUTE,
            ConnectionMode.ALWAYS_ON_EXECUTE,
            ConnectionMode.DEFAULT,
        ],
    )
    def test_connect_with_connect_before_execute(self, mode):
        engine = self.MockConnectedEngine()
        assert engine.is_connected == False
        runtime = LegacyRuntime(engine, connection_mode=mode)
        runtime.connect_engine(ConnectionMode.CONNECT_BEFORE_EXECUTE)
        assert engine.is_connected == True

    @pytest.mark.parametrize("mode", [ConnectionMode.ALWAYS, ConnectionMode.MANUAL])
    def test_connect_without_connect_before_execute(self, mode):
        engine = self.MockConnectedEngine()
        runtime = LegacyRuntime(engine, connection_mode=mode)
        engine.shutdown()
        assert engine.is_connected == False
        runtime.connect_engine(ConnectionMode.CONNECT_BEFORE_EXECUTE)
        assert engine.is_connected == False

    @pytest.mark.parametrize(
        "mode", [ConnectionMode.DISCONNECT_AFTER_EXECUTE, ConnectionMode.DEFAULT]
    )
    def test_disconnect_with_disconnect_after_execute(self, mode):
        engine = self.MockConnectedEngine()
        engine.startup()
        runtime = LegacyRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True
        runtime.disconnect_engine(ConnectionMode.DISCONNECT_AFTER_EXECUTE)
        assert engine.is_connected == False

    @pytest.mark.parametrize(
        "mode",
        [ConnectionMode.ALWAYS, ConnectionMode.MANUAL, ConnectionMode.ALWAYS_ON_EXECUTE],
    )
    def test_disconnect_without_disconnect_after_execute(self, mode):
        engine = self.MockConnectedEngine()
        engine.startup()
        runtime = LegacyRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True
        runtime.disconnect_engine(ConnectionMode.DISCONNECT_AFTER_EXECUTE)
        assert engine.is_connected == True
