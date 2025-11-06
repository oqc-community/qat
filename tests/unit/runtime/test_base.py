# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.core.config.configure import get_config
from qat.runtime import BaseRuntime
from qat.runtime.connection import ConnectionMode

from tests.unit.utils.engines import MockConnectedEngine

qatconfig = get_config()


class TestBaseRuntime:
    @pytest.mark.parametrize(
        "mode", [ConnectionMode.CONNECT_AT_BEGINNING, ConnectionMode.ALWAYS]
    )
    def test_connect_on_startup_with_connect_at_beginning(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = MockConnectedEngine()
        assert engine.is_connected == False
        # Ignore F841 as the connection is automatically closed on __del__
        runtime = BaseRuntime(engine, connection_mode=mode)  # noqa: F841
        assert engine.is_connected == True

    @pytest.mark.parametrize(
        "mode",
        [
            ConnectionMode.ALWAYS_ON_EXECUTE,
            ConnectionMode.MANUAL,
            ConnectionMode.DEFAULT,
        ],
    )
    def test_connect_on_startup_without_connect_at_beginning(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = MockConnectedEngine()
        assert engine.is_connected == False
        BaseRuntime(engine, connection_mode=mode)
        assert engine.is_connected == False

    @pytest.mark.parametrize(
        "mode",
        [
            ConnectionMode.DISCONNECT_AT_END,
            ConnectionMode.ALWAYS,
            ConnectionMode.ALWAYS_ON_EXECUTE,
        ],
    )
    def test_disconnect_on_exit_with_disconnect_at_end(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = MockConnectedEngine()
        engine.connect()
        runtime = BaseRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True
        del runtime
        assert engine.is_connected == False

    @pytest.mark.parametrize("mode", [ConnectionMode.MANUAL, ConnectionMode.DEFAULT])
    def test_disconnect_on_exit_without_disconnect_at_end(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = MockConnectedEngine()
        engine.connect()
        runtime = BaseRuntime(engine, connection_mode=mode)
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
    def test_connect_with_connect_before_execute(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = MockConnectedEngine()
        assert engine.is_connected == False
        runtime = BaseRuntime(engine, connection_mode=mode)
        runtime.connect_engine(ConnectionMode.CONNECT_BEFORE_EXECUTE)
        assert engine.is_connected == True

    @pytest.mark.parametrize("mode", [ConnectionMode.ALWAYS, ConnectionMode.MANUAL])
    def test_connect_without_connect_before_execute(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = MockConnectedEngine()
        runtime = BaseRuntime(engine, connection_mode=mode)
        engine.disconnect()
        assert engine.is_connected == False
        runtime.connect_engine(ConnectionMode.CONNECT_BEFORE_EXECUTE)
        assert engine.is_connected == False

    @pytest.mark.parametrize(
        "mode", [ConnectionMode.DISCONNECT_AFTER_EXECUTE, ConnectionMode.DEFAULT]
    )
    def test_disconnect_with_disconnect_after_execute(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = MockConnectedEngine()
        engine.connect()
        runtime = BaseRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True
        runtime.disconnect_engine(ConnectionMode.DISCONNECT_AFTER_EXECUTE)
        assert engine.is_connected == False

    @pytest.mark.parametrize(
        "mode",
        [
            ConnectionMode.ALWAYS,
            ConnectionMode.MANUAL,
            ConnectionMode.ALWAYS_ON_EXECUTE,
        ],
    )
    def test_disconnect_without_disconnect_after_execute(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = MockConnectedEngine()
        engine.connect()
        runtime = BaseRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True
        runtime.disconnect_engine(ConnectionMode.DISCONNECT_AFTER_EXECUTE)
        assert engine.is_connected == True
