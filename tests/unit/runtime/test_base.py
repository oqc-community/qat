# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.engines import ConnectionMixin, NativeEngine
from qat.purr.compiler.instructions import AcquireMode
from qat.purr.qatconfig import qatconfig
from qat.runtime import BaseRuntime, ResultsAggregator
from qat.runtime.connection import ConnectionMode
from qat.runtime.executables import AcquireData


class TestBaseRuntime:
    class MockConnectedEngine(NativeEngine, ConnectionMixin):
        is_connected: bool = False

        def execute(self, *args):
            pass

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

    @pytest.mark.parametrize("shots", [0, 254, 999, 1000, 1001, 2000, 2073])
    def test_number_of_batches_gives_correct_number_of_batches(self, shots):
        # tests that non-divisable batching gives warning.
        if shots % 1000 != 0:
            with pytest.warns():
                batches = BaseRuntime.number_of_batches(shots, 1000)
        else:
            batches = BaseRuntime.number_of_batches(shots, 1000)
        assert batches == np.ceil(shots / 1000)

    @pytest.mark.parametrize("shots", [0, 254, 999, 1000, 1001, 2000, 2073])
    def test_number_of_batches_with_no_max(self, shots):
        batches = BaseRuntime.number_of_batches(shots, 0)
        assert batches == 1

    def test_validate_total_shots_raises_value_error(self):
        qat_max_shots = qatconfig.MAX_REPEATS_LIMIT
        with pytest.raises(ValueError):
            BaseRuntime.validate_max_shots(qat_max_shots + 1)
        BaseRuntime.validate_max_shots(qat_max_shots)

    @pytest.mark.parametrize(
        "mode", [ConnectionMode.CONNECT_AT_BEGINNING, ConnectionMode.ALWAYS]
    )
    def test_connect_on_startup_with_connect_at_beginning(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = self.MockConnectedEngine()
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
        engine = self.MockConnectedEngine()
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
        engine = self.MockConnectedEngine()
        engine.connect()
        runtime = BaseRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True
        del runtime
        assert engine.is_connected == False

    @pytest.mark.parametrize("mode", [ConnectionMode.MANUAL, ConnectionMode.DEFAULT])
    def test_disconnect_on_exit_without_disconnect_at_end(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = self.MockConnectedEngine()
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
        engine = self.MockConnectedEngine()
        assert engine.is_connected == False
        runtime = BaseRuntime(engine, connection_mode=mode)
        runtime.connect_engine(ConnectionMode.CONNECT_BEFORE_EXECUTE)
        assert engine.is_connected == True

    @pytest.mark.parametrize("mode", [ConnectionMode.ALWAYS, ConnectionMode.MANUAL])
    def test_connect_without_connect_before_execute(self, monkeypatch, mode):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        engine = self.MockConnectedEngine()
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
        engine = self.MockConnectedEngine()
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
        engine = self.MockConnectedEngine()
        engine.connect()
        runtime = BaseRuntime(engine, connection_mode=mode)
        assert engine.is_connected == True
        runtime.disconnect_engine(ConnectionMode.DISCONNECT_AFTER_EXECUTE)
        assert engine.is_connected == True


class TestResultsAggregator:
    @pytest.mark.parametrize("max_shots", [None, 9999, 10000])
    def test_raw(self, max_shots):
        acquires = [
            AcquireData(
                length=254, position=0, mode=AcquireMode.RAW, output_variable="test"
            )
        ]
        aggregator = ResultsAggregator(acquires)
        for _ in range(10):
            aggregator.update({"test": np.ones((1000, 254))})
        results = aggregator.results(max_shots)
        max_shots = max_shots if max_shots else 10000
        assert len(results) == 1
        assert "test" in results
        assert np.shape(results["test"]) == (max_shots, 254)
        assert np.allclose(results["test"], 1.0)

    @pytest.mark.parametrize("max_shots", [None, 9999, 10000])
    def test_integrator(self, max_shots):
        acquires = [
            AcquireData(
                length=254,
                position=0,
                mode=AcquireMode.INTEGRATOR,
                output_variable="test",
            )
        ]
        aggregator = ResultsAggregator(acquires)
        for _ in range(10):
            aggregator.update({"test": np.ones(1000)})
        results = aggregator.results(max_shots)
        max_shots = max_shots if max_shots else 10000
        assert len(results) == 1
        assert "test" in results
        assert np.shape(results["test"]) == (max_shots,)
        assert np.allclose(results["test"], 1.0)

    @pytest.mark.parametrize("max_shots", [None, 9999, 10000])
    def test_scope(self, max_shots):
        acquires = [
            AcquireData(
                length=254, position=0, mode=AcquireMode.SCOPE, output_variable="test"
            )
        ]
        aggregator = ResultsAggregator(acquires)
        for _ in range(10):
            aggregator.update({"test": np.ones(254)})
        results = aggregator.results(max_shots)
        max_shots = max_shots if max_shots else 10000
        assert len(results) == 1
        assert "test" in results
        assert np.shape(results["test"]) == (254,)
        assert np.allclose(results["test"], 1.0)
