# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.engines.waveform_v1 import EchoEngine
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import AcquireMode
from qat.purr.qatconfig import qatconfig
from qat.runtime import BaseRuntime, ResultsAggregator
from qat.runtime.executables import AcquireDataStruct


class TestBaseRuntime:

    def test_invalid_engine_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        model = get_default_echo_hardware()
        with pytest.raises(ValueError):
            BaseRuntime(model)

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

    def test_context_manager(self, monkeypatch):
        def set_connected(self, val):
            self.connected = val

        monkeypatch.setattr(BaseRuntime, "__abstractmethods__", set())
        monkeypatch.setattr(EchoEngine, "startup", lambda x: set_connected(x, True))
        monkeypatch.setattr(EchoEngine, "shutdown", lambda x: set_connected(x, False))

        # doesn't connect automatically
        engine = EchoEngine()
        assert not hasattr(engine, "connected")

        # doesn't connect with init
        BaseRuntime(engine)
        assert not hasattr(engine, "connected")

        # connects with context manager
        with BaseRuntime(engine):
            assert hasattr(engine, "connected")
            assert engine.connected == True

        # disconnects at the end
        assert hasattr(engine, "connected")
        assert engine.connected == False


class TestResultsAggregator:

    @pytest.mark.parametrize("max_shots", [None, 9999, 10000])
    def test_raw(self, max_shots):
        acquires = [
            AcquireDataStruct(
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
            AcquireDataStruct(
                length=254, position=0, mode=AcquireMode.INTEGRATOR, output_variable="test"
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
            AcquireDataStruct(
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
