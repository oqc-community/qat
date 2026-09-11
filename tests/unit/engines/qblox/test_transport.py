# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import asyncio

import pytest

from qat.engines.qblox import transport as engines_transport
from qat.purr.backends.qblox import transport as purr_transport

# The engines and purr backends ship identical copies of the transport cleanup helpers,
# so every behaviour is exercised against both modules.
TRANSPORT_MODULES = [
    pytest.param(engines_transport, id="engines"),
    pytest.param(purr_transport, id="purr"),
]


class _FakeModule:
    def __init__(self, loop=None):
        self._loop = loop


class _FakeTransport:
    def __init__(self, loop=None, modules=None):
        self._loop = loop
        self._modules = {} if modules is None else modules


class _FakeDriver:
    def __init__(self, transport):
        self._transport = transport
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


@pytest.fixture
def make_loop():
    """Yield a factory for event loops and close any that survive the test."""
    created = []

    def _make():
        loop = asyncio.new_event_loop()
        created.append(loop)
        return loop

    yield _make

    for loop in created:
        if not loop.is_closed():
            loop.close()


@pytest.mark.parametrize("transport", TRANSPORT_MODULES)
class TestCollectTransportLoops:
    def test_collects_cluster_and_module_loops(self, transport, make_loop):
        cluster_loop = make_loop()
        module_loop = make_loop()
        driver = _FakeDriver(
            _FakeTransport(loop=cluster_loop, modules={"1": _FakeModule(module_loop)})
        )

        assert transport._collect_transport_loops(driver) == {cluster_loop, module_loop}

    def test_deduplicates_shared_loop(self, transport, make_loop):
        shared = make_loop()
        driver = _FakeDriver(
            _FakeTransport(loop=shared, modules={"1": _FakeModule(shared)})
        )

        assert transport._collect_transport_loops(driver) == {shared}

    def test_returns_empty_when_transport_absent(self, transport):
        assert transport._collect_transport_loops(_FakeDriver(None)) == set()

    def test_skips_none_modules_and_non_loops(self, transport, make_loop):
        cluster_loop = make_loop()
        driver = _FakeDriver(
            _FakeTransport(
                loop=cluster_loop,
                modules={"1": None, "2": _FakeModule(None), "3": _FakeModule("nope")},
            )
        )

        assert transport._collect_transport_loops(driver) == {cluster_loop}


@pytest.mark.parametrize("transport", TRANSPORT_MODULES)
class TestCloseCluster:
    def test_none_driver_is_noop(self, transport):
        transport.close_cluster(None)  # must not raise

    def test_closes_driver_and_idle_loops(self, transport, make_loop):
        cluster_loop = make_loop()
        module_loop = make_loop()
        driver = _FakeDriver(
            _FakeTransport(loop=cluster_loop, modules={"1": _FakeModule(module_loop)})
        )

        transport.close_cluster(driver)

        assert driver.close_calls == 1
        assert cluster_loop.is_closed()
        assert module_loop.is_closed()

    def test_skips_already_closed_loop(self, transport, make_loop):
        closed_loop = make_loop()
        closed_loop.close()
        driver = _FakeDriver(_FakeTransport(loop=closed_loop))

        transport.close_cluster(driver)  # must not raise on an already-closed loop

        assert driver.close_calls == 1

    def test_loops_closed_even_when_driver_close_raises(self, transport, make_loop):
        cluster_loop = make_loop()

        class _BoomDriver(_FakeDriver):
            def close(self):
                super().close()
                raise RuntimeError("boom")

        driver = _BoomDriver(_FakeTransport(loop=cluster_loop))

        with pytest.raises(RuntimeError, match="boom"):
            transport.close_cluster(driver)

        assert driver.close_calls == 1
        assert cluster_loop.is_closed()
