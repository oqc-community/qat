# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
"""Close a Qblox cluster together with the asyncio loops its transports leak.

``qblox-instruments`` 1.3.x runs its transport and SCPI layers on dedicated
:mod:`asyncio` event loops, one per cluster transport and one per module transport.
``Cluster.close()`` stops the connection and discards the transport without closing those
loops, so the loops and their self-pipe sockets survive until garbage collection. When
collection happens the interpreter emits a :class:`ResourceWarning`, which pytest promotes
to an error under its ``filterwarnings`` configuration and attributes to whichever test is
running at the time.

The helpers are plain module-level functions so any owner of a ``Cluster`` can reclaim its
transports by calling :func:`close_cluster` on the driver. This is a self-contained copy of
the engines helpers so the legacy ``purr`` backend does not depend on ``qat.engines``.
"""

from asyncio import AbstractEventLoop

from qblox_instruments import Cluster

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def close_cluster(driver: Cluster | None) -> None:
    """Close a Qblox cluster driver and the event loops leaked by its transports.

    The transport loops are collected before ``driver.close()`` discards them and closed
    afterwards, so no loops or sockets outlive the driver.

    :param driver: The ``qblox_instruments.Cluster`` driver to close, or ``None``.
    """
    if driver is None:
        return

    loops = _collect_transport_loops(driver)
    try:
        driver.close()
    finally:
        for loop in loops:
            try:
                if loop.is_running() or loop.is_closed():
                    continue
                loop.close()
            except Exception as e:  # noqa: BLE001 - teardown must not raise
                log.warning(f"Failed to close Qblox transport event loop: {e}")


def _collect_transport_loops(driver: Cluster) -> set[AbstractEventLoop]:
    """Return the event loops owned by a cluster driver's transports.

    :param driver: A ``qblox_instruments.Cluster`` driver.
    :returns: The cluster transport loop and every module transport loop.
    """
    loops: set[AbstractEventLoop] = set()

    transport = getattr(driver, "_transport", None)
    if transport is None:
        return loops

    loop = getattr(transport, "_loop", None)
    if isinstance(loop, AbstractEventLoop):
        loops.add(loop)

    for module in getattr(transport, "_modules", {}).values():
        if module is None:
            continue
        module_loop = getattr(module, "_loop", None)
        if isinstance(module_loop, AbstractEventLoop):
            loops.add(module_loop)

    return loops
