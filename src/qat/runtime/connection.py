# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from enum import Flag, auto


class ConnectionMode(Flag):
    """The connection mode can be used to specify how a Runtime should handle connections.

    The method for connection handling will depend in the context in which the Runtime is
    used. For example, if a user only requires to send jobs sporadically, then a connection
    that is created before execution and ended after execution might be desired. On the
    contrary, if there is a stream of frequent jobs, the connection likely wants to be held
    indefinitely.

    There are different flags that can be specified:

    #. `ConnectionMode.CONNECT_BEFORE_EXECUTE`: Specifies that the Runtime should attempt
        to connect the engine each time execute is called.
    #. `ConnectionMode.DISCONNECT_AFTER_EXECUTE`: Specifies that the Runtime should attempt
        to disconnect the engine after each execute.
    #. `ConnectionMode.CONNECT_AT_BEGINNING`: Specifies that the Runtime should attempt to
        connect the engine as soon as its instantiated.
    #. `ConnectionMode.DISCONNECT_AT_END`: Specifies that the Runtime should attempt to
        disconnect the engine when the Runtime is deleted (or at the end of a context
        manager).

    Some predefined composite modes:

    #. `ConnectionMode.MANUAL`: Connection handling is done externally: the Runtime will
        make no attempt to manage it.
    #. `ConnectionMode.DEFAULT`: The connection will be created before each execution and
        disconnected immediately after the execution is complete.
    #. `ConnectionMode.ALWAYS`: A connection is created at instantiation of the Runtime,
        and is kept throughout the lifetime of the Runtime. It will not try to connect at
        execution.
    #. `ConnectionMode.ALWAYS_ON_EXECUTE`: A connection is attempted each time execution is
        called. However, the Runtime will not attempt to disconnect after execution is
        complete. This allows for external disconnection of the engine during e.g. idle
        periods, and for the connection to be restored when needed.

    .. warning::
        If a Runtime is not used with `ConnectionMode.DEFAULT` or
        `ConnectionMode.ALWAYS_ON_EXECUTE`, then external interference can cause undesired
        behaviour. Furthermore, the specifics of each mode might have slight variations
        depending on which Runtime is used (which should be well documented).
    """

    # Individual flags
    CONNECT_BEFORE_EXECUTE = auto()
    DISCONNECT_AFTER_EXECUTE = auto()
    CONNECT_AT_BEGINNING = auto()
    DISCONNECT_AT_END = auto()

    # Some pre-defined composite options
    MANUAL = auto()
    DEFAULT = CONNECT_BEFORE_EXECUTE | DISCONNECT_AFTER_EXECUTE
    ALWAYS = CONNECT_AT_BEGINNING | DISCONNECT_AT_END
    ALWAYS_ON_EXECUTE = CONNECT_BEFORE_EXECUTE | DISCONNECT_AT_END
