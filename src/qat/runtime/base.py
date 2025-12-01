# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import abc
from contextlib import contextmanager

from qat.core.pass_base import PassManager
from qat.engines import ConnectionMixin, NativeEngine
from qat.model.target_data import TargetData
from qat.runtime.connection import ConnectionMode
from qat.runtime.passes.transform import (
    AcquisitionPostprocessing,
    AssignResultsTransform,
    InlineResultsProcessingTransform,
)


class BaseRuntime(abc.ABC):
    """Provides a Base class to build on for runtimes of varying complexities.

    A runtime provides the means to execute quantum programs. It can take on various
    responsibilities, including interfacing the execution engine and post-processing of
    results. Runtimes are designed to fit a specific purpose. For example, the
    :class:`SimpleRuntime` provides the means to execute already compiled programs
    :class:`Executable`. In the future, there will be support for hybrid runtimes that take
    on both compilation and execution responsibilities.
    """

    def __init__(
        self,
        engine: NativeEngine,
        results_pipeline: PassManager | None = None,
        connection_mode: ConnectionMode = ConnectionMode.DEFAULT,
    ):
        """
        :param engine: The execution engine for a target machine.
        :param results_pipeline: Optionally provided a pipeline for results processing. If
            not provided, a default pipeline is provided.
        :param connection_mode: Specifies how the connection is maintained.
        """
        self.engine = engine
        self.connection_mode = connection_mode
        self.connect_engine(ConnectionMode.CONNECT_AT_BEGINNING)

        if not results_pipeline:
            results_pipeline = self.default_pipeline()
        self.results_pipeline = results_pipeline

    @abc.abstractmethod
    def execute(package, *args, **kwargs): ...

    def default_pipeline(self, target_data: TargetData | None = None):
        return (
            PassManager()
            | AcquisitionPostprocessing(target_data)
            | InlineResultsProcessingTransform()
            | AssignResultsTransform()
        )

    def connect_engine(self, flag: ConnectionMode) -> bool | None:
        """Connect the engine according to the connection mode."""
        if not isinstance(self.engine, ConnectionMixin):
            return None

        if flag in self.connection_mode:
            if not self.engine.is_connected:
                self.engine.connect()
        return self.engine.is_connected

    def disconnect_engine(self, flag: ConnectionMode) -> bool | None:
        """Disconnect the engine according to the connection mode."""
        if not isinstance(self.engine, ConnectionMixin):
            return None

        if flag in self.connection_mode:
            if self.engine.is_connected:
                self.engine.disconnect()
        return self.engine.is_connected

    @contextmanager
    def _hold_connection(self):
        """Context manager to establish a connection which is held for the duration of the
        execution, according to the connection mode provided.

        If required, the engine will always be disconnected at the end of the context,
        even if execution is interrupted by an exception.
        """

        self.connect_engine(ConnectionMode.CONNECT_BEFORE_EXECUTE)
        try:
            yield self
        finally:
            self.disconnect_engine(ConnectionMode.DISCONNECT_AFTER_EXECUTE)

    def __del__(self):
        self.disconnect_engine(ConnectionMode.DISCONNECT_AT_END)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.disconnect_engine(ConnectionMode.DISCONNECT_AT_END)
