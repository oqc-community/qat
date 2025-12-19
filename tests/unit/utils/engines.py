# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
from pydantic import BaseModel, validate_call

from qat.engines.model import RequiresHardwareModelMixin
from qat.engines.native import ConnectionMixin, NativeEngine
from qat.engines.zero import ZeroEngine
from qat.executables import Executable


class CblamConfig(BaseModel):
    host: str
    username: str | None = None
    token: str | None = None
    timeout: int | None = None


class InitableEngine(NativeEngine):
    @validate_call
    def __init__(
        self,
        x="default_x",
        y="default_y",
        z: int = 3,
        cblam: CblamConfig | None = None,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.cblam = cblam

    def execute(self, package: Executable) -> dict[str, np.ndarray]:
        return {}


class BrokenEngine(NativeEngine):
    @validate_call
    def __init__(
        self,
        x="default_x",
        y="default_y",
        z: int = 3,
        cblam: CblamConfig | None = None,
    ):
        raise ValueError("This engine is broken intentionally.")

    def execute(self, package: Executable) -> dict[str, np.ndarray]:
        return {}


class MockEngineWithModel(ZeroEngine, RequiresHardwareModelMixin):
    """A mock engine that takes a model."""

    def __init__(self, model):
        self._model = model
        self.num_changes: int = 0

    def _update_model(self, model):
        self._model = model
        self.num_changes += 1


class MockConnectedEngine(NativeEngine, ConnectionMixin):
    def __init__(self):
        self._is_connected = False
        self.connections = 0

    def execute(self, *args, **kwargs):
        return {}

    def connect(self):
        self._is_connected = True
        self.connections += 1

    def disconnect(self):
        self._is_connected = False

    @property
    def is_connected(self):
        return self._is_connected
