# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.engines.native import (
    ConnectionMixin as ConnectionMixin,
    NativeEngine as NativeEngine,
)
from qat.engines.zero import ZeroEngine as ZeroEngine

DefaultEngine = ZeroEngine
