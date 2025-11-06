# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.runtime.base import BaseRuntime as BaseRuntime
from qat.runtime.legacy import LegacyRuntime as LegacyRuntime
from qat.runtime.simple import SimpleRuntime as SimpleRuntime

DefaultRuntime = SimpleRuntime
