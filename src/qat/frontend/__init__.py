# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.frontend.base import BaseFrontend
from qat.frontend.fallthrough import FallthroughFrontend
from qat.frontend.custom import CustomFrontend
from qat.frontend.auto import AutoFrontend
from qat.frontend.qasm import Qasm2Frontend, Qasm3Frontend
from qat.frontend.qir import QIRFrontend

DefaultFrontend = AutoFrontend
