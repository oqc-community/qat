# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.frontend.auto import (
    AutoFrontend as AutoFrontend,
    AutoFrontendWithFlattenedIR as AutoFrontendWithFlattenedIR,
)
from qat.frontend.base import BaseFrontend as BaseFrontend
from qat.frontend.custom import CustomFrontend as CustomFrontend
from qat.frontend.fallthrough import FallthroughFrontend as FallthroughFrontend
from qat.frontend.qasm import Qasm2Frontend as Qasm2Frontend, Qasm3Frontend as Qasm3Frontend
from qat.frontend.qat_ir import QatFrontend as QatFrontend
from qat.frontend.qir import QIRFrontend as QIRFrontend

DefaultFrontend = AutoFrontend
