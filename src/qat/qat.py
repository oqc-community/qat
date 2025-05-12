# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd
import warnings

warnings.simplefilter("always", DeprecationWarning)
warnings.warn(
    "module 'qat.qat' is deprecated, please use 'qat.purr.qat instead.'",
    DeprecationWarning,
    stacklevel=2,
)

from qat.purr.qat import *  # noqa: E402, F403
