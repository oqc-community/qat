# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import warnings

warnings.simplefilter("default", DeprecationWarning)
warnings.warn(
    "module 'qat.purr.utils.serializer' is deprecated, please use "
    "'compiler_config.serialiser instead.'",
    DeprecationWarning,
)

from compiler_config.serialiser import *  # fmt: skip
