# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.pipelines.legacy.echo import get_pipeline
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware

legacy_rtcs2 = get_pipeline(get_default_RTCS_hardware(), name="legacy_rtcs2")
