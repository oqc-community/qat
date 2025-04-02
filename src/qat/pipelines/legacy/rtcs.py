# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.model.loaders.legacy import RTCSModelLoader
from qat.pipelines.legacy.echo import get_pipeline

legacy_rtcs2 = get_pipeline(RTCSModelLoader().load(), name="legacy_rtcs2")
