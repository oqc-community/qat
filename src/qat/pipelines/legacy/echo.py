# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.core.pipeline import Pipeline
from qat.pipelines.legacy.base import get_pipeline as get_legacy_pipeline
from qat.purr.backends.echo import get_default_echo_hardware


def get_pipeline(model, name="legacy_echo") -> Pipeline:
    return get_legacy_pipeline(model, name)


legacy_echo8 = get_pipeline(get_default_echo_hardware(qubit_count=8), name="legacy_echo8")
legacy_echo16 = get_pipeline(
    get_default_echo_hardware(qubit_count=16), name="legacy_echo16"
)
legacy_echo32 = get_pipeline(
    get_default_echo_hardware(qubit_count=32), name="legacy_echo32"
)
