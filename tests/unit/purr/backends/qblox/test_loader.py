# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pathlib import Path

from qat.purr.backends.qblox.execution.executor import CompositeExecutor, LeafExecutor
from qat.purr.backends.qblox.loader import load_executor


def test_load_executor(testpath):
    filepath = Path(
        testpath,
        "files",
        "config",
        "instrument_info.csv",
    )

    composite = load_executor(filepath)
    assert composite
    assert isinstance(composite, CompositeExecutor)
    assert len(composite.components) == 4
    assert all([isinstance(comp, LeafExecutor) for comp in composite.components.values()])
