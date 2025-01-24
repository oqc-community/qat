# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import csv
import os

from qat.purr.backends.qblox.execution.executor import CompositeExecutor, LeafExecutor
from qat.purr.backends.qblox.execution.instrument_base import InstrumentModel


def load_executor(instrument_info_csv: str):
    """
    Builds a ControlHardware object wrapping an arbitrary fleet of Qblox clusters defined as CSV
    """

    if not os.path.exists(instrument_info_csv):
        raise ValueError(f"File '{instrument_info_csv}' not found!")

    executor = CompositeExecutor()
    with open(instrument_info_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            instrument = InstrumentModel.model_validate(row)
            executor.add(LeafExecutor(instrument))

    return executor
