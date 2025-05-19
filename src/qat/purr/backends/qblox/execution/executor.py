# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from typing import List

from qat.purr.backends.qblox.codegen import QbloxPackage
from qat.purr.backends.qblox.execution.instrument_base import (
    CompositeInstrument,
    LeafInstrument,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class LeafExecutor(LeafInstrument):
    def upload(self, packages: List[QbloxPackage]):
        pass

    def playback(self):
        pass

    def collect(self):
        pass


class CompositeExecutor(CompositeInstrument):
    def upload(self, packages: List[QbloxPackage]):
        pass

    def playback(self, *args, **kwargs):
        pass

    def collect(self, *args, **kwargs):
        pass
