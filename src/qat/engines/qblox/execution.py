# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import json
import os
from dataclasses import asdict
from datetime import datetime

from qat.backend.qblox.acquisition import Acquisition
from qat.backend.qblox.execution import QbloxProgram
from qat.backend.qblox.visualisation import plot_playback, plot_program
from qat.engines import NativeEngine
from qat.instrument.base import InstrumentConcept
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QbloxEngine(NativeEngine[QbloxProgram]):
    def __init__(self, instrument: InstrumentConcept):
        self.instrument: InstrumentConcept = instrument

        self.plot_program = False
        self.dump_program = False
        self.plot_playback = False

    def execute(self, program: QbloxProgram, **kwargs) -> dict[str, list[Acquisition]]:
        if self.plot_program:
            plot_program(program)

        if self.dump_program:
            for pulse_channel_id, pkg in program.packages.items():
                filename = f"schedules/target_{pulse_channel_id}_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as f:
                    f.write(json.dumps(asdict(pkg.sequence)))

        self.instrument.setup(program)
        playback: dict[str, list[Acquisition]] = self.instrument.playback()

        if self.plot_playback:
            plot_playback(playback)

        return playback
