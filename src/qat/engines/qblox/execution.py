# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import json
import os
from dataclasses import asdict
from datetime import datetime
from functools import reduce
from itertools import groupby
from typing import Dict, List

from qat.backend.qblox.acquisition import Acquisition
from qat.backend.qblox.execution import QbloxExecutable
from qat.backend.qblox.visualisation import plot_executable
from qat.engines import NativeEngine
from qat.engines.qblox.instrument import InstrumentConcept
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.backends.qblox.visualisation import plot_playback
from qat.purr.compiler.devices import PulseChannel


class QbloxEngine(NativeEngine):
    def __init__(self, instrument: InstrumentConcept, model: QbloxLiveHardwareModel):
        self.instrument: InstrumentConcept = instrument
        self.model: QbloxLiveHardwareModel = model

        self.plot_executable = False
        self.dump_executable = False
        self.plot_playback = False

    @staticmethod
    def combine_playbacks(playbacks: Dict[PulseChannel, List[Acquisition]]):
        """
        Combines acquisition objects from multiple acquire instructions in multiple readout targets.
        Notice that :meth:`groupby` preserves (original) relative order, which makes it honour
        the (sequential) lexicographical order of the loop nest:

        playback[target]["acq_0"] contains (potentially) a list of acquisitions collected in the same
        order as the order in which the packages were sent to the FPGA.

        Although acquisition names are enough for unicity in practice, the playback's structure
        distinguishes different (multiple) acquisitions per readout target, thus making it more robust.
        """

        playback: Dict[PulseChannel, Dict[str, Acquisition]] = {}
        for target, acquisitions in playbacks.items():
            groups_by_name = groupby(acquisitions, lambda acquisition: acquisition.name)
            playback[target] = {
                name: reduce(
                    lambda acq1, acq2: Acquisition.accumulate(acq1, acq2),
                    acqs,
                    Acquisition(),
                )
                for name, acqs in groups_by_name
            }

        return playback

    def execute(
        self, executable: QbloxExecutable
    ) -> Dict[PulseChannel, Dict[str, Acquisition]]:
        if self.plot_executable:
            plot_executable(executable)

        if self.dump_executable:
            for channel_id, pkg in executable.packages.items():
                filename = f"schedules/target_{channel_id}_@_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as f:
                    f.write(json.dumps(asdict(pkg.sequence)))

        self.instrument.setup(executable, self.model)
        self.instrument.playback()
        playbacks: Dict[PulseChannel, List[Acquisition]] = self.instrument.collect()

        if self.plot_playback:
            plot_playback(playbacks)

        playback: Dict[PulseChannel, Dict[str, Acquisition]] = self.combine_playbacks(
            playbacks
        )
        return playback
