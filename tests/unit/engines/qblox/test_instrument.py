# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from pathlib import Path

import pytest
from qblox_instruments import (
    Cluster,
    SequencerStates,
    SequencerStatuses,
    SequencerStatusFlags,
)

from qat.backend.base import BaseBackend
from qat.backend.passes.purr.analysis import (
    BindingPass,
    TILegalisationPass,
    TriagePass,
)
from qat.backend.passes.purr.transform import DesugaringPass
from qat.backend.qblox.codegen import QbloxBackend1
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.backend.qblox.passes.analysis import QbloxLegalisationPass
from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.engines.qblox.instrument import (
    CompositeInstrument,
    LeafInstrument,
    load_instrument,
)
from qat.middleend.passes.purr.transform import (
    DeviceUpdateSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.middleend.passes.purr.validation import InstructionValidation, ReadoutValidation
from qat.purr.utils.logger import get_default_logger

from tests.unit.utils.builder_nuggets import resonator_spect

log = get_default_logger()


class TestInstrument:
    def middleend_pipeline(self, model):
        target_data = QbloxTargetData.default()
        return (
            PassManager()
            | PhaseOptimisation()
            | PostProcessingSanitisation()
            | DeviceUpdateSanitisation()
            | InstructionValidation(target_data)
            | ReadoutValidation(model)
            | RepeatSanitisation(model, target_data)
            | ScopeSanitisation()
            | ReturnSanitisation()
            | DesugaringPass()
            | TriagePass()
            | BindingPass()
            | TILegalisationPass()
            | QbloxLegalisationPass()
        )

    def _do_emit(self, model, backend: BaseBackend, builder):
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        self.middleend_pipeline(model).run(builder, res_mgr, met_mgr)
        return backend.emit(builder, res_mgr, met_mgr)

    def test_load_instrument(self, testpath):
        filepath = Path(
            testpath,
            "files",
            "config",
            "instrument_info.csv",
        )

        composite = load_instrument(filepath)
        assert composite
        assert isinstance(composite, CompositeInstrument)
        assert len(composite.components) == 8
        assert all(
            [isinstance(comp, LeafInstrument) for comp in composite.components.values()]
        )

    @pytest.mark.parametrize("instrument", [None], indirect=True)
    def test_instrument_lifecycle(self, instrument):
        assert instrument.is_connected  # connected already after resolution of the fixture
        assert instrument.driver is not None
        assert isinstance(instrument.driver, Cluster)

        instrument.disconnect()
        assert not instrument.is_connected
        assert instrument.driver is None

        instrument.connect()
        assert instrument.is_connected
        assert instrument.driver is not None
        assert isinstance(instrument.driver, Cluster)

    @pytest.mark.parametrize("model", [None], indirect=True)
    @pytest.mark.parametrize("instrument", [None], indirect=True)
    def test_setup(self, model, instrument):
        backend = QbloxBackend1(model)
        builder = resonator_spect(model)
        ordered_executables = self._do_emit(model, backend, builder)

        for executable in ordered_executables.values():
            assert len(instrument.allocations) == 0
            instrument.setup(executable, model)
            assert len(instrument.allocations) == 1

            # Unexpecting wf mem as the square pulse in res spec is done via Q1ASM
            # Expecting acquisition specification
            channel, sequencer = next(((k, v) for k, v in instrument.allocations.items()))
            assert not sequencer.get_waveforms()
            assert sequencer.get_acquisitions()

            # Expecting injected configuration such as integration length (aka. acquire width)
            acq_pkg = next(
                (
                    pkg
                    for channel_id, pkg in executable.packages.items()
                    if channel.full_id() == channel_id
                )
            )
            assert (
                sequencer.integration_length_acq()
                == acq_pkg.sequencer_config.square_weight_acq.integration_length
            )
            instrument.allocations.clear()

    @pytest.mark.parametrize("model", [None], indirect=True)
    @pytest.mark.parametrize("instrument", [None], indirect=True)
    def test_playback(self, model, instrument):
        backend = QbloxBackend1(model)
        builder = resonator_spect(model)
        ordered_executables = self._do_emit(model, backend, builder)
        for executable in ordered_executables.values():
            # Fail if no resource had been allocated leading up to playback
            with pytest.raises(ValueError):
                instrument.playback()

            instrument.setup(executable, model)
            channel, sequencer = next(((k, v) for k, v in instrument.allocations.items()))

            instrument.playback()  # IDLE ---arm--> ARMED ---start--> STOPPED

            # Expecting STOPPED after "arm" and "start" steps
            sequencer_status = sequencer.get_sequencer_status()
            assert sequencer_status.status == SequencerStatuses.OKAY
            assert sequencer_status.state == SequencerStates.STOPPED
            assert SequencerStatusFlags.ACQ_BINNING_DONE in sequencer_status.info_flags
            assert not sequencer.sync_en()

            instrument.allocations.clear()

    @pytest.mark.parametrize("model", [None], indirect=True)
    @pytest.mark.parametrize("instrument", [None], indirect=True)
    def test_collect(self, model, instrument):
        backend = QbloxBackend1(model)
        builder = resonator_spect(model)
        ordered_executables = self._do_emit(model, backend, builder)
        for executable in ordered_executables.values():
            instrument.setup(executable, model)
            instrument.playback()
            results = instrument.collect()
            assert results

            # Expect resource cleanup after playback
            assert not instrument.allocations
