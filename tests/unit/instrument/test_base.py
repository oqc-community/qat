# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from uuid import uuid4

import pytest
from qblox_instruments import (
    Cluster,
    SequencerStates,
    SequencerStatuses,
    SequencerStatusFlags,
)

from qat.backend.qblox.codegen import QbloxBackend1, QbloxBackend2
from qat.model.loaders.qblox import create_instrument
from qat.purr.utils.logger import get_default_logger
from qat.utils.uuid import temporary_uuid_seed

from tests.unit.backend.qblox.utils import do_emit
from tests.unit.utils.builder_nuggets import resonator_spect

log = get_default_logger()

pytestmark = pytest.mark.usefixtures("tmp_cwd")


class TestInstrument:
    @pytest.mark.parametrize("address", [None])
    def test_instrument_lifecycle(self, function_seed, address):
        with temporary_uuid_seed(function_seed):
            id = f"test_{uuid4()}".replace("-", "_")

        instrument = create_instrument(id, id, address)
        assert not instrument.is_connected
        assert instrument.driver is None

        instrument.connect()
        assert instrument.is_connected
        assert instrument.driver is not None
        assert isinstance(instrument.driver, Cluster)

        instrument.disconnect()
        assert not instrument.is_connected
        assert instrument.driver is None

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    @pytest.mark.parametrize("qblox_instrument", [None], indirect=True)
    @pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
    def test_setup(self, qblox_model, qblox_instrument, backend_type):
        builder = resonator_spect(qblox_model)
        executable = do_emit(qblox_model, backend_type, builder)

        for program in executable.programs:
            assert len(qblox_instrument.id2seq) == 0
            qblox_instrument.setup(program)
            assert len(qblox_instrument.id2seq) == 1

            # Unexpecting wf mem as the square pulse in res spec is done via Q1ASM
            # Expecting acquisition specification
            pulse_channel_id, sequencer = next(
                ((k, v) for k, v in qblox_instrument.id2seq.items())
            )
            assert not sequencer.get_waveforms()
            assert sequencer.get_acquisitions()

            # Expecting injected configuration such as integration length (aka. acquire width)
            acq_pkg = program.packages[pulse_channel_id]
            assert (
                sequencer.integration_length_acq()
                == acq_pkg.seq_config.square_weight_acq.integration_length
            )
            qblox_instrument.id2seq.clear()

            # Expecting allocated modules to be marked as dirty
            for pkg in program.packages.values():
                module = getattr(qblox_instrument.driver, f"module{pkg.slot_idx}")
                assert qblox_instrument.modules[module]

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    @pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
    @pytest.mark.parametrize("qblox_instrument", [None], indirect=True)
    def test_playback(self, qblox_model, qblox_instrument, backend_type):
        builder = resonator_spect(qblox_model)
        executable = do_emit(qblox_model, backend_type, builder)
        for program in executable.programs:
            # Fail if no resource had been allocated leading up to playback
            with pytest.raises(ValueError, match="No allocations found"):
                qblox_instrument.playback()

            qblox_instrument.setup(program)
            _, sequencer = next(((k, v) for k, v in qblox_instrument.id2seq.items()))

            # IDLE ---arm--> ARMED ---start--> STOPPED
            results = qblox_instrument.playback()
            assert results

            # Expecting STOPPED after "arm" and "start" steps
            sequencer_status = sequencer.get_sequencer_status()
            assert sequencer_status.status == SequencerStatuses.OKAY
            assert sequencer_status.state == SequencerStates.STOPPED
            assert SequencerStatusFlags.ACQ_BINNING_DONE in sequencer_status.info_flags
            assert not sequencer.sync_en()

            # Expect resource cleanup after playback
            assert not qblox_instrument.id2seq

            # Expecting allocated modules to be marked as clean
            for pkg in program.packages.values():
                module = getattr(qblox_instrument.driver, f"module{pkg.slot_idx}")
                assert not qblox_instrument.modules[module]
