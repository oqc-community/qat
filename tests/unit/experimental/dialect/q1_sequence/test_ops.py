# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1.ir.ops import NopOp, StopOp
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    make_acquisition,
    make_waveform,
    make_weight,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp


class TestSequenceOpConstruction:
    def test_minimal(self):
        seq = SequenceOp("ch0", [StopOp()])
        assert seq.channel_id.data == "ch0"
        assert len(seq.waveforms) == 0
        assert len(seq.weights) == 0
        assert len(seq.acquisitions) == 0

    def test_with_body_ops(self):
        seq = SequenceOp("ch0", [NopOp(), StopOp()])
        ops = list(seq.body.block.ops)
        assert len(ops) == 2

    def test_string_attr_channel_id(self):
        seq = SequenceOp(StringAttr("ch0"), [StopOp()])
        assert seq.channel_id.data == "ch0"

    def test_with_waveforms(self):
        wf = make_waveform("wf0", 0, [0.1, 0.2])
        seq = SequenceOp("ch0", [StopOp()], waveforms=ArrayAttr([wf]))
        assert len(seq.waveforms) == 1
        assert seq.waveforms.data[0].waveform_name.data == "wf0"

    def test_with_weights(self):
        w = make_weight("w0", 0, [1.0, 0.0])
        seq = SequenceOp("ch0", [StopOp()], weights=ArrayAttr([w]))
        assert len(seq.weights) == 1
        assert seq.weights.data[0].weight_name.data == "w0"

    def test_with_acquisitions(self):
        acq = make_acquisition("acq0", 0, 1)
        seq = SequenceOp("ch0", [StopOp()], acquisitions=ArrayAttr([acq]))
        assert len(seq.acquisitions) == 1
        assert seq.acquisitions.data[0].num_bins.data == 1

    def test_with_all_tables(self):
        wf = make_waveform("wf0", 0, [0.5])
        w = make_weight("w0", 0, [1.0])
        acq = make_acquisition("acq0", 0, 2)
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            waveforms=ArrayAttr([wf]),
            weights=ArrayAttr([w]),
            acquisitions=ArrayAttr([acq]),
        )
        assert len(seq.waveforms) == 1
        assert len(seq.weights) == 1
        assert len(seq.acquisitions) == 1


class TestSequenceOpVerify:
    def test_valid(self):
        seq = SequenceOp("ch0", [StopOp()])
        seq.verify_()

    def test_empty_channel_id_fails(self):
        seq = SequenceOp("", [StopOp()])
        with pytest.raises(VerifyException, match="channel_id must be non-empty"):
            seq.verify_()

    def test_duplicate_waveform_indices_fail(self):
        wf0 = make_waveform("a", 0, [0.1])
        wf1 = make_waveform("b", 0, [0.2])
        seq = SequenceOp("ch0", [StopOp()], waveforms=ArrayAttr([wf0, wf1]))
        with pytest.raises(VerifyException, match="Duplicate index 0 in waveforms"):
            seq.verify_()

    def test_duplicate_weight_indices_fail(self):
        w0 = make_weight("a", 1, [1.0])
        w1 = make_weight("b", 1, [0.5])
        seq = SequenceOp("ch0", [StopOp()], weights=ArrayAttr([w0, w1]))
        with pytest.raises(VerifyException, match="Duplicate index 1 in weights"):
            seq.verify_()

    def test_duplicate_acquisition_indices_fail(self):
        a0 = make_acquisition("a", 2, 1)
        a1 = make_acquisition("b", 2, 1)
        seq = SequenceOp("ch0", [StopOp()], acquisitions=ArrayAttr([a0, a1]))
        with pytest.raises(
            VerifyException,
            match="Duplicate index 2 in acquisitions",
        ):
            seq.verify_()

    def test_distinct_indices_pass(self):
        wf0 = make_waveform("a", 0, [0.1])
        wf1 = make_waveform("b", 1, [0.2])
        seq = SequenceOp("ch0", [StopOp()], waveforms=ArrayAttr([wf0, wf1]))
        seq.verify_()

    def test_missing_terminator_fails(self):
        seq = SequenceOp("ch0", [NopOp()])
        with pytest.raises(VerifyException, match="must end with a terminator"):
            seq.verify_()

    def test_duplicate_waveform_names_fail(self):
        wf0 = make_waveform("same", 0, [0.1])
        wf1 = make_waveform("same", 1, [0.2])
        seq = SequenceOp("ch0", [StopOp()], waveforms=ArrayAttr([wf0, wf1]))
        with pytest.raises(VerifyException, match="Duplicate name 'same' in waveforms"):
            seq.verify_()

    def test_duplicate_weight_names_fail(self):
        w0 = make_weight("dup", 0, [1.0])
        w1 = make_weight("dup", 1, [0.5])
        seq = SequenceOp("ch0", [StopOp()], weights=ArrayAttr([w0, w1]))
        with pytest.raises(VerifyException, match="Duplicate name 'dup' in weights"):
            seq.verify_()

    def test_duplicate_acquisition_names_fail(self):
        a0 = make_acquisition("acq", 0, 1)
        a1 = make_acquisition("acq", 1, 2)
        seq = SequenceOp("ch0", [StopOp()], acquisitions=ArrayAttr([a0, a1]))
        with pytest.raises(
            VerifyException,
            match="Duplicate name 'acq' in acquisitions",
        ):
            seq.verify_()
