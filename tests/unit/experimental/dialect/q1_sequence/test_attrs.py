# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    Float64Type,
    IntegerType,
    Signedness,
    StringAttr,
    VectorType,
)
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionAttr,
    WaveformAttr,
    WeightAttr,
    f32,
    make_acquisition,
    make_waveform,
    make_weight,
)
from qat.experimental.dialect.q1_sequence.ir.imm_desc import (
    AcqTableIndex,
    BinCountImm,
    WaveformTableIndex,
    WeightTableIndex,
)


class TestWaveformAttr:
    @pytest.mark.parametrize(
        ("element_type", "values", "should_fail"),
        [
            pytest.param(f32, [0.1, 0.2], False, id="f32-accepted"),
            pytest.param(Float64Type(), [0.1, 0.2], True, id="f64-rejected"),
            pytest.param(
                IntegerType(32, Signedness.SIGNED),
                [1, 2],
                True,
                id="i32-rejected",
            ),
        ],
    )
    def test_data_type(self, element_type, values, should_fail):
        vec = VectorType(element_type, [len(values)])
        data = DenseIntOrFPElementsAttr.from_list(vec, values)
        if should_fail:
            with pytest.raises(VerifyException):
                WaveformAttr(StringAttr("wf"), WaveformTableIndex(0), data)
        else:
            wf = WaveformAttr(StringAttr("wf"), WaveformTableIndex(0), data)
            wf.verify()

    @pytest.mark.parametrize(
        ("values", "should_fail"),
        [
            pytest.param([0.0, 0.5, -0.5], False, id="within-range"),
            pytest.param([-1.0, 1.0], False, id="boundary"),
            pytest.param([1.1], True, id="above-max"),
            pytest.param([-1.1], True, id="below-min"),
        ],
    )
    def test_data_range(self, values, should_fail):
        if should_fail:
            with pytest.raises(VerifyException, match="out of DAC range"):
                make_waveform("wf", 0, values)
        else:
            wf = make_waveform("wf", 0, values)
            wf.verify()


class TestWeightAttr:
    @pytest.mark.parametrize(
        ("element_type", "values", "should_fail"),
        [
            pytest.param(f32, [1.0, 0.0], False, id="f32-accepted"),
            pytest.param(Float64Type(), [1.0, 0.0], True, id="f64-rejected"),
            pytest.param(
                IntegerType(32, Signedness.SIGNED),
                [1, 2],
                True,
                id="i32-rejected",
            ),
        ],
    )
    def test_data_type(self, element_type, values, should_fail):
        vec = VectorType(element_type, [len(values)])
        data = DenseIntOrFPElementsAttr.from_list(vec, values)
        if should_fail:
            with pytest.raises(VerifyException):
                WeightAttr(StringAttr("w"), WeightTableIndex(0), data)
        else:
            w = WeightAttr(StringAttr("w"), WeightTableIndex(0), data)
            w.verify()

    @pytest.mark.parametrize(
        ("values", "should_fail"),
        [
            pytest.param([0.0, 0.5, -0.5], False, id="within-range"),
            pytest.param([-1.0, 1.0], False, id="boundary"),
            pytest.param([1.1], True, id="above-max"),
            pytest.param([-1.1], True, id="below-min"),
        ],
    )
    def test_data_range(self, values, should_fail):
        if should_fail:
            with pytest.raises(VerifyException, match="out of ADC range"):
                make_weight("w", 0, values)
        else:
            w = make_weight("w", 0, values)
            w.verify()


class TestAcquisitionAttr:
    def test_construction(self):
        a = AcquisitionAttr(StringAttr("a"), AcqTableIndex(5), BinCountImm(100))
        assert a.index.data == 5
        assert a.num_bins.data == 100

    def test_construction_via_helper(self):
        a = make_acquisition("acq", 3, 42)
        assert a.acquisition_name.data == "acq"
        assert a.index.data == 3
        assert a.num_bins.data == 42
