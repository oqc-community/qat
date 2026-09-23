# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pathlib import Path

import numpy as np
import pytest

from qat.backend.qblox import acquisition as current_acquisition
from qat.backend.qblox.acquisition import Acquisition as CurrentAcquisition
from qat.backend.qblox.target_data import QRM_DATA
from qat.purr.backends.qblox.acquisition import (
    Acquisition,
    BinnedAcqData,
    BinnedAndScopeAcqData,
    IntegData,
    PathData,
    ScopeAcqData,
)


def _current_acquisition(
    name: str = "readout",
    index: int = 0,
    path0: list[float] | None = None,
    path1: list[float] | None = None,
) -> current_acquisition.Acquisition:
    return CurrentAcquisition(
        name=name,
        index=index,
        acquisition=current_acquisition.BinnedAndScopeAcqData(
            scope=current_acquisition.ScopeAcqData(
                path0=current_acquisition.PathData(
                    avg_cnt=4,
                    **{"out-of-range": True},
                    data=path0 or [1.0],
                ),
                path1=current_acquisition.PathData(
                    avg_cnt=2,
                    **{"out-of-range": True},
                    data=path1 or [2.0],
                ),
            ),
            bins=current_acquisition.BinnedAcqData(
                avg_cnt=[1],
                integration=current_acquisition.IntegData(path0=[3.0], path1=[4.0]),
                threshold=[1.0],
            ),
        ),
    )


def test_current_acquisition_concatenates_data():
    first = _current_acquisition()
    second = _current_acquisition(path0=[5.0], path1=[6.0])

    result = first + second

    assert result.name == "readout"
    assert result.index == 0
    assert result.acquisition.scope.path0.avg_cnt == 4
    assert result.acquisition.scope.path1.avg_cnt == 2
    assert result.acquisition.scope.path0.oor
    assert result.acquisition.scope.path1.oor
    np.testing.assert_array_equal(result.acquisition.scope.path0.data, [1.0, 5.0])
    np.testing.assert_array_equal(result.acquisition.scope.path1.data, [2.0, 6.0])
    np.testing.assert_array_equal(result.acquisition.bins.avg_cnt, [1, 1])
    np.testing.assert_array_equal(result.acquisition.bins.integration.path0, [3.0, 3.0])
    np.testing.assert_array_equal(result.acquisition.bins.integration.path1, [4.0, 4.0])
    np.testing.assert_array_equal(result.acquisition.bins.threshold, [1.0, 1.0])


def test_current_acquisition_handles_empty_operands():
    acquisition = _current_acquisition()

    assert acquisition + CurrentAcquisition() == acquisition
    assert CurrentAcquisition() + acquisition == acquisition


def test_current_acquisition_rejects_incompatible_operands():
    acquisition = _current_acquisition()

    with pytest.raises(TypeError, match="Can only add acquisitions"):
        acquisition + object()
    with pytest.raises(ValueError, match="Expected the same index"):
        acquisition + _current_acquisition(index=1)
    with pytest.raises(ValueError, match="Expected the same name"):
        acquisition + _current_acquisition(name="other")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (current_acquisition.PathData(), object()),
        (
            current_acquisition.PathData(avg_cnt=1),
            current_acquisition.PathData(avg_cnt=2),
        ),
        (
            current_acquisition.PathData(**{"out-of-range": True}),
            current_acquisition.PathData(**{"out-of-range": False}),
        ),
        (
            current_acquisition.PathData(data=[1.0]),
            current_acquisition.PathData(data=[1.0, 2.0]),
        ),
        (
            current_acquisition.PathData(data=[1.0]),
            current_acquisition.PathData(data=[2.0]),
        ),
        (current_acquisition.IntegData(), object()),
        (
            current_acquisition.IntegData(path0=[1.0]),
            current_acquisition.IntegData(path0=[1.0, 2.0]),
        ),
        (
            current_acquisition.IntegData(path0=[1.0]),
            current_acquisition.IntegData(path0=[2.0]),
        ),
        (
            current_acquisition.IntegData(path1=[1.0]),
            current_acquisition.IntegData(path1=[1.0, 2.0]),
        ),
        (
            current_acquisition.IntegData(path1=[1.0]),
            current_acquisition.IntegData(path1=[2.0]),
        ),
        (current_acquisition.BinnedAcqData(), object()),
        (
            current_acquisition.BinnedAcqData(avg_cnt=[1]),
            current_acquisition.BinnedAcqData(avg_cnt=[1, 2]),
        ),
        (
            current_acquisition.BinnedAcqData(avg_cnt=[1]),
            current_acquisition.BinnedAcqData(avg_cnt=[2]),
        ),
        (
            current_acquisition.BinnedAcqData(
                integration=current_acquisition.IntegData(path0=[1.0])
            ),
            current_acquisition.BinnedAcqData(
                integration=current_acquisition.IntegData(path0=[2.0])
            ),
        ),
        (
            current_acquisition.BinnedAcqData(threshold=[1.0]),
            current_acquisition.BinnedAcqData(threshold=[1.0, 2.0]),
        ),
        (
            current_acquisition.BinnedAcqData(threshold=[1.0]),
            current_acquisition.BinnedAcqData(threshold=[2.0]),
        ),
    ],
)
def test_current_acquisition_data_equality_rejects_differences(left, right):
    assert left != right


class TestAcquisition:
    def test_default_constructors(self):
        assert Acquisition() == Acquisition()

        acquisition = Acquisition()

        assert not acquisition.index
        assert acquisition.acquisition == BinnedAndScopeAcqData()

        scope_data = acquisition.acquisition.scope
        assert scope_data == ScopeAcqData()

        assert scope_data.path0 == PathData()
        assert not scope_data.path0.avg_cnt
        assert scope_data.path0.data.size == 0
        assert not scope_data.path0.oor

        assert scope_data.path1 == PathData()
        assert not scope_data.path1.avg_cnt
        assert scope_data.path1.data.size == 0
        assert not scope_data.path1.oor

        bin_data = acquisition.acquisition.bins
        assert bin_data == BinnedAcqData()
        assert bin_data.avg_cnt.size == 0
        assert bin_data.threshold.size == 0
        assert bin_data.integration == IntegData()

        integ_data = bin_data.integration
        assert integ_data.path0.size == 0
        assert integ_data.path1.size == 0

    @pytest.mark.parametrize(
        "acquisition, integ_length",
        [
            ("acquisition_1.json", QRM_DATA.max_sample_size_scope_acquisitions),
            ("acquisition_2.json", 800),
        ],
    )
    def test_deserialisation(self, testpath, acquisition, integ_length):
        file_path = Path(testpath, "files", "payload", acquisition)
        with open(file_path) as f:
            acquisition = Acquisition.model_validate_json(f.read())

        assert acquisition.index == 0

        scope_data = acquisition.acquisition.scope
        assert len(scope_data.path0.data) == len(scope_data.path1.data)
        assert len(scope_data.path0.data) == integ_length

        bin_data = acquisition.acquisition.bins
        assert len(bin_data.integration.path0) == len(bin_data.integration.path1)
        assert len(bin_data.threshold) == len(bin_data.integration.path0)

        json_str = acquisition.model_dump_json()
        deserialised_acquisition = Acquisition.model_validate_json(json_str)

        assert deserialised_acquisition == acquisition

    @pytest.mark.parametrize("acquisition", ["acquisition_2.json"])
    def test_serialisation(self, testpath, acquisition):
        file_path = Path(testpath, "files", "payload", acquisition)
        with open(file_path) as f:
            blob = f.read()
            # On-disk datafile has an trailing linefeed due to pre-commmit formatting
            blob = blob.strip()
            acquisition = Acquisition.model_validate_json(blob)
            assert acquisition.model_dump_json(indent=2) == blob

    @pytest.mark.parametrize("acq0", ["acquisition_1.json", "acquisition_2.json"])
    def test_addition(self, testpath, acq0):
        file_path = Path(testpath, "files", "payload", acq0)
        with open(file_path) as f:
            blob = f.read()
            acq0 = Acquisition.model_validate_json(blob)

            assert acq0 + Acquisition() == acq0
            assert Acquisition() + acq0 == acq0

            double_acq0 = acq0 + acq0
            assert double_acq0.index == acq0.index
            assert double_acq0.name == acq0.name

            scope0 = acq0.acquisition.scope
            scope1 = double_acq0.acquisition.scope
            assert scope1.path0.avg_cnt == scope0.path0.avg_cnt
            assert scope1.path0.oor == scope0.path0.oor
            assert np.all(
                scope1.path0.data == np.append(scope0.path0.data, scope0.path0.data)
            )

            assert scope1.path1.avg_cnt == scope0.path1.avg_cnt
            assert scope1.path1.oor == scope0.path1.oor
            assert np.all(
                scope1.path1.data == np.append(scope0.path1.data, scope0.path1.data)
            )

            bin_data0 = acq0.acquisition.bins
            bin_data1 = double_acq0.acquisition.bins
            assert np.all(
                bin_data1.avg_cnt == np.append(bin_data0.avg_cnt, bin_data0.avg_cnt)
            )
            assert np.all(
                bin_data1.integration.path0
                == np.append(bin_data0.integration.path0, bin_data0.integration.path0)
            )
            assert np.all(
                bin_data1.integration.path1
                == np.append(bin_data0.integration.path1, bin_data0.integration.path1)
            )
            assert np.all(
                bin_data1.threshold == np.append(bin_data0.threshold, bin_data0.threshold)
            )

    @pytest.mark.parametrize(
        "binned_acq",
        ["binned_acq_1.json", "binned_acq_2.json"],
    )
    def test_binned_acq_data(self, testpath, binned_acq):
        file_path = Path(testpath, "files", "payload", binned_acq)
        with open(file_path) as f:
            bins = BinnedAcqData.model_validate_json(f.read())

        assert bins
        assert len(bins.integration.path0) == len(bins.integration.path1)
        assert len(bins.threshold) == len(bins.integration.path0)
