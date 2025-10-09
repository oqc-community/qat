from pathlib import Path

import numpy as np
import pytest

from qat.purr.backends.qblox.acquisition import (
    Acquisition,
    BinnedAcqData,
    BinnedAndScopeAcqData,
    IntegData,
    PathData,
    ScopeAcqData,
)
from qat.purr.backends.qblox.constants import Constants


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
            ("acquisition_1.json", Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS),
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
