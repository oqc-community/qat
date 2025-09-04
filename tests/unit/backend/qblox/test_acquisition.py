from pathlib import Path

from qat.backend.qblox.acquisition import (
    Acquisition,
    BinnedAcqData,
    BinnedAndScopeAcqData,
    IntegData,
    PathData,
    ScopeAcqData,
)
from qat.backend.qblox.config.constants import Constants


def test_default_constructors():
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


def test_serialisation(testpath):
    file_path = Path(testpath, "files", "acquisition", "qblox_payload.json")
    with open(file_path) as f:
        acquisition = Acquisition.model_validate_json(f.read())

    assert acquisition.index == 0

    scope_data = acquisition.acquisition.scope
    assert len(scope_data.path0.data) == len(scope_data.path1.data)
    assert len(scope_data.path0.data) == Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS

    bin_data = acquisition.acquisition.bins
    assert len(bin_data.integration.path0) == len(bin_data.integration.path1)
    assert len(bin_data.threshold) == len(bin_data.integration.path0)

    json_str = acquisition.model_dump_json()
    deserialised_acquisition = Acquisition.model_validate_json(json_str)

    assert deserialised_acquisition == acquisition
