from pathlib import Path

from qat.purr.backends.qblox.acquisition import (
    AcqData,
    Acquisition,
    BinnedAcqData,
    IntegData,
    PathData,
    ScopeAcqData,
)
from qat.purr.backends.qblox.constants import Constants


def test_default_constructors():
    acquisition = Acquisition()

    assert not acquisition.index
    assert acquisition.acq_data == AcqData()

    scope_data = acquisition.acq_data.scope
    assert scope_data == ScopeAcqData()

    assert scope_data.i == PathData()
    assert not scope_data.i.avg_count
    assert not scope_data.i.data
    assert not scope_data.i.oor

    assert scope_data.q == PathData()
    assert not scope_data.q.avg_count
    assert not scope_data.q.data
    assert not scope_data.q.oor

    bin_data = acquisition.acq_data.bins
    assert bin_data == BinnedAcqData()
    assert not bin_data.avg_count
    assert not bin_data.threshold
    assert bin_data.integration == IntegData()

    integ_data = bin_data.integration
    assert not integ_data.i
    assert not integ_data.q


def test_serialisation(testpath):
    file_path = Path(testpath, "files", "acquisition", "qblox_payload.json")
    with open(file_path) as f:
        acquisition = Acquisition.model_validate_json(f.read())

    assert acquisition.index == 0

    scope_data = acquisition.acq_data.scope
    assert len(scope_data.i.data) == len(scope_data.q.data)
    assert len(scope_data.i.data) == Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS

    bin_data = acquisition.acq_data.bins
    assert len(bin_data.integration.i) == len(bin_data.integration.q)
    assert len(bin_data.threshold) == len(bin_data.integration.i)
