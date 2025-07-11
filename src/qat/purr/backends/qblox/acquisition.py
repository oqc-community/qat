# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from qat.utils.pydantic import FloatNDArray, IntNDArray


class PathData(BaseModel):
    """
    This object wraps the actual data as a list of samples, the number
    of averages performed by the hardware (if any), and whether the hw observed any out-of-range samples.
    """

    avg_cnt: int = None
    oor: bool = Field(alias="out-of-range", default=False)
    data: FloatNDArray = np.empty(0, dtype=float)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.avg_cnt != other.avg_cnt:
            return False
        if self.oor != other.oor:
            return False
        if self.data.size != other.data.size or np.any(self.data != other.data):
            return False
        return True


class IntegData(BaseModel):
    """
    Path 0 refers to I while Path 1 refers to Q
    """

    path0: FloatNDArray = np.empty(0, dtype=float)
    path1: FloatNDArray = np.empty(0, dtype=float)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.path0.size != other.path0.size or np.any(self.path0 != other.path0):
            return False
        if self.path1.size != other.path1.size or np.any(self.path1 != other.path1):
            return False
        return True


class ScopeAcqData(BaseModel):
    """
    Path 0 refers to I while Path 1 refers to Q. Their lengths are statically equal
    to :class:`Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS`
    """

    path0: PathData = PathData()
    path1: PathData = PathData()


class BinnedAcqData(BaseModel):
    """
    Binned data is data that's been acquired and processed via different routes such as squared acquisition,
    weighed integration. Processing here refers to steps like averaging, rotation, and thresholding
    which are executed by the hardware.
    """

    avg_cnt: IntNDArray = np.empty(0, dtype=int)
    integration: IntegData = IntegData()
    threshold: FloatNDArray = np.empty(0, dtype=float)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.avg_cnt.size != other.avg_cnt.size or np.any(self.avg_cnt != other.avg_cnt):
            return False
        if self.integration != other.integration:
            return False
        if self.threshold.size != other.threshold.size or np.any(
            self.threshold != other.threshold
        ):
            return False
        return True


class BinnedAndScopeAcqData(BaseModel):
    """
    The actual acquisition data, it represents the value associated with the key "acquisition"
    in the acquisition blob returned by Qblox. This object contains scope data and binned data.
    """

    bins: BinnedAcqData = BinnedAcqData()
    scope: ScopeAcqData = ScopeAcqData()


class Acquisition(BaseModel):
    """
    Represents a single acquisition. In Qblox terminology, this object contains scope, integrated, and threshold
    data all at once. It's up to the SW layer to pick up what it needs and adapt it to its flow.

    An acquisition contains is described by a name, index, and blob data represented by :class:`AcqData`
    """

    name: Optional[str] = None
    index: int = None
    acquisition: BinnedAndScopeAcqData = BinnedAndScopeAcqData()

    @staticmethod
    def accumulate(acq1: "Acquisition", acq2: "Acquisition"):
        if acq1 == Acquisition():
            return acq2.model_copy(deep=True)

        if acq2 == Acquisition():
            return acq1.model_copy(deep=True)

        if acq1 == acq2:
            return acq1.model_copy(deep=True)

        result = Acquisition()

        if acq1.index != acq2.index:
            raise ValueError(
                f"Expected the same index but got {acq1.index} != {acq2.index}"
            )
        result.index = acq1.index

        if acq1.name != acq2.name:
            raise ValueError(f"Expected the same name but got {acq1.name} != {acq2.name}")
        result.name = acq1.name

        scope_data1 = acq1.acquisition.scope
        scope_data2 = acq2.acquisition.scope
        scope_data = result.acquisition.scope

        scope_data.path0.avg_cnt = min(
            scope_data1.path0.avg_cnt or 0, scope_data2.path0.avg_cnt or 0
        )
        scope_data.path0.oor = scope_data1.path0.oor and scope_data2.path0.oor
        scope_data.path0.data = np.append(scope_data1.path0.data, scope_data2.path0.data)
        scope_data.path1.avg_cnt = min(
            scope_data1.path1.avg_cnt or 0, scope_data2.path1.avg_cnt or 0
        )
        scope_data.path1.oor = scope_data1.path1.oor and scope_data2.path1.oor
        scope_data.path1.data = np.append(scope_data1.path1.data, scope_data2.path1.data)

        bin_data1 = acq1.acquisition.bins
        bin_data2 = acq2.acquisition.bins
        bin_data = result.acquisition.bins

        bin_data.avg_cnt = np.append(bin_data1.avg_cnt, bin_data2.avg_cnt)
        bin_data.integration.path0 = np.append(
            bin_data1.integration.path0, bin_data2.integration.path0
        )
        bin_data.integration.path1 = np.append(
            bin_data1.integration.path1, bin_data2.integration.path1
        )
        bin_data.threshold = np.append(bin_data1.threshold, bin_data2.threshold)

        return result
