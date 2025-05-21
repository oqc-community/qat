# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


import numpy as np
from pydantic import BaseModel, Field

from qat.utils.pydantic import FloatNDArray, IntNDArray


class PathData(BaseModel):
    """
    This object wraps the actual data as a list of samples, the number
    of averages performed by the hardware (if any), and whether the hw observed any out-of-range samples.
    """

    avg_count: int = Field(alias="avg_cnt", default=None)
    oor: bool = Field(alias="out-of-range", default=None)
    data: FloatNDArray = Field(alias="data", default=np.empty(0, dtype=float))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.avg_count != other.avg_count:
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

    i: FloatNDArray = Field(alias="path0", default=np.empty(0, dtype=float))
    q: FloatNDArray = Field(alias="path1", default=np.empty(0, dtype=float))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.i.size != other.i.size or np.any(self.i != other.i):
            return False
        if self.q.size != other.q.size or np.any(self.q != other.q):
            return False
        return True


class ScopeAcqData(BaseModel):
    """
    Path 0 refers to I while Path 1 refers to Q. Their lengths are statically equal
    to :class:`Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS`
    """

    i: PathData = Field(alias="path0", default=PathData())
    q: PathData = Field(alias="path1", default=PathData())


class BinnedAcqData(BaseModel):
    """
    Binned data is data that's been acquired and processed via different routes such as squared acquisition,
    weighed integration. Processing here refers to steps like averaging, rotation, and thresholding
    which are executed by the hardware.
    """

    avg_count: IntNDArray = Field(alias="avg_cnt", default=np.empty(0, dtype=int))
    integration: IntegData = Field(alias="integration", default=IntegData())
    threshold: FloatNDArray = Field(alias="threshold", default=np.empty(0, dtype=float))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.avg_count.size != other.avg_count.size or np.any(
            self.avg_count != other.avg_count
        ):
            return False
        if self.integration != other.integration:
            return False
        if self.threshold.size != other.threshold.size or np.any(
            self.threshold != other.threshold
        ):
            return False
        return True


class AcqData(BaseModel):
    """
    The actual acquisition data, it represents the value associated with the key "acquisition"
    in the acquisition blob returned by Qblox. This object contains scope data and binned data.
    """

    bins: BinnedAcqData = Field(alias="bins", default=BinnedAcqData())
    scope: ScopeAcqData = Field(alias="scope", default=ScopeAcqData())


class Acquisition(BaseModel):
    """
    Represents a single acquisition. In Qblox terminology, this object contains scope, integrated, and threshold
    data all at once. It's up to the SW layer to pick up what it needs and adapt it to its flow.

    An acquisition contains is described by a name, index, and blob data represented by :class:`AcqData`
    """

    name: str = None
    index: int = Field(alias="index", default=None)
    acq_data: AcqData = Field(alias="acquisition", default=AcqData())

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

        scope_data1 = acq1.acq_data.scope
        scope_data2 = acq2.acq_data.scope
        scope_data = result.acq_data.scope

        scope_data.i.avg_count = min(
            scope_data1.i.avg_count or 0, scope_data2.i.avg_count or 0
        )
        scope_data.i.oor = scope_data1.i.oor and scope_data2.i.oor
        scope_data.i.data = np.append(scope_data1.i.data, scope_data2.i.data)
        scope_data.q.avg_count = min(
            scope_data1.q.avg_count or 0, scope_data2.q.avg_count or 0
        )
        scope_data.q.oor = scope_data1.q.oor and scope_data2.q.oor
        scope_data.q.data = np.append(scope_data1.q.data, scope_data2.q.data)

        bin_data1 = acq1.acq_data.bins
        bin_data2 = acq2.acq_data.bins
        bin_data = result.acq_data.bins

        bin_data.avg_count = np.append(bin_data1.avg_count, bin_data2.avg_count)
        bin_data.integration.i = np.append(bin_data1.integration.i, bin_data2.integration.i)
        bin_data.integration.q = np.append(bin_data1.integration.q, bin_data2.integration.q)
        bin_data.threshold = np.append(bin_data1.threshold, bin_data2.threshold)

        return result
