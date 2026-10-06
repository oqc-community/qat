# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Oxford Quantum Circuits Ltd

import numpy as np
from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from qat.utils.pydantic import FloatNDArray, IntNDArray


class PathData(BaseModel):
    """This object wraps the actual data as a list of samples, the number of averages
    performed by the hardware (if any), and whether the hw observed any out-of-range
    samples."""

    avg_cnt: int = None
    oor: bool = Field(alias="out-of-range", default=False)
    data: FloatNDArray = Field(default_factory=lambda: FloatNDArray([]))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.avg_cnt != other.avg_cnt:
            return False
        if self.oor != other.oor:
            return False
        return not (self.data.size != other.data.size or np.any(self.data != other.data))


class IntegData(BaseModel):
    """Path 0 refers to I while Path 1 refers to Q."""

    path0: FloatNDArray = Field(default_factory=lambda: FloatNDArray([]))
    path1: FloatNDArray = Field(default_factory=lambda: FloatNDArray([]))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.path0.size != other.path0.size or np.any(self.path0 != other.path0):
            return False
        return not (
            self.path1.size != other.path1.size or np.any(self.path1 != other.path1)
        )


class ScopeAcqData(BaseModel):
    """Raw scope traces returned by a Qblox readout module.

    Paths 0 and 1 are the I/Q pair for the first physical input. QRC modules additionally
    return paths 2 and 3 for the second physical input. Paths 2 and 3 are independently
    optional because this result model also accepts QRM and partial acquisition payloads.
    Their lengths are statically equal to the target's maximum scope acquisition size.
    """

    path0: PathData = PathData()
    path1: PathData = PathData()
    path2: PathData | None = None
    path3: PathData | None = None

    @model_serializer(mode="wrap")
    def _serialize_scope_paths(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        scope_data = handler(self)
        if self.path2 is None:
            scope_data.pop("path2", None)
        if self.path3 is None:
            scope_data.pop("path3", None)
        return scope_data


class BinnedAcqData(BaseModel):
    """Binned data is data that's been acquired and processed via different routes such as
    squared acquisition, weighed integration.

    Processing here refers to steps like averaging, rotation, and thresholding which are
    executed by the hardware.
    """

    avg_cnt: IntNDArray = Field(default_factory=lambda: IntNDArray([]))
    integration: IntegData = IntegData()
    threshold: FloatNDArray = Field(default_factory=lambda: FloatNDArray([]))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.avg_cnt.size != other.avg_cnt.size or np.any(self.avg_cnt != other.avg_cnt):
            return False
        if self.integration != other.integration:
            return False
        return not (
            self.threshold.size != other.threshold.size
            or np.any(self.threshold != other.threshold)
        )


class BinnedAndScopeAcqData(BaseModel):
    """The actual acquisition data, it represents the value associated with the key
    "acquisition" in the acquisition blob returned by Qblox.

    This object contains scope data and binned data.
    """

    bins: BinnedAcqData = BinnedAcqData()
    scope: ScopeAcqData = ScopeAcqData()


class Acquisition(BaseModel):
    """Represents a single acquisition. In Qblox terminology, this object contains scope,
    integrated, and threshold data all at once. It's up to the SW layer to pick up what it
    needs and adapt it to its flow.

    An acquisition contains is described by a name, index, and blob data represented by
    :class:`AcqData`
    """

    name: str | None = None
    index: int = None
    acquisition: BinnedAndScopeAcqData = BinnedAndScopeAcqData()

    def __add__(self, other: "Acquisition") -> "Acquisition":
        """Acquisition addition follows concatenation semantics such as the case for
        strings.

        A few important details that might be adjusted in the future:
            + Resulting scope_data.path0.avg_cnt is taken as the minimum of the two
                Reason for the underestimation is to remain conservative and on the safe
                side (Can raise if strictness is required)
            + Resulting scope_data.path0.oor follows "AND" semantics
        """

        if not isinstance(other, Acquisition):
            raise TypeError(f"Can only add acquisitions, got {type(other)}")

        if self == Acquisition():
            return other.model_copy(deep=True)

        if other == Acquisition():
            return self.model_copy(deep=True)

        result = Acquisition()

        if self.index != other.index:
            raise ValueError(
                f"Expected the same index but got {self.index} != {other.index}"
            )
        result.index = self.index

        if self.name != other.name:
            raise ValueError(f"Expected the same name but got {self.name} != {other.name}")
        result.name = self.name

        scope_data1 = self.acquisition.scope
        scope_data2 = other.acquisition.scope
        result.acquisition.scope = ScopeAcqData(
            path0=_merge_path_data(scope_data1.path0, scope_data2.path0),
            path1=_merge_path_data(scope_data1.path1, scope_data2.path1),
            path2=_merge_optional_path_data(scope_data1.path2, scope_data2.path2),
            path3=_merge_optional_path_data(scope_data1.path3, scope_data2.path3),
        )

        bin_data1 = self.acquisition.bins
        bin_data2 = other.acquisition.bins
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


def _merge_path_data(left: PathData, right: PathData) -> PathData:
    return PathData(
        avg_cnt=min(left.avg_cnt or 0, right.avg_cnt or 0),
        **{"out-of-range": left.oor and right.oor},
        data=np.append(left.data, right.data),
    )


def _merge_optional_path_data(
    left: PathData | None, right: PathData | None
) -> PathData | None:
    if left is None:
        return right.model_copy(deep=True) if right is not None else None
    if right is None:
        return left.model_copy(deep=True)
    return _merge_path_data(left, right)
