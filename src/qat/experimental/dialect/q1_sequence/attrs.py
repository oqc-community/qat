# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from typing import Literal, TypeAlias

from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    Float32Type,
    IntegerAttr,
    IntegerType,
    Signedness,
    StringAttr,
    VectorType,
)
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition, param_def
from xdsl.utils.exceptions import VerifyException

UI32: TypeAlias = IntegerType[Literal[32], Literal[Signedness.UNSIGNED]]

ui32: UI32 = UI32(32, Signedness.UNSIGNED)
f32 = Float32Type()


@irdl_attr_definition
class WaveformAttr(ParametrizedAttribute):
    """A waveform entry in a Qblox sequence's waveforms dictionary.

    :param waveform_name: Waveform name.
    :param index: Index referenced by play ops (wave0/wave1).
    :param data: Float32 samples, each sample in [-1.0, 1.0] represents DAC range.
    """

    name = "q1_sequence.waveform"

    waveform_name: StringAttr = param_def(StringAttr)
    index: IntegerAttr[UI32] = param_def(IntegerAttr[UI32])

    # Qblox API accepts int|float (f64), but f32 suffices for Qblox DACs.
    data: DenseIntOrFPElementsAttr[Float32Type] = param_def(
        DenseIntOrFPElementsAttr[Float32Type]
    )

    def verify(self) -> None:
        for v in self.data.iter_values():
            if not -1.0 <= v <= 1.0:
                raise VerifyException(
                    f"Waveform sample {v} is out of DAC range [-1.0, 1.0]"
                )


@irdl_attr_definition
class WeightAttr(ParametrizedAttribute):
    """A weight entry in a Qblox sequence's weights dictionary.

    Weights are per-sample integration coefficients applied to the
    demodulated signal during weighted acquisition
    (``acquire_weighed``). Each coefficient multiplies the
    corresponding 1 ns ADC sample before summation, enabling
    matched-filter or optimal-discrimination readout schemes.

    A sequencer holds up to 32 weight arrays sharing a budget of
    16 384 samples (i.e. 16 384 ns at 1 GSa/s).

    :param weight_name: Weight name.
    :param index: Index referenced by ``acquire_weighed``
        (0-31, up to 32 weight arrays per sequencer).
    :param data: Float32 coefficients, each in [-1.0, 1.0].
    """

    name = "q1_sequence.weight"

    weight_name: StringAttr = param_def(StringAttr)
    index: IntegerAttr[UI32] = param_def(IntegerAttr[UI32])

    # Qblox API accepts int|float (f64), but f32 suffices for Qblox ADCs.
    data: DenseIntOrFPElementsAttr[Float32Type] = param_def(
        DenseIntOrFPElementsAttr[Float32Type]
    )

    def verify(self) -> None:
        for v in self.data.iter_values():
            if not -1.0 <= v <= 1.0:
                raise VerifyException(
                    f"Weight coefficient {v} is out of ADC range [-1.0, 1.0]"
                )


@irdl_attr_definition
class AcquisitionAttr(ParametrizedAttribute):
    """An acquisition entry in a Qblox sequence's acquisitions dictionary.

    :param acquisition_name: Acquisition name.
    :param index: Index referenced by acquire ops (acq_idx).
    :param num_bins: Number of acquisition bins.
    """

    name = "q1_sequence.acquisition"

    acquisition_name: StringAttr = param_def(StringAttr)
    index: IntegerAttr[UI32] = param_def(IntegerAttr[UI32])
    num_bins: IntegerAttr[UI32] = param_def(IntegerAttr[UI32])


def _make_dense_floats(
    values: list[float],
) -> DenseIntOrFPElementsAttr:
    vec_type = VectorType(f32, [len(values)])
    return DenseIntOrFPElementsAttr.from_list(vec_type, values)


def make_waveform(name: str, index: int, samples: list[float]) -> WaveformAttr:
    """Creates a ``WaveformAttr`` from Python primitives.

    :param name: Waveform name.
    :param index: Table index.
    :param samples: Float samples in [-1.0, 1.0].
    :returns: A verified ``WaveformAttr``.
    """
    return WaveformAttr(
        StringAttr(name),
        IntegerAttr(index, ui32),
        _make_dense_floats(samples),
    )


def make_weight(name: str, index: int, coeffs: list[float]) -> WeightAttr:
    """Creates a ``WeightAttr`` from Python primitives.

    :param name: Weight name.
    :param index: Table index.
    :param coeffs: Float coefficients in [-1.0, 1.0].
    :returns: A verified ``WeightAttr``.
    """
    return WeightAttr(
        StringAttr(name),
        IntegerAttr(index, ui32),
        _make_dense_floats(coeffs),
    )


def make_acquisition(name: str, index: int, num_bins: int) -> AcquisitionAttr:
    """Creates an ``AcquisitionAttr`` from Python primitives.

    :param name: Acquisition name.
    :param index: Table index.
    :param num_bins: Number of acquisition bins.
    :returns: An ``AcquisitionAttr``.
    """
    return AcquisitionAttr(
        StringAttr(name),
        IntegerAttr(index, ui32),
        IntegerAttr(num_bins, ui32),
    )
