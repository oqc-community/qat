# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd
"""Data tables and physical configuration attributes of the ``q1_sequence`` dialect.

The configuration attributes mirror the QCoDeS parameters exposed by the Qblox Instruments
driver. See the `Qblox Instruments API reference
<https://docs.qblox.com/en/main/products/qblox_instruments/api/>`_.

:class:`ModuleConfigAttr` describes the analogue chain, keyed by physical lane: one
:class:`OutputConfigAttr` per used output and one :class:`InputConfigAttr` per used input,
alongside the module's :class:`LocalOscillatorConfigAttr` entries.
:class:`SequencerConfigAttr` describes the digital chain and carries the lossless connection
that binds its sequencer to physical outputs, acquisition inputs and a local oscillator.

Every optional value is represented by :class:`~xdsl.dialects.builtin.NoneAttr` so that an
absent value stays distinguishable from a configured one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import partial

from typing_extensions import Self
from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    DenseIntOrFPElementsAttr,
    Float32Type,
    Float64Type,
    FloatAttr,
    IntAttr,
    NoneAttr,
    NoneType,
    StringAttr,
    VectorType,
)
from xdsl.ir import (
    Attribute,
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
)
from xdsl.irdl import irdl_attr_definition, param_def
from xdsl.parser import AttrParser
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.common.attribute_converters import (
    OptionalBool,
    OptionalFloat,
    OptionalInt,
    OptionalString,
    as_array,
    as_bool,
    as_float,
    as_int,
    as_int_array,
    as_optional,
    as_string,
)
from qat.experimental.dialect.q1_sequence.ir.imm_desc import (
    AcqTableIndex,
    BinCountImm,
    IntegrationLengthImm,
    SequencerIndexAttr,
    SlotIndexAttr,
    WaveformTableIndex,
    WeightTableIndex,
)
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    QbloxModuleKind,
    SignalPath,
    connection_input_ids,
    connection_output_ids,
)
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    Q1SequencerType,
)

f32 = Float32Type()

_as_required_string = partial(as_string, required=True)
_as_required_int = partial(as_int, required=True)
_CONTROL_SEQUENCER_SPEC = DEFAULT_QBLOX_TARGET.sequencer_spec(Q1SequencerType.control)
_NCO_MIN_FREQUENCY = _CONTROL_SEQUENCER_SPEC.nco_min_frequency_hz
_NCO_MAX_FREQUENCY = _CONTROL_SEQUENCER_SPEC.nco_max_frequency_hz


def _as_integration_length(
    value: IntegrationLengthImm | int | None,
) -> IntegrationLengthImm | NoneAttr:
    if isinstance(value, Attribute):
        return value
    return NoneAttr() if value is None else IntegrationLengthImm(value)


def _as_slot_index(value: SlotIndexAttr | int) -> SlotIndexAttr:
    return value if isinstance(value, SlotIndexAttr) else SlotIndexAttr(value)


def _as_module_kind(
    value: QbloxModuleKindAttr | QbloxModuleKind,
) -> QbloxModuleKindAttr:
    return value if isinstance(value, QbloxModuleKindAttr) else QbloxModuleKindAttr(value)


def _as_signal_path(
    value: SignalPathAttr | SignalPath | None,
) -> SignalPathAttr | NoneAttr:
    if value is None:
        return NoneAttr()
    return value if isinstance(value, SignalPathAttr) else SignalPathAttr(value)


def _as_required_signal_path(
    value: SignalPathAttr | SignalPath,
) -> SignalPathAttr:
    if value is None:
        raise TypeError("Acquisition connection path is required")
    return value if isinstance(value, SignalPathAttr) else SignalPathAttr(value)


@irdl_attr_definition
class QbloxModuleKindAttr(EnumAttribute[QbloxModuleKind], SpacedOpaqueSyntaxAttribute):
    """Attribute carrying a :class:`QbloxModuleKind`."""

    name = "q1_sequence.module_kind"


@irdl_attr_definition
class WaveformAttr(ParametrizedAttribute):
    """A waveform entry in a Qblox sequence's waveforms dictionary.

    :param waveform_name: Waveform name.
    :param index: Index referenced by play operations. Range ``[0, 1023]``.
    :param data: Float32 samples, each sample in [-1.0, 1.0] represents DAC range.
    """

    name = "q1_sequence.waveform"

    waveform_name: StringAttr = param_def(StringAttr)
    index: WaveformTableIndex = param_def(WaveformTableIndex)

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

    Each coefficient multiplies one demodulated ADC sample before summation. A sequencer
    accepts at most 32 weight arrays and 16 384 samples.

    :param weight_name: Weight name.
    :param index: Index referenced by ``acquire_weighted``. Range ``[0, 31]``.
    :param data: Float32 coefficients, each in [-1.0, 1.0].
    """

    name = "q1_sequence.weight"

    weight_name: StringAttr = param_def(StringAttr)
    index: WeightTableIndex = param_def(WeightTableIndex)

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
    :param index: Acquisition index. Range ``[0, 31]``.
    :param num_bins: Number of acquisition bins. Range ``[0, 7_000_000]``.
    """

    name = "q1_sequence.acquisition"

    acquisition_name: StringAttr = param_def(StringAttr)
    index: AcqTableIndex = param_def(AcqTableIndex)
    num_bins: BinCountImm = param_def(BinCountImm)


class ConfigAttr(ParametrizedAttribute):
    """Base for configuration attributes whose absent parameters are ``NoneAttr``.

    xDSL prints both :class:`~xdsl.dialects.builtin.NoneAttr` and
    :class:`~xdsl.dialects.builtin.NoneType` as ``none`` but always parses ``none`` back as
    a :class:`~xdsl.dialects.builtin.NoneType`, so absent parameters are normalised on parse
    to keep the configuration round-trippable.
    """

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        return [
            NoneAttr() if isinstance(parameter, NoneType) else parameter
            for parameter in super().parse_parameters(parser)
        ]

    def _merge(
        self,
        bound: Self,
        path: str = "",
        preserve_fields: frozenset[str] = frozenset(),
        preserve_existing: bool = False,
    ) -> Self:
        merged: list[Attribute] = []
        for (name, _), existing_value, bound_value in zip(
            type(self).get_irdl_definition().parameters,
            self.parameters,
            bound.parameters,
            strict=True,
        ):
            parameter_path = f"{path}.{name}" if path else name
            preserve_parameter = preserve_existing or name in preserve_fields
            if isinstance(existing_value, NoneAttr):
                merged.append(bound_value)
            elif isinstance(bound_value, NoneAttr):
                merged.append(existing_value)
            elif type(existing_value) is type(bound_value) and isinstance(
                existing_value, ConfigAttr
            ):
                merged.append(
                    existing_value._merge(
                        bound_value, parameter_path, preserve_fields, preserve_parameter
                    )
                )
            elif existing_value != bound_value and not preserve_parameter:
                raise VerifyException(
                    f"Sequencer configuration conflict at {parameter_path}."
                )
            else:
                merged.append(existing_value)
        return type(self).new(merged)


@irdl_attr_definition
class NcoConfigAttr(ConfigAttr):
    """Numerically controlled oscillator configuration of one sequencer.

    :param frequency: NCO frequency in Hz. Range ``[-500e6, 500e6]``.
    :param phase_offs: NCO phase offset in degrees.
    :param prop_delay_comp: Propagation delay compensation in nanoseconds.
    :param prop_delay_comp_en: Whether propagation delay compensation is enabled.
    """

    name = "q1_sequence.nco_config"

    frequency: OptionalFloat = param_def(converter=as_float)
    phase_offs: OptionalFloat = param_def(converter=as_float)
    prop_delay_comp: OptionalInt = param_def(converter=as_int)
    prop_delay_comp_en: OptionalBool = param_def(converter=as_bool)

    def __init__(
        self,
        frequency: FloatAttr[Float64Type] | float | None = None,
        phase_offs: FloatAttr[Float64Type] | float | None = None,
        prop_delay_comp: IntAttr | int | None = None,
        prop_delay_comp_en: BoolAttr | bool | None = None,
    ):
        super().__init__(frequency, phase_offs, prop_delay_comp, prop_delay_comp_en)

    def verify(self) -> None:
        if isinstance(self.frequency, FloatAttr) and not (
            _NCO_MIN_FREQUENCY <= self.frequency.value.data <= _NCO_MAX_FREQUENCY
        ):
            raise VerifyException(
                f"NCO frequency {self.frequency.value.data} is outside "
                f"[{_NCO_MIN_FREQUENCY:.0f}, {_NCO_MAX_FREQUENCY:.0f}] Hz"
            )


@irdl_attr_definition
class AwgConfigAttr(ConfigAttr):
    """Arbitrary waveform generator configuration of one sequencer.

    :param gain_path0: Gain applied to AWG path 0.
    :param gain_path1: Gain applied to AWG path 1.
    :param offset_path0: Offset applied to AWG path 0.
    :param offset_path1: Offset applied to AWG path 1.
    :param mod_en: Whether AWG modulation is enabled.
    """

    name = "q1_sequence.awg_config"

    gain_path0: OptionalFloat = param_def(converter=as_float)
    gain_path1: OptionalFloat = param_def(converter=as_float)
    offset_path0: OptionalFloat = param_def(converter=as_float)
    offset_path1: OptionalFloat = param_def(converter=as_float)
    mod_en: OptionalBool = param_def(converter=as_bool)

    def __init__(
        self,
        gain_path0: FloatAttr[Float64Type] | float | None = None,
        gain_path1: FloatAttr[Float64Type] | float | None = None,
        offset_path0: FloatAttr[Float64Type] | float | None = None,
        offset_path1: FloatAttr[Float64Type] | float | None = None,
        mod_en: BoolAttr | bool | None = None,
    ):
        super().__init__(gain_path0, gain_path1, offset_path0, offset_path1, mod_en)


@irdl_attr_definition
class UnweightedAcquireConfigAttr(ConfigAttr):
    """Unweighted acquisition configuration of one sequencer.

    :param integration_length: Integration length in samples, a multiple of 4.
    """

    name = "q1_sequence.unweighted_acquisition_config"

    integration_length: IntegrationLengthImm | NoneAttr = param_def(
        converter=_as_integration_length
    )

    def __init__(self, integration_length: IntegrationLengthImm | int | None = None):
        super().__init__(integration_length)


@irdl_attr_definition
class ThresholdedAcquireConfigAttr(ConfigAttr):
    """Thresholded acquisition configuration of one sequencer.

    :param rotation: Phase rotation applied to the integration result, in degrees.
    :param threshold: Threshold discretizing the phase-rotated integration result.
    """

    name = "q1_sequence.thresholded_acq_config"

    rotation: OptionalFloat = param_def(converter=as_float)
    threshold: OptionalFloat = param_def(converter=as_float)

    def __init__(
        self,
        rotation: FloatAttr[Float64Type] | float | None = None,
        threshold: FloatAttr[Float64Type] | float | None = None,
    ):
        super().__init__(rotation, threshold)


@irdl_attr_definition
class AcquireConfigAttr(ConfigAttr):
    """Acquisition path configuration of one sequencer.

    :param auto_bin_incr_en: Whether the bin index auto-increments across triggers.
    :param demod_en_acq: Whether demodulation is enabled on the acquisition path.
    """

    name = "q1_sequence.acquire_config"

    auto_bin_incr_en: OptionalBool = param_def(converter=as_bool)
    demod_en_acq: OptionalBool = param_def(converter=as_bool)

    def __init__(
        self,
        auto_bin_incr_en: BoolAttr | bool | None = None,
        demod_en_acq: BoolAttr | bool | None = None,
    ):
        super().__init__(auto_bin_incr_en, demod_en_acq)


@irdl_attr_definition
class MarkerOverrideConfigAttr(ConfigAttr):
    """Marker override configuration of one sequencer.

    When enabled, the override takes priority over the ``set_mrk`` instruction. It drives
    the marker outputs and the RF switches, which must be enabled for RF operation.

    :param marker_ovr_en: Whether marker overriding is enabled.
    :param marker_ovr_value: Marker override value, one bit per marker channel.
    """

    name = "q1_sequence.marker_override_config"

    marker_ovr_en: OptionalBool = param_def(converter=as_bool)
    marker_ovr_value: OptionalInt = param_def(converter=as_int)

    def __init__(
        self,
        marker_ovr_en: BoolAttr | bool | None = None,
        marker_ovr_value: IntAttr | int | None = None,
    ):
        super().__init__(marker_ovr_en, marker_ovr_value)


@irdl_attr_definition
class MixerCorrectionConfigAttr(ConfigAttr):
    """Mixer imbalance correction applied to one sequencer's AWG path.

    :param phase_offset: Mixer phase imbalance correction in degrees.
    :param gain_ratio: Mixer gain imbalance correction.
    """

    name = "q1_sequence.mixer_correction_config"

    phase_offset: OptionalFloat = param_def(converter=as_float)
    gain_ratio: OptionalFloat = param_def(converter=as_float)

    def __init__(
        self,
        phase_offset: FloatAttr[Float64Type] | float | None = None,
        gain_ratio: FloatAttr[Float64Type] | float | None = None,
    ):
        super().__init__(phase_offset, gain_ratio)


@irdl_attr_definition
class RealTimePredistortionConfigAttr(ConfigAttr):
    """Real-time predistortion filters applied to one physical output.

    Each filter accepts ``'bypassed'`` to disable it, or ``'delay_comp'`` to bypass it while
    delaying the output as if it were applied.

    :param fir_out: Finite-impulse-response filter configuration.
    :param exp_overshoot_0_out: Exponential-overshoot filter 0 configuration.
    :param exp_overshoot_1_out: Exponential-overshoot filter 1 configuration.
    :param exp_overshoot_2_out: Exponential-overshoot filter 2 configuration.
    :param exp_overshoot_3_out: Exponential-overshoot filter 3 configuration.
    """

    name = "q1_sequence.real_time_predistortion_config"

    fir_out: OptionalString = param_def(converter=as_string)
    exp_overshoot_0_out: OptionalString = param_def(converter=as_string)
    exp_overshoot_1_out: OptionalString = param_def(converter=as_string)
    exp_overshoot_2_out: OptionalString = param_def(converter=as_string)
    exp_overshoot_3_out: OptionalString = param_def(converter=as_string)

    def __init__(
        self,
        fir_out: StringAttr | str | None = None,
        exp_overshoot_0_out: StringAttr | str | None = None,
        exp_overshoot_1_out: StringAttr | str | None = None,
        exp_overshoot_2_out: StringAttr | str | None = None,
        exp_overshoot_3_out: StringAttr | str | None = None,
    ):
        super().__init__(
            fir_out,
            exp_overshoot_0_out,
            exp_overshoot_1_out,
            exp_overshoot_2_out,
            exp_overshoot_3_out,
        )


@irdl_attr_definition
class OutputSignalConfigAttr(ConfigAttr):
    """Signal conditioning applied to one physical output.

    :param attenuation: Output attenuation in dB.
    :param offset: Baseband output offset in V.
    :param offset_path_0: Offset in mV applied to output path 0, the I component.
    :param offset_path_1: Offset in mV applied to output path 1, the Q component.
    """

    name = "q1_sequence.output_signal_config"

    attenuation: OptionalFloat = param_def(converter=as_float)
    offset: OptionalFloat = param_def(converter=as_float)
    offset_path_0: OptionalFloat = param_def(converter=as_float)
    offset_path_1: OptionalFloat = param_def(converter=as_float)

    def __init__(
        self,
        attenuation: FloatAttr[Float64Type] | float | None = None,
        offset: FloatAttr[Float64Type] | float | None = None,
        offset_path_0: FloatAttr[Float64Type] | float | None = None,
        offset_path_1: FloatAttr[Float64Type] | float | None = None,
    ):
        super().__init__(attenuation, offset, offset_path_0, offset_path_1)


@irdl_attr_definition
class InputSignalConfigAttr(ConfigAttr):
    """Signal conditioning applied to one physical input.

    :param attenuation: Input attenuation in dB.
    :param gain: Input gain in dB.
    :param offset: Baseband input offset in V.
    :param offset_path_0: Offset in V applied to input path 0, the I component.
    :param offset_path_1: Offset in V applied to input path 1, the Q component.
    """

    name = "q1_sequence.input_signal_config"

    attenuation: OptionalFloat = param_def(converter=as_float)
    gain: OptionalFloat = param_def(converter=as_float)
    offset: OptionalFloat = param_def(converter=as_float)
    offset_path_0: OptionalFloat = param_def(converter=as_float)
    offset_path_1: OptionalFloat = param_def(converter=as_float)

    def __init__(
        self,
        attenuation: FloatAttr[Float64Type] | float | None = None,
        gain: FloatAttr[Float64Type] | float | None = None,
        offset: FloatAttr[Float64Type] | float | None = None,
        offset_path_0: FloatAttr[Float64Type] | float | None = None,
        offset_path_1: FloatAttr[Float64Type] | float | None = None,
    ):
        super().__init__(attenuation, gain, offset, offset_path_0, offset_path_1)


@irdl_attr_definition
class ScopeAcquireConfigAttr(ConfigAttr):
    """Trace acquisition configuration of one physical input.

    :param sequencer_select: Index of the sequencer whose acquisitions the module writes
        into its scope memory when using sequencer trigger mode. Qblox exposes this as a
        sequencer id rather than a per-sequencer flag, so the identity of the selection is
        preserved here instead of being reduced to a boolean.
    :param enable_average_mode: Whether scope acquisition averaging is enabled.
    """

    name = "q1_sequence.scope_acquire_config"

    sequencer_select: SequencerIndexAttr | NoneAttr = param_def(converter=as_optional)
    enable_average_mode: OptionalBool = param_def(converter=as_bool)

    def __init__(
        self,
        sequencer_select: SequencerIndexAttr | int | None = None,
        enable_average_mode: BoolAttr | bool | None = None,
    ):
        if isinstance(sequencer_select, int) and not isinstance(sequencer_select, bool):
            sequencer_select = SequencerIndexAttr(sequencer_select)
        super().__init__(sequencer_select, enable_average_mode)


@irdl_attr_definition
class LocalOscillatorConfigAttr(ConfigAttr):
    """A local oscillator owned by one physical module.

    :param oscillator_id: Identifier the module's lanes reference the oscillator by.
    :param frequency: Oscillator frequency in Hz.
    :param enable: Whether the oscillator is enabled.
    """

    name = "q1_sequence.local_oscillator_config"

    oscillator_id: StringAttr = param_def(converter=_as_required_string)
    frequency: IntAttr = param_def(converter=_as_required_int)
    enable: OptionalBool = param_def(converter=as_bool)

    def __init__(
        self,
        oscillator_id: StringAttr | str,
        frequency: IntAttr | int,
        enable: BoolAttr | bool | None = None,
    ):
        super().__init__(oscillator_id, frequency, enable)

    def verify(self) -> None:
        if not self.oscillator_id.data:
            raise VerifyException(
                "LocalOscillatorConfigAttr requires a non-empty oscillator_id"
            )


@irdl_attr_definition
class OutputConfigAttr(ConfigAttr):
    """Analogue configuration of one physical module output.

    :param output_id: Zero-based physical output. For example, ``0`` denotes ``out0``.
    :param pulse_shaping: Pulse-shaping filters applied to the output.
    :param output_signal: Signal conditioning applied to the output.
    """

    name = "q1_sequence.output_config"

    output_id: IntAttr = param_def(converter=_as_required_int)
    pulse_shaping: RealTimePredistortionConfigAttr | NoneAttr = param_def(
        converter=as_optional
    )
    output_signal: OutputSignalConfigAttr | NoneAttr = param_def(converter=as_optional)

    def __init__(
        self,
        output_id: IntAttr | int,
        pulse_shaping: RealTimePredistortionConfigAttr | None = None,
        output_signal: OutputSignalConfigAttr | None = None,
    ):
        super().__init__(output_id, pulse_shaping, output_signal)

    def verify(self) -> None:
        if self.output_id.data < 0:
            raise VerifyException("OutputConfigAttr output_id must be non-negative")


@irdl_attr_definition
class InputConfigAttr(ConfigAttr):
    """Analogue configuration of one physical module input.

    :param input_id: Zero-based physical input. For example, ``0`` denotes ``in0``.
    :param input_signal: Signal conditioning applied to the input.
    :param scope_acquire: Scope acquisition configuration of the input.
    """

    name = "q1_sequence.input_config"

    input_id: IntAttr = param_def(converter=_as_required_int)
    input_signal: InputSignalConfigAttr | NoneAttr = param_def(converter=as_optional)
    scope_acquire: ScopeAcquireConfigAttr | NoneAttr = param_def(converter=as_optional)

    def __init__(
        self,
        input_id: IntAttr | int,
        input_signal: InputSignalConfigAttr | None = None,
        scope_acquire: ScopeAcquireConfigAttr | None = None,
    ):
        super().__init__(input_id, input_signal, scope_acquire)

    def verify(self) -> None:
        if self.input_id.data < 0:
            raise VerifyException("InputConfigAttr input_id must be non-negative")


@irdl_attr_definition
class SignalPathAttr(EnumAttribute[SignalPath], SpacedOpaqueSyntaxAttribute):
    """Sequencer I/Q signal path selected by a connection."""

    name = "q1_sequence.sequencer_path"


@irdl_attr_definition
class DirectionKindAttr(EnumAttribute[DirectionKind], SpacedOpaqueSyntaxAttribute):
    """Direction prefix accepted by the Qblox sequencer connection API."""

    name = "q1_sequence.direction_kind"


@irdl_attr_definition
class ConnectionAttr(ParametrizedAttribute):
    """One connection string accepted by the Qblox sequencer connection API.

    A connection names one I/O port for real mode or an I and a Q port for complex mode, in
    every direction. Which combinations a module kind accepts is verified against the target
    description rather than here.

    :param direction: Whether the connection carries output, input, or bidirectional data.
    :param port_ids: Ordered I/O ports connected to the sequencer I/Q paths.
    """

    name = "q1_sequence.connection"

    direction: DirectionKindAttr = param_def()
    port_ids: ArrayAttr[IntAttr] = param_def(converter=as_int_array)

    def __init__(
        self,
        direction: DirectionKindAttr | DirectionKind,
        port_ids: ArrayAttr[IntAttr] | Iterable[IntAttr | int],
    ):
        super().__init__(
            (
                direction
                if isinstance(direction, DirectionKindAttr)
                else DirectionKindAttr(direction)
            ),
            port_ids,
        )

    def verify(self) -> None:
        ids = [port_id.data for port_id in self.port_ids]
        if not ids:
            raise VerifyException("ConnectionAttr requires at least one port_id")
        if any(port_id < 0 for port_id in ids):
            raise VerifyException("ConnectionAttr port_ids must be non-negative")
        if len(ids) != len(set(ids)):
            raise VerifyException("ConnectionAttr port_ids must be distinct")
        if len(ids) > 2:
            raise VerifyException("Connections support at most two port_ids")

    @property
    def connection(self) -> str:
        """Return the connection string accepted by ``connect_sequencer``."""

        return self.direction.data.value + "_".join(
            str(port_id.data) for port_id in self.port_ids
        )

    @property
    def output_ids(self) -> tuple[int, ...]:
        """Return the physical outputs this connection drives.

        An ``ioX_Y`` connection drives every listed lane; see
        :func:`~qat.experimental.system_data.hardware.qblox.models.connection_output_ids`.
        """

        return connection_output_ids(
            self.direction.data, [port_id.data for port_id in self.port_ids]
        )

    @property
    def input_ids(self) -> tuple[int, ...]:
        """Return the physical inputs this connection acquires from.

        An ``ioX_Y`` connection acquires on every listed lane; see
        :func:`~qat.experimental.system_data.hardware.qblox.models.connection_input_ids`.
        """

        return connection_input_ids(
            self.direction.data, [port_id.data for port_id in self.port_ids]
        )


@irdl_attr_definition
class OutputPathConnectionAttr(ParametrizedAttribute):
    """One physical output connected to a sequencer output path.

    :param output_id: Physical output configured by the ``outN`` field.
    :param path: Qblox sequencer path carried by the connection.
    """

    name = "q1_sequence.output_path_connection"

    output_id: IntAttr = param_def(converter=_as_required_int)
    path: SignalPathAttr | NoneAttr = param_def(converter=_as_signal_path)

    def __init__(
        self,
        output_id: IntAttr | int,
        path: SignalPathAttr | SignalPath | None = None,
    ):
        super().__init__(output_id, path)

    def verify(self) -> None:
        if self.output_id.data < 0:
            raise VerifyException("OutputPathConnectionAttr output_id must be non-negative")


@irdl_attr_definition
class AcquisitionPathConnectionAttr(ParametrizedAttribute):
    """Physical input connected to one Qblox acquisition path.

    :param input_id: Physical input selected by the acquisition path.
    :param path: Acquisition path receiving the input.
    """

    name = "q1_sequence.acquisition_path_connection"

    input_id: IntAttr = param_def(converter=_as_required_int)
    path: SignalPathAttr = param_def(converter=_as_required_signal_path)

    def __init__(
        self,
        input_id: IntAttr | int,
        path: SignalPathAttr | SignalPath,
    ):
        super().__init__(input_id, path)

    def verify(self) -> None:
        if self.input_id.data < 0:
            raise VerifyException(
                "AcquisitionPathConnectionAttr input_id must be non-negative"
            )


@irdl_attr_definition
class SequencerConfigAttr(ConfigAttr):
    """Configuration and physical connections of one sequencer.

    The connections bind the sequencer to the analogue lanes configured by the attached
    :class:`ModuleConfigAttr`. Output connections retain fan-out and Qblox signal-path
    selection. Acquisition connections retain each ``acq_I``/``acq_Q`` input selection.

    :param port_id: Canonical port the sequencer drives.
    :param carrier_frequency: Carrier frequency in Hz produced by the local oscillator and
        the NCO together.
    :param connections: Exact ordered entries in the supplied ``bulk_value``.
    :param output_path_connections: Output selection for each output path.
    :param acquisition_path_connections: Input selection for each acquisition path.
    :param acquisition_enabled: Explicit acquisition enable state.
    :param disabled_outputs: Outputs explicitly configured as off.
    :param disabled_acquisition_paths: Acquisition paths configured as off.
    :param acquisition_disabled: Whether the combined acquisition path is configured as off.
    :param local_oscillator_id: Local oscillator mixed with the NCO, absent for baseband.
    :param enable_sync: Whether the sequencer joins party-line synchronisation.
    :param nco: Numerically controlled oscillator configuration.
    :param awg: Arbitrary waveform generator configuration.
    :param mixer: Mixer imbalance correction applied to the AWG path.
    :param marker_switch: Marker override configuration.
    :param unweighted_acquire: Unweighted acquisition configuration.
    :param acquire: Acquisition path configuration.
    :param thresholded_acquire: Thresholded acquisition configuration.
    """

    name = "q1_sequence.sequencer_config"

    port_id: OptionalString = param_def(converter=as_string)
    carrier_frequency: OptionalFloat = param_def(converter=as_float)
    connections: ArrayAttr[ConnectionAttr] | NoneAttr = param_def(converter=as_optional)
    output_path_connections: ArrayAttr[OutputPathConnectionAttr] | NoneAttr = param_def(
        converter=as_optional
    )
    acquisition_path_connections: ArrayAttr[AcquisitionPathConnectionAttr] | NoneAttr = (
        param_def(converter=as_optional)
    )
    acquisition_enabled: OptionalBool = param_def(converter=as_bool)
    disabled_outputs: ArrayAttr[IntAttr] | NoneAttr = param_def(converter=as_optional)
    disabled_acquisition_paths: ArrayAttr[SignalPathAttr] | NoneAttr = param_def(
        converter=as_optional
    )
    acquisition_disabled: OptionalBool = param_def(converter=as_bool)
    local_oscillator_id: OptionalString = param_def(converter=as_string)
    enable_sync: OptionalBool = param_def(converter=as_bool)
    nco: NcoConfigAttr | NoneAttr = param_def(converter=as_optional)
    awg: AwgConfigAttr | NoneAttr = param_def(converter=as_optional)
    mixer: MixerCorrectionConfigAttr | NoneAttr = param_def(converter=as_optional)
    marker_switch: MarkerOverrideConfigAttr | NoneAttr = param_def(converter=as_optional)
    unweighted_acquire: UnweightedAcquireConfigAttr | NoneAttr = param_def(
        converter=as_optional
    )
    acquire: AcquireConfigAttr | NoneAttr = param_def(converter=as_optional)
    thresholded_acquire: ThresholdedAcquireConfigAttr | NoneAttr = param_def(
        converter=as_optional
    )

    def __init__(
        self,
        port_id: StringAttr | str | None = None,
        carrier_frequency: FloatAttr[Float64Type] | float | None = None,
        connections: ArrayAttr[ConnectionAttr] | Iterable[ConnectionAttr] | None = None,
        output_path_connections: (
            ArrayAttr[OutputPathConnectionAttr] | Iterable[OutputPathConnectionAttr] | None
        ) = None,
        acquisition_path_connections: (
            ArrayAttr[AcquisitionPathConnectionAttr]
            | Iterable[AcquisitionPathConnectionAttr]
            | None
        ) = None,
        acquisition_enabled: BoolAttr | bool | None = None,
        disabled_outputs: ArrayAttr[IntAttr] | Iterable[IntAttr | int] | None = None,
        disabled_acquisition_paths: (
            ArrayAttr[SignalPathAttr] | Iterable[SignalPathAttr | SignalPath] | None
        ) = None,
        acquisition_disabled: BoolAttr | bool | None = None,
        local_oscillator_id: StringAttr | str | None = None,
        enable_sync: BoolAttr | bool | None = None,
        nco: NcoConfigAttr | None = None,
        awg: AwgConfigAttr | None = None,
        mixer: MixerCorrectionConfigAttr | None = None,
        marker_switch: MarkerOverrideConfigAttr | None = None,
        unweighted_acquire: UnweightedAcquireConfigAttr | None = None,
        acquire: AcquireConfigAttr | None = None,
        thresholded_acquire: ThresholdedAcquireConfigAttr | None = None,
    ):
        super().__init__(
            port_id,
            carrier_frequency,
            (
                ArrayAttr(connections)
                if connections is not None and not isinstance(connections, ArrayAttr)
                else connections
            ),
            (
                ArrayAttr(output_path_connections)
                if output_path_connections is not None
                and not isinstance(output_path_connections, ArrayAttr)
                else output_path_connections
            ),
            (
                ArrayAttr(acquisition_path_connections)
                if acquisition_path_connections is not None
                and not isinstance(acquisition_path_connections, ArrayAttr)
                else acquisition_path_connections
            ),
            acquisition_enabled,
            as_int_array(disabled_outputs) if disabled_outputs is not None else None,
            (
                ArrayAttr(
                    path if isinstance(path, SignalPathAttr) else SignalPathAttr(path)
                    for path in disabled_acquisition_paths
                )
                if disabled_acquisition_paths is not None
                and not isinstance(disabled_acquisition_paths, ArrayAttr)
                else disabled_acquisition_paths
            ),
            acquisition_disabled,
            local_oscillator_id,
            enable_sync,
            nco,
            awg,
            mixer,
            marker_switch,
            unweighted_acquire,
            acquire,
            thresholded_acquire,
        )

    @property
    def integration_length(self) -> IntegrationLengthImm | None:
        """Configured unweighted integration length, if any."""

        if isinstance(self.unweighted_acquire, UnweightedAcquireConfigAttr) and isinstance(
            self.unweighted_acquire.integration_length, IntegrationLengthImm
        ):
            return self.unweighted_acquire.integration_length
        return None

    @property
    def has_acquisition_config(self) -> bool:
        """Whether any acquisition-only configuration is present."""

        return (
            (
                isinstance(self.acquisition_path_connections, ArrayAttr)
                and bool(self.acquisition_path_connections)
            )
            or not isinstance(self.acquisition_enabled, NoneAttr)
            or (
                isinstance(self.disabled_acquisition_paths, ArrayAttr)
                and bool(self.disabled_acquisition_paths)
            )
            or not isinstance(self.acquisition_disabled, NoneAttr)
            or not isinstance(self.unweighted_acquire, NoneAttr)
            or not isinstance(self.acquire, NoneAttr)
            or not isinstance(self.thresholded_acquire, NoneAttr)
        )

    def with_unweighted_acquire(
        self, unweighted_acquire: UnweightedAcquireConfigAttr
    ) -> SequencerConfigAttr:
        """Return a copy with a new unweighted acquisition configuration.

        :param unweighted_acquire: The unweighted acquisition configuration to apply.
        :returns: The updated sequencer configuration.
        """

        return SequencerConfigAttr(
            port_id=self.port_id,
            carrier_frequency=self.carrier_frequency,
            connections=None
            if isinstance(self.connections, NoneAttr)
            else self.connections,
            output_path_connections=(
                None
                if isinstance(self.output_path_connections, NoneAttr)
                else self.output_path_connections
            ),
            acquisition_path_connections=(
                None
                if isinstance(self.acquisition_path_connections, NoneAttr)
                else self.acquisition_path_connections
            ),
            acquisition_enabled=self.acquisition_enabled,
            disabled_outputs=(
                None
                if isinstance(self.disabled_outputs, NoneAttr)
                else self.disabled_outputs
            ),
            disabled_acquisition_paths=(
                None
                if isinstance(self.disabled_acquisition_paths, NoneAttr)
                else self.disabled_acquisition_paths
            ),
            acquisition_disabled=self.acquisition_disabled,
            local_oscillator_id=self.local_oscillator_id,
            enable_sync=self.enable_sync,
            nco=self.nco,
            awg=self.awg,
            mixer=self.mixer,
            marker_switch=self.marker_switch,
            unweighted_acquire=unweighted_acquire,
            acquire=self.acquire,
            thresholded_acquire=self.thresholded_acquire,
        )

    def merge_bound(self, bound: SequencerConfigAttr) -> SequencerConfigAttr:
        """Merge resolved values while retaining program-owned acquisition data.

        :param bound: Configuration resolved from the hardware description.
        :returns: The merged sequencer configuration.
        """

        return self._merge(bound, preserve_fields=frozenset({"unweighted_acquire"}))

    def verify(self) -> None:
        if isinstance(self.port_id, StringAttr) and not self.port_id.data:
            raise VerifyException("SequencerConfigAttr port_id must be non-empty")
        connections = (
            list(self.connections) if isinstance(self.connections, ArrayAttr) else []
        )
        connection_tokens = [connection.connection for connection in connections]
        if len(connection_tokens) != len(set(connection_tokens)):
            raise VerifyException("SequencerConfigAttr has duplicate connections")
        occupied_outputs: set[int] = set()
        occupied_inputs: set[int] = set()
        for connection in connections:
            if overlap := occupied_outputs.intersection(connection.output_ids):
                raise VerifyException(
                    "SequencerConfigAttr has overlapping output connections: "
                    f"{sorted(overlap)}"
                )
            occupied_outputs.update(connection.output_ids)
            if overlap := occupied_inputs.intersection(connection.input_ids):
                raise VerifyException(
                    "SequencerConfigAttr has overlapping input connections: "
                    f"{sorted(overlap)}"
                )
            occupied_inputs.update(connection.input_ids)
        output_path_connections = (
            [connection.output_id.data for connection in self.output_path_connections]
            if isinstance(self.output_path_connections, ArrayAttr)
            else []
        )
        if len(output_path_connections) != len(set(output_path_connections)):
            raise VerifyException(
                "SequencerConfigAttr has multiple paths for the same physical output"
            )
        acquisition_paths = (
            [connection.path.data for connection in self.acquisition_path_connections]
            if isinstance(self.acquisition_path_connections, ArrayAttr)
            else []
        )
        if len(acquisition_paths) != len(set(acquisition_paths)):
            raise VerifyException(
                "SequencerConfigAttr has duplicate acquisition connection paths"
            )
        disabled_outputs = (
            [output_id.data for output_id in self.disabled_outputs]
            if isinstance(self.disabled_outputs, ArrayAttr)
            else []
        )
        if any(output_id < 0 for output_id in disabled_outputs):
            raise VerifyException(
                "SequencerConfigAttr disabled outputs must be non-negative"
            )
        if len(disabled_outputs) != len(set(disabled_outputs)):
            raise VerifyException("SequencerConfigAttr has duplicate disabled outputs")
        connected_outputs = occupied_outputs | set(output_path_connections)
        if overlap := connected_outputs & set(disabled_outputs):
            raise VerifyException(
                "SequencerConfigAttr outputs cannot be both connected and disabled: "
                f"{sorted(overlap)}"
            )
        disabled_acquisition_paths = (
            [path.data for path in self.disabled_acquisition_paths]
            if isinstance(self.disabled_acquisition_paths, ArrayAttr)
            else []
        )
        if len(disabled_acquisition_paths) != len(set(disabled_acquisition_paths)):
            raise VerifyException(
                "SequencerConfigAttr has duplicate disabled acquisition paths"
            )
        if overlap := set(acquisition_paths) & set(disabled_acquisition_paths):
            raise VerifyException(
                "SequencerConfigAttr acquisition paths cannot be both connected and "
                f"disabled: {sorted(path.value for path in overlap)}"
            )
        acquisition_disabled = (
            bool(self.acquisition_disabled.value.data)
            if not isinstance(self.acquisition_disabled, NoneAttr)
            else False
        )
        acquisition_enabled = (
            bool(self.acquisition_enabled.value.data)
            if not isinstance(self.acquisition_enabled, NoneAttr)
            else False
        )
        if acquisition_enabled and acquisition_disabled:
            raise VerifyException(
                "SequencerConfigAttr acquisition cannot be both enabled and disabled"
            )
        if acquisition_disabled and (occupied_inputs or acquisition_paths):
            raise VerifyException(
                "SequencerConfigAttr acquisition cannot be disabled while input "
                "connections are active"
            )


@irdl_attr_definition
class ModuleConfigAttr(ConfigAttr):
    """Authoritative analogue configuration of one physical Qblox module.

    :param slot_idx: Cluster slot the module occupies.
    :param instrument_id: Cluster instrument containing the module.
    :param kind: Module kind.
    :param outputs: Configuration of every physical output in use.
    :param inputs: Configuration of every physical input in use.
    :param local_oscillators: Local oscillators the module's lanes reference.
    """

    name = "q1_sequence.module_config"

    slot_idx: SlotIndexAttr = param_def(converter=_as_slot_index)
    instrument_id: StringAttr = param_def(converter=_as_required_string)
    kind: QbloxModuleKindAttr = param_def(converter=_as_module_kind)
    outputs: ArrayAttr[OutputConfigAttr] = param_def(converter=as_array)
    inputs: ArrayAttr[InputConfigAttr] = param_def(converter=as_array)
    local_oscillators: ArrayAttr[LocalOscillatorConfigAttr] = param_def(converter=as_array)

    def __init__(
        self,
        slot_idx: SlotIndexAttr | int,
        instrument_id: StringAttr | str,
        kind: QbloxModuleKindAttr | QbloxModuleKind,
        outputs: ArrayAttr[OutputConfigAttr] | Iterable[OutputConfigAttr] = (),
        inputs: ArrayAttr[InputConfigAttr] | Iterable[InputConfigAttr] = (),
        local_oscillators: (
            ArrayAttr[LocalOscillatorConfigAttr] | Iterable[LocalOscillatorConfigAttr]
        ) = (),
    ):
        super().__init__(slot_idx, instrument_id, kind, outputs, inputs, local_oscillators)

    def verify(self) -> None:
        if not self.instrument_id.data:
            raise VerifyException("ModuleConfigAttr instrument_id must be non-empty")

        module_spec = DEFAULT_QBLOX_TARGET.module_spec(self.kind.data)
        oscillator_ids = [
            oscillator.oscillator_id.data for oscillator in self.local_oscillators
        ]
        if len(oscillator_ids) != len(set(oscillator_ids)):
            raise VerifyException("ModuleConfigAttr has duplicate local oscillator ids")

        for lane_name, lane_ids, limit in (
            (
                "output_id",
                [output.output_id.data for output in self.outputs],
                module_spec.output_count,
            ),
            (
                "input_id",
                [config_input.input_id.data for config_input in self.inputs],
                module_spec.input_count,
            ),
        ):
            if len(lane_ids) != len(set(lane_ids)):
                raise VerifyException(f"ModuleConfigAttr has duplicate {lane_name}")
            for lane_id in lane_ids:
                if lane_id >= limit:
                    raise VerifyException(
                        f"ModuleConfigAttr {lane_name} {lane_id} is invalid for "
                        f"{self.kind.data.value}"
                    )


def make_dense_floats(
    values: list[float],
) -> DenseIntOrFPElementsAttr:
    vec_type = VectorType(f32, [len(values)])
    return DenseIntOrFPElementsAttr.from_list(vec_type, values)


def make_waveform(name: str, index: int, samples: list[float]) -> WaveformAttr:
    """Creates a ``WaveformAttr`` from Python primitives.

    :param name: Waveform name.
    :param index: Table index in ``[0, 1023]``.
    :param samples: Float samples in [-1.0, 1.0].
    :returns: A verified ``WaveformAttr``.
    """
    return WaveformAttr(
        StringAttr(name),
        WaveformTableIndex(index),
        make_dense_floats(samples),
    )


def make_weight(name: str, index: int, coeffs: list[float]) -> WeightAttr:
    """Creates a ``WeightAttr`` from Python primitives.

    :param name: Weight name.
    :param index: Table index in ``[0, 31]``.
    :param coeffs: Float coefficients in [-1.0, 1.0].
    :returns: A verified ``WeightAttr``.
    """
    return WeightAttr(
        StringAttr(name),
        WeightTableIndex(index),
        make_dense_floats(coeffs),
    )


def make_acquisition(name: str, index: int, num_bins: int) -> AcquisitionAttr:
    """Creates an ``AcquisitionAttr`` from Python primitives.

    :param name: Acquisition name.
    :param index: Table index in ``[0, 31]``.
    :param num_bins: Number of acquisition bins in ``[0, 7_000_000]``.
    :returns: An ``AcquisitionAttr``.
    """
    return AcquisitionAttr(
        StringAttr(name),
        AcqTableIndex(index),
        BinCountImm(num_bins),
    )


def make_sequencer_config(
    integration_length: int | None = None,
    port_id: str | None = None,
    carrier_frequency: float | None = None,
    connections: Iterable[ConnectionAttr] | None = None,
    output_path_connections: Iterable[OutputPathConnectionAttr] | None = None,
    acquisition_path_connections: Iterable[AcquisitionPathConnectionAttr] | None = None,
    acquisition_enabled: bool | None = None,
    disabled_outputs: Iterable[int] | None = None,
    disabled_acquisition_paths: Iterable[SignalPath] | None = None,
    acquisition_disabled: bool | None = None,
    local_oscillator_id: str | None = None,
    enable_sync: bool | None = None,
    nco: NcoConfigAttr | None = None,
    awg: AwgConfigAttr | None = None,
    mixer: MixerCorrectionConfigAttr | None = None,
    marker_switch: MarkerOverrideConfigAttr | None = None,
    acquire: AcquireConfigAttr | None = None,
    thresholded_acquire: ThresholdedAcquireConfigAttr | None = None,
) -> SequencerConfigAttr:
    """Create a sequencer configuration from Python values and nested attributes.

    :param integration_length: Unweighted acquisition integration length in samples.
    :param port_id: Canonical port the sequencer drives.
    :param carrier_frequency: Carrier frequency in Hz.
    :param connections: Connections accepted by ``connect_sequencer``.
    :param output_path_connections: Physical outputs selected for output signal paths.
    :param acquisition_path_connections: Physical inputs selected for acquisition paths.
    :param acquisition_enabled: Explicit acquisition enable state.
    :param disabled_outputs: Physical outputs explicitly configured as off.
    :param disabled_acquisition_paths: Acquisition paths explicitly configured as off.
    :param acquisition_disabled: Whether the combined acquisition path is off.
    :param local_oscillator_id: Local oscillator mixed with the sequencer NCO.
    :param enable_sync: Whether the sequencer joins party-line synchronisation.
    :param nco: Numerically controlled oscillator configuration.
    :param awg: Arbitrary waveform generator configuration.
    :param mixer: Mixer correction configuration.
    :param marker_switch: Marker override configuration.
    :param acquire: Acquisition path configuration.
    :param thresholded_acquire: Thresholded acquisition configuration.
    :returns: A ``SequencerConfigAttr``.
    """

    return SequencerConfigAttr(
        port_id=port_id,
        carrier_frequency=carrier_frequency,
        connections=connections,
        output_path_connections=output_path_connections,
        acquisition_path_connections=acquisition_path_connections,
        acquisition_enabled=acquisition_enabled,
        disabled_outputs=disabled_outputs,
        disabled_acquisition_paths=disabled_acquisition_paths,
        acquisition_disabled=acquisition_disabled,
        local_oscillator_id=local_oscillator_id,
        enable_sync=enable_sync,
        nco=nco,
        awg=awg,
        mixer=mixer,
        marker_switch=marker_switch,
        unweighted_acquire=(
            UnweightedAcquireConfigAttr(integration_length)
            if integration_length is not None
            else None
        ),
        acquire=acquire,
        thresholded_acquire=thresholded_acquire,
    )


def make_module_config(
    slot_idx: int,
    instrument_id: str,
    kind: QbloxModuleKind,
    outputs: Iterable[OutputConfigAttr] = (),
    inputs: Iterable[InputConfigAttr] = (),
    local_oscillators: Iterable[LocalOscillatorConfigAttr] = (),
) -> ModuleConfigAttr:
    """Create a module configuration from Python primitives.

    :param slot_idx: Cluster slot the module occupies.
    :param instrument_id: Cluster instrument containing the module.
    :param kind: Module kind.
    :param outputs: Physical output configurations.
    :param inputs: Physical input configurations.
    :param local_oscillators: Local oscillators referenced by the module.
    :returns: A ``ModuleConfigAttr``.
    """

    return ModuleConfigAttr(
        slot_idx,
        instrument_id,
        kind,
        outputs,
        inputs,
        local_oscillators,
    )
