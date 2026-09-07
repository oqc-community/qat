# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Dependency-neutral target description for Qblox hardware.

The records in this module model the Q1 sequencer and module architecture documented by
Qblox. They describe a compilation target, not the installed hardware or its calibrated
configuration.

See:

* https://docs.qblox.com/en/main/products/architecture/sequencers/sequencer.html
* https://docs.qblox.com/en/main/products/qblox_instruments/api/sequencer.html
* https://docs.qblox.com/en/main/products/architecture/modules/index.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from frozendict import frozendict

from qat.experimental.system_data.hardware.qblox.models import QbloxModuleKind


class Q1SequencerType(str, Enum):
    """Q1 sequencer types defined by Qblox."""

    control = "control"
    readout = "readout"


class Q1SequencerFeature(str, Enum):
    """Feature predicates exposed by a Q1 sequencer type."""

    awg = "awg"
    acquisition = "acquisition"


@dataclass(frozen=True, slots=True)
class Q1AsmSpec:
    """Immediate and register limits of the Q1ASM instruction set."""

    min_gain: int = -(1 << 15)
    max_gain: int = (1 << 15) - 1
    min_offset: int = -(1 << 15)
    max_offset: int = (1 << 15) - 1
    max_wait_time_ns: int = (1 << 16) - 4
    register_size: int = (1 << 32) - 1
    loop_unroll_threshold: int = 4


@dataclass(frozen=True, slots=True)
class ReadoutSpec:
    """Limits of the acquisition path added by a readout sequencer."""

    min_integration_length_samples: int = 4
    max_integration_length_samples: int = (1 << 24) - 4
    min_acquisition_threshold: int = -((1 << 24) - 4)
    max_acquisition_threshold: int = (1 << 24) - 4
    weight_sample_capacity: int = 16_384


@dataclass(frozen=True, slots=True)
class Q1SequencerSpec:
    """Static limits and features of one Q1 sequencer type."""

    type: Q1SequencerType
    instruction_capacity: int
    clock_period_ns: int = 4
    sample_rate_hz: int = 1_000_000_000
    nco_min_frequency_hz: float = -500_000_000.0
    nco_max_frequency_hz: float = 500_000_000.0
    nco_phase_steps: int = 1_000_000_000
    nco_phase_steps_per_degree: float = 1_000_000_000 / 360
    nco_frequency_steps_per_hz: int = 4
    nco_frequency_limit_steps: int = 2_000_000_000
    register_count: int = 64
    waveform_sample_capacity: int = 16_384
    readout: ReadoutSpec | None = None

    def __post_init__(self) -> None:
        if (self.type is Q1SequencerType.readout) != (self.readout is not None):
            raise ValueError("Readout sequencer specifications require readout-path limits")

    @property
    def features(self) -> frozenset[Q1SequencerFeature]:
        """Return the feature predicates implied by this sequencer type."""

        features = {Q1SequencerFeature.awg}
        if self.type is Q1SequencerType.readout:
            features.add(Q1SequencerFeature.acquisition)
        return frozenset(features)

    def supports(self, feature: Q1SequencerFeature) -> bool:
        """Return whether this sequencer type provides ``feature``."""

        return feature in self.features


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    """Physical sequencer bank and channel map of one Qblox module kind."""

    kind: QbloxModuleKind
    sequencers: tuple[Q1SequencerType, ...]
    output_count: int
    input_count: int
    output_channel_map: frozendict[int, tuple[int, ...]]
    input_channel_map: frozendict[int, tuple[int, ...]] = field(default_factory=frozendict)
    marker_count: int = 0
    acquisition_memory_bins: int | None = None
    supports_mixer_correction: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.sequencers, tuple):
            raise TypeError("Module sequencers must be an immutable tuple")
        if not isinstance(self.output_channel_map, frozendict) or not isinstance(
            self.input_channel_map, frozendict
        ):
            raise TypeError("Module channel maps must be immutable frozendict values")
        if any(
            not isinstance(indices, tuple)
            for channel_map in (self.output_channel_map, self.input_channel_map)
            for indices in channel_map.values()
        ):
            raise TypeError("Module channel-map sequencer indices must be tuples")
        if set(self.output_channel_map) != set(range(self.output_count)):
            raise ValueError(f"{self.kind.value} output channel map is incomplete")
        if set(self.input_channel_map) != set(range(self.input_count)):
            raise ValueError(f"{self.kind.value} input channel map is incomplete")

        indices = set(range(len(self.sequencers)))
        if any(
            set(routed_indices) - indices
            for routed_indices in (
                *self.output_channel_map.values(),
                *self.input_channel_map.values(),
            )
        ):
            raise ValueError(
                f"{self.kind.value} channel map references an invalid sequencer"
            )
        if any(
            self.sequencers[index] is not Q1SequencerType.readout
            for routed_indices in self.input_channel_map.values()
            for index in routed_indices
        ):
            raise ValueError(
                f"{self.kind.value} input channel map references a control sequencer"
            )
        has_readout = Q1SequencerType.readout in self.sequencers
        if has_readout != (self.acquisition_memory_bins is not None):
            raise ValueError(
                f"{self.kind.value} acquisition memory must match its sequencer types"
            )

    @property
    def sequencer_count(self) -> int:
        """Return the number of physical Q1 sequencers."""

        return len(self.sequencers)

    def sequencer_indices(self, type_: Q1SequencerType | None = None) -> tuple[int, ...]:
        """Return all sequencer indices, optionally filtered by Q1 sequencer type."""

        return tuple(
            index
            for index, sequencer_type in enumerate(self.sequencers)
            if type_ is None or sequencer_type is type_
        )


@dataclass(frozen=True, slots=True)
class SequencerTarget:
    """Module-bound view of one physical Q1 sequencer."""

    index: int
    spec: Q1SequencerSpec
    output_channels: tuple[int, ...]
    input_channels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QbloxTargetDescription:
    """Immutable Qblox target specification and typed query boundary."""

    q1asm: Q1AsmSpec
    sequencer_specs: frozendict[Q1SequencerType, Q1SequencerSpec]
    module_specs: frozendict[QbloxModuleKind, ModuleSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.sequencer_specs, frozendict) or not isinstance(
            self.module_specs, frozendict
        ):
            raise TypeError("Target specification maps must be immutable frozendict values")
        if set(self.sequencer_specs) != set(Q1SequencerType):
            raise ValueError("Target description must define every Q1 sequencer type")
        if any(type_ is not spec.type for type_, spec in self.sequencer_specs.items()):
            raise ValueError("Sequencer specification keys must match their types")
        if set(self.module_specs) != set(QbloxModuleKind):
            raise ValueError("Target description must define every Qblox module kind")
        if any(kind is not spec.kind for kind, spec in self.module_specs.items()):
            raise ValueError("Module specification keys must match their kinds")

    def module(self, kind: QbloxModuleKind) -> ModuleSpec:
        """Return the target specification for ``kind``."""

        return self.module_specs[kind]

    def sequencer_spec(self, type_: Q1SequencerType) -> Q1SequencerSpec:
        """Return the shared target specification for a Q1 sequencer type."""

        return self.sequencer_specs[type_]

    def sequencer(self, kind: QbloxModuleKind, index: int) -> SequencerTarget:
        """Return the module-bound target view of a physical sequencer."""

        module = self.module(kind)
        if index < 0 or index >= module.sequencer_count:
            raise ValueError(f"Sequencer index {index} is invalid for {kind.value}")
        return SequencerTarget(
            index=index,
            spec=self.sequencer_spec(module.sequencers[index]),
            output_channels=tuple(
                channel
                for channel, indices in module.output_channel_map.items()
                if index in indices
            ),
            input_channels=tuple(
                channel
                for channel, indices in module.input_channel_map.items()
                if index in indices
            ),
        )

    def supports(
        self,
        kind: QbloxModuleKind,
        index: int,
        feature: Q1SequencerFeature,
    ) -> bool:
        """Return whether a physical sequencer provides ``feature``."""

        return self.sequencer(kind, index).spec.supports(feature)

    def output_sequencers(self, kind: QbloxModuleKind, output: int) -> tuple[int, ...]:
        """Return sequencers routable to a physical output channel."""

        try:
            return self.module(kind).output_channel_map[output]
        except KeyError as error:
            raise ValueError(
                f"Output channel out{output} is invalid for {kind.value}"
            ) from error

    def input_sequencers(self, kind: QbloxModuleKind, input_: int) -> tuple[int, ...]:
        """Return sequencers routable from a physical input channel."""

        try:
            return self.module(kind).input_channel_map[input_]
        except KeyError as error:
            raise ValueError(
                f"Input channel in{input_} is invalid for {kind.value}"
            ) from error


_CONTROL = Q1SequencerType.control
_READOUT = Q1SequencerType.readout
_SIX_CONTROL = (_CONTROL,) * 6
_SIX_READOUT = (_READOUT,) * 6

# TODO: Remove the overlapping legacy QbloxTargetData constants once all experimental
# lowering consumers use QbloxTargetDescription.
DEFAULT_QBLOX_TARGET = QbloxTargetDescription(
    q1asm=Q1AsmSpec(),
    sequencer_specs=frozendict(
        {
            _CONTROL: Q1SequencerSpec(type=_CONTROL, instruction_capacity=16_384),
            _READOUT: Q1SequencerSpec(
                type=_READOUT,
                instruction_capacity=12_288,
                readout=ReadoutSpec(),
            ),
        }
    ),
    module_specs=frozendict(
        {
            QbloxModuleKind.qcm: ModuleSpec(
                kind=QbloxModuleKind.qcm,
                sequencers=_SIX_CONTROL,
                output_count=4,
                input_count=0,
                output_channel_map=frozendict(
                    {output: tuple(range(6)) for output in range(4)}
                ),
                marker_count=4,
            ),
            QbloxModuleKind.qcm_rf: ModuleSpec(
                kind=QbloxModuleKind.qcm_rf,
                sequencers=_SIX_CONTROL,
                output_count=2,
                input_count=0,
                output_channel_map=frozendict(
                    {output: tuple(range(6)) for output in range(2)}
                ),
                marker_count=2,
            ),
            QbloxModuleKind.qrm: ModuleSpec(
                kind=QbloxModuleKind.qrm,
                sequencers=_SIX_READOUT,
                output_count=2,
                input_count=2,
                output_channel_map=frozendict(
                    {output: tuple(range(6)) for output in range(2)}
                ),
                input_channel_map=frozendict(
                    {input_: tuple(range(6)) for input_ in range(2)}
                ),
                marker_count=4,
                acquisition_memory_bins=3_000_000,
            ),
            QbloxModuleKind.qrm_rf: ModuleSpec(
                kind=QbloxModuleKind.qrm_rf,
                sequencers=_SIX_READOUT,
                output_count=1,
                input_count=1,
                output_channel_map=frozendict({0: tuple(range(6))}),
                input_channel_map=frozendict({0: tuple(range(6))}),
                marker_count=2,
                acquisition_memory_bins=3_000_000,
            ),
            QbloxModuleKind.qrc: ModuleSpec(
                kind=QbloxModuleKind.qrc,
                sequencers=(_READOUT,) * 8 + (_CONTROL,) * 4,
                output_count=6,
                input_count=2,
                output_channel_map=frozendict(
                    {
                        0: tuple(range(8)),
                        1: tuple(range(8)),
                        2: (0, 4, *range(8, 12)),
                        3: (1, 5, *range(8, 12)),
                        4: (2, 6, *range(8, 12)),
                        5: (3, 7, *range(8, 12)),
                    }
                ),
                input_channel_map=frozendict(
                    {input_: tuple(range(8)) for input_ in range(2)}
                ),
                marker_count=1,
                acquisition_memory_bins=7_000_000,
                supports_mixer_correction=False,
            ),
        }
    ),
)
