# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Base dataclasses and interface for hardware-specific derived views of canonical system
data.

This module provides the building blocks for describing a complete RF tone generation setup:
a :class:`Generator` groups the :class:`Sequencer` instances and
:class:`LocalOscillator` instances required to produce a single tone (e.g. LO at
4.4 GHz + sequencer NCO at 240 MHz = 4.64 GHz output). The
:class:`HardwareViewInterface` interface ties these together into a derived view of
:class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData`, enforcing
that every concrete backend view exposes its generators and the acquisition limit drawn
from the canonical system data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.derived.interface import DerivedViewInterface


@dataclass(frozen=True, slots=True)
class Sequencer(ABC):
    """Abstract hardware sequencer description.

    Concrete hardware sequencer classes can expose arbitrary capabilities via ``fields``
    and optional numeric bounds via ``min_values``/``max_values``.

    :ivar fields: Hardware-specific sequencer fields keyed by string name.
    :ivar min_values: Optional minimum bounds keyed by field name.
    :ivar max_values: Optional maximum bounds keyed by field name.
    """

    fields: dict[str, object] = field(default_factory=dict)
    min_values: dict[str, float] = field(default_factory=dict)
    max_values: dict[str, float] = field(default_factory=dict)

    @classmethod
    @abstractmethod
    def kind(cls) -> str:
        """Return the hardware-specific sequencer kind."""


@dataclass(frozen=True, slots=True)
class LocalOscillator:
    """Placeholder for hardware-specific local oscillator attributes."""

    id: str
    frequency: int


@dataclass(frozen=True, slots=True)
class Generator:
    """Hardware generator representing a complete tone generation unit.

    Contains sequencers and local oscillators that work together to generate a tone.

    :ivar port_id: Port ID this generator is derived from.
    :ivar sample_time: Sample time in picoseconds.
    :ivar sequencers: Tuple of sequencers that drive this generator.
    :ivar local_oscillators: Tuple of local oscillators used by this generator.
    """

    port_id: str
    sample_time: int
    sequencers: tuple[Sequencer, ...] = ()
    local_oscillators: tuple[LocalOscillator, ...] = ()


class HardwareViewInterface(DerivedViewInterface[CanonicalSystemData], ABC):
    """Abstract interface for hardware-specific derived views of canonical system data.

    Concrete backend views must implement :meth:`derive` (inherited from
    :class:`~qat.experimental.system_data.derived.interface.DerivedViewInterface`) and
    expose :attr:`acquire_limit` and :attr:`generators`.

    :ivar acquire_limit: Maximum allowed acquisitions for a single execution batch.
    :ivar generators: Generators available in this hardware view.
    """

    acquire_limit: int
    generators: tuple[Generator, ...]
