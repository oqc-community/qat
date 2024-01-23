# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from qat.purr.compiler.devices import MeasureChannel
from qat.purr.compiler.waveforms import AbstractWaveform
from qat.purr.integrations.qasm import (
    extern_port_name,
    get_frame_mappings,
    get_port_mappings,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class FeatureMetadata:
    """Exposed metadata for various languages, integrations and features."""

    def to_json_dict(self):
        """
        Turn object into a JSON-amenable dictionary to return from a web service. All
        returned dictionaries should be able to be merged together without additional
        effort, so make sure a root node with the name of the feature is available.
        """


class Scale(Enum):
    """
    SI units of frequency
    """

    NANO = "ns"
    MICRO = "us"
    MILLI = "ms"
    DEFAULT = ""
    KILO = "k"
    MEGA = "M"
    GIGA = "G"
    TERA = "T"


class Unit(Enum):
    """
    Physical SI units.
    """

    TIME = "s"
    FREQUENCY = "Hz"


# TODO: Should be two different objects to deal with time/frequency isntead of an enum.
@dataclass
class Quantity:
    amount: float
    unit: Unit
    scale: Scale

    def __repr__(self):
        return f"{self.amount} {self.scale.value}{self.unit.value}"


@dataclass
class Constraints:
    pulse_control_contraints: str = None
    max_scale: float = 1
    max_waveform_amplitude: float = 1
    awg_frequency_bandwidth: str = Quantity(1, Unit.FREQUENCY, Scale.DEFAULT)
    max_pulse_duration: str = Quantity(120, Unit.TIME, Scale.MICRO)
    min_pulse_duration: str = Quantity(8, Unit.TIME, Scale.NANO)
    min_pulse_time: str = Quantity(8, Unit.TIME, Scale.NANO)


class OpenPulseFeatures(FeatureMetadata):
    def __init__(self):
        self.ports = dict()
        self.frames = dict()
        self.waveforms = dict()
        self.constraints: Optional[Constraints] = None

    @staticmethod
    def from_hardware(hardware):
        feature = OpenPulseFeatures()
        feature.for_hardware(hardware)
        return feature

    def for_hardware(self, hardware):
        for frame_name, frame in get_frame_mappings(hardware).items():
            if isinstance(frame, MeasureChannel):
                qb = [frame.related_qubit]
            else:
                qb = frame.related_devices

            self.frames[frame_name] = dict(
                qubits=qb if qb is not None else [],
                port_id=extern_port_name(frame.physical_channel),
                frequency=frame.frequency,
                bandwidth_centre=frame.baseband_frequency,
                phase=0.0,
            )

        for port_name, port in get_port_mappings(hardware).items():
            qb = port.related_qubit
            self.ports[port_name] = dict(
                direction="two-way" if port.acquire_allowed else "one-way",
                type="port_type_1",
                associated_qubits=[qb] if qb is not None else [],
            )

        self.waveforms = {
            key: vars(value.waveform_definition)
            for key, value in AbstractWaveform.actual_waveforms.items()
        }

        self.constraints = Constraints()

    def to_json_dict(self):
        return {
            "open_pulse": {
                "version": "0",
                "ports": self.ports,
                "frames": self.frames,
                "waveforms": self.waveforms,
                "constraints": self.constraints.__dict__,
            }
        }
