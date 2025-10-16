# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from __future__ import annotations

import re
from typing import Any

from pydantic import Field
from pydantic_extra_types.semantic_version import SemanticVersion
from semver import Version

from qat.ir.waveforms import Waveform
from qat.model.device import Resonator
from qat.model.hardware_model import PhysicalHardwareModel
from qat.utils.pydantic import (
    CalibratablePositiveFloat,
    NoExtraFieldsFrozenModel,
    NoExtraFieldsModel,
    find_all_subclasses,
)
from qat.utils.units import Frequency, Scale, Time

VERSION = Version(0, 0, 1)


class FeatureMetadata(NoExtraFieldsFrozenModel):
    """
    Exposed metadata for various languages, integrations and features.
    """

    name: str
    description: str
    version: SemanticVersion = Field(frozen=True, repr=False, default=VERSION)
    enabled: bool = True


class OpenPulseConstraints(NoExtraFieldsModel):
    pulse_control_contraints: str = None
    max_scale: float = 1.0
    max_waveform_amplitude: float = 1.0
    awg_frequency_bandwidth: Frequency = Frequency(amount=1, scale=Scale.DEFAULT)
    max_pulse_duration: Time = Time(amount=120, scale=Scale.MICRO)
    min_pulse_duration: Time = Time(amount=8, scale=Scale.NANO)
    min_pulse_time: Time = Time(amount=8, scale=Scale.NANO)


class OpenPulseFrame(NoExtraFieldsModel):
    qubits: list[str]
    port_id: str
    frequency: CalibratablePositiveFloat
    bandwidth_centre: CalibratablePositiveFloat
    phase: float = 0.0


class OpenPulsePort(NoExtraFieldsModel):
    direction: str
    type: str
    associated_qubits: list[int]


class OpenPulseFeatures(FeatureMetadata):
    """
    Features specific to OpenPulse.
    """

    ports: dict[str, OpenPulsePort]
    frames: dict[str, Any]
    waveforms: dict[str, Any]
    constraints: OpenPulseConstraints | None = None

    @staticmethod
    def from_hardware(model: PhysicalHardwareModel) -> OpenPulseFeatures:
        if not isinstance(model, PhysicalHardwareModel):
            raise ValueError(
                "OpenPulseFeatures can only be generated from a `model` of type `PhysicalHardwareModel`."
            )

        # Generate frames from the hardware model.
        frames = {}
        for qubit_id, qubit in model.qubits.items():
            for pulse_channel in qubit.all_qubit_and_resonator_pulse_channels:
                pulse_ch_name = pulse_channel.__class__.__name__
                ignored = any(
                    ignored_pc in pulse_ch_name.lower()
                    for ignored_pc in ["freqshift", "secondstate"]
                )

                if not ignored:
                    pulse_ch_type = re.sub(r"PulseChannel$", "", pulse_ch_name)
                    pulse_ch_type = CamelCase_to_snake(pulse_ch_type)

                    qubits = [f"Q{qubit_id}"]
                    device = model.device_for_pulse_channel_id(pulse_channel.uuid)

                    # Generate a unique frame name for each pulse channel.
                    frame_name = (
                        f"r{qubit_id}_"
                        if isinstance(device, Resonator)
                        else f"q{qubit_id}_"
                    )

                    if hasattr(
                        pulse_channel, "auxiliary_qubit"
                    ):  # CR and CRC pulse channels
                        frame_name += f"q{pulse_channel.auxiliary_qubit}_"
                        qubits.append(f"Q{pulse_channel.auxiliary_qubit}")

                    frame_name += pulse_ch_type

                    frames[frame_name] = OpenPulseFrame(
                        qubits=qubits,
                        port_id=f"channel_{device.physical_channel.name_index}",
                        frequency=pulse_channel.frequency,
                        bandwidth_centre=device.physical_channel.baseband.frequency,
                        phase=0.0,
                    )

        # Generate ports from the hardware model.
        ports = {}
        for qubit_id, qubit in model.qubits.items():
            port_name_q = f"channel_{qubit.physical_channel.name_index}"
            port_name_r = f"channel_{qubit.resonator.physical_channel.name_index}"
            ports[port_name_q] = OpenPulsePort(
                direction="one-way",
                type="port_type_1",
                associated_qubits=[qubit_id],
            )
            ports[port_name_r] = OpenPulsePort(
                direction="two-way",
                type="port_type_1",
                associated_qubits=[qubit_id],
            )

        # Generate waveforms from the available waveform classes.
        waveforms = find_all_subclasses(Waveform)
        waveforms = {CamelCase_to_snake(waveform.name()): "test" for waveform in waveforms}

        constraints = OpenPulseConstraints()

        return OpenPulseFeatures(
            name="OpenPulse",
            description="Features for OpenPulse integration.",
            version=VERSION,
            ports=ports,
            frames=frames,
            waveforms=waveforms,
            constraints=constraints,
        )

    def to_json_dict(self):
        # This method is to keep backwards compatibility with purr OpenPulse features.
        blob = self.model_dump()
        return {"open_pulse": blob}


def CamelCase_to_snake(name: str) -> str:
    # Insert underscore before each capital letter (except the first), then lowercase
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
