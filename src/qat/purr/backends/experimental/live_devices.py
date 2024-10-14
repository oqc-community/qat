from __future__ import annotations

from typing import Dict

from pydantic import model_validator

from qat.purr.compiler.experimental.devices import PhysicalBaseband, PhysicalChannel
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.pydantic import WarnOnExtraFieldsModel

log = get_default_logger()


class DCBiasChannel(WarnOnExtraFieldsModel):
    """
    This is generic DC Bias Channel class, It would ONLY accept DC bias card as
    instrument which needs to have get_voltage and set_voltage function.

    Attributes
        channel_idx: Index of the DC bias channel.
        bias_value:
    """

    channel_idx: int | None = None
    bias_value: float | int = 0
    instrument: Instrument | None = None
    min_value: int = -1
    max_value: int = 1

    @model_validator(mode="after")
    def check_min_max_values(self):
        if self.min_value > self.max_value:
            raise ValueError(
                f"Min value {self.min_value} of {self.__class__.name} with instrument {str(self.instrument)} is larger than its max value {self.max_value}."
            )
        return self


class LivePhysicalBaseband(PhysicalBaseband):
    """
    A wrapper over the PhysicalBaseband, that connects to a live instrument.
    """

    channel_idx: int | None = None
    instrument: Instrument | None = None


class ControlHardwareChannel(PhysicalChannel):
    """
    Wrapper over a PhysicalChannel, that maps to a live instrument channel.
    This (and derived) object should contain hardware specific information.
    """

    hardware_id: str
    dcbiaschannel_pair: Dict[str, DCBiasChannel]
    switch_ch: Instrument | None = None


class Instrument(WarnOnExtraFieldsModel):
    """
    An interface for any live instrument. It requires a unique address (IP address, USB VISA address, etc.).
    To avoid saving driver specific data in the calibration files, the actual drivers should be a property of
    this object, so the calibration will skip it.
    """

    address: str
    id: str | None = None
    is_connected: bool = False

    @model_validator(mode="after")
    def check_id(self):
        if self.id is None:
            self.id = self.address
        return self

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id}, address={self.address})"


class ControlHardware(Instrument):
    channels: Dict[str, ControlHardwareChannel] = {}

    def add_physical_channel(self, *physical_channel: ControlHardwareChannel):
        for pc in physical_channel:
            if pc.id not in self.channels:
                self.channels[pc.id] = pc
            else:
                log.warning(
                    f"Physical channel {str(physical_channel)} already in channels of {str(self)}."
                )

        return physical_channel
