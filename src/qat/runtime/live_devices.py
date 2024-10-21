from __future__ import annotations

import uuid
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Dict, List

from pydantic import PrivateAttr, model_validator

from qat.model.devices import PhysicalBaseband, PhysicalChannel
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import WarnOnExtraFieldsModel

log = get_default_logger()


class AbstractInstrument(WarnOnExtraFieldsModel):
    """
    A dataclass for any live instrument. It requires a unique address (IP address, USB VISA address, etc.).
    To avoid saving driver specific data in the calibration files, the actual drivers should be a property of
    this object, so the calibration will skip it.
    """

    id: str | None = None
    _uuid: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))
    is_connected: bool = False

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id})"


class Instrument(AbstractInstrument):
    address: str
    _driver: Any | None

    @model_validator(mode="after")
    def check_id(self):
        if self.id is None:
            self.id = self.address
        return self

    @property
    def driver(self):
        return self._driver

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id}, address={self.address})"

    def __iter__(self):
        yield self


class CompositeInstrument(AbstractInstrument, Iterable):
    instruments: List[AbstractInstrument]

    def add(self, instrument: AbstractInstrument):
        if self._get_index(instrument):
            raise KeyError(f"Instrument with id {instrument.id} is already present.")

        self.instruments.append(instrument)

    def remove(self, instrument: AbstractInstrument):
        if idx := self._get_index(instrument):
            self.instruments.pop(idx)
        else:
            raise KeyError(f"Instrument with id {instrument.id} not found.")

    def _get_index(self, instrument: AbstractInstrument):
        for idx, instr in enumerate(self.instruments):
            if instr.id == instrument.id:
                return idx

    def __iter__(self):
        for instrument in self.instruments:
            yield from instrument


class InstrumentConnectionManager(WarnOnExtraFieldsModel):
    """
    Interface to implement the connection and disconnection of an instrument to a unique address.

    Args:
        instrument: The (composite) instrument that can be connected or disconnected.
    """

    instrument: AbstractInstrument

    def all_instruments_connected(self):
        return all([sub_instrument.is_connected for sub_instrument in self.instrument])

    def connect_instrument(self, instrument: Instrument):
        for sub_instrument in self.instrument:
            if sub_instrument.__uuid == instrument._uuid:
                log.info(
                    f"{type(sub_instrument).__name__} with ID {sub_instrument.id} connected."
                )
                sub_instrument.is_connected = True

    def connect_instruments(self):
        for sub_instrument in self.instrument:
            log.info(
                f"{type(sub_instrument).__name__} with ID {sub_instrument.id} connected."
            )
            sub_instrument.is_connected = True

    def disconnect_instruments(self):
        connected = []
        for sub_instrument in self.instrument:
            if sub_instrument.driver is not None:
                try:
                    sub_instrument.driver.close()
                    sub_instrument.driver = None
                    sub_instrument.is_connected = False
                    log.info(
                        f"{type(sub_instrument).__name__} with ID {sub_instrument.id} disconnected."
                    )
                except BaseException as e:
                    log.warning(
                        f"Failed to close instrument {type(sub_instrument).__name__} at: "
                        f"{sub_instrument.address} ID: {sub_instrument.id}\n{str(e)}"
                    )
                connected.append(False)

        return all(connected)

    def disconnect_instrument(self, instrument: Instrument):
        for sub_instrument in self.instrument:
            if sub_instrument.__uuid == instrument._uuid:
                log.info(
                    f"{type(sub_instrument).__name__} with ID {sub_instrument.id} disconnected."
                )
                sub_instrument.is_connected = False


class DCBiasChannel(WarnOnExtraFieldsModel):
    """
    This is generic DC Bias Channel class, It would ONLY accept DC bias card as
    instrument which needs to have get_voltage and set_voltage function.

    Attributes:
        channel_idx: The index of the DC bias channel.
        bias_value: The numerical value for the bias.
        instrument: The live DC bias card connected to this bias channel.
        min_value: ???
        max_value: ???
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


class DCBiasChannelPair(WarnOnExtraFieldsModel):
    I: DCBiasChannel
    Q: DCBiasChannel


class LivePhysicalBaseband(PhysicalBaseband):
    """
    A wrapper over the PhysicalBaseband, that connects to a live instrument.

    Attributes:
        channel_idx: The index of the channel associated with the baseband.
        instrument: The live instrument connected to this baseband.
    """

    channel_idx: int | None = None
    instrument: Instrument | None = None

    @property
    def instrument_id(self):
        if self.instrument:
            return self.instrument.id
        else:
            return None


class ControlHardwareChannel(PhysicalChannel):
    """
    Wrapper over a PhysicalChannel, that maps to a live instrument channel.
    This (and derived) object should contain hardware specific information.

    Attributes:
        channel_idx: The index of this channel.
        dcbiaschannel_pair: The DC bias channels for the I and Q components.
        switch_ch: ID of the switch channel.
    """

    channel_idx: int
    dcbiaschannel_pair: DCBiasChannelPair
    switch_ch: str | None = None

    @model_validator(mode="after")
    def check_id(self):
        if str(self.channel_idx + 1) not in self.id:
            raise ValueError(
                f"The channel index of the control hardware channel {self.channel_idx} should"
                f"be in agreement with the id of the control hardware channel {self.id}."
            )


class ControlHardware(Instrument):
    channels: Dict[str, ControlHardwareChannel] = {}

    def add_physical_channel(self, *physical_channel: ControlHardwareChannel):
        for pc in physical_channel:
            if pc.id not in self.channels:
                self.channels[pc.id] = pc
            else:
                log.warning(
                    f"Physical channel {str(physical_channel)} already in {str(self)}."
                )

    @property
    def channels(self):
        return self.channels

    @abstractmethod
    def start_playback(self): ...
