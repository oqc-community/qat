from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Dict, List

from pydantic import model_validator

from qat.model.devices import PhysicalBaseband, PhysicalChannel
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.pydantic import WarnOnExtraFieldsModel

log = get_default_logger()


class AbstractInstrument(WarnOnExtraFieldsModel):
    """
    A dataclass for any live instrument. It requires a unique address (IP address, USB VISA address, etc.).
    To avoid saving driver specific data in the calibration files, the actual drivers should be a property of
    this object, so the calibration will skip it.
    """

    id: str | None = None
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
        instruments: The instruments that can be connected or disconnected.
    """

    instruments: AbstractInstrument

    def instrument_connected(self, name: str):
        return self.instruments[name].is_connected

    def all_instruments_connected(self):
        return all([instrument.is_connected for instrument in self.instruments])

    def connect(self):
        for instrument in self.instruments:
            log.info(f"{type(instrument).__name__} with ID {instrument.id} connected.")
            instrument.is_connected = True

        return self.all_instruments_connected

    def disconnect(self):
        connected = []
        for instrument in self.instruments:
            if instrument.driver is not None:
                try:
                    instrument.driver.close()
                    instrument.driver = None
                    instrument.is_connected = False
                    log.info(f"{type(instrument).__name__} with ID {self.id} disconnected.")
                except BaseException as e:
                    log.warning(
                        f"Failed to close instrument {type(instrument).__name__} at: "
                        f"{instrument.address} ID: {instrument.id}\n{str(e)}"
                    )
                connected.append(False)

        return all(connected)


class DCBiasChannel(WarnOnExtraFieldsModel):
    """
    This is generic DC Bias Channel class, It would ONLY accept DC bias card as
    instrument which needs to have get_voltage and set_voltage function.

    Attributes:
        channel_idx: The index of the DC bias channel.
        bias_value:
        instrument: The live instrument connected to this bias channel.
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

    channel_idx: The index of the channel associated with this channel.
    dcbiaschannel_pair: ???
    switch_ch: ???
    """

    channel_idx: int
    dcbiaschannel_pair: Dict[str, DCBiasChannel]
    switch_ch: Instrument | None = None

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

        return physical_channel

    @abstractmethod
    def start_playback(self): ...
