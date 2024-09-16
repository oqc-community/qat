# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from abc import abstractmethod
from typing import Dict, Union

from qat.purr.compiler.devices import Calibratable, PhysicalBaseband, PhysicalChannel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class Instrument(Calibratable):
    """
    An interface for any live instrument. It requires a unique address (IP address, USB VISA address, etc.).
    It is derived from Calibratable so the instruments will be saved in the calibration files. To avoid saving
    driver specific data in the calibration files, the actual drivers should be a property of this object, so the
    calibration will skip it.
    """

    def __init__(self, address, id_=None):
        super().__init__()
        self.id = id_ if id_ else address
        self.address = address
        self._driver = None
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            self.disconnect()
        self.is_connected = True
        log.info(f"{type(self).__name__} with ID {self.id} connected")
        return self.is_connected

    def close(self):
        return self.disconnect()

    def disconnect(self):
        if self.driver is not None:
            try:
                self.driver.close()
                self._driver = None
                self.is_connected = False
                log.info(f"{type(self).__name__} with ID {self.id} disconnected")
            except BaseException as e:
                log.warning(
                    f"Failed to close instrument at: {self.address} ID: {self.id}\n{str(e)}"
                )
        return self.is_connected

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, obj):
        if self._driver is not None:
            self.disconnect()
        self._driver = obj

    def __getstate__(self) -> Dict:
        results = super(Instrument, self).__getstate__()
        results["_driver"] = None
        return results

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._driver = None
        self.is_connected = False

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id}, address={self.address})"


class ControlHardwareChannel(PhysicalChannel):
    """
    Wrapper over a PhysicalChannel, that maps to a live instrument channel. This (and
    derived) object should contain hardware specific information.
    """

    def __init__(
        self, id_, hardware_id, dcbiaschannel_pair, switch_ch=None, *args, **kwargs
    ):
        super().__init__(id_, *args, **kwargs)
        self.hardware_id = hardware_id
        self.dcbiaschannel_pair: Dict[str, DCBiasChannel] = dcbiaschannel_pair
        self.switch_ch = switch_ch


class ControlHardware(Instrument):
    """
    The actual control hardware. For now, there is only one per LiveHardwareModel. It
    contains all the physical channels, since in most of the cases, you interact with
    the control unit rather than with the individual channels.
    """

    def __init__(self, id_=None):
        super().__init__(address=None, id_=id_)
        self.channels: Dict[str, ControlHardwareChannel] = {}

    def add_physical_channel(self, *physical_channel: ControlHardwareChannel):
        for physical_channel in physical_channel:
            if physical_channel.id not in self.channels:
                self.channels[physical_channel.id] = physical_channel

        return physical_channel

    def set_data(self, data):
        pass

    def start_playback(self, repetitions, repetition_time):
        pass


class LivePhysicalBaseband(PhysicalBaseband):
    """
    A wrapper over the PhysicalBaseband, that connects to a live instrument.
    """

    def __init__(
        self,
        id_,
        frequency,
        if_frequency,
        instrument: Instrument = None,
        channel_idx=None,
    ):
        self.instrument = instrument
        self.instrument_id = instrument.id if instrument is not None else None
        self.channel_idx = channel_idx
        super().__init__(id_, frequency, if_frequency)

    def connect_to_instrument(self):
        """
        Delayed connection to an instrument, in order to allow instantiating a
        LivePhysicalBaseband without having the live instrument ready.
        """
        if self.instrument is None:
            raise ValueError(f"Cannot connect to instrument '{self.instrument}'!")
        if not self.instrument.is_connected:
            self.instrument.connect()

    @abstractmethod
    def turn_on(self):
        pass

    @abstractmethod
    def turn_off(self):
        pass

    @abstractmethod
    def get_output_state(self):
        pass

    def set_frequency(self, value):
        self.frequency = value


class DCBiasChannel(Calibratable):
    """
    This is generic DC Bias Channel class, It would ONLY accept DC bias card as
    instrument which needs to have get_voltage and set_voltage function.
    """

    def __init__(
        self,
        channel_idx=None,
        bias_value: Union[float, int] = 0,
        instrument: Instrument = None,
    ):
        super().__init__()
        self.channel_idx = channel_idx
        self.instrument = instrument
        self._bias_value = bias_value
        self.max_value = 1
        self.min_value = -1

    def connect_to_instrument(self):
        if self.instrument is None:
            raise ValueError(
                "Cannot connect to instrument 'None', please define/add a DC bias Card!"
            )
        if not self.instrument.is_connected:
            self.instrument.connect()

    @property
    def bias_value(self):
        return self._bias_value

    @bias_value.setter
    def bias_value(self, value):
        self._bias_value = value
