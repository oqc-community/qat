# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from __future__ import annotations

import os
import sys
from enum import Enum, auto
from typing import Dict, List, Optional, Set, TypeVar, Union

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
from jsonpickle import Pickler, Unpickler
from jsonpickle.util import is_picklable

from qat.purr.utils.logger import get_default_logger

# TODO: Remove the setting of the recursion limit as soon as the pickling of hardware is
# not blocked by the default limit.
sys.setrecursionlimit(2000)

log = get_default_logger()
jsonpickle_numpy.register_handlers()


def build_resonator(
    resonator_id,
    resonator_channel,
    *args,
    measure_fixed_if=False,
    acquire_fixed_if=False,
    **kwargs,
):
    """Helper method to build a resonator with default channels."""
    reson = Resonator(resonator_id, resonator_channel)
    kwargs.pop("fixed_if", None)
    reson.create_pulse_channel(
        ChannelType.measure, *args, fixed_if=measure_fixed_if, **kwargs
    )
    reson.create_pulse_channel(
        ChannelType.acquire, *args, fixed_if=acquire_fixed_if, **kwargs
    )
    return reson


def build_qubit(
    index,
    resonator,
    physical_channel,
    drive_freq,
    second_state_freq=None,
    measure_amp: float = 1.0,
    fixed_drive_if=False,
    qubit_id=None,
):
    """
    Helper method tp build a qubit with assumed default values on the channels. Modelled
    after the live hardware.
    """
    qubit = Qubit(index, resonator, physical_channel, drive_amp=measure_amp, id_=qubit_id)
    qubit.create_pulse_channel(
        ChannelType.drive,
        frequency=drive_freq,
        scale=(1.0e-8 + 0.0j),
        fixed_if=fixed_drive_if,
    )

    qubit.create_pulse_channel(
        ChannelType.second_state, frequency=second_state_freq, scale=(1.0e-8 + 0.0j)
    )

    return qubit


def add_cross_resonance(qubit_one: Qubit, qubit_two: Qubit):
    """Adds cross-resonance couplings and channels between these two qubits."""

    def couple_qubits(qone, qtwo):
        """Builds the channels on qone that will all coupling to qtwo."""
        qone.add_coupled_qubit(qtwo)

        twobit_drive = qtwo.get_drive_channel()
        qone.create_pulse_channel(
            ChannelType.cross_resonance,
            frequency=twobit_drive.frequency,
            scale=(1.0e-7 + 0.0j),
            auxiliary_devices=[qtwo],
        )

        qtwo.create_pulse_channel(
            ChannelType.cross_resonance_cancellation,
            frequency=twobit_drive.frequency,
            scale=(0.0 + 0.0j),
            auxiliary_devices=[qone],
        )

    couple_qubits(qubit_one, qubit_two)
    couple_qubits(qubit_two, qubit_one)


class ChannelType(Enum):
    drive = auto()
    measure = auto()
    second_state = auto()
    cross_resonance = auto()
    cross_resonance_cancellation = auto()
    acquire = auto()
    freq_shift = auto()
    macq = auto()

    def __repr__(self):
        return self.name


class QuantumComponent:
    """
    Base class for any logical object which can act as a target of a quantum action - a
    Qubit or various channels for a simple example.
    """

    def __init__(self, id_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if id_ is None:
            id_ = ""
        self.id = str(id_)

    def full_id(self):
        return self.id

    def __repr__(self):
        return f"{self.full_id()}"


class Calibratable:
    """
    Allows this object to be loaded as a calibrated object when we get pickled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_calibrated = False

    def _is_serializable(self, val):
        """
        Checks whether this field value/name should be serialized as a calibration
        value.
        """
        if isinstance(val, (List, tuple)):
            return all([self._is_serializable(val_) for val_ in val])

        if isinstance(val, dict):
            return all([self._is_serializable(val_) for val_ in val.keys()]) and all(
                [self._is_serializable(val_) for val_ in val.values()]
            )

        if isinstance(val, Enum):
            return self._is_serializable(val.value)

        # This checks the field name in JSON is valid, which it'll always be for us.
        return is_picklable("", val)

    def __getstate__(self) -> Dict:
        """
        Returns a dictionary that will be used to pickle the object. By default
        everything that is considered valid by jsonpickle will be added.

        Override this method if you want to explicitly remove things from the
        serialization.
        """
        results = {}
        for key, value in self.__dict__.items():
            if self._is_serializable(value):
                results[key] = value

        return results

    def get_calibration(self):
        return jsonpickle.encode(self, indent=4, context=CyclicRefPickler())

    @staticmethod
    def load_calibration(calibration_string):
        reconstituted = jsonpickle.decode(calibration_string, context=CyclicRefUnpickler())
        if isinstance(reconstituted, str):
            raise ValueError("Loading from calibration string failed. Please regenerate.")

        return reconstituted

    def save_calibration_to_file(self, file_path, use_cwd=False):
        """
        Saves calibration to passed-in file path.
        use_cwd appends the working directory to any non-absolute path, otherwise it'll
        search for common calibration config folders before defaulting to the one in
        source code.
        """
        if use_cwd:
            file_path = os.path.join(os.getcwd(), file_path)

        with open(file_path, "w") as f:
            f.write(self.get_calibration())

    @staticmethod
    def load_calibration_from_file(file_path):
        """
        Looks for this calibration in the passed-in directory or the default calibration
        save location.
        """
        # Check to see if we're just trying to find a file on the working directory.
        cwd_path = os.path.join(os.getcwd(), file_path)
        if os.path.isfile(cwd_path):
            file_path = cwd_path

        with open(file_path, "r") as f:
            return Calibratable.load_calibration(f.read())


class CyclicRefPickler(Pickler):
    """
    Adds reference ID to each object that requires it, for use when re-pickling and
    generating accurate circular references.
    """

    ref_field = "py/obj_ref_id"

    def _flatten_obj_instance(self, obj):
        state = super()._flatten_obj_instance(obj)
        state[self.ref_field] = id(obj)
        return state

    def _log_ref(self, obj):
        is_new = super()._log_ref(obj)
        # We don't want complex' or enums to be referenced. Just create a new one on
        # each instance.
        if isinstance(obj, (complex, Enum, np.number)):
            return True

        if is_new:
            obj_id = id(obj)
            self._objs[obj_id] = obj_id

        return is_new


class FakeList(dict):
    """
    This is a patch-up object because originally the reference-counting mechanism for
    jsonpickle used incrementing ID's and relied upon the graph to be deterministic.

    This isn't the case in some situations, so we want to be able to match up object
    reference by their memory pointer instead. But we don't really want a list a few
    billion elements long for many reasons.

    This simply exposes the methods and functionality jsonpickle relies upon in the
    list, while also allowing us to use very large ints as lookup targets.
    """

    def append(self, val):
        """Appends a value on to the dictionary with key=length."""
        self[len(self)] = val

    def __getitem__(self, item):
        """
        Gets the value, but transforms item-not-found error into an IndexError as
        functionality is triggered upon this exception being thrown (and caught).
        """
        try:
            return super().__getitem__(item)
        except KeyError:
            raise IndexError()


class CyclicRefUnpickler(Unpickler):
    """
    Makes sure cyclic references are picked up correctly. Objects need to be pickled
    with CyclicRefPickler otherwise the tags required for this more precise detection
    won't exist.
    """

    _objs: FakeList

    def reset(self):
        """
        Replace _objs with our own dictionary upon every reset. This is called from
        the constructor.
        """
        super().reset()
        self._objs = FakeList()

    def _restore_object_instance_variables(self, obj, instance):
        # If we see an object with our custom reference pointer replace the incremental
        # number that the dictionary uses to refer to it with the original reference id.
        #
        # The existing code takes care of the reference mapping as long as the py/ref
        # points to the ref_field value. We also strip the custom tag from the
        # dictionary.
        if CyclicRefPickler.ref_field in obj:
            existing_id = obj.pop(CyclicRefPickler.ref_field)
            instance_id = id(instance)
            inc_mapping = self._obj_to_idx[instance_id]
            self._obj_to_idx[instance_id] = existing_id
            del self._objs[inc_mapping]
            self._objs[existing_id] = instance

        return super()._restore_object_instance_variables(obj, instance)


class PhysicalChannel(QuantumComponent, Calibratable):
    def __init__(
        self,
        id_: str,
        sample_time: float,
        baseband: PhysicalBaseband,
        block_size: Optional[int] = None,
        phase_offset: float = 0.0,
        imbalance: float = 1.0,
        acquire_allowed: bool = False,
        pulse_channel_min_frequency: float = 0.0,
        pulse_channel_max_frequency: float = np.inf,
    ):
        super().__init__(id_)
        self.sample_time: float = sample_time
        self.baseband: PhysicalBaseband = baseband
        self.block_size: int = block_size or 1
        self.phase_offset: float = phase_offset
        self.imbalance: float = imbalance

        self.acquire_allowed: bool = acquire_allowed
        self.pulse_channel_min_frequency: float = pulse_channel_min_frequency
        self.pulse_channel_max_frequency: float = pulse_channel_max_frequency

    def create_pulse_channel(
        self,
        id_: str,
        frequency=0.0,
        bias=0.0 + 0.0j,
        scale=1.0 + 0.0j,
        fixed_if: bool = False,
    ):
        pulse_channel = PulseChannel(id_, self, frequency, bias, scale, fixed_if)

        return pulse_channel

    def create_freq_shift_pulse_channel(
        self,
        id_: str,
        frequency=0.0,
        bias=0.0 + 0.0j,
        scale=1.0 + 0.0j,
        amp=0.0,
        active: bool = True,
        fixed_if: bool = False,
    ):
        pulse_channel = FreqShiftPulseChannel(
            id_, self, frequency, bias, scale, amp, active, fixed_if
        )

        return pulse_channel

    @property
    def block_time(self):
        return self.block_size * self.sample_time

    @property
    def baseband_frequency(self):
        return self.baseband.frequency

    @property
    def baseband_if_frequency(self):
        return self.baseband.if_frequency


class PhysicalBaseband(QuantumComponent, Calibratable):
    def __init__(self, id_: str, frequency: float, if_frequency: float = 250e6):
        super().__init__(id_)
        self.frequency: float = frequency
        self.if_frequency: Optional[float] = if_frequency


class PulseChannel(QuantumComponent, Calibratable):
    """Models a pulse channel on a particular device."""

    def __init__(
        self,
        id_: str,
        physical_channel: PhysicalChannel,
        frequency=0.0,
        bias=0.0 + 0.0j,
        scale=1.0 + 0.0j,
        fixed_if: bool = False,
        **kwargs,
    ):
        super().__init__(id_, **kwargs)
        self.physical_channel: PhysicalChannel = physical_channel

        self.frequency: float = frequency
        self.bias: complex = bias
        self.scale: complex = scale

        self.fixed_if: bool = fixed_if

        if frequency < self.min_frequency or frequency > self.max_frequency:
            raise ValueError(
                f"Pulse channel frequency '{frequency}' must be between the bounds "
                f"({self.min_frequency}, {self.max_frequency}) on physical "
                f"channel with id {self.full_id()}."
            )

    @property
    def sample_time(self):
        return self.physical_channel.sample_time

    @property
    def block_size(self):
        return self.physical_channel.block_size

    @property
    def block_time(self):
        return self.physical_channel.block_time

    @property
    def phase_offset(self):
        return self.physical_channel.phase_offset

    @property
    def imbalance(self):
        return self.physical_channel.imbalance

    @property
    def acquire_allowed(self):
        return self.physical_channel.acquire_allowed

    @property
    def baseband_frequency(self):
        return self.physical_channel.baseband_frequency

    @property
    def baseband_if_frequency(self):
        return self.physical_channel.baseband_if_frequency

    @property
    def physical_channel_id(self):
        return self.physical_channel.full_id()

    @property
    def min_frequency(self):
        return self.physical_channel.pulse_channel_min_frequency

    @property
    def max_frequency(self):
        return self.physical_channel.pulse_channel_max_frequency

    def partial_id(self):
        return self.id

    def full_id(self):
        return self.physical_channel_id + "." + self.partial_id()

    def __eq__(self, other):
        if not isinstance(other, PulseChannel):
            return False

        return self.full_id() == other.full_id()

    def __hash__(self):
        return hash(self.full_id())


class FreqShiftPulseChannel(PulseChannel):
    def __init__(
        self,
        id_: str,
        physical_channel: PhysicalChannel,
        frequency=0.0,
        bias=0.0 + 0.0j,
        scale=1.0 + 0.0j,
        amp=0.0,
        active: bool = True,
        fixed_if: bool = False,
        **kwargs,
    ):
        super().__init__(id_, physical_channel, frequency, bias, scale, fixed_if, **kwargs)
        self.amp: float = amp
        self.active: bool = active


class QubitCoupling(Calibratable):
    def __init__(self, direction, quality=1):
        """
        Direction of coupling stated in a tuple: (4,5) means we have a  4 -> 5 coupling.
        Quality is the quality-level of the coupling.
        """
        super().__init__()
        self.direction = tuple(direction)
        self.quality = 1 if quality < 1 else quality

    def __repr__(self):
        return f"{str(self.direction)} @{self.quality}"


class PulseChannelView(PulseChannel):
    """
    Each quantum device will have a unique view of a PulseChannel, which this class
    helps encapsulate. For example, a PulseChannel may be the driving channel for one
    qubit but to another qubit the same PulseChannel might be driving its second state.
    We need to be able to share pulse channel instances with different usages in this
    particular case.

    Functionally this acts as an opaque wrapper to PulseChannel, forwarding all calls to
    the wrapped object.
    """

    # noinspection PyMissingConstructor
    def __init__(
        self,
        pulse_channel: PulseChannel,
        channel_type: ChannelType,
        auxiliary_devices: List[QuantumDevice] = None,
    ):
        """
        pulse_channel: The PulseChannel this device is viewing.
        channel_type: How this devices intends the PulseChannel to be used.
        auxiliary_devices: Any extra devices this PulseChannel could be effecting except
                           the current one.
                           For example in cross resonance pulses.
        """
        # We aren't calling super().__init__ on purpose. Inheriting is purely for show
        # (and typing).
        if auxiliary_devices is None:
            auxiliary_devices = []
        self.pulse_channel: PulseChannel = pulse_channel
        self.channel_type: ChannelType = channel_type
        self.auxiliary_devices: List[QuantumDevice] = auxiliary_devices
        if not any(self._pulse_channel_attributes):
            self._pulse_channel_attributes = {
                val
                for val in dir(pulse_channel)
                if not val.startswith("__") and not val.endswith("__")
            }

            if any(set(vars(self)).intersection(self._pulse_channel_attributes)):
                raise ValueError(
                    "Pulse channel has attributes that will shadow wrapping class."
                )

    _pulse_channel_attributes: Set[str] = set()

    def __getattr__(self, item):
        if item in self._pulse_channel_attributes:
            return getattr(self.pulse_channel, item)
        return super().__getattribute__(item)

    def __setattr__(self, key, value):
        if key in self._pulse_channel_attributes:
            return setattr(self.pulse_channel, key, value)
        return super().__setattr__(key, value)


class QuantumDevice(QuantumComponent, Calibratable):
    """A physical device whose main form of operation involves pulse channels."""

    multi_device_pulse_channel_types = (
        ChannelType.cross_resonance,
        ChannelType.cross_resonance_cancellation,
    )

    def __init__(
        self,
        id_: str,
        physical_channel: PhysicalChannel,
        measure_device: QuantumDevice = None,
    ):
        super().__init__(id_)
        self.measure_device: QuantumDevice = measure_device
        self.pulse_channels: Dict[str, Union[PulseChannel, PulseChannelView]] = {}
        self.default_pulse_channel_type = ChannelType.measure
        self.physical_channel: PhysicalChannel = physical_channel

    def create_pulse_channel(
        self,
        channel_type: ChannelType,
        frequency=0.0,
        bias=0.0 + 0.0j,
        scale=1.0 + 0.0j,
        amp=0.0,
        active: bool = True,
        fixed_if: bool = False,
        auxiliary_devices: List[QuantumDevice] = None,
        id_: str = None,
    ):
        if auxiliary_devices is None:
            auxiliary_devices = []
        if id_ is None:
            id_ = self._create_pulse_channel_id(channel_type, [self] + auxiliary_devices)
        if channel_type == ChannelType.freq_shift:
            pulse_channel = self.physical_channel.create_freq_shift_pulse_channel(
                id_, frequency, bias, scale, amp, active, fixed_if
            )
        else:
            pulse_channel = self.physical_channel.create_pulse_channel(
                id_, frequency, bias, scale, fixed_if
            )
        self.add_pulse_channel(pulse_channel, channel_type, auxiliary_devices)

        return pulse_channel

    def _create_pulse_channel_id(
        self, channel_type: ChannelType, auxiliary_devices: List[QuantumDevice]
    ):
        if (
            channel_type in self.multi_device_pulse_channel_types
            and len(auxiliary_devices) == 0
        ):
            raise ValueError(
                f"Channel type {channel_type.name} requires at least one "
                "auxillary_device"
            )
        return ".".join(
            [str(x.full_id()) for x in (auxiliary_devices)] + [channel_type.name]
        )

    def add_pulse_channel(
        self,
        pulse_channel: PulseChannel,
        channel_type: ChannelType,
        auxiliary_devices: List[QuantumDevice] = None,
    ):
        if auxiliary_devices is None:
            auxiliary_devices = []

        if pulse_channel.physical_channel != self.physical_channel:
            raise ValueError(
                "Pulse channel has physical channel and device physical channel must "
                f"be equal. Device {self} has physical channel {self.physical_channel} "
                "while pulse channel has physical channel "
                f"{pulse_channel.physical_channel}"
            )

        id_ = self._create_pulse_channel_id(channel_type, auxiliary_devices)
        if id_ in self.pulse_channels:
            raise KeyError(f"Pulse channel with id '{id_}' already exists.")

        self.pulse_channels[id_] = PulseChannelView(
            pulse_channel, channel_type, auxiliary_devices
        )

    PTType = TypeVar("PTType", bound=PulseChannel, covariant=True)

    def get_pulse_channel(
        self,
        channel_type: ChannelType = None,
        auxiliary_devices: List[QuantumDevice] = None,
    ) -> PTType:
        if channel_type is None:
            channel_type = self.default_pulse_channel_type
        if auxiliary_devices is None:
            auxiliary_devices = []
        if not isinstance(auxiliary_devices, List):
            auxiliary_devices = [auxiliary_devices]

        id_ = self._create_pulse_channel_id(channel_type, auxiliary_devices)
        if id_ not in self.pulse_channels:
            raise KeyError(
                f"Pulse channel with stored as '{id_}' not found on device '{self.id}'."
            )

        return self.pulse_channels[id_]

    def get_default_pulse_channel(self):
        return self.get_pulse_channel(self.default_pulse_channel_type)

    def get_auxiliary_devices(self, pulse_channel: PulseChannel):
        if isinstance(pulse_channel, PulseChannelView):
            return pulse_channel.auxiliary_devices
        return []


class Resonator(QuantumDevice):
    """Models a resonator on a chip. Can be connected to multiple qubits."""

    def get_measure_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.measure)

    def get_acquire_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.acquire)


class Qubit(QuantumDevice):
    """
    Class modelling our superconducting qubit and holds all information relating to
    them.
    """

    measure_device: Resonator

    def __init__(
        self,
        index: int,
        resonator: Resonator,
        physical_channel: PhysicalChannel,
        coupled_qubits: List[Qubit] = None,
        drive_amp: float = 1.0,
        id_=None,
    ):
        super().__init__(id_ or f"Q{index}", physical_channel, resonator)
        self.index = index
        self.coupled_qubits: List[Qubit] = []
        self.mean_z_map_args = [1.0, 0.0]
        self.discriminator = [0.0]
        self.pulse_hw_zx_pi_4 = dict()
        self.default_pulse_channel_type = ChannelType.drive

        for qubit in coupled_qubits or []:
            self.add_coupled_qubit(qubit)

        self.pulse_hw_x_pi_2 = {
            "shape": PulseShapeType.GAUSSIAN,
            "width": 100e-9,
            "rise": 1.0 / 3.0,
            "amp": 0.25 / (100e-9 * 1.0 / 3.0 * np.pi**0.5),
            "drag": 0.0,
            "phase": 0.0,
        }

        self.pulse_measure = {
            "shape": PulseShapeType.SQUARE,
            "width": 1.0e-6,
            "amp": drive_amp,
            "amp_setup": 0.0,
            "rise": 0.0,
        }

        self.measure_acquire = {
            "delay": 180e-9,
            "sync": True,
            "width": 1e-6,
            "weights": None,
            "use_weights": False,
        }

    def add_coupled_qubit(self, qubit: Qubit):
        if qubit is None:
            return

        if qubit not in self.coupled_qubits:
            self.coupled_qubits.append(qubit)

        if qubit.full_id() not in self.pulse_hw_zx_pi_4:
            self.pulse_hw_zx_pi_4[qubit.full_id()] = {
                "shape": PulseShapeType.SOFT_SQUARE,
                "width": 125e-9,
                "rise": 10e-9,
                "amp": 1e6,
                "drag": 0.0,
                "phase": 0.0,
            }

    def get_drive_channel(self):
        return self.get_pulse_channel(ChannelType.drive)

    def get_measure_channel(self) -> PulseChannel:
        return self.measure_device.get_measure_channel()

    def get_acquire_channel(self) -> PulseChannel:
        return self.measure_device.get_acquire_channel()

    def get_second_state_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.second_state)

    def get_freq_shift_channel(self) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.freq_shift)

    def get_cross_resonance_channel(self, linked_qubits: List[Qubit]) -> PulseChannel:
        return self.get_pulse_channel(ChannelType.cross_resonance, linked_qubits)

    def get_cross_resonance_cancellation_channel(
        self, linked_qubits: List[Qubit]
    ) -> PulseChannel:
        return self.get_pulse_channel(
            ChannelType.cross_resonance_cancellation, linked_qubits
        )

    def get_all_channels(self):
        """
        Returns all channels associated with this qubit, including resonator channel and
        other auxiliary devices that act as if they are on this object.
        """
        return [
            *self.pulse_channels.values(),
            self.get_measure_channel(),
            self.get_acquire_channel(),
        ]


class PulseShapeType(Enum):
    SQUARE = "square"
    GAUSSIAN = "gaussian"
    SOFT_SQUARE = "soft_square"
    BLACKMAN = "blackman"
    SETUP_HOLD = "setup_hold"
    SOFTER_SQUARE = "softer_square"
    EXTRA_SOFT_SQUARE = "extra_soft_square"
    SOFTER_GAUSSIAN = "softer_gaussian"
    ROUNDED_SQUARE = "rounded_square"
    GAUSSIAN_DRAG = "gaussian_drag"
    GAUSSIAN_ZERO_EDGE = "gaussian_zero_edge"
    GAUSSIAN_SQUARE = "gaussian_square"
    SECH = "sech"
    SIN = "sin"
    COS = "cos"

    def __repr__(self):
        return self.value


def _strip_aliases(key: str, build_unaliased=False):
    if "<" in key:
        lhs = key.split("<")
        if len(lhs) > 2:
            raise ValueError(
                f"Alias {key} has more than 1 angle bracket. "
                "Nested or multiple alias' are unsupported."
            )

        rhs = lhs[1].split(">")
        if len(rhs) != 2:
            raise ValueError(
                f"Multiple (or no) closing angle brackets detected in for alias {key}."
            )

        if build_unaliased:
            return lhs[0] + rhs[1]

        return rhs[0]

    return key


MaxPulseLength = 1e-3  # seconds
