from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Union

import numpy as np
from pydantic import Field

from qat.model.model import QuantumHardwareModel
from qat.purr.backends.utilities import get_axis_map
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import SweepIterator
from qat.purr.compiler.instructions import AcquireMode, Instruction
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.pydantic import WarnOnExtraFieldsModel
from qat.runtime.live_devices import ControlHardware, InstrumentConnectionManager

log = get_default_logger()


class InstructionExecutionEngine(WarnOnExtraFieldsModel, ABC):
    """
    Base class for an engine that can execute instructions.

    Attributes:
        model: The hardware model.
        max_instruction_len: The max length an instruction can have.
    """

    model: QuantumHardwareModel
    max_instruction_len: int = Field(ge=1, default=2000000)

    @abstractmethod
    def execute(self, instructions: List[Instruction]) -> Dict[str, Any]: ...


class QuantumExecutionEngine(InstructionExecutionEngine):
    """
    Engine that can execute instructions on a quantum hardware model.
    """

    def execute(self, instructions: QatFile):
        """Executes this qat file against this current hardware."""
        return self._common_execute(instructions)

    ### !!! This method is probably going to be refactored according to the new pass manager feature.
    def _common_execute(
        self, instructions: List[Instruction], interrupt: Interrupt = NullInterrupt()
    ):
        """Executes this qat file against this current hardware."""
        pass

    @abstractmethod
    def _execute_on_hardware(
        self, sweep_iterator, package: QatFile, interrupt: Interrupt = NullInterrupt()
    ) -> Union[Dict[str, List[float]], Dict[str, np.ndarray]]: ...

    def _execute_with_interrupt(
        self, instructions: QatFile, interrupt: Interrupt = NullInterrupt()
    ):
        """Executes this qat file against this current hardware.
        Execution allows for interrupts triggered by events.
        """
        return self._common_execute(instructions, interrupt)

    def __repr__(self):
        return f"{type(self).__name__}(model={str(self.model)})"


class EchoEngine(QuantumExecutionEngine):
    """
    A backend that just returns default values.
    Primarily used for testing and no-backend situations.
    """

    def _execute_on_hardware(
        self,
        sweep_iterator: SweepIterator,
        package: QatFile,
        interrupt: Interrupt = NullInterrupt(),
    ) -> Dict[str, np.ndarray]:
        results = {}
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            metadata = {"sweep_iteration": sweep_iterator.get_current_sweep_iteration()}
            interrupt.if_triggered(metadata, throw=True)

            position_map = self.create_duration_timeline(package.instructions)
            pulse_channel_buffers = self.build_pulse_channel_buffers(position_map, True)
            buffers = self.build_physical_channel_buffers(pulse_channel_buffers)
            aq_map = self.build_acquire_list(position_map)

            repeats = package.repeat.repeat_count
            for channel_id, aqs in aq_map.items():
                for aq in aqs:
                    # just echo the output pulse back for now
                    response = buffers[aq.physical_channel.full_id()][
                        aq.start : aq.start + aq.samples
                    ]
                    if aq.mode != AcquireMode.SCOPE:
                        if repeats > 0:
                            response = np.tile(response, repeats).reshape((repeats, -1))

                    response_axis = get_axis_map(aq.mode, response)
                    for pp in package.get_pp_for_variable(aq.output_variable):
                        response, response_axis = self.run_post_processing(
                            pp, response, response_axis
                        )

                    var_result = results.setdefault(
                        aq.output_variable,
                        np.empty(
                            sweep_iterator.get_results_shape(response.shape),
                            response.dtype,
                        ),
                    )
                    sweep_iterator.insert_result_at_sweep_position(var_result, response)


class LiveDeviceEngine(QuantumExecutionEngine):
    """
    Backend that hooks up the logical hardware model to our QPU's, currently hardcoded to particular fridges.
    This will only work when run on a machine physically connected to a QPU.
    """

    instruments: InstrumentConnectionManager
    startup_engine: bool = True
    control_hardware: ControlHardware | None = None

    def __init__(self, **data):
        super.__init__(**data)
        if self.startup_engine:
            self.startup()

    def startup(self):
        connected = self.instruments.connect()
        return connected

    def shutdown(self):
        connected = self.instruments.disconnect()
        return connected

    def execute(self):
        pass


def _complex_to_binary(number: complex):
    """Base calculation for changing a complex measurement to binary form."""
    return 0 if number.real > 0 else 1


def _binary(results_list):
    """Changes all measurements to binary format."""
    if not isinstance(results_list, Iterable):
        return [results_list]

    results = []
    for item in results_list:
        if isinstance(item, Iterable):
            results.append(_binary(item))
        elif isinstance(item, complex):  # If we're a flat register, just append.
            results.append(_complex_to_binary(item))
        elif isinstance(item, float):
            results.append(0 if item > 0 else 1)
        else:
            results.append(item)
    return results


def _binary_average(results_list):
    """
    Averages all repeat results and returns a definitive 1/0 for each qubit measurement.
    """
    # If we have many sweeps/repeats loop through all of them and sum.
    if all([isinstance(val, list) for val in results_list]):
        binary_results = [_binary_average(nested) for nested in results_list]
    else:
        binary_results = _binary(results_list)

    return 1 if sum(binary_results) >= (len(binary_results) / 2) else 0


def _numpy_array_to_list(array):
    """Transform numpy arrays to a normal list."""
    if isinstance(array, np.ndarray):
        numpy_list: List = array.tolist()
        if len(numpy_list) == 1:
            return numpy_list[0]
        return numpy_list
    elif isinstance(array, List):
        list_list = [_numpy_array_to_list(val) for val in array]
        if len(list_list) == 1:
            return list_list[0]
        return list_list
    else:
        return array