from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, Iterable, List, Union

import numpy as np
from pydantic import Field

from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.experimental.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Instruction
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.pydantic import WarnOnExtraFieldsModel

log = get_default_logger()


class InstructionExecutionEngine(WarnOnExtraFieldsModel, ABC):
    """
    Engine that can optimise, validate and execute instructions
    on a hardware model.

    Attributes:
        model: The hardware model.
        max_instruction_len: The max length an instruction can have.
    """

    model: QuantumHardwareModel
    max_instruction_len: int = Field(ge=1, default=2000000)

    @abstractmethod
    def execute(self, instructions: List[Instruction]) -> Dict[str, Any]: ...


class QuantumInstructionExecutionEngine(InstructionExecutionEngine):
    startup_engine: bool = True

    def _generate_repeat_batches(self, repeats: int) -> List[int]:
        """
        Batches together executions if we have more than the amount the hardware can
        handle at once. A repeat limit of -1 disables this functionality and just runs
        everything at once.
        """
        batch_limit = self.model.repeat_limit
        if batch_limit == -1:
            return [repeats]

        list_expansion_ratio = int(
            (Decimal(repeats) / Decimal(batch_limit)).to_integral_value(ROUND_DOWN)
        )
        remainder = repeats % batch_limit
        batches = [batch_limit] * list_expansion_ratio
        if remainder > 0:
            batches.append(remainder)

        return batches

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
