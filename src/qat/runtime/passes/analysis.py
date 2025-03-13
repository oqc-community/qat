# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import re
from dataclasses import dataclass, field
from typing import Dict, List, Union

from compiler_config.config import CompilerConfig

from qat.core.pass_base import AnalysisPass
from qat.core.result_base import ResultInfoMixin, ResultManager
from qat.purr.backends.calibrations.remote import find_calibration
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Acquire
from qat.purr.compiler.runtime import CalibrationWithArgs
from qat.runtime.executables import Executable


@dataclass
class CalibrationAnalysisResult(ResultInfoMixin):
    calibration_executables: List[CalibrationWithArgs] = field(default_factory=list)


class CalibrationAnalysis(AnalysisPass):
    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):
        cal_blocks = [find_calibration(arg) for arg in compiler_config.active_calibrations]
        res_mgr.add(CalibrationAnalysisResult(cal_blocks))
        return ir


@dataclass
class IndexMappingResult(ResultInfoMixin):
    mapping: Dict[str, int]


class IndexMappingAnalysis(AnalysisPass):
    """Determines a mapping from classical bit registers to qubits.

    Searches through the acquires of the package and determines their associated qubit.
    Also looks for classical registers of the form :code:`<clreg_name>[<clreg_index>]`.

    Supports both :class:`Executable` packages and :class:`InstructionBuilder`.
    """

    def __init__(self, model: QuantumHardwareModel):
        """:param model: The hardware model is needed for the qubit mapping."""
        # TODO: searching for classical registers feels a little shaky. Guessing there are
        # changes to make at a higher level to faciliate improvements here.
        # TODO: support with pydantic instructions and hardware.
        # TODO: should the output_variable -> qubit mapping be separate from the classical
        # register extraction? The former might be useful for a compiliation analysis pass.
        # Only used here right now, so let's worry about this later.
        self.model = model

    def run(
        self,
        acquisitions: Dict[str, any],
        res_mgr: ResultManager,
        *args,
        package: Union[InstructionBuilder, Executable],
        **kwargs,
    ):
        """
        :param acquisitions: The dictionary of results acquired from the target machine.
        :param res_mgr: The results manager to save the mapping.
        :param package: The executable program containing the results-processing
            information should be passed as a keyword argument.
        """
        # Determine a mapping from output variable - > qubit index.
        if isinstance(package, Executable):
            var_to_channel_map = self.var_to_physical_channel_executable(package)
        else:
            var_to_channel_map = self.var_to_physical_channel_qat_ir(package)
        var_to_qubit_map = self.var_to_qubit_map(var_to_channel_map)

        # Search for classical registers defined in e.g. QASM
        pattern = re.compile(r"(.*)\[([0-9]+)\]")
        for var in list(var_to_qubit_map.keys()):
            result = pattern.match(var)
            if result:
                var_to_qubit_map[result.group(2)] = var_to_qubit_map.pop(var)

        res_mgr.add(IndexMappingResult(mapping=var_to_qubit_map))
        return acquisitions

    @staticmethod
    def var_to_physical_channel_executable(package: Executable) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for channel_id, channel_data in package.channel_data.items():
            for acquire in channel_data.acquires:
                mapping[acquire.output_variable] = channel_id
        return mapping

    @staticmethod
    def var_to_physical_channel_qat_ir(package: InstructionBuilder) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for instruction in package.instructions:
            if not isinstance(instruction, Acquire):
                continue
            mapping[instruction.output_variable] = instruction.channel.physical_channel_id
        return mapping

    def var_to_qubit_map(self, mapping: Dict[str, str]):
        chan_to_qubit_map: Dict[str, int] = {}
        for chan in set(mapping.values()):
            qubits = [
                qubit
                for qubit in self.model.qubits
                if qubit.get_acquire_channel().physical_channel_id == chan
            ]

            if len(qubits) == 0:
                raise ValueError(f"Could not find any qubits with acquire channel {chan}.")
            chan_to_qubit_map[chan] = qubits[0].index

        var_to_qubit_map = {var: chan_to_qubit_map[chan] for var, chan in mapping.items()}
        return var_to_qubit_map
