# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import re
from dataclasses import dataclass

from qat.core.pass_base import AnalysisPass
from qat.core.result_base import ResultInfoMixin, ResultManager
from qat.executables import Executable
from qat.ir.instruction_builder import InstructionBuilder
from qat.ir.measure import Acquire
from qat.model.hardware_model import PhysicalHardwareModel


@dataclass
class IndexMappingResult(ResultInfoMixin):
    mapping: dict[str, int]


class IndexMappingAnalysis(AnalysisPass):
    """Determines a mapping from classical bit registers to qubits.

    Searches through the acquires of the package and determines their associated qubit.
    Also looks for classical registers of the form :code:`<clreg_name>[<clreg_index>]`.

    Supports both :class:`Executable` packages and :class:`InstructionBuilder`.
    """

    def __init__(self, model: PhysicalHardwareModel):
        """:param model: The hardware model is needed for the qubit mapping."""

        self.model = model
        pulse_to_phys_channel_map: dict[str, str] = {}
        for qubit in self.model.qubits.values():
            for pulse_ch in qubit.all_pulse_channels:
                pulse_to_phys_channel_map[pulse_ch.uuid] = qubit.physical_channel.uuid

            for pulse_ch in qubit.resonator.all_pulse_channels:
                pulse_to_phys_channel_map[pulse_ch.uuid] = (
                    qubit.resonator.physical_channel.uuid
                )
        self.pulse_to_phys_channel_map = pulse_to_phys_channel_map

    def run(
        self,
        acquisitions: dict[str, any],
        res_mgr: ResultManager,
        *args,
        package: InstructionBuilder | Executable,
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
            var_to_phys_channel_map = self.var_to_physical_channel_executable(package)
        else:
            var_to_phys_channel_map = self.var_to_physical_channel_qat_ir(package)
        var_to_qubit_map = self.var_to_qubit_map(var_to_phys_channel_map)

        # Search for classical registers defined in e.g. QASM.
        pattern = re.compile(r"(.*)\[([0-9]+)\]")
        for var in list(var_to_qubit_map.keys()):
            result = pattern.match(var)
            if result:
                var_to_qubit_map[result.group(2)] = var_to_qubit_map.pop(var)

        res_mgr.add(IndexMappingResult(mapping=var_to_qubit_map))
        return acquisitions

    @staticmethod
    def var_to_physical_channel_executable(package: Executable) -> dict[str, str]:
        return {var: acquire.physical_channel for var, acquire in package.acquires.items()}

    def var_to_physical_channel_qat_ir(self, package: InstructionBuilder) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for instruction in package.instructions:
            if isinstance(instruction, Acquire):
                mapping[instruction.output_variable] = self.pulse_to_phys_channel_map[
                    instruction.target
                ]
        return mapping

    def var_to_qubit_map(self, mapping: dict[str, str]):
        """
        Maps the variables in the mapping to qubit indices.
        """

        var_to_qubit_map: dict[str, int] = {}
        for var, phys_ch_id in mapping.items():
            acquired_qubits = [
                idx
                for idx, qubit in self.model.qubits.items()
                if qubit.resonator.physical_channel.uuid == phys_ch_id
            ]

            if len(acquired_qubits) == 0:
                raise ValueError(
                    f"Could not find any qubits with acquire channel {phys_ch_id}."
                )
            var_to_qubit_map[var] = acquired_qubits[0]

        return var_to_qubit_map
