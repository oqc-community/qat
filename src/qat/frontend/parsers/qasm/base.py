# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import abc
import itertools as it
from typing import Any, Iterable

from compiler_config.config import InlineResultsProcessing, Languages
from pydantic import Field

from qat.frontend.register import BitRegister, CregIndexValue, QubitRegister, Registers
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import Variable
from qat.model.device import Qubit
from qat.purr.utils.logger import get_default_logger
from qat.utils.pydantic import NoExtraFieldsModel, ValidatedDict, ValidatedList

log = get_default_logger()


class QasmContext(NoExtraFieldsModel):
    """
    Container object for all data relating to the scope/pass of QASM currently under
    analysis.
    """

    registers: Registers = Registers()
    variables: ValidatedDict[str, Variable] = Field(
        default_factory=lambda: ValidatedDict[str, Variable]()
    )
    gates: dict[str, Any] = Field(default_factory=dict)


class ParseResults(NoExtraFieldsModel):
    """
    Results object for attempted parse. When coerced to a boolean matches
    against if parse was successful.
    """

    success: bool
    errors: str = ""

    def __bool__(self):
        return self.success


class AbstractParser(abc.ABC):
    def __init__(self):
        self.results_format = InlineResultsProcessing.Program
        self._cached_parses: dict[int, Any] = dict()

    def can_parse(self, qasm: str) -> ParseResults:
        try:
            self._fetch_or_parse(qasm)
            return ParseResults(success=True)
        except Exception as ex:
            return ParseResults(success=False, errors=str(ex))

    def parser_language(self) -> Languages:
        return Languages.Empty

    @abc.abstractmethod
    def _fetch_or_parse(self): ...

    @abc.abstractmethod
    def walk_node(self, node, context: QasmContext, builder, **kwargs):
        pass

    def _add_delay(
        self, delay: float, qubits: list[Qubit], builder: QuantumInstructionBuilder
    ):
        registers = self._expand_to_match_registers(qubits, flatten_results=True)
        for qubit in registers:
            builder.delay(qubit, delay)

    def _add_qreg(
        self,
        reg_name: str,
        reg_length: int,
        context: QasmContext,
        builder: QuantumInstructionBuilder,
    ):
        if reg_name in context.registers.quantum:
            return
        index_range = self._get_qreg_index_range(reg_length, context, builder)
        qubits = []
        for idx in index_range:
            qubit = builder.hw.qubit_with_index(idx)
            qubits.append(qubit)
        context.registers.quantum[reg_name] = QubitRegister(qubits=qubits)

    def _add_creg(self, reg_name: str, reg_length: int, context: QasmContext):
        if reg_name in context.registers.classic:
            return
        context.registers.classic[reg_name] = BitRegister(
            bits=[
                CregIndexValue(register_name=reg_name, index=idx, value=0)
                for idx in range(reg_length)
            ]
        )

    def _add_measure(
        self,
        qubits: list[Qubit],
        clbits: list[CregIndexValue],
        builder: QuantumInstructionBuilder,
    ):
        registers = self._expand_to_match_registers(qubits, clbits, flatten_results=True)

        for qubit, creg in registers:
            qubit_idx = builder._qubit_index_by_uuid[qubit.uuid]

            res_id = f"{creg}_{qubit_idx}"
            builder.measure_single_shot_z(qubit, output_variable=res_id)
            creg.value = res_id
            builder.results_processing(res_id, self.results_format)

    def _add_unitary(
        self,
        theta: float,
        phi: float,
        lamb: float,
        qubit_or_register: list[Qubit | QubitRegister],
        builder: QuantumInstructionBuilder,
    ):
        """Unitary in QASM terms is just ``U(...)``."""
        qubits = self._expand_to_match_registers(qubit_or_register, flatten_results=True)
        for qubit in qubits:
            builder.Z(qubit, lamb).Y(qubit, theta).Z(qubit, phi)

    def _add_cnot(
        self,
        control_qbs: list[Qubit],
        target_qbs: list[Qubit],
        builder: QuantumInstructionBuilder,
    ):
        for cqb, tqb in self._expand_to_match_registers(
            control_qbs, target_qbs, flatten_results=True
        ):
            builder.cnot(cqb, tqb)

    def _add_reset(self, qubits: list[Qubit], builder: QuantumInstructionBuilder):
        qubits = self._expand_to_match_registers(qubits, flatten_results=True)
        builder.reset(qubits)

    def _add_if(self, left, right, if_body, context, builder):
        raise NotImplementedError("Conditional logic is currently not implemented.")

    def _add_ecr(self, qubits: list[Qubit], builder: QuantumInstructionBuilder):
        if len(qubits) != 2:
            raise ValueError(
                "Definition of ECR intrinsic is wrong. Can only take 2 qubits as arguments."
            )
        builder.ECR(qubits[0], qubits[1])

    def _get_qreg_index_range(
        self, reg_length: int, context: QasmContext, builder: QuantumInstructionBuilder
    ):
        next_free = 0
        available_indices = list(builder.hw.qubits.keys())
        max_used = max(
            [-1]
            + [
                builder._qubit_index_by_uuid[qubit.uuid]
                for qubit_reg in context.registers.quantum.values()
                for qubit in qubit_reg.qubits
            ]
        )

        if max_used > -1:
            next_free = available_indices.index(max_used) + 1
        index_range = available_indices[next_free : next_free + reg_length]
        if len(index_range) < reg_length:
            raise ValueError("Attempted to allocate more qubits than available.")

        return index_range

    def _is_register_target(self, values: list[Any]):
        """
        Does it look like the passed-in qubit or parameter list contains a register.
        """
        return isinstance(values, Iterable) and any(
            isinstance(val, (QubitRegister, BitRegister)) for val in values
        )

    def _curate_register_input(self, input: list | Qubit) -> list:
        """
        If the input is a list of (qu)bit registers, curate the input such that
        we are left with a list of (qu)bits. Do nothing if a list of (qu)bits is passed.
        """
        if isinstance(input, list | ValidatedList):
            if len(input) == 1 and isinstance(
                register := input[0], QubitRegister | BitRegister
            ):
                return register.contents
            else:
                return list(input)
        elif isinstance(input, Qubit):
            return [input]
        else:
            raise TypeError("Please provide a list as input.")

    def _expand_to_match_registers(self, *args, tuple_return=True, flatten_results=False):
        """
        Expands and zips registers/non-registers together so they can be processed.

        QASM treats any registers of bits or qubits as calling the gate * register size
        times with each individual register value. This is a helper method for
        centralizing this expansion/mapping.

        :Example:

        .. code-block:: python

            [[q1, q2, q3], [q4, q5, q6],[c1]]

        With the first 2 lists being expanded qubit registers, should resolve into:

        .. code-block:: python

            [(q1, q4, c1), (q2, q5, c1), (q3, q6, c1)]
        """
        curated_args = []
        for arg in args:
            curated_args.append(self._curate_register_input(arg))

        if len(curated_args) > 1:
            # Each list needs to be of equal length, repeat element if necessary.
            curated_args = [
                arg if len(arg) > 1 else arg * len(max(curated_args, key=len))
                for arg in curated_args
            ]
            return list(it.zip_longest(*curated_args))
        else:
            return curated_args[0]
