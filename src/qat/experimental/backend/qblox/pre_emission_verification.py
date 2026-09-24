# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Pre-emission verification for configured Qblox Q1 sequence IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, NoneAttr
from xdsl.ir import Attribute, ParametrizedAttribute, SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException, VerifyException

from qat.backend.qblox.target_data import (
    TARGET_DATA,
    ModuleDescription,
    QbloxTargetData,
    SequencerDescription,
)
from qat.experimental.dialect.q1 import (
    ACQUISITION_OP_TYPES,
    AcquireImmImmImmOp,
    AcquireImmRsImmOp,
    AcquireWeightedImmImmImmImmImmOp,
    LabelOp,
    PlayImmImmImmOp,
)
from qat.experimental.dialect.q1.ir.abstract_ops import Q1AsmOperation
from qat.experimental.dialect.q1.ir.reg_desc import Q1RegisterType
from qat.experimental.dialect.q1.transforms.reg_alloc import (
    LinearScanRegisterAllocationPass,
)
from qat.experimental.dialect.q1_cf.transforms.linearise_q1_cf import LineariseQ1CfToQ1Pass
from qat.experimental.dialect.q1_sequence.ir.attrs import NcoConfigAttr
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPass
from qat.experimental.system_data.hardware.qblox.models import QbloxModuleKind
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    Q1SequencerFeature,
    Q1SequencerType,
    SequencerTarget,
)


def _check_register(value: SSAValue, operation_name: str) -> None:
    register = value.type
    if isinstance(register, Q1RegisterType) and not register.is_allocated:
        raise PassFailedException(f"{operation_name} contains an unallocated Q1 register.")


def _check_attribute(attribute: Attribute, operation_name: str) -> None:
    if isinstance(attribute, Q1RegisterType) and not attribute.is_allocated:
        raise PassFailedException(f"{operation_name} contains an unallocated Q1 register.")
    if isinstance(attribute, ParametrizedAttribute):
        for parameter in attribute.parameters:
            _check_attribute(parameter, operation_name)


class _QbloxPreEmissionVerifier:
    """Reject IR that is not a configured, allocated, flat Qblox Q1 sequence module.

    The fixed target description owns structural topology and capabilities.
    ``QbloxTargetData`` supplies configurable numeric limits.
    """

    def __init__(self, target_data: QbloxTargetData):
        self.target_data = target_data

    @staticmethod
    def _sequencer_target(sequence: SequenceOp) -> SequencerTarget:
        """Return the configured physical sequencer target."""

        if sequence.module_config is None or sequence.seq_idx is None:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} is missing its Qblox configuration."
            )
        return DEFAULT_QBLOX_TARGET.sequencer(
            sequence.module_config.kind.data, sequence.seq_idx.data
        )

    def _is_readout_sequencer(self, sequence: SequenceOp) -> bool:
        """Return whether the physical sequencer supports acquisition operations.

        The configured module kind and sequencer index identify the physical sequencer.
        Structural capabilities come from the fixed repository target description.
        """

        return self._sequencer_target(sequence).sequencer_spec.supports(
            Q1SequencerFeature.acquisition
        )

    def _sequencer_data(self, sequence: SequenceOp) -> SequencerDescription:
        """Return numeric limits for the configured physical sequencer."""

        if self._sequencer_target(sequence).sequencer_spec.type is Q1SequencerType.readout:
            return self.target_data.READOUT_SEQUENCER_DATA
        return self.target_data.CONTROL_SEQUENCER_DATA

    def _module_data(self, kind: QbloxModuleKind) -> ModuleDescription:
        """Return numeric limits for ``kind`` from the legacy target data."""

        return {
            QbloxModuleKind.qcm: self.target_data.QCM_DATA,
            QbloxModuleKind.qcm_rf: self.target_data.QCM_RF_DATA,
            QbloxModuleKind.qrm: self.target_data.QRM_DATA,
            QbloxModuleKind.qrm_rf: self.target_data.QRM_RF_DATA,
            QbloxModuleKind.qrc: self.target_data.QRC_DATA,
        }[kind]

    def verify(self, op: ModuleOp) -> list[SequenceOp]:
        top_level = list(op.body.block.ops)
        if not top_level:
            raise PassFailedException(
                "Qblox emission requires at least one top-level q1_sequence.sequence."
            )

        sequences: list[SequenceOp] = []
        for child in top_level:
            if not isinstance(child, SequenceOp):
                raise PassFailedException(
                    "Qblox emission requires every top-level operation to be "
                    f"q1_sequence.sequence. Found {child.name}."
                )
            self._check_sequence(child)
            sequences.append(child)
        self._check_allocations(sequences)
        self._check_table_capacity(sequences)
        self._check_instruction_capacity(sequences)
        self._check_acquisition_capacity(sequences)
        return sequences

    def _check_sequence(self, sequence: SequenceOp) -> None:
        if (
            sequence.instrument_id is None
            or sequence.slot_idx is None
            or sequence.seq_idx is None
            or sequence.sequencer_config is None
            or sequence.module_config is None
        ):
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} is missing its Qblox allocation "
                "or configuration."
            )
        if len(sequence.body.blocks) != 1:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} must contain exactly one block "
                "before Qblox emission."
            )

        block = sequence.body.block
        for argument in block.args:
            _check_register(argument, sequence.name)

        for nested in block.walk():
            if not isinstance(nested, Q1AsmOperation):
                raise PassFailedException(
                    f"Qblox emission does not permit residual operation {nested.name}."
                )
            if nested.regions:
                raise PassFailedException(
                    f"Qblox emission requires flat Q1ASM. {nested.name} contains a region."
                )
            for operand in nested.operands:
                _check_register(operand, nested.name)
            for result in nested.results:
                _check_register(result, nested.name)
            for attribute in (*nested.attributes.values(), *nested.properties.values()):
                _check_attribute(attribute, nested.name)
        for table_name, table in (
            ("waveform", sequence.waveforms),
            ("weight", sequence.weights),
            ("acquisition", sequence.acquisitions),
        ):
            for entry in table:
                try:
                    entry.verify()
                except VerifyException as exc:
                    raise PassFailedException(
                        f"Sequence {sequence.channel_id.data!r} has an invalid "
                        f"{table_name} table entry: {exc}"
                    ) from exc
        try:
            sequence.verify()
        except VerifyException as exc:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} failed Qblox verification: {exc}"
            ) from exc
        self._check_nco_frequency(sequence)

    def _check_nco_frequency(self, sequence: SequenceOp) -> None:
        """Validate configured NCO frequency against the selected sequencer data."""

        sequencer_config = sequence.sequencer_config
        if sequencer_config is None or not isinstance(sequencer_config.nco, NcoConfigAttr):
            return
        frequency_attr = sequencer_config.nco.frequency
        if isinstance(frequency_attr, NoneAttr):
            return
        frequency = frequency_attr.value.data
        sequencer_data = self._sequencer_data(sequence)
        if not isfinite(frequency) or not (
            sequencer_data.nco_min_freq <= frequency <= sequencer_data.nco_max_freq
        ):
            sequencer_type = self._sequencer_target(sequence).sequencer_spec.type
            raise PassFailedException(
                f"NCO frequency {frequency} is outside "
                f"[{sequencer_data.nco_min_freq:.0f}, "
                f"{sequencer_data.nco_max_freq:.0f}] Hz for the selected "
                f"{sequencer_type.value} sequencer."
            )

    @staticmethod
    def _check_allocations(sequences: list[SequenceOp]) -> None:
        """Validate that sequence allocations are unique."""

        allocated: set[tuple[str, int, int]] = set()
        for sequence in sequences:
            if (
                sequence.instrument_id is None
                or sequence.slot_idx is None
                or sequence.seq_idx is None
            ):
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} is missing its allocation."
                )
            address = (
                sequence.instrument_id.data,
                sequence.slot_idx.data,
                sequence.seq_idx.data,
            )
            if address in allocated:
                raise PassFailedException(
                    f"Multiple sequences use Qblox allocation {address}."
                )
            allocated.add(address)

    def _check_acquisition_capacity(self, sequences: list[SequenceOp]) -> None:
        """Validate sequencer acquisition roles and aggregate module bin capacity."""

        # TODO(COMPILER-1458): Verify the per-sequencer interval between acquisition
        # starts against the utilisation-dependent Qblox firmware limit.
        modules = {
            (sequence.instrument_id.data, sequence.slot_idx.data): sequence.module_config
            for sequence in sequences
            if sequence.instrument_id is not None
            and sequence.slot_idx is not None
            and sequence.module_config is not None
        }
        totals: dict[tuple[str, int], int] = {}
        for sequence in sequences:
            if sequence.instrument_id is None or sequence.slot_idx is None:
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} is missing its module "
                    "allocation."
                )
            key = (sequence.instrument_id.data, sequence.slot_idx.data)
            acquisition_ops = [
                nested
                for nested in sequence.body.block.walk()
                if isinstance(nested, ACQUISITION_OP_TYPES)
            ]
            acquisition_indices = {
                acquisition.index.data for acquisition in sequence.acquisitions
            }
            missing_indices = sorted(
                {
                    acquisition.acq_idx.data
                    for acquisition in acquisition_ops
                    if acquisition.acq_idx.data not in acquisition_indices
                }
            )
            if missing_indices:
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} uses acquisition indices "
                    f"{missing_indices} that are absent from its acquisition table."
                )
            if (
                any(
                    isinstance(acquisition, AcquireImmImmImmOp | AcquireImmRsImmOp)
                    for acquisition in acquisition_ops
                )
                and sequence.sequencer_config.integration_length is None
            ):
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} uses square-weight "
                    "acquisition without a configured integration length."
                )
            has_acquisition = bool(sequence.acquisitions) or bool(acquisition_ops)
            is_readout = self._is_readout_sequencer(sequence)
            if has_acquisition and not is_readout:
                raise PassFailedException(
                    f"Qblox allocation ({key[0]!r}, {key[1]}, "
                    f"{sequence.seq_idx.data if sequence.seq_idx is not None else '?'}) "
                    "does not support acquisitions."
                )
            sequencer_config = sequence.sequencer_config
            acquisition_disabled = not isinstance(
                sequencer_config.acquisition_disabled, NoneAttr
            ) and bool(sequencer_config.acquisition_disabled.value.data)
            acquisition_explicitly_disabled = not isinstance(
                sequencer_config.acquisition_enabled, NoneAttr
            ) and not bool(sequencer_config.acquisition_enabled.value.data)
            if has_acquisition and (
                acquisition_disabled or acquisition_explicitly_disabled
            ):
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} uses acquisitions while its "
                    "acquisition path is explicitly disabled."
                )
            has_acquisition_route = (
                isinstance(sequencer_config.acquisition_path_connections, ArrayAttr)
                and bool(sequencer_config.acquisition_path_connections)
            ) or (
                isinstance(sequencer_config.connections, ArrayAttr)
                and any(connection.input_ids for connection in sequencer_config.connections)
            )
            if has_acquisition and not has_acquisition_route:
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} uses acquisitions without a "
                    "configured acquisition route."
                )
            totals[key] = totals.get(key, 0) + sum(
                acquisition.num_bins.data for acquisition in sequence.acquisitions
            )

        for key, total in totals.items():
            kind = modules[key].kind.data
            limit = getattr(self._module_data(kind), "max_binned_acquisitions", None)
            if limit is None:
                continue
            if total > limit:
                raise PassFailedException(
                    f"Qblox module {key} allocates {total} acquisition bins, exceeding "
                    f"the {limit} bin capacity for {kind.value}."
                )

    def _check_table_capacity(self, sequences: list[SequenceOp]) -> None:
        """Validate per-sequencer waveform and integration-weight sample capacity."""

        for sequence in sequences:
            sequencer_data = self._sequencer_data(sequence)
            waveform_limit = sequencer_data.max_sample_size_waveforms
            waveform_indices = {waveform.index.data for waveform in sequence.waveforms}
            weight_indices = {weight.index.data for weight in sequence.weights}
            for nested in sequence.body.block.walk():
                if isinstance(nested, PlayImmImmImmOp):
                    missing = sorted(
                        {nested.wave0.data, nested.wave1.data} - waveform_indices
                    )
                    if missing:
                        raise PassFailedException(
                            f"Sequence {sequence.channel_id.data!r} uses waveform indices "
                            f"{missing} that are absent from its waveform table."
                        )
                if isinstance(nested, AcquireWeightedImmImmImmImmImmOp):
                    missing = sorted(
                        {nested.weight_idx0.data, nested.weight_idx1.data} - weight_indices
                    )
                    if missing:
                        raise PassFailedException(
                            f"Sequence {sequence.channel_id.data!r} uses weight indices "
                            f"{missing} that are absent from its weight table."
                        )
            waveform_total = sum(len(waveform.data) for waveform in sequence.waveforms)
            if waveform_total > waveform_limit:
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} contains {waveform_total} "
                    f"waveform samples, exceeding the {waveform_limit} sample sequencer "
                    "capacity."
                )
            weight_total = sum(len(weight.data) for weight in sequence.weights)
            weight_limit = (
                sequencer_data.max_sample_size_waveforms
                if self._is_readout_sequencer(sequence)
                else 0
            )
            if weight_total > weight_limit:
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} contains {weight_total} "
                    f"weight samples, exceeding the {weight_limit} sample sequencer "
                    "capacity."
                )

    def _check_instruction_capacity(self, sequences: list[SequenceOp]) -> None:
        """Validate the number of emitted instructions for each sequencer."""

        for sequence in sequences:
            if sequence.module_config is None:
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} is missing its Qblox "
                    "configuration."
                )
            sequencer_data = self._sequencer_data(sequence)
            sequencer_type = self._sequencer_target(sequence).sequencer_spec.type
            instruction_count = sum(
                not isinstance(instruction, LabelOp)
                for instruction in sequence.body.block.ops
            )
            if instruction_count > sequencer_data.max_num_instructions:
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} contains "
                    f"{instruction_count} instructions, exceeding the "
                    f"{sequencer_data.max_num_instructions} instruction capacity for its "
                    f"{sequencer_type.value} sequencer."
                )


def verify_qblox_pre_emission(
    op: ModuleOp,
    target_data: QbloxTargetData = TARGET_DATA,
) -> list[SequenceOp]:
    """Verify that a module is ready for Qblox program emission.

    :param op: Configured Q1 sequence module to verify.
    :param target_data: Qblox numeric limits used for final emission validation.
    :returns: The verified top-level sequences in program order.
    :raises PassFailedException: If the module violates the Qblox emission contract.
    """

    return _QbloxPreEmissionVerifier(target_data).verify(op)


@dataclass(frozen=True)
class QbloxPreEmissionVerificationPass(OrderedPass, ModulePass):
    """Pipeline wrapper for Qblox pre-emission verification."""

    name = "qblox-pre-emission-verification"
    target_data: QbloxTargetData = field(default=TARGET_DATA)

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        return frozenset({LinearScanRegisterAllocationPass, LineariseQ1CfToQ1Pass})

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        verify_qblox_pre_emission(op, self.target_data)
