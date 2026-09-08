# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Pre-emission verification for configured Qblox Q1 sequence IR."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, NoneAttr
from xdsl.ir import Attribute, ParametrizedAttribute, SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException, VerifyException

from qat.experimental.dialect.q1 import (
    ACQUISITION_OP_TYPES,
    AcquireImmImmImmOp,
    AcquireImmRsImmOp,
    AcquireWeightedImmImmImmImmImmOp,
    PlayImmImmImmOp,
)
from qat.experimental.dialect.q1.ir.abstract_ops import Q1AsmOperation
from qat.experimental.dialect.q1.ir.reg_desc import Q1RegisterType
from qat.experimental.dialect.q1.transforms.reg_alloc import (
    LinearScanRegisterAllocationPass,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPass
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    Q1SequencerFeature,
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

    Waveform, weight and acquisition-bin capacities are read from the authoritative Qblox
    target description. Physical module configuration is attached directly to each sequence;
    no duplicate module-level registry is required.
    """

    def verify(self, op: ModuleOp) -> None:
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
                    f"q1_sequence.sequence; found {child.name}."
                )
            self._check_sequence(child)
            sequences.append(child)
        self._check_allocations(sequences)
        self._check_table_capacity(sequences)
        self._check_acquisition_capacity(sequences)

    @staticmethod
    def _check_sequence(sequence: SequenceOp) -> None:
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
            for operand in nested.operands:
                _check_register(operand, nested.name)
            for result in nested.results:
                _check_register(result, nested.name)
            for attribute in (*nested.attributes.values(), *nested.properties.values()):
                _check_attribute(attribute, nested.name)
        try:
            sequence.verify()
        except VerifyException as exc:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} failed Qblox verification: {exc}"
            ) from exc

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

    @staticmethod
    def _check_acquisition_capacity(sequences: list[SequenceOp]) -> None:
        """Validate sequencer acquisition roles and aggregate module bin capacity."""

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
            module = modules[key]
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
            is_readout = sequence.seq_idx is not None and DEFAULT_QBLOX_TARGET.supports(
                module.kind.data,
                sequence.seq_idx.data,
                Q1SequencerFeature.acquisition,
            )
            if has_acquisition and not is_readout:
                raise PassFailedException(
                    f"Qblox allocation ({key[0]!r}, {key[1]}, "
                    f"{sequence.seq_idx.data if sequence.seq_idx is not None else '?'}) "
                    "does not support acquisitions."
                )
            config = sequence.sequencer_config
            acquisition_disabled = not isinstance(
                config.acquisition_disabled, NoneAttr
            ) and bool(config.acquisition_disabled.value.data)
            acquisition_explicitly_disabled = not isinstance(
                config.acquisition_enabled, NoneAttr
            ) and not bool(config.acquisition_enabled.value.data)
            if has_acquisition and (
                acquisition_disabled or acquisition_explicitly_disabled
            ):
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} uses acquisitions while its "
                    "acquisition path is explicitly disabled."
                )
            has_acquisition_route = (
                isinstance(config.acquisition_path_connections, ArrayAttr)
                and bool(config.acquisition_path_connections)
            ) or (
                isinstance(config.connections, ArrayAttr)
                and any(connection.input_ids for connection in config.connections)
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
            limit = DEFAULT_QBLOX_TARGET.module_spec(kind).acquisition_memory_bins
            if limit is None:
                continue
            if total > limit:
                raise PassFailedException(
                    f"Qblox module {key} allocates {total} acquisition bins, exceeding "
                    f"the {limit} bin capacity for {kind.value}."
                )

    @staticmethod
    def _check_table_capacity(sequences: list[SequenceOp]) -> None:
        """Validate per-sequencer waveform and integration-weight sample capacity."""

        for sequence in sequences:
            if sequence.module_config is None or sequence.seq_idx is None:
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} is missing its Qblox "
                    "configuration."
                )
            sequencer_spec = DEFAULT_QBLOX_TARGET.sequencer(
                sequence.module_config.kind.data, sequence.seq_idx.data
            ).sequencer_spec
            waveform_limit = sequencer_spec.waveform_sample_capacity
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
                sequencer_spec.readout.weight_sample_capacity
                if sequencer_spec.readout is not None
                else 0
            )
            if weight_total > weight_limit:
                raise PassFailedException(
                    f"Sequence {sequence.channel_id.data!r} contains {weight_total} "
                    f"weight samples, exceeding the {weight_limit} sample sequencer "
                    "capacity."
                )


def verify_qblox_pre_emission(op: ModuleOp) -> None:
    """Verify that a module is ready for Qblox program emission.

    :param op: Configured Q1 sequence module to verify.
    :raises PassFailedException: If the module violates the Qblox emission contract.
    """

    _QbloxPreEmissionVerifier().verify(op)


@dataclass(frozen=True)
class QbloxPreEmissionVerificationPass(OrderedPass, ModulePass):
    """Pipeline wrapper for Qblox pre-emission verification."""

    name = "qblox-pre-emission-verification"

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        return frozenset({LinearScanRegisterAllocationPass})

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        verify_qblox_pre_emission(op)
