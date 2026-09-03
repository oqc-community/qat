# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import singledispatchmethod

import numpy as np
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue

from qat.experimental.dialect.pulse.ir import (
    AcquisitionType,
    CallKernelOp,
    DiscriminateOp,
    EqualiseAttr,
    EqualiseOp,
    IQResultType,
    RealThresholdPolicyAttr,
)
from qat.experimental.dialect.pulse.ir.ops import KernelOp
from qat.experimental.dialect.results.ir import (
    CreateOp,
    ExtractOp,
    MapOp,
    PostSelectOp,
    RecordType,
    ResultsCollectionType,
    YieldOp,
)
from qat.experimental.frontend.importer.pulse.builder import PulseKernelBuilder
from qat.experimental.frontend.importer.pulse.post_processing import PostSelectionBuilder
from qat.experimental.waveforms.shapes.gaussian import GaussianWaveformShape
from qat.experimental.waveforms.shapes.gaussian_square import GaussianSquareWaveformShape
from qat.experimental.waveforms.shapes.rounded_square import RoundedSquareWaveformShape
from qat.experimental.waveforms.shapes.sech import SechWaveformShape
from qat.experimental.waveforms.shapes.setup_hold import SetupHoldWaveformShape
from qat.experimental.waveforms.shapes.sinusoidal import SinusoidalWaveformShape
from qat.experimental.waveforms.shapes.soft_square import SoftSquareWaveformShape
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    Assign,
    CustomPulse,
    Delay,
    DeviceUpdate,
    EndSweep,
    PhaseReset,
    PhaseSet,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    Pulse,
    QuantumInstruction,
    Repeat,
    Return,
    Sweep,
    Synchronize,
    Variable,
)

_KERNEL_NAME = "program"


@dataclass
class _PurrAnalysis:
    """A container for the analysis of a PuRR program which will be used to correctly
    assemble the module.

    :ivar quantum_instructions: The list of quantum instructions in the program.
    :ivar pulse_channels: The set of pulse channels used in the program.
    :ivar acquisition_types: A mapping from acquisition variable names to their acquisition
        modes.
    :ivar repeat: The repeat instruction, if present.
    :ivar sweeps: The list of sweep instructions in the program.
    :ivar device_updates: The list of device update instructions in the program.
    :ivar post_processing: A mapping from acquisition variable names to their post-
        processing instructions.
    :ivar assigns: The list of assign instructions in the program.
    :ivar returns: The set of variable names to return from the program.
    """

    quantum_instructions: list[QuantumInstruction] = field(default_factory=list)
    pulse_channels: set[PulseChannel] = field(default_factory=set)
    acquisition_types: dict[str, AcquireMode] = field(default_factory=dict)
    repeat: Repeat | None = None
    sweeps: list[Sweep] = field(default_factory=list)
    device_updates: list[DeviceUpdate] = field(default_factory=list)
    post_processing: dict[str, list[PostProcessing]] = field(
        default_factory=lambda: defaultdict(list)
    )
    assigns: list[Assign] = field(default_factory=list)
    returns: set[str] = field(default_factory=set)

    @property
    def number_of_shots(self) -> int:
        """Return the number of shots to run, based on the repeat instruction."""
        if self.repeat is None:
            return 1
        return self.repeat.repeat_count

    @classmethod
    def from_builder(cls, builder: QuantumInstructionBuilder) -> _PurrAnalysis:
        """Partition the instructions in *builder* into pulse instructions, post-processing
        instructions, and return variables."""

        analysis = cls()

        for instruction in builder.instructions:
            if isinstance(instruction, Repeat):
                if analysis.repeat is not None:
                    raise ValueError("Multiple repeat instructions are not supported.")
                analysis.repeat = instruction
                continue
            if isinstance(instruction, PostProcessing):
                key = instruction.quantum_targets[0].output_variable
                analysis.post_processing[key].append(instruction)
                continue
            if isinstance(instruction, Return):
                analysis.returns.update(instruction.variables)
                continue
            if isinstance(instruction, Sweep):
                analysis.sweeps.append(instruction)
                continue
            if isinstance(instruction, EndSweep):
                raise ValueError("EndSweep is not a supported instruction.")
            if isinstance(instruction, Assign):
                analysis.assigns.append(instruction)
                continue
            if isinstance(instruction, Variable):
                raise ValueError("Standalone variable instructions are not supported.")
            if isinstance(instruction, DeviceUpdate):
                analysis.device_updates.append(instruction)
                continue
            if isinstance(instruction, Acquire):
                key = instruction.output_variable
                analysis.acquisition_types[key] = instruction.mode
            if isinstance(instruction, QuantumInstruction):
                for target in instruction.quantum_targets:
                    analysis.pulse_channels.add(target)
                analysis.quantum_instructions.append(instruction)

        return analysis


class PurrImporter:
    """Takes a PuRR builder and produces a module representing that program.

    From a high-level, the module consists of two objects:

    1. A kernel that contains the pulse instructions and any acquisition instructions,
       embedded within the shot loop. This kernel returns a results collection.
    2. A main function that calls the kernel and applies any post-processing to the results
       through a ``results.map`` operation. The main function returns the results collection
       after post-processing has been applied.

    The importer does not currently support sweeps, or general device updates and
    parameterised operation operands.

    The importer starts by doing an analysis walk through the builder, understanding what
    results processing operations happen, and collecting unique pulse channels (frames).
    This allows us to structure the program in a way that does not necessarily map onto the
    builder in a linear fashion, but instead allows us to produce the structure described
    above.
    """

    def __init__(
        self,
        post_selection_builder: PostSelectionBuilder | None = None,
    ) -> None:
        """Initialise the importer.

        :param post_selection_builder: Optional post-selection builder. When provided
            and enabled, a :class:`~qat.experimental.dialect.results.ir.PostSelectOp`
            is inserted after the results map in the main function, filtering shots
            whose discriminated states are disallowed. Pass
            ``PostSelectionBuilder(PostProcessing.derive(system_data),
            enabled=config.post_selection)`` at the call site.
        """
        self._waveform_index = 0
        self._post_selection_builder = post_selection_builder
        self._label_to_channel: dict[str, str] = {}

    def build(
        self,
        purr_ir: QuantumInstructionBuilder,
    ) -> ModuleOp:
        """Translate *purr_ir* into a module containing a kernel and a main function.

        :param purr_ir: The purr IR to translate.
        """
        self._label_to_channel = {}
        analysis = _PurrAnalysis.from_builder(purr_ir)
        if analysis.sweeps:
            raise NotImplementedError("Sweep instructions are not yet supported.")

        kernel = self._build_kernel(analysis)
        kernel_collection_type = kernel.function_type.outputs.data[0]
        main = self._build_main(
            analysis, kernel_collection_type, self._post_selection_builder
        )
        return ModuleOp(ops=[kernel, main])

    def _build_kernel(self, analysis: _PurrAnalysis) -> KernelOp:
        """Build the pulse kernel from analyzed quantum instructions."""

        frequency_updates = self._resolve_device_updates(analysis)
        builder = self._initialise_builder(
            analysis.number_of_shots, analysis.pulse_channels, frequency_updates
        )
        for instruction in analysis.quantum_instructions:
            self.translate(instruction, builder)
        return builder.finalize()

    def _build_main(
        self,
        analysis: _PurrAnalysis,
        kernel_collection_type: ResultsCollectionType,
        post_selection_builder: PostSelectionBuilder | None = None,
    ) -> func.FuncOp:
        """Build the main function that calls the kernel and maps results.

        The function structure is:

        1. :class:`~qat.experimental.dialect.pulse.ir.CallKernelOp` — executes the
           pulse kernel, returning a raw IQ results collection.
        2. :class:`~qat.experimental.dialect.results.ir.MapOp` — discriminates each
           raw IQ record into integer state labels.
        3. :class:`~qat.experimental.dialect.results.ir.PostSelectOp` — (optional)
           discards shots whose discriminated states are disallowed. Only emitted when
           *post_selection_builder* is provided, enabled, and at least one acquire
           channel has disallowed states in the system data.
        4. :class:`~xdsl.dialects.func.ReturnOp` — returns the final collection.

        :param analysis: Analysed purr program.
        :param kernel_collection_type: The result type of the kernel, used to type the
            :class:`~qat.experimental.dialect.pulse.ir.CallKernelOp`.
        :param post_selection_builder: Optional post-selection builder.
        """
        entry_block = Block()
        call = CallKernelOp(_KERNEL_NAME, [], [kernel_collection_type])
        entry_block.add_ops([call])
        collection = call.result[0]
        map_op = self._build_results_map(collection, analysis)

        ops = [map_op]
        final_result = self._build_post_selection(post_selection_builder, ops)
        ops.append(func.ReturnOp(final_result))
        entry_block.add_ops(ops)
        main = func.FuncOp("main", ((), (final_result.type,)), Region(entry_block))
        return main

    def _build_post_selection(
        self,
        post_selection_builder: PostSelectionBuilder | None,
        ops: list,
    ) -> SSAValue:
        """Optionally append a :class:`~qat.experimental.dialect.results.ir.PostSelectOp` to
        *ops* and return the SSA value that should be wired to the return op.

        :param post_selection_builder: Optional post-selection builder.
        :param ops: ``ops[0]`` must be the :class:`~qat.experimental.dialect.results.ir.MapOp`.
        :returns: The SSA value to pass to ``func.ReturnOp``.
        """
        if post_selection_builder is not None:
            post_select = post_selection_builder.apply(
                ops[0].result, self._label_to_channel
            )
            if isinstance(post_select, PostSelectOp):
                ops.append(post_select)
                final_result = post_select.result
            else:
                final_result = ops[0].result
        else:
            final_result = ops[0].result
        return final_result

    def _initialise_builder(
        self,
        shots: int,
        pulse_channels: set[PulseChannel],
        frequency_updates: dict[str, float],
    ) -> PulseKernelBuilder:
        """Creates the builder for the kernel, adding all pulse channels as frames."""

        builder = PulseKernelBuilder(_KERNEL_NAME, shots=shots)
        for channel in pulse_channels:
            frame_name = self._frame_key(channel)
            if frame_name in frequency_updates:
                frequency = frequency_updates[frame_name]
            else:
                frequency = float(self._resolve_numeric(channel.frequency))
            builder.create_frame(
                self._frame_key(channel), frequency, channel.physical_channel_id
            )
        return builder

    def _resolve_device_updates(self, analysis: _PurrAnalysis) -> dict[str, float]:
        """Analyse the device updates to see if any can be applied.

        Only currently supports static frequency assignments.
        """

        frequency_updates: dict[str, float] = {}
        for update in analysis.device_updates:
            purr_device = update.target
            purr_device_attribute = update.attribute
            purr_value = update.value

            if not isinstance(purr_device, PulseChannel):
                raise ValueError(f"Unsupported device {purr_device} for device update.")

            if purr_device_attribute != "frequency":
                raise ValueError(
                    f"Unsupported pulse channel attribute {purr_device_attribute} in "
                    f"device update."
                )

            if isinstance(purr_value, Variable):
                raise NotImplementedError(
                    "Variable resolution is not yet supported in the device update."
                )

            fid = self._frame_key(purr_device)
            if fid in frequency_updates:
                raise ValueError(f"Multiple frequency updates for pulse channel {fid}.")
            frequency_updates[fid] = float(purr_value)

        return frequency_updates

    def _build_results_map(self, collection: SSAValue, analysis: _PurrAnalysis) -> MapOp:
        """Build a results.map operation that applies record-level post-processing."""
        body = Block(arg_types=[RecordType(collection.type.schema)])
        record = body.args[0]
        ssa_map = self._build_acquisition_value_map(body, record, analysis)
        self._add_assign_results(body, ssa_map, analysis)
        record_value = self._build_return_record(body, ssa_map, analysis)
        result_collection_type = ResultsCollectionType(
            record_value.type.schema,
            collection.type.size,
        )

        # Yield the record to the map operation
        body.add_op(YieldOp(record_value))
        return MapOp(collection, body, result_collection_type)

    def _build_acquisition_value_map(
        self,
        body: Block,
        record: SSAValue,
        analysis: _PurrAnalysis,
    ) -> dict[str, SSAValue]:
        """Extract acquisition values and apply per-key post-processing chains."""
        selected_keys = list(analysis.acquisition_types.keys())

        unknown_post_processing_keys = set(analysis.post_processing) - set(selected_keys)
        if unknown_post_processing_keys:
            missing = ", ".join(sorted(unknown_post_processing_keys))
            raise ValueError(
                "Post-processing references output variables with no prior acquisition "
                f"found in the environment: {missing}."
            )

        # Build the post-processing chains for each acquisition
        ssa_map: dict[str, SSAValue] = {}
        for key in selected_keys:
            if (
                analysis.post_processing[key]
                and analysis.acquisition_types[key] != AcquireMode.INTEGRATOR
            ):
                raise ValueError(
                    "Post-processing expects an IQResultType. Ensure the acquire has "
                    "mode INTEGRATOR."
                )

            extract_op = ExtractOp.value_from_record(record, key)
            body.add_op(extract_op)
            value = extract_op.result
            for pp in analysis.post_processing[key]:
                operation, value = self._convert_post_processing(pp, value)
                body.add_op(operation)
            ssa_map[key] = value

        return ssa_map

    def _add_assign_results(
        self,
        body: Block,
        ssa_map: dict[str, SSAValue],
        analysis: _PurrAnalysis,
    ) -> None:
        """Materialize assign instructions into tuple values in the map body."""
        for assign in analysis.assigns:
            assign_vars = self._resolve_assign_values(assign, ssa_map)
            if not isinstance(assign.value, list):
                # Scalar assign behaves as an alias: add a second key to the same SSA value.
                ssa_map[assign.name] = assign_vars[0]
                continue

            operation = CreateOp.for_tuple(assign_vars)
            ssa_map[assign.name] = operation.result
            body.add_op(operation)

    @staticmethod
    def _resolve_assign_values(
        assign: Assign,
        ssa_map: dict[str, SSAValue],
    ) -> list[SSAValue]:
        """Resolve assign input names to SSA values from the current result map."""
        assign_values = assign.value if isinstance(assign.value, list) else [assign.value]

        assign_vars: list[SSAValue] = []
        for val in assign_values:
            if isinstance(val, Variable):
                val = val.name
            if not isinstance(val, str):
                raise ValueError(f"Cannot assign value {val} in assign instruction.")
            if val not in ssa_map:
                raise ValueError(
                    f"Assign value {val} not found in post-processing results."
                )
            assign_vars.append(ssa_map[val])

        return assign_vars

    @staticmethod
    def _build_return_record(
        body: Block, ssa_map: dict[str, SSAValue], analysis: _PurrAnalysis
    ) -> SSAValue:
        """Create and return the final output record SSA value for the map body."""
        # Use the returns to create the new record
        returns = sorted(analysis.returns)
        if len(analysis.returns) == 0:
            # Just return everything if none is provided
            returns = list(ssa_map.keys())
        if not set(returns).issubset(set(ssa_map.keys())):
            raise ValueError(
                "Return variables must be a subset of the post-processing results."
            )
        values = [ssa_map[key] for key in returns]
        operation = CreateOp.for_record(returns, values)
        body.add_op(operation)
        return operation.result

    @staticmethod
    def _get_acquisition_type(acquire_mode: AcquireMode) -> Attribute:
        """Map an acquire mode to its corresponding result attribute type."""
        if acquire_mode == AcquireMode.INTEGRATOR:
            return IQResultType()
        if acquire_mode == AcquireMode.SCOPE:
            raise NotImplementedError(
                "Scope mode is not yet supported by the PurrImporter."
            )
        return AcquisitionType()

    @staticmethod
    def _convert_post_processing(
        instruction: PostProcessing, value: SSAValue
    ) -> tuple[Operation, SSAValue]:
        """Convert one post-processing step into an IR operation and output SSA value."""
        match instruction.process:
            case PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
                args = instruction.args
                if len(args) != 2:
                    raise ValueError(
                        f"LINEAR_MAP_COMPLEX_TO_REAL expects 2 arguments, got {len(args)}."
                    )

                affine_attr = EqualiseAttr(
                    linear_coefficient=0.5 * args[0],
                    conjugate_coefficient=0.5 * np.conj(args[0]),
                    translation=np.real(args[1]),
                )
                op = EqualiseOp(value, affine_attr)
                result = op.result
            case PostProcessType.DISCRIMINATE:
                if len(instruction.args) != 1:
                    raise ValueError(
                        f"DISCRIMINATE expects 1 argument, got {len(instruction.args)}."
                    )
                op = DiscriminateOp(
                    value,
                    RealThresholdPolicyAttr(threshold=instruction.args[0]),
                )
                result = op.result
            case _:
                raise ValueError(f"Unsupported post-processing type {instruction.process}.")

        return op, result

    @staticmethod
    def _frame_key(quantum_target: PulseChannel) -> str:
        """Return the unique frame identifier for a pulse channel."""
        return quantum_target.partial_id()

    def _resolve_numeric(self, value) -> float | int | complex:
        """Resolve a numeric-like value, rejecting unresolved runtime variables."""
        if isinstance(value, Variable):
            raise NotImplementedError("Variable resolution is not yet supported.")
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float | int | complex):
            return value
        raise ValueError(f"Unsupported value type {type(value)} for {value!r}.")

    def _get_waveform_name(self) -> str:
        """Generate a unique waveform name for emitted pulse waveforms."""
        self._waveform_index += 1
        return f"waveform_{self._waveform_index}"

    def _create_waveform(self, purr_waveform: Pulse, builder: PulseKernelBuilder) -> str:
        """Create a backend waveform from a PuRR pulse and return its symbol name."""
        waveform_name = self._get_waveform_name()
        width = float(self._resolve_numeric(purr_waveform.width))
        amplitude = self._resolve_numeric(purr_waveform.amp)
        drag = self._resolve_numeric(purr_waveform.drag)

        match purr_waveform.shape:
            case PulseShapeType.SQUARE:
                builder.create_square_waveform(waveform_name, amplitude, width)
            case PulseShapeType.GAUSSIAN:
                shape = GaussianWaveformShape.from_gaussian_waveform(
                    rise=float(self._resolve_numeric(purr_waveform.rise))
                )
                builder.create_gaussian_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_breadth,
                    shape.regularize,
                    drag,
                )
            case PulseShapeType.SOFT_SQUARE:
                shape = SoftSquareWaveformShape.from_soft_square_waveform(
                    rise=float(self._resolve_numeric(purr_waveform.rise)),
                    width=width,
                )
                builder.create_soft_square_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_top_width,
                    shape.fractional_rise,
                    shape.regularize,
                    drag,
                )
            case PulseShapeType.SOFTER_SQUARE:
                shape = SoftSquareWaveformShape.from_softer_square_waveform(
                    std_dev=float(self._resolve_numeric(purr_waveform.std_dev)),
                    rise=float(self._resolve_numeric(purr_waveform.rise)),
                    width=width,
                )
                builder.create_soft_square_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_top_width,
                    shape.fractional_rise,
                    shape.regularize,
                    drag,
                )
            case PulseShapeType.EXTRA_SOFT_SQUARE:
                shape = SoftSquareWaveformShape.from_extra_soft_square_waveform(
                    std_dev=float(self._resolve_numeric(purr_waveform.std_dev)),
                    rise=float(self._resolve_numeric(purr_waveform.rise)),
                    width=width,
                )
                builder.create_soft_square_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_top_width,
                    shape.fractional_rise,
                    shape.regularize,
                    drag,
                )
            case PulseShapeType.GAUSSIAN_SQUARE:
                shape = GaussianSquareWaveformShape.from_legacy(
                    std_dev=float(self._resolve_numeric(purr_waveform.std_dev)),
                    width=width,
                    square_width=float(self._resolve_numeric(purr_waveform.square_width)),
                    zero_at_edges=bool(self._resolve_numeric(purr_waveform.zero_at_edges)),
                )
                builder.create_gaussian_square_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_rise,
                    shape.fractional_top_width,
                    shape.regularize,
                    drag,
                )
            case PulseShapeType.SOFTER_GAUSSIAN:
                shape = GaussianWaveformShape.from_softer_gaussian_waveform(
                    rise=float(self._resolve_numeric(purr_waveform.rise))
                )
                builder.create_gaussian_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_breadth,
                    shape.regularize,
                    drag,
                )
            case PulseShapeType.BLACKMAN:
                builder.create_blackman_waveform(waveform_name, amplitude, width, drag)
            case PulseShapeType.SETUP_HOLD:
                shape = SetupHoldWaveformShape.from_legacy(
                    amp_setup=self._resolve_numeric(purr_waveform.amp_setup),
                    amp=amplitude,
                    rise=float(self._resolve_numeric(purr_waveform.rise)),
                    width=width,
                )
                builder.create_setup_hold_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.setup,
                    shape.rise_location,
                )
            case PulseShapeType.ROUNDED_SQUARE:
                shape = RoundedSquareWaveformShape.from_legacy(
                    rise=float(self._resolve_numeric(purr_waveform.rise)),
                    std_dev=float(self._resolve_numeric(purr_waveform.std_dev)),
                    width=width,
                )
                builder.create_rounded_square_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_top_width,
                    shape.fractional_rise,
                    drag,
                )
            case PulseShapeType.GAUSSIAN_DRAG:
                shape = GaussianWaveformShape.from_gaussian_zero_edge_waveform(
                    std_dev=float(self._resolve_numeric(purr_waveform.std_dev)),
                    width=width,
                    zero_at_edges=bool(self._resolve_numeric(purr_waveform.zero_at_edges)),
                )
                drag_coefficient = float(self._resolve_numeric(purr_waveform.beta))
                builder.create_gaussian_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_breadth,
                    shape.regularize,
                    drag_coefficient,
                )
            case PulseShapeType.GAUSSIAN_ZERO_EDGE:
                shape = GaussianWaveformShape.from_gaussian_zero_edge_waveform(
                    std_dev=float(self._resolve_numeric(purr_waveform.std_dev)),
                    width=width,
                    zero_at_edges=bool(self._resolve_numeric(purr_waveform.zero_at_edges)),
                )
                builder.create_gaussian_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_breadth,
                    shape.regularize,
                    drag,
                )
            case PulseShapeType.SECH:
                shape = SechWaveformShape.from_legacy(
                    std_dev=float(self._resolve_numeric(purr_waveform.std_dev)),
                    width=width,
                )
                builder.create_sech_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.fractional_breadth,
                    shape.regularize,
                    drag,
                )
            case PulseShapeType.COS:
                shape = SinusoidalWaveformShape.from_frequency(
                    frequency=float(self._resolve_numeric(purr_waveform.frequency)),
                    width=width,
                    internal_phase=float(
                        self._resolve_numeric(purr_waveform.internal_phase)
                    )
                    + math.pi / 2.0,
                )
                builder.create_sinusoidal_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.number_of_periods,
                    shape.internal_phase,
                    drag,
                )
            case PulseShapeType.SIN:
                shape = SinusoidalWaveformShape.from_frequency(
                    frequency=float(self._resolve_numeric(purr_waveform.frequency)),
                    width=width,
                    internal_phase=float(
                        self._resolve_numeric(purr_waveform.internal_phase)
                    ),
                )
                builder.create_sinusoidal_waveform(
                    waveform_name,
                    amplitude,
                    width,
                    shape.number_of_periods,
                    shape.internal_phase,
                    drag,
                )
            case _:
                raise ValueError(f"Unsupported shape, {purr_waveform.shape}.")

        return waveform_name

    @singledispatchmethod
    def translate(
        self, instruction: QuantumInstruction, builder: PulseKernelBuilder
    ) -> None:
        raise ValueError(f"{instruction} not a supported instruction.")

    @translate.register
    def _(self, value: PhaseSet, builder: PulseKernelBuilder) -> None:
        for target in value.quantum_targets:
            builder.phase_set(
                self._frame_key(target), float(self._resolve_numeric(value.phase))
            )

    @translate.register
    def _(self, value: PhaseReset, builder: PulseKernelBuilder) -> None:
        for target in value.quantum_targets:
            builder.phase_set(self._frame_key(target), 0.0)

    @translate.register
    def _(self, value: PhaseShift, builder: PulseKernelBuilder) -> None:
        for target in value.quantum_targets:
            builder.phase_shift(
                self._frame_key(target), float(self._resolve_numeric(value.phase))
            )

    @translate.register
    def _(self, value: Delay, builder: PulseKernelBuilder) -> None:
        for target in value.quantum_targets:
            builder.wait(self._frame_key(target), float(self._resolve_numeric(value.time)))

    @translate.register
    def _(self, value: Synchronize, builder: PulseKernelBuilder) -> None:
        frame_names = [self._frame_key(target) for target in value.quantum_targets]
        builder.synchronize(*frame_names)

    @translate.register
    def _(self, value: Pulse, builder: PulseKernelBuilder) -> None:
        frame_name = self._frame_key(value.quantum_targets[0])
        waveform_name = self._create_waveform(value, builder)
        builder.pulse(frame_name, waveform_name)

    @translate.register
    def _(self, value: CustomPulse, builder: PulseKernelBuilder) -> None:
        frame_name = self._frame_key(value.quantum_targets[0])
        waveform_name = self._get_waveform_name()
        builder.create_custom_waveform(
            waveform_name,
            list(value.samples),
            float(self._resolve_numeric(value.duration)),
        )
        builder.pulse(frame_name, waveform_name)

    @translate.register
    def _(self, value: Acquire, builder: PulseKernelBuilder) -> None:
        frame_name = self._frame_key(value.quantum_targets[0])
        self._label_to_channel[value.output_variable] = frame_name
        weights = None
        if value.filter is not None:
            if not isinstance(value.filter, CustomPulse):
                raise ValueError(
                    f"Acquire filter must be a CustomPulse, got "
                    f"{type(value.filter).__name__}."
                )
            weights = list(value.filter.samples)

        if value.mode == AcquireMode.SCOPE:
            raise NotImplementedError(
                "Scope mode is not yet supported by the PurrImporter."
            )

        builder.acquire(
            frame_name,
            value.output_variable,
            float(self._resolve_numeric(value.duration)),
            weights,
            integrate=value.mode == AcquireMode.INTEGRATOR,
        )
