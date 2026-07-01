# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod

from bidict import bidict
from xdsl.dialects import func
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.interpreters.scf import scf
from xdsl.ir import Block, Operation, Region, SSAValue

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    FrameType,
    PhaseOp,
    PulseOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
    WaitOp,
)


class BaseLinearImporter(ABC):
    """Abstract base for linear, instruction-stream importers into the Pulse dialect.

    Provides the common machinery shared by concrete importers (e.g.
    :class:`~qat.experimental.dialect.frontend.importer.purr.PurrImporter`):

    * :class:`ModuleOp` is progressively populated. The module is
      initialised with a ``main`` function as an entry-point.
      All translated operations are inserted into the body of this
      function.
    * This class tracks the current insertion block (initially the
      body of ``main``) and environment variables to build a
      structure control flow from a flat IR. Environment variables
      maps source-language variables the latest SSA value.
    * Handle opening and closing ``scf.for`` regions which pass every
      tracked environment variable into the loop body as iter-args /
      block arguments and yield them all back out so the environment
      map. Threading global access of variables through blocks. This
      will be optimised in a later pass (COMPILER-1253).

    The environment is intentionally global (no scoped stack): every
    binding is visible everywhere and updates done inside an
    ``scf.for`` body are reflected in the enclosing scope through the
    loop's iter-args/results.  This is sufficient for structured
    ``for`` loops; ``if`` statements will need some more careful
    tracking of variables.
    """

    module: ModuleOp
    _current_block: Block
    _current_environment_variables: bidict[str, SSAValue]

    def __init__(self) -> None:
        """Initialise an empty module with a ``main`` function and empty environment.

        Creates a fresh :class:`ModuleOp` containing a single
        :class:`func.FuncOp` named ``main``, sets the function
        body's block as the current insertion point, and
        installs an empty :class:`bidict` as the
        environment-variable map.
        """
        main_block = Block()
        main = func.FuncOp("main", ((), ()), Region(main_block))
        self.module = ModuleOp(ops=[main])
        self._current_block = main_block
        self._current_environment_variables = bidict()

    def _add_final_return(self) -> None:
        """Terminate the enclosing ``main`` function with a ``func.return``.

        Intended to be called by :meth:`build` implementations after
        all source instructions have been translated, so that the
        produced ``main`` :class:`func.FuncOp` ends with a valid
        terminator and the module is well-formed.

        :raises ValueError: If the current insertion block is not
            directly inside a :class:`func.FuncOp`, or if that
            function is not named ``main`` (indicating a scope
            mismatch -- e.g. an unclosed ``scf.for`` region).
        """
        parent_op = self._current_block.parent_op()
        if not isinstance(parent_op, func.FuncOp):
            raise ValueError("Scope error: not in function at final return.")
        sym_name = parent_op.properties["sym_name"].data
        if sym_name != "main":
            raise ValueError(f"Scope error: final function not main, got {sym_name!r}.")
        self._current_block.add_ops([func.ReturnOp()])

    @abstractmethod
    def build(self, ir) -> ModuleOp:
        """Translate a source-language IR object into Pulse-dialect ops.

        :param ir: The source IR to import (e.g. a Purr
            ``QuantumInstructionBuilder``).
        :returns: The populated :class:`ModuleOp`.
        """
        ...

    def _add_ops(self, ops: list[Operation]) -> None:
        """Append *ops* to the current block and update frame tracking.

        Any operation that carries a ``frame`` or ``frames`` attribute
        (excluding :class:`CreateFrameOp`) is forwarded to
        :meth:`update_frames_from_ops` so that the environment-variable
        map stays in sync with the latest SSA values.

        :param ops: Sequence of xDSL operations to insert.
        """
        self._current_block.add_ops(ops)
        ops_with_frame_operands = [
            op
            for op in ops
            if any(isinstance(operand.type, FrameType) for operand in op.operands)
        ]
        self.update_frames_from_ops(ops_with_frame_operands)

    @abstractmethod
    def create_frame(self, target, freq_op: Operation | SSAValue | None) -> SSAValue:
        """Create a Pulse-dialect frame for the given source target.

        :param target: The source-language object describing the
            channel to be turned into a frame.
        :param freq_op: An optional pre-built op or SSA value providing
            the frame's frequency.  When ``None`` the implementation
            should synthesise an appropriate constant from the target.
        :returns: The SSA value representing the newly created frame.
        """
        ...

    @abstractmethod
    def get_frames(self, quantum_targets) -> list[SSAValue]:
        """Look up or create Pulse-dialect frames for source-language frame objects.

        Implementations are responsible for converting frequency-carrying source objects
        into frame SSA values, reusing already-created frames where possible.

        :param quantum_targets: Source-language frequency-carrying objects.
        :returns: Ordered list of SSA frame values corresponding to each target.
        """
        ...

    def update_frames_from_ops(self, ops: list[Operation]) -> None:
        """Refresh the environment-variable map after frame-mutating ops.

        For operations that produce a new frame SSA value (e.g.
        :class:`PulseOp`), the binding for the consumed frame is
        replaced with the freshly produced result so subsequent
        translations see the latest SSA value for each frame.

        :param ops: Frame-carrying operations whose results need
            tracking.
        :raises KeyError: If a consumed frame cannot be found in the
            environment-variable map.
        """
        for op in ops:
            if isinstance(
                op,
                PulseOp
                | StartContinuousWaveformOp
                | StopContinuousWaveformOp
                | WaitOp
                | PhaseOp,
            ):
                key = self._current_environment_variables.inverse.pop(op.frame)
                self._current_environment_variables[key] = op.result
            elif isinstance(op, AcquireOp):
                key = self._current_environment_variables.inverse.pop(op.frame)
                self._current_environment_variables[key] = op.frame_result
            elif isinstance(op, SynchronizeOp):
                for old_frame, new_frame in zip(op.frames, op.results, strict=False):
                    key = self._current_environment_variables.inverse.pop(old_frame)
                    self._current_environment_variables[key] = new_frame

    def enter_for_loop(self, start: int, stop: int, step: int) -> None:
        """Open a new ``scf.for`` region threading all env vars as iter-args.

        Creates ``arith.constant`` ops for *start*, *stop*, and *step*,
        gathers every currently-tracked environment variable, and
        emits an ``scf.for`` op whose body block takes an index
        induction variable followed by one block argument per
        environment variable.  The loop body is set as the current
        insertion point.

        :param start: Loop lower bound (inclusive).
        :param stop: Loop upper bound (exclusive).
        :param step: Loop step size.
        """
        index_type = IndexType()
        start_op = ArithConstantOp.from_int_and_width(start, index_type)
        stop_op = ArithConstantOp.from_int_and_width(stop, index_type)
        step_op = ArithConstantOp.from_int_and_width(step, index_type)
        current_vars = list(self._current_environment_variables.values())
        loop = scf.ForOp(
            start_op,
            stop_op,
            step_op,
            current_vars,
            Block(arg_types=[IndexType(), *[v.type for v in current_vars]]),
        )
        self._add_ops([start_op, stop_op, step_op, loop])
        self._current_block = loop.body.block

    def exit_for_loop(self) -> None:
        """Close the innermost ``scf.for`` region and propagate results out.

        Emits an ``scf.yield`` for every currently-tracked environment
        variable (in the same order they were passed in as iter-args),
        restores the parent block as the current insertion point, and
        rebinds each environment variable to the corresponding result
        of the enclosing ``scf.for`` op so the post-loop environment
        observes the loop's effects.
        """
        current_vars = list(self._current_environment_variables.values())
        yield_op = scf.YieldOp(*current_vars)
        self._add_ops([yield_op])

        for_op = self._current_block.parent_op()

        for old_var, new_var in zip(current_vars, for_op.results, strict=False):
            key = self._current_environment_variables.inverse.pop(old_var)
            self._current_environment_variables[key] = new_var

        self._current_block = self._current_block.parent_block()

    @abstractmethod
    def translate(self, instruction) -> None:
        """Dispatch a single source-language instruction.

        Concrete subclasses typically implement this with
        :func:`functools.singledispatchmethod` so that handlers are
        selected by the instruction's runtime type.

        :param instruction: The source-language instruction to
            translate into Pulse-dialect operations.
        """
        ...
