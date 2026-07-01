# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :mod:`qat.experimental.dialect.frontend.importer.base`."""

import pytest
from bidict import ValueDuplicationError
from xdsl.dialects import func
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i32
from xdsl.interpreters.scf import scf
from xdsl.ir import SSAValue

from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrameType,
    FrequencyAttr,
    PhaseAttr,
    PhaseSetOp,
    PulseOp,
    SquareWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
)
from qat.experimental.frontend.importer.base import BaseLinearImporter


def _i32(value: int) -> SSAValue:
    return SSAValue.get(ArithConstantOp(IntegerAttr(value, i32), i32))


def _const_freq(value: float = 1e9) -> ConstantOp:
    return ConstantOp(FrequencyAttr(value))


def _const_phase(value: float = 0.0) -> ConstantOp:
    return ConstantOp(PhaseAttr(value))


def _const_time(value: float = 1e-7) -> ConstantOp:
    return ConstantOp(TimeAttr(value))


def _const_amp(value: float = 0.5) -> ConstantOp:
    return ConstantOp(AmplitudeAttr(value))


def _main_block(imp: "BaseLinearImporter"):
    """Return the body block of the module's top-level ``main`` function."""
    [main] = list(imp.module.body.block.ops)
    assert isinstance(main, func.FuncOp)
    return main.body.block


class _DummyImporter(BaseLinearImporter):
    """Minimal concrete :class:`BaseLinearImporter` for testing.

    Frames are keyed by a simple string; each target passed to
    :meth:`get_frames` is a ``(key, frequency)`` tuple so the tests can
    fully control which frame is being tracked.
    """

    def build(self, ir):  # pragma: no cover - never used in tests
        return self.module

    def create_frame(self, dummy_target: tuple[str, float], freq_op=None):
        """Dummy create-frame for testing frame tracking."""
        q_id, frequency = dummy_target
        if freq_op is None:
            freq_op = _const_freq(frequency)
        frame_op = CreateFrameOp(freq_op, StringAttr(q_id))
        self._current_block.add_ops([freq_op, frame_op])
        return frame_op.result

    def get_frames(self, dummy_targets: list[tuple[str, float]]):
        """Dummy get-frames for testing frame tracking."""
        frames = []
        for target in dummy_targets:
            q_id = target[0]
            frame = self._current_environment_variables.get(q_id)
            if frame is None:
                frame = self.create_frame(target)
                self._current_environment_variables[q_id] = frame
            frames.append(frame)
        return frames

    def translate(self, instruction):  # pragma: no cover - not exercised
        raise NotImplementedError


class TestBaseLinearImporterEnvironment:
    def test_initial_environment_is_empty(self):
        imp = _DummyImporter()
        assert dict(imp._current_environment_variables) == {}

    def test_assign_then_lookup(self):
        imp = _DummyImporter()
        ssa = _i32(3)
        imp._current_environment_variables["x"] = ssa
        assert imp._current_environment_variables["x"] is ssa
        assert imp._current_environment_variables.inverse[ssa] == "x"

    def test_bidict_enforces_unique_values(self):
        imp = _DummyImporter()
        ssa = _i32(1)
        imp._current_environment_variables["x"] = ssa
        with pytest.raises(ValueDuplicationError):
            imp._current_environment_variables["y"] = ssa

    def test_forceput_overwrites_existing_value(self):
        imp = _DummyImporter()
        ssa1 = _i32(1)
        ssa2 = _i32(2)
        imp._current_environment_variables["x"] = ssa1
        imp._current_environment_variables.forceput("x", ssa2)
        assert imp._current_environment_variables["x"] is ssa2
        assert imp._current_environment_variables.inverse.get(ssa1) is None

    def test_get_missing_returns_none(self):
        imp = _DummyImporter()
        assert imp._current_environment_variables.get("missing") is None


class TestBaseLinearImporterAddOps:
    def test_add_ops_appends_to_current_block(self):
        imp = _DummyImporter()
        c = ArithConstantOp(IntegerAttr(1, i32), i32)
        imp._add_ops([c])
        assert list(imp._current_block.ops) == [c]

    def test_frame_is_created_and_tracked(self):
        imp = _DummyImporter()
        [frame] = imp.get_frames([("q0", 4.2e9)])
        assert isinstance(frame.type, FrameType)
        assert imp._current_environment_variables.get("q0") is frame

    def test_pulse_op_rebinds_frame(self):
        imp = _DummyImporter()
        [frame] = imp.get_frames([("q0", 3.8e9)])
        width = _const_time(1e-7)
        amp = _const_amp(0.5)
        wave = SquareWaveformOp(width, amp)
        pulse = PulseOp(frame, wave)
        imp._add_ops([width, amp, wave, pulse])
        assert imp._current_environment_variables.get("q0") is pulse.result
        # Old SSA value no longer in the map.
        assert imp._current_environment_variables.inverse.get(frame) is None
        # Assert new SSA is in the map.
        assert imp._current_environment_variables.inverse.get(pulse.result) == "q0"

    def test_wait_op_rebinds_frame(self):
        imp = _DummyImporter()
        [frame] = imp.get_frames([("q0", 4.2e9)])
        dur = _const_time(1e-7)
        wait = WaitOp(frame, dur)
        imp._add_ops([dur, wait])
        assert imp._current_environment_variables.inverse.get(frame) is None
        assert imp._current_environment_variables.get("q0") is wait.result

    def test_sync_op_rebinds_all_frames(self):
        imp = _DummyImporter()
        frames = imp.get_frames([("q0", 3.6e9), ("q1", 4.2e9)])
        sync = SynchronizeOp(*frames)
        imp._add_ops([sync])
        for frame in frames:
            assert imp._current_environment_variables.inverse.get(frame) is None
        assert imp._current_environment_variables.get("q0") is sync.results[0]
        assert imp._current_environment_variables.get("q1") is sync.results[1]

    def test_phase_op_rebinds_frame(self):
        imp = _DummyImporter()
        [frame] = imp.get_frames([("q0", 4.2e9)])
        phase = ConstantOp(PhaseAttr(0.25))
        phase_set = PhaseSetOp(frame, phase)
        imp._add_ops([phase, phase_set])
        # The old frame SSA value should be evicted from the map and the
        # phase op's result registered in its place.
        assert imp._current_environment_variables.inverse.get(frame) is None
        assert imp._current_environment_variables.get("q0") is phase_set.result

    def test_acquire_op_rebinds_frame(self):
        imp = _DummyImporter()
        [frame] = imp.get_frames([("q0", 4.2e9)])
        dur = _const_time(1e-6)
        acquire = AcquireOp(frame, dur)
        imp._add_ops([dur, acquire])
        assert imp._current_environment_variables.inverse.get(frame) is None
        # AcquireOp rebinds via ``frame_result`` (not ``result``).
        assert imp._current_environment_variables.get("q0") is acquire.frame_result

    def test_update_frames_from_ops_raises_for_unknown_frame(self):
        imp = _DummyImporter()
        # Construct an ad-hoc CreateFrameOp inside main but never register
        # its SSA value in the environment.
        freq = _const_freq()
        orphan = CreateFrameOp(freq, StringAttr("q_orphan"))
        imp._current_block.add_ops([freq, orphan])

        wait_dur = _const_time(1e-7)
        wait = WaitOp(orphan.result, wait_dur)
        with pytest.raises(KeyError):
            imp._add_ops([wait_dur, wait])

    def test_add_ops_ignores_ops_without_frame_operands(self):
        imp = _DummyImporter()
        [frame] = imp.get_frames([("q0", 4.2e9)])
        baseline = dict(imp._current_environment_variables)
        # ``arith.constant`` carries no FrameType operands; the env map
        # must remain untouched.
        c = ArithConstantOp(IntegerAttr(7, i32), i32)
        imp._add_ops([c])
        assert dict(imp._current_environment_variables) == baseline
        assert c in list(imp._current_block.ops)


class TestBaseLinearImporterForLoops:
    def test_enter_for_loop_switches_block(self):
        imp = _DummyImporter()
        parent_block = imp._current_block
        env = imp._current_environment_variables
        imp.enter_for_loop(0, 10, 1)

        assert imp._current_block is not parent_block
        assert imp._current_block.parent_block() is parent_block
        # The env is a single flat bidict; it is the same object
        # before/after entering the loop (no scoped stack).
        assert imp._current_environment_variables is env

        for_op = parent_block.last_op
        assert isinstance(for_op, scf.ForOp)
        assert imp._current_block is for_op.body.block

    def test_exit_for_loop_returns_block(self):
        imp = _DummyImporter()
        env = imp._current_environment_variables
        imp.enter_for_loop(0, 10, 1)
        imp.exit_for_loop()
        assert imp._current_block is _main_block(imp)
        assert imp._current_environment_variables is env

    def test_enter_for_loop_captures_env_vars_as_iter_args(self):
        imp = _DummyImporter()
        [outer_frame] = imp.get_frames([("q0", 4.2e9)])
        imp.enter_for_loop(0, 10, 1)

        for_op = _main_block(imp).last_op
        assert isinstance(for_op, scf.ForOp)
        # iter-args are snapshotted from the env at enter time.
        assert len(for_op.iter_args) == 1
        assert for_op.iter_args[0] is outer_frame
        # The loop body block has [index, frame_block_arg].
        assert len(imp._current_block.args) == 2

    def test_exit_for_loop_yields_and_propagates_body_vars(self):
        imp = _DummyImporter()
        [outer_frame] = imp.get_frames([("q0", 4.2e9)])
        imp.enter_for_loop(0, 10, 1)
        dur = _const_time(1e-7)
        wait = WaitOp(outer_frame, dur)
        imp._add_ops([dur, wait])
        assert imp._current_environment_variables.get("q0") is wait.result

        imp.exit_for_loop()

        for_op = _main_block(imp).last_op
        # iter-args were captured at enter time (the outer frame), not
        # the body-yielded value.
        assert len(for_op.iter_args) == 1
        assert for_op.iter_args[0] is outer_frame
        # The body's terminator yields the current env values.
        yield_op = for_op.regions[0].last_block.last_op
        assert isinstance(yield_op, scf.YieldOp)
        assert list(yield_op.arguments) == [wait.result]
        # After exit, env is rebound to the loop's results.
        assert imp._current_environment_variables.get("q0") is for_op.results[0]

    def test_nested_for_loops(self):
        imp = _DummyImporter()
        [outer_frame] = imp.get_frames([("q0", 1e9)])

        imp.enter_for_loop(0, 4, 1)
        d1 = _const_time(1e-7)
        w1 = WaitOp(outer_frame, d1)
        imp._add_ops([d1, w1])

        imp.enter_for_loop(0, 2, 1)
        d2 = _const_time(2e-7)
        w2 = WaitOp(w1.result, d2)
        imp._add_ops([d2, w2])
        inner_for_body = imp._current_block

        imp.exit_for_loop()
        assert imp._current_block is not inner_for_body
        outer_loop = _main_block(imp).last_op
        assert isinstance(outer_loop, scf.ForOp)
        assert imp._current_block is outer_loop.body.block

        inner_for = imp._current_block.last_op
        assert isinstance(inner_for, scf.ForOp)
        # Inner loop captured env["q0"] == w1.result at enter time.
        assert inner_for.iter_args[0] is w1.result
        assert imp._current_environment_variables.get("q0") is inner_for.results[0]

        imp.exit_for_loop()
        assert imp._current_block is _main_block(imp)
        # Outer loop captured env["q0"] == outer_frame at enter time.
        assert outer_loop.iter_args[0] is outer_frame
        assert imp._current_environment_variables.get("q0") is outer_loop.results[0]

    def test_sequential_for_loops(self):
        imp = _DummyImporter()
        [frame] = imp.get_frames([("q0", 1e9)])

        imp.enter_for_loop(0, 4, 1)
        d1 = _const_time(1e-7)
        w1 = WaitOp(frame, d1)
        imp._add_ops([d1, w1])
        imp.exit_for_loop()

        first_for = _main_block(imp).last_op
        assert isinstance(first_for, scf.ForOp)
        assert first_for.iter_args[0] is frame
        assert imp._current_environment_variables.get("q0") is first_for.results[0]

        imp.enter_for_loop(0, 2, 1)
        d2 = _const_time(2e-7)
        live = imp._current_environment_variables.get("q0")
        w2 = WaitOp(live, d2)
        imp._add_ops([d2, w2])
        imp.exit_for_loop()

        for_ops = [op for op in _main_block(imp).ops if isinstance(op, scf.ForOp)]
        assert len(for_ops) == 2
        # The second loop's iter-args were snapshotted from the env at
        # its enter time, which is the first loop's result.
        assert for_ops[1].iter_args[0] is first_for.results[0]
        assert imp._current_environment_variables.get("q0") is for_ops[1].results[0]


class TestBaseLinearImporterMainFunction:
    def test_module_contains_single_main_function(self):
        imp = _DummyImporter()
        top_ops = list(imp.module.body.block.ops)
        assert len(top_ops) == 1
        main = top_ops[0]
        assert isinstance(main, func.FuncOp)
        # Default signature is `() -> ()`.
        assert main.sym_name.data == "main"
        assert list(main.function_type.inputs) == []
        assert list(main.function_type.outputs) == []
        # The body block is initially empty.
        assert list(main.body.block.ops) == []

    def test_current_block_is_main_body_initially(self):
        imp = _DummyImporter()
        assert imp._current_block is _main_block(imp)

    def test_translated_ops_are_inserted_into_main_body(self):
        imp = _DummyImporter()
        c = ArithConstantOp(IntegerAttr(1, i32), i32)
        imp._add_ops([c])
        assert c in list(_main_block(imp).ops)
        # The module body still contains only the main FuncOp.
        assert len(list(imp.module.body.block.ops)) == 1

    def test_add_final_return_appends_func_return(self):
        imp = _DummyImporter()
        imp._add_final_return()
        last = _main_block(imp).last_op
        assert isinstance(last, func.ReturnOp)
        assert list(last.arguments) == []

    def test_add_final_return_raises_when_not_in_main(self):
        imp = _DummyImporter()
        # Open an scf.for loop; the current block is now the loop body,
        # whose parent op is `scf.for` (not a FuncOp).
        imp.enter_for_loop(0, 1, 1)
        with pytest.raises(ValueError, match="not in function"):
            imp._add_final_return()

    def test_add_final_return_raises_when_function_not_main(self):
        imp = _DummyImporter()
        [main] = list(imp.module.body.block.ops)
        # Rename the function to anything other than "main".
        main.properties["sym_name"] = StringAttr("foo")
        with pytest.raises(ValueError, match="not main"):
            imp._add_final_return()


class TestBaseLinearImporterEmptyEnvLoops:
    """Loop helpers must also work when no env vars are currently tracked."""

    def test_enter_for_loop_with_empty_env_has_no_iter_args(self):
        imp = _DummyImporter()
        assert dict(imp._current_environment_variables) == {}
        imp.enter_for_loop(0, 5, 1)

        for_op = _main_block(imp).last_op
        assert isinstance(for_op, scf.ForOp)
        assert list(for_op.iter_args) == []
        # Only the induction variable is in the body block args.
        assert len(imp._current_block.args) == 1

    def test_exit_for_loop_with_empty_env_yields_nothing(self):
        imp = _DummyImporter()
        imp.enter_for_loop(0, 5, 1)
        imp.exit_for_loop()

        for_op = _main_block(imp).last_op
        assert isinstance(for_op, scf.ForOp)
        yield_op = for_op.regions[0].last_block.last_op
        assert isinstance(yield_op, scf.YieldOp)
        assert list(yield_op.arguments) == []
        assert imp._current_block is _main_block(imp)
