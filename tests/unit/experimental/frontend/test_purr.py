# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the PuRR frontend, including invocation of the PuRR pipeline and the importer.

These tests just operate at the surface-level; deep tests are done for the respective
components that make this frontend.
"""

from dataclasses import replace
from json import loads

import pytest
from compiler_config.config import CompilerConfig
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreters.scf import scf

from qat.experimental.dialect.pulse.ir import KernelOp, WaitOp
from qat.experimental.dialect.results.ir.ops import PostSelectOp
from qat.experimental.frontend.purr import PurrFrontend
from qat.experimental.system_data.canonical.schema import (
    MaxLikelihoodDiscriminateParams,
    MaxLikelihoodMethodData,
)
from qat.experimental.system_data.materialisers.boundary import materialise
from qat.model.loaders.purr import EchoModelLoader


@pytest.fixture
def echo_model():
    """Create an EchoModelLoader model with 2 qubits."""
    return EchoModelLoader(qubit_count=2).load()


@pytest.fixture
def builder(echo_model):
    program = echo_model.create_builder()
    q0 = echo_model.qubits[0]
    program.had(q0)
    program.measure(q0)
    return program


def _ops(module: ModuleOp):
    """Return all operations in a built module, including nested region ops."""

    return list(module.walk())


def _ops_of_type(module: ModuleOp, op_type):
    return [op for op in _ops(module) if isinstance(op, op_type)]


@pytest.fixture
def canonical_model_from_echo(echo_model):
    """Create canonical system data directly from the echo model.

    Converts the PuRR echo model to canonical representation via the materialisation
    boundary, ensuring exact channel parity with the echo model. Injects a
    max-likelihood discriminator with one disallowed state on acquire modes so
    post-selection tests can deterministically assert :class:`PostSelectOp`
    emission when enabled.
    """
    json = echo_model.get_calibration()
    canonical = materialise(source_payload=loads(json))

    # Ensure acquire channels carry at least one disallowed state so post-selection
    # tests deterministically emit PostSelectOp when enabled.
    method = MaxLikelihoodMethodData(
        states=(
            (0, MaxLikelihoodDiscriminateParams(location=0.0 + 0.0j)),
            (-1, MaxLikelihoodDiscriminateParams(location=2.0 + 0.0j)),
        )
    )

    qubits = []
    for qubit in canonical.qubits:
        modes = tuple(
            replace(mode, post_process_method=method)
            if mode.channel_id.endswith(".acquire")
            else mode
            for mode in qubit.modes
        )
        qubits.append(replace(qubit, modes=modes))

    return replace(canonical, qubits=tuple(qubits))


def test_check_and_return_returns_the_source_if_ib():
    """Tests that the ``check_and_return`` method succeeds for a
    QuantumInstructionBuilder."""
    model = EchoModelLoader(qubit_count=2).load()
    src = model.create_builder()

    frontend = PurrFrontend()

    assert frontend.check_and_return_source(src) is src


def test_canonical_systems_data_has_passive_reset_time(canonical_model_from_echo):
    """Tests the passive reset time is available."""

    passive_reset = next(
        (
            reset
            for reset in canonical_model_from_echo.reset_methods
            if reset.type == "passive"
        ),
        None,
    )
    assert passive_reset is not None, "Passive reset not found."

    duration_entry = next(
        (
            attribute
            for attribute in passive_reset.attributes
            if attribute.key == "duration"
        ),
        None,
    )
    assert duration_entry is not None
    assert duration_entry.value > 0


def _canonical_model_without_passive_reset(canonical_model_from_echo):
    """Return a copy of the canonical model with passive reset metadata removed."""

    return replace(
        canonical_model_from_echo,
        reset_methods=tuple(
            reset
            for reset in canonical_model_from_echo.reset_methods
            if reset.type != "passive"
        ),
        default_reset_method=None,
    )


def _canonical_model_with_passive_reset_duration(
    canonical_model_from_echo, duration_seconds: float
):
    """Return a copy of the canonical model with a custom passive reset duration."""

    duration_ps = int(round(duration_seconds * 1e12))
    reset_methods = []
    for reset in canonical_model_from_echo.reset_methods:
        if reset.type != "passive":
            reset_methods.append(reset)
            continue

        updated_attributes = tuple(
            replace(attribute, value=duration_ps)
            if attribute.key == "duration"
            else attribute
            for attribute in reset.attributes
        )
        reset_methods.append(replace(reset, attributes=updated_attributes))

    return replace(canonical_model_from_echo, reset_methods=tuple(reset_methods))


def _wait_durations(module):
    return [wait.duration.owner.value.value.data for wait in _ops_of_type(module, WaitOp)]


def test_passive_reset_time_from_canonical_data_is_used(builder, canonical_model_from_echo):
    """Tests the frontend uses the canonical passive reset duration when present."""

    frontend_model = _canonical_model_with_passive_reset_duration(
        canonical_model_from_echo, 2e-3
    )
    frontend = PurrFrontend(model=frontend_model)

    module = frontend.emit(builder)

    assert any(duration == pytest.approx(2e-3) for duration in _wait_durations(module))


def test_passive_reset_time_falls_back_to_builder_model(builder, canonical_model_from_echo):
    """Tests the frontend falls back to the PuRR builder model when canonical metadata is
    missing."""

    frontend_model = _canonical_model_without_passive_reset(canonical_model_from_echo)
    frontend = PurrFrontend(model=frontend_model)

    module = frontend.emit(builder)

    assert any(
        duration == pytest.approx(builder.model.default_repetition_period)
        for duration in _wait_durations(module)
    )


@pytest.mark.parametrize("src", [None, 123, "not-an-instruction-builder", object()])
def test_check_and_return_returns_false_for_non_ib(src):
    """Tests that the ``check_and_return`` method returns ``False`` for a non-
    QuantumInstructionBuilder source."""
    frontend = PurrFrontend()

    assert frontend.check_and_return_source(src) is False


@pytest.mark.parametrize("src", [None, 123, "not-an-instruction-builder", object()])
def test_emit_raises_type_error_for_non_ib(src):
    """Tests that ``emit`` rejects sources that are not QuantumInstructionBuilder."""
    frontend = PurrFrontend()

    with pytest.raises(TypeError, match="PuRR frontend does not support object of type"):
        frontend.emit(src)


def test_emitted_ir_is_a_module(builder):
    """Tests the PuRR frontend emits a module."""
    frontend = PurrFrontend()
    module = frontend.emit(builder)
    assert isinstance(module, ModuleOp)


def test_pipeline_is_invoked(builder):
    """Tests the pipeline is invoked by building IR without a repeat, a compiler config with
    a custom number of shots, and testing the produced IR has that amount of shots in the
    for loop.

    Searches for the KernelOp, and looks for the For loop within.
    """
    requested_shots = 7
    frontend = PurrFrontend()

    module = frontend.emit(builder, compiler_config=CompilerConfig(repeats=requested_shots))

    kernel = next(op for op in module.walk() if isinstance(op, KernelOp))
    kernel_ops = list(kernel.body.block.ops)
    [for_op] = [op for op in kernel_ops if isinstance(op, scf.ForOp)]

    assert isinstance(for_op.ub.owner, ArithConstantOp)
    assert for_op.ub.owner.value.value.data == requested_shots


def test_shots_default_to_one_when_repeat_not_added(builder):
    """Tests that shots default to one when a repeat is not added via compiler config."""
    frontend = PurrFrontend(run_purr_pipeline=False)

    module = frontend.emit(builder)

    kernel = next(op for op in module.walk() if isinstance(op, KernelOp))
    kernel_ops = list(kernel.body.block.ops)
    [for_op] = [op for op in kernel_ops if isinstance(op, scf.ForOp)]

    assert isinstance(for_op.ub.owner, ArithConstantOp)
    assert for_op.ub.owner.value.value.data == 1


def test_post_selection_enabled_with_canonical_model(builder, canonical_model_from_echo):
    """Tests that post-selection operations are added for acquisition channels when enabled.

    Verifies that PostSelectOp ops exist in the emitted module, with channel coverage
    matching the acquire channels from the echo model used in the program.
    """
    frontend = PurrFrontend(model=canonical_model_from_echo)

    module = frontend.emit(builder, compiler_config=CompilerConfig(post_selection=True))

    assert isinstance(module, ModuleOp)
    post_select_ops = [op for op in module.walk() if isinstance(op, PostSelectOp)]
    assert len(post_select_ops) == 1
    predicate = post_select_ops[0].predicates.data[0]
    assert {s.data for s in predicate.disallowed_values.data} == {-1}


def test_post_selection_disabled_with_canonical_model(builder, canonical_model_from_echo):
    """Tests that no post-selection operations are added when disabled, even with a
    model."""
    frontend = PurrFrontend(model=canonical_model_from_echo)

    module = frontend.emit(builder, compiler_config=CompilerConfig(post_selection=False))

    assert isinstance(module, ModuleOp)
    # Check that PostSelectOp does not exist when post-selection is disabled
    post_select_ops = [op for op in module.walk() if isinstance(op, PostSelectOp)]
    assert len(post_select_ops) == 0, (
        "Expected no PostSelectOp when post-selection is disabled"
    )


def test_post_selection_disabled_without_model(builder):
    """Tests that post-selection can be disabled without providing a canonical model."""
    frontend = PurrFrontend()

    module = frontend.emit(builder, compiler_config=CompilerConfig(post_selection=False))

    assert isinstance(module, ModuleOp)
    # Check that PostSelectOp does not exist when post-selection is disabled
    post_select_ops = [op for op in module.walk() if isinstance(op, PostSelectOp)]
    assert len(post_select_ops) == 0, (
        "Expected no PostSelectOp when post-selection is disabled"
    )


def test_post_selection_enabled_without_model_raises_error(builder):
    """Tests that enabling post-selection without providing a canonical model raises
    ValueError."""
    frontend = PurrFrontend()

    with pytest.raises(
        ValueError,
        match=(
            "Canonical system data description must be provided if post-selection is "
            "enabled"
        ),
    ):
        frontend.emit(builder, compiler_config=CompilerConfig(post_selection=True))


def test_default_compiler_config_has_post_selection_disabled(builder):
    """Tests that post-selection is disabled by default in compiler config."""
    frontend = PurrFrontend()

    # Should emit successfully without a model since post_selection defaults to False
    module = frontend.emit(builder)

    assert isinstance(module, ModuleOp)
    # Check that PostSelectOp does not exist with default config
    post_select_ops = [op for op in module.walk() if isinstance(op, PostSelectOp)]
    assert len(post_select_ops) == 0, (
        "Expected no PostSelectOp with default post-selection disabled"
    )
