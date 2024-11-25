from functools import reduce
from operator import mul
from queue import Queue
from threading import Event, Thread
from typing import List
from unittest.mock import create_autospec

import pytest
from compiler_config.config import CompilerConfig

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.experimental.frontends import QASMFrontend, QIRFrontend
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.interrupt import BasicInterrupt, InterruptError
from qat.purr.compiler.runtime import get_builder
from qat.qat import _execute_with_metrics

from tests.qat.qasm_utils import get_qasm2
from tests.qat.test_qir import _get_qir_path
from tests.qat.utils.models import raises_thread_exception


def _get_qasm_path(file_name):
    return get_qasm2(file_name)


@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
        get_default_qiskit_hardware,
    ],
)
def test_execute_using_interruptable_QIRFrontend(get_hardware):
    config = CompilerConfig()
    config.results_format.binary_count()
    frontend = QIRFrontend()
    hardware = get_hardware(4)
    path = _get_qir_path("generator-bell.ll")
    res, _ = _execute_with_metrics(frontend, path, hardware, config)
    assert sum(res.values()) == 1000
    assert "00" in res.keys()


@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
        get_default_qiskit_hardware,
    ],
)
def test_execute_using_interruptable_QasmFrontend(get_hardware):
    config = CompilerConfig()
    config.results_format.binary_count()
    frontend = QASMFrontend()
    hardware = get_hardware(4)
    path = _get_qasm_path("ghz_2.qasm")
    res, _ = _execute_with_metrics(frontend, path, hardware, config)
    assert sum(res["b"].values()) == 1000
    assert "00" in res["b"].keys()


# TODO: Test with simulator when implemented.
@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
    ],
)
def test_interrupt_triggered_on_batch(get_hardware):
    config = CompilerConfig()
    path = _get_qasm_path("ghz_2.qasm")
    hardware = get_hardware(2)
    frontend = QASMFrontend()
    instructions, _ = frontend.parse(path, hardware, config)
    assert isinstance(instructions, (InstructionBuilder, List))
    queue = Queue(maxsize=1)
    event = Event()
    interrupt = BasicInterrupt(event, queue)
    with raises_thread_exception(InterruptError):
        t = Thread(
            target=frontend.execute,
            args=(instructions, hardware, config, interrupt),
            kwargs={"interrupt": interrupt},
            daemon=True,
        )
        # Set event before starting thread to prevent race conditions
        interrupt.trigger()
        t.start()
        t.join()
    why_finished = interrupt.queue.get(block=False)
    assert why_finished["interrupted"]
    assert why_finished["metadata"]["batch_iteration"] == 0


# TODO: Test with simulator when implemented.
@pytest.mark.parametrize(
    "kill_index, repeat_count, repeat_limit",
    [(0, 10, 5), (3, 20, 5), (0, 10, 10), (0, 10, 20)],
)
@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
    ],
)
def test_interrupt_triggered_on_batch_n(
    kill_index, repeat_count, repeat_limit, get_hardware
):
    num_batches = repeat_count // repeat_limit
    if repeat_count % repeat_limit > 0:
        num_batches = num_batches + 1

    side_effects = [False] * (2 * num_batches)  # Account for sweep checkpoints
    side_effects[2 * kill_index] = True  # Account for sweep checkpoints

    hardware = get_hardware(2)
    # Force more than one batch in the batches
    hardware.repeat_limit = repeat_limit
    hardware.default_repeat_count = repeat_count

    config = CompilerConfig()
    path = _get_qasm_path("ghz_2.qasm")
    frontend = QASMFrontend()
    instructions, _ = frontend.parse(path, hardware, config)
    assert isinstance(instructions, (InstructionBuilder, List))
    mock_event = create_autospec(Event)

    # Trigger cancellation on the nth batch
    mock_event.is_set.side_effect = side_effects
    interrupt = BasicInterrupt(mock_event)
    with raises_thread_exception(InterruptError):
        t = Thread(
            target=frontend.execute,
            args=(instructions, hardware, config, interrupt),
            kwargs={"interrupt": interrupt},
            daemon=True,
        )
        t.start()
        t.join()
    why_finished = interrupt.queue.get(block=False)
    assert why_finished["interrupted"]
    assert why_finished["metadata"]["batch_iteration"] == kill_index


@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
        get_default_qiskit_hardware,
    ],
)
def test_interrupt_not_triggered_on_n_batches(get_hardware):

    hardware = get_hardware(2)
    # Force more than one batch in the batches
    hardware.repeat_limit = 10
    hardware.default_repeat_count = 20

    config = CompilerConfig()
    path = _get_qasm_path("ghz_2.qasm")
    frontend = QASMFrontend()
    instructions, _ = frontend.parse(path, hardware, config)
    assert isinstance(instructions, (InstructionBuilder, List))
    interrupt = BasicInterrupt()

    t = Thread(
        target=frontend.execute,
        args=(instructions, hardware, config, interrupt),
        kwargs={"interrupt": interrupt},
        daemon=True,
    )
    t.start()
    t.join()
    assert interrupt.queue.qsize() == 0, "No interrupt expected"


def _builder_1d_sweep_example(hw):
    # 1 sweep instruction, with measure single shot
    qubit = hw.get_qubit(0)
    amps = [i * 1e6 for i in range(5)]
    builder = (
        get_builder(hw)
        .sweep(SweepValue("amp", amps))
        .pulse(
            qubit.get_drive_channel(),
            width=100e-9,
            shape=PulseShapeType.SQUARE,
            amp=Variable("amp"),
        )
        .measure_single_shot_z(qubit)
    )
    return builder, [len(amps)]


# TODO: Test with simulator when implemented.
@pytest.mark.parametrize(
    "batch_n, repeat_count, repeat_limit, sweep_m",
    [(0, 10, 5, [0]), (1, 15, 5, [0]), (3, 20, 5, [1]), (0, 10, 10, [4]), (0, 10, 20, [3])],
)
@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
    ],
)
def test_interrupt_triggered_on_sweep_m_1d(
    batch_n, repeat_count, repeat_limit, sweep_m, get_hardware
):
    num_batches = repeat_count // repeat_limit
    if repeat_count % repeat_limit > 0:
        num_batches = num_batches + 1

    hardware = get_hardware(2)
    # Force more than one batch in the batches
    hardware.repeat_limit = repeat_limit
    hardware.default_repeat_count = repeat_count
    instructions, sweep_size = _builder_1d_sweep_example(hardware)
    assert isinstance(instructions, (InstructionBuilder, List))
    switerator_length = reduce(mul, sweep_size, 1)

    coordinates = 1
    for i in range(len(sweep_m) - 1):
        coordinates = coordinates + sweep_m[i] * reduce(mul, sweep_size[i + 1 :], 1)
    coordinates = coordinates + sweep_m[-1]

    side_effects = [False] * (
        num_batches * (switerator_length + 1)
    )  # Account for sweep checkpoints
    # side_effects[batch_n * (switerator_length + 1)] = True  # Batch checkpoints are separated by sweep checkpoints
    side_effects[batch_n * (switerator_length + 1) + coordinates] = True

    config = CompilerConfig()
    frontend = QASMFrontend()
    mock_event = create_autospec(Event)

    # Trigger cancellation on the nth batch
    mock_event.is_set.side_effect = side_effects
    interrupt = BasicInterrupt(mock_event)
    with raises_thread_exception(InterruptError):
        t = Thread(
            target=frontend.execute,
            args=(instructions, hardware, config, interrupt),
            kwargs={"interrupt": interrupt},
            daemon=True,
        )
        t.start()
        t.join()
    why_finished = interrupt.queue.get(block=False)
    assert why_finished["interrupted"]
    assert why_finished["metadata"]["batch_iteration"] == batch_n
    assert why_finished["metadata"]["sweep_iteration"] == sweep_m


def _builder_2d_sweep_example(hw):
    # 2 sweep instructions
    qubit = hw.get_qubit(0)
    amps = [i * 1e6 for i in range(5)]
    widths = [i * 100e-9 for i in range(1, 4)]
    builder = (
        get_builder(hw)
        .sweep(SweepValue("amp", amps))
        .sweep(SweepValue("width", widths))
        .pulse(
            qubit.get_drive_channel(),
            width=Variable("width"),
            shape=PulseShapeType.SQUARE,
            amp=Variable("amp"),
        )
        .measure_single_shot_z(qubit)
    )
    return builder, [len(amps), len(widths)]


# TODO: Test with simulator when implemented.
@pytest.mark.parametrize(
    "batch_n, repeat_count, repeat_limit, sweep_m",
    [
        (0, 10, 5, [0, 0]),
        (1, 15, 5, [0, 2]),
        (3, 20, 5, [1, 1]),
        (0, 10, 10, [4, 2]),
        (0, 10, 20, [3, 0]),
    ],
)
@pytest.mark.parametrize(
    "get_hardware",
    [
        get_default_echo_hardware,
    ],
)
def test_interrupt_triggered_on_sweep_m_2d(
    batch_n, repeat_count, repeat_limit, sweep_m, get_hardware
):
    num_batches = repeat_count // repeat_limit
    if repeat_count % repeat_limit > 0:
        num_batches = num_batches + 1

    hardware = get_hardware(2)
    # Force more than one batch in the batches
    hardware.repeat_limit = repeat_limit
    hardware.default_repeat_count = repeat_count
    instructions, sweep_size = _builder_2d_sweep_example(hardware)
    assert isinstance(instructions, (InstructionBuilder, List))
    switerator_length = reduce(mul, sweep_size, 1)

    coordinates = 1
    for i in range(len(sweep_m) - 1):
        coordinates = coordinates + sweep_m[i] * reduce(mul, sweep_size[i + 1 :], 1)
    coordinates = coordinates + sweep_m[-1]

    side_effects = [False] * (
        num_batches * (switerator_length + 1)
    )  # Account for sweep checkpoints
    # side_effects[batch_n * (switerator_length + 1)] = True  # Batch checkpoints are separated by sweep checkpoints
    side_effects[batch_n * (switerator_length + 1) + coordinates] = True

    config = CompilerConfig()
    frontend = QASMFrontend()
    mock_event = create_autospec(Event)

    # Trigger cancellation on the nth batch
    mock_event.is_set.side_effect = side_effects
    interrupt = BasicInterrupt(mock_event)
    with raises_thread_exception(InterruptError):
        t = Thread(
            target=frontend.execute,
            args=(instructions, hardware, config, interrupt),
            kwargs={"interrupt": interrupt},
            daemon=True,
        )
        t.start()
        t.join()
    why_finished = interrupt.queue.get(block=False)
    assert why_finished["interrupted"]
    assert why_finished["metadata"]["batch_iteration"] == batch_n
    assert why_finished["metadata"]["sweep_iteration"] == sweep_m
