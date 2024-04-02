from os.path import abspath, dirname, join
from queue import Queue
from threading import Event, Thread
from typing import List
from unittest.mock import create_autospec

import pytest

from qat import _parse_and_execute
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import CompilerConfig
from qat.purr.compiler.experimental.frontends import QASMFrontend, QIRFrontend
from qat.purr.compiler.interrupt import BasicInterrupt


def _get_qir_path(file_name):
    return join(abspath(join(dirname(__file__), "files", "qir", file_name)))


def _get_qasm_path(file_name):
    return join(abspath(join(dirname(__file__), "files", "qasm", "qasm2", file_name)))


def test_execute_using_interruptable_QIRFrontend():
    config = CompilerConfig()
    config.results_format.binary_count()
    frontend = QIRFrontend()
    hardware = get_default_echo_hardware(4)
    path = _get_qir_path("generator-bell.ll")
    res = _parse_and_execute(frontend, path, hardware, config)
    assert res[0]["00"] == 1000


def test_execute_using_interruptable_QasmFrontend():
    config = CompilerConfig()
    config.results_format.binary_count()
    frontend = QASMFrontend()
    hardware = get_default_echo_hardware(4)
    path = _get_qasm_path("ghz_2.qasm")
    res = _parse_and_execute(frontend, path, hardware, config)
    assert res[0]["b"]["00"] == 1000


def test_interrupt_triggered_on_batch():
    config = CompilerConfig()
    path = _get_qasm_path("ghz_2.qasm")
    hardware = get_default_echo_hardware(4)
    frontend = QASMFrontend()
    instructions, _ = frontend.parse(path, hardware, config)
    assert isinstance(instructions, (InstructionBuilder, List))
    queue = Queue(maxsize=1)
    event = Event()
    interrupt = BasicInterrupt(event, queue)
    t = Thread(
        target=frontend.execute_with_interrupt,
        args=(instructions, hardware, config, interrupt),
        daemon=True,
    )
    # Set event before starting thread to prevent race conditions
    interrupt.trigger()
    t.start()
    t.join()
    why_finished = interrupt.queue.get(block=False)
    assert why_finished["interrupted"]
    assert why_finished["iteration"] == 0


@pytest.mark.parametrize(
    "kill_index, repeat_count, repeat_limit",
    [(0, 10, 5), (3, 20, 5), (0, 10, 10), (0, 10, 20)],
)
def test_interrupt_triggered_on_batch_n(kill_index, repeat_count, repeat_limit):
    num_batches = repeat_count // repeat_limit
    if repeat_count % repeat_limit > 0:
        num_batches = num_batches + 1

    side_effects = [False] * num_batches
    side_effects[kill_index] = True

    hardware = get_default_echo_hardware(4)
    # Force more than one batch in the batches
    hardware.shot_limit = repeat_limit
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
    t = Thread(
        target=frontend.execute_with_interrupt,
        args=(instructions, hardware, config, interrupt),
        daemon=True,
    )
    t.start()
    t.join()
    why_finished = interrupt.queue.get(block=False)
    assert why_finished["interrupted"]
    assert why_finished["iteration"] == kill_index


def test_interrupt_not_triggered_on_n_batches():

    hardware = get_default_echo_hardware(2)
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
        target=frontend.execute_with_interrupt,
        args=(instructions, hardware, config, interrupt),
        daemon=True,
    )
    t.start()
    t.join()
    assert interrupt.queue.qsize() == 0, "No interrupt expected"
