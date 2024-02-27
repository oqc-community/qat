from unittest.mock import create_autospec
from scc.backends.echo import get_default_echo_hardware
from scc.compiler.config import CompilerConfig
from scc.qat import _execute_with_metrics
from os.path import join, dirname, abspath, exists
from scc.compiler.experimental.frontends import QIRFrontend, QASMFrontend
from scc.compiler.interrupt import BasicInterrupt
from queue import Queue
from threading import Thread, Event
import pytest


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
    x = exists(path)
    res = _execute_with_metrics(frontend, path, hardware, config)
    assert res[0]['00'] == 1000


def test_execute_using_interruptable_QasmFrontend():
    config = CompilerConfig()
    config.results_format.binary_count()
    frontend = QASMFrontend()
    hardware = get_default_echo_hardware(4)
    path = _get_qasm_path("basic.qasm")
    res = _execute_with_metrics(frontend, path, hardware, config)
    assert res[0]['c']['00'] == 1000


def test_interrupt_triggered_on_batch():
    config = CompilerConfig()
    path = _get_qasm_path("basic.qasm")
    hardware = get_default_echo_hardware(4)
    frontend = QASMFrontend()
    instructions = frontend.parse(path, hardware, config)
    queue = Queue(maxsize=1)
    event = Event()
    interrupt = BasicInterrupt(event, queue)
    t = Thread(
        target=frontend.execute_with_interrupt,
        args=(instructions, hardware, config, interrupt),
        daemon=True
    )
    # Set event before starting thread to prevent race conditions
    interrupt.trigger()
    t.start()
    t.join()
    why_finished = interrupt.queue.get(block=False)
    assert why_finished["interrupted"]
    assert why_finished["iteration"] == 0


@pytest.mark.parametrize("kill_index", range(3))
def test_interrupt_triggered_on_batch_n(kill_index):
    side_effects = [False] * 3
    side_effects[kill_index] = True

    hardware = get_default_echo_hardware(4)
    # Force more than one batch in the batches
    hardware.repeat_limit = 10
    hardware.default_repeat_count = 20

    config = CompilerConfig()
    path = _get_qasm_path("basic.qasm")
    frontend = QASMFrontend()
    instructions = frontend.parse(path, hardware, config)
    mock_event = create_autospec(Event)

    # Trigger cancellation on the nth batch
    mock_event.is_set.side_effect = side_effects
    interrupt = BasicInterrupt(mock_event)
    t = Thread(
        target=frontend.execute_with_interrupt,
        args=(instructions, hardware, config, interrupt),
        daemon=True
    )
    t.start()
    t.join()
    why_finished = interrupt.queue.get(block=False)
    assert why_finished["interrupted"]
    assert why_finished["iteration"] == kill_index


def test_interrupt_not_triggered_on_n_batches():

    hardware = get_default_echo_hardware(4)
    # Force more than one batch in the batches
    hardware.repeat_limit = 10
    hardware.default_repeat_count = 20

    config = CompilerConfig()
    path = _get_qasm_path("basic.qasm")
    frontend = QASMFrontend()
    instructions = frontend.parse(path, hardware, config)
    interrupt = BasicInterrupt()

    t = Thread(
        target=frontend.execute_with_interrupt,
        args=(instructions, hardware, config, interrupt),
        daemon=True
    )
    t.start()
    t.join()
    assert interrupt.queue.qsize() == 0, "No interrupt expected"
