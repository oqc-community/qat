# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from functools import reduce
from operator import mul
from queue import Queue
from threading import Event, Thread
from typing import List
from unittest.mock import create_autospec

import numpy as np
import pytest
from compiler_config.config import (
    CompilerConfig,
    MetricsType,
    Qasm2Optimizations,
    QuantumResultsFormat,
    TketOptimizations,
)

import qat.purr.compiler.experimental.frontends as experimental_frontends
import qat.purr.compiler.frontends as core_frontends
from qat.purr.backends.echo import EchoEngine, get_default_echo_hardware
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.backends.realtime_chip_simulator import qutip_available
from qat.purr.compiler.builders import InstructionBuilder, QuantumInstructionBuilder
from qat.purr.compiler.devices import PulseShapeType, QubitCoupling
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.interrupt import BasicInterrupt, InterruptError
from qat.purr.compiler.metrics import CompilationMetrics
from qat.purr.compiler.runtime import execute_instructions, get_builder
from qat.purr.qat import _execute_with_metrics, execute, execute_qasm, fetch_frontend

from tests.unit.purr.integrations.test_qir import _get_qir_path
from tests.unit.utils.models import raises_thread_exception
from tests.unit.utils.qasm_qir import (
    ProgramFileType,
    get_qasm2,
    get_qasm3,
    get_test_file_path,
)


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
    frontend = experimental_frontends.QIRFrontend()
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
    frontend = experimental_frontends.QASMFrontend()
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
    frontend = experimental_frontends.QASMFrontend()
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
    frontend = experimental_frontends.QASMFrontend()
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
    frontend = experimental_frontends.QASMFrontend()
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
    frontend = experimental_frontends.QASMFrontend()
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
    frontend = experimental_frontends.QASMFrontend()
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


class TestExecutionFrontend:
    def test_invalid_paths(self):
        with pytest.raises(ValueError):
            execute("/very/wrong.qasm")

    def test_valid_qasm_path(self):
        hardware = get_default_echo_hardware(2)
        execute(get_test_file_path(ProgramFileType.QASM2, "basic.qasm"), hardware=hardware)

    def test_quality_couplings(self):
        qasm_string = get_qasm2("basic.qasm")
        hardware = get_default_echo_hardware(8)
        hardware.qubit_direction_couplings = [
            QubitCoupling((0, 1)),
            QubitCoupling((1, 2), quality=10),
            QubitCoupling((2, 3), quality=10),
            QubitCoupling((4, 3), quality=10),
            QubitCoupling((4, 5), quality=10),
            QubitCoupling((6, 5), quality=10),
            QubitCoupling((7, 6), quality=7),
            QubitCoupling((0, 7)),
        ]

        results = execute_qasm(qasm_string, hardware=hardware)

        assert results is not None
        assert len(results) == 1
        assert len(results["c"]) == 2

    def test_quality_couplings_all_off(self):
        qasm_string = get_qasm2("basic.qasm")
        hardware = get_default_echo_hardware(8)
        hardware.qubit_direction_couplings = [
            QubitCoupling((0, 1)),
            QubitCoupling((1, 2), quality=10),
            QubitCoupling((2, 3), quality=10),
            QubitCoupling((4, 3), quality=10),
            QubitCoupling((4, 5), quality=10),
            QubitCoupling((6, 5), quality=10),
            QubitCoupling((7, 6), quality=7),
            QubitCoupling((0, 7)),
        ]

        config = CompilerConfig()
        config.optimizations = Qasm2Optimizations().disable()
        results = execute_qasm(qasm_string, hardware, config)

        assert results is not None
        assert len(results) == 1
        assert len(results["c"]) == 2

    @pytest.mark.skip(
        "Tket incorrectly fails verification with remapping off. Assert this is wrong, "
        "then fix upstream."
    )
    def test_quality_couplings_some_off(self):
        qasm_string = get_qasm2("basic.qasm")
        hardware = get_default_echo_hardware(8)
        hardware.qubit_direction_couplings = [
            QubitCoupling((0, 1)),
            QubitCoupling((1, 2), quality=10),
            QubitCoupling((2, 3), quality=10),
            QubitCoupling((4, 3), quality=10),
            QubitCoupling((4, 5), quality=10),
            QubitCoupling((6, 5), quality=10),
            QubitCoupling((7, 6), quality=7),
            QubitCoupling((0, 7)),
        ]

        config = CompilerConfig()
        config.tket_optimizations = (
            config.tket_optimizations & ~TketOptimizations.DefaultMappingPass
        )
        results = execute_qasm(qasm_string, hardware, config)

        assert results is not None
        assert len(results) == 1
        assert len(results["c"]) == 2

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_primitives(self):
        qasm_string = get_qasm2("primitives.qasm")
        results = execute_qasm(qasm_string)

        assert len(results) == 1
        assert "c" in results
        assert results["c"] == [1, 1]

    def test_engine_as_model(self):
        qasm_string = get_qasm2("ghz.qasm")
        engine = EchoEngine(get_default_echo_hardware(5))
        results = execute_qasm(qasm_string, engine)

        assert len(results) == 1
        assert "b" in results
        assert results["b"] == [0, 0, 0, 0]

    def test_ghz(self):
        qasm_string = get_qasm2("ghz.qasm")
        hardware = get_default_echo_hardware(5)
        results = execute_qasm(qasm_string, hardware)
        assert len(results) == 1
        assert "b" in results
        assert results["b"] == [0, 0, 0, 0]

    def test_basic_binary(self):
        qasm_string = get_qasm2("basic_results_formats.qasm")
        hardware = get_default_echo_hardware(8)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 2
        assert "ab" in results
        assert "c" in results
        assert results["ab"] == [0, 0]
        assert results["c"][1] == 0
        assert results["c"][0] in (1, 0)

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_basic_binary_count(self):
        qasm_string = get_qasm2("basic_results_formats.qasm")
        config = CompilerConfig()
        config.results_format = QuantumResultsFormat().binary_count()
        results = execute_qasm(qasm_string, compiler_config=config)
        assert "ab" in results
        assert "c" in results

        # ab is unmeasured, will always be empty.
        assert len(results["ab"]) == 1
        assert results["ab"]["00"] == 1000

        # c[1] is unmeasured, so one bit will always be static.
        assert len(results["c"]) == 2
        assert (results["c"]["10"] + results["c"]["00"]) == 1000

    def test_ecr(self):
        qasm_string = get_qasm2("ecr.qasm")
        hardware = get_default_echo_hardware(3)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["meas"] == [0, 0]

    def test_device_revert(self):
        hw = get_default_echo_hardware(4)
        drive = hw.get_qubit(0).get_drive_channel()
        original_drive_value = drive.frequency

        freq_array = np.linspace(4e9, 6e9, 10)
        builder = (
            get_builder(hw)
            .sweep(SweepValue("drive_freq", freq_array))
            .device_assign(drive, "frequency", Variable("drive_freq"))
        )
        builder.measure_mean_signal(hw.get_qubit(0))
        execute_instructions(hw, builder)

        assert drive.frequency == original_drive_value

    def test_ecr_exists(self):
        qasm_string = get_qasm2("ecr_exists.qasm")
        hardware = get_default_echo_hardware(2)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["meas"] == [0, 0]

    def test_example(self):
        qasm_string = get_qasm2("example.qasm")
        hardware = get_default_echo_hardware(9)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 2
        assert results["c"] == [0, 0, 0]
        assert results["d"] == [0, 0, 0]

    def test_example_if(self):
        qasm_string = get_qasm2("example_if.qasm")
        hardware = get_default_echo_hardware(2)
        with pytest.raises(ValueError):
            execute_qasm(qasm_string, hardware=hardware)

    def test_invalid_custom_gate(self):
        qasm_string = get_qasm2("invalid_custom_gate.qasm")
        hardware = get_default_echo_hardware(5)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c"] == [0, 0, 0]

    def test_invalid_mid_circuit_measure(self):
        qasm_string = get_qasm2("invalid_mid_circuit_measure.qasm")
        hardware = get_default_echo_hardware(2)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c"] == [0, 0]

    def test_mid_circuit_measure(self):
        qasm_string = get_qasm2("mid_circuit_measure.qasm")
        hardware = get_default_echo_hardware(3)
        with pytest.raises(ValueError):
            execute_qasm(qasm_string, hardware=hardware)

    def test_more_basic(self):
        qasm_string = get_qasm2("more_basic.qasm")
        hardware = get_default_echo_hardware(6)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c"] == [0, 0]

    def test_move_measurements(self):
        qasm_string = get_qasm2("move_measurements.qasm")
        hardware = get_default_echo_hardware(12)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c"] == [0, 0, 0]

    def test_order_cregs(self):
        qasm_string = get_qasm2("ordered_cregs.qasm")
        hardware = get_default_echo_hardware(4)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 3
        assert results["a"] == [0, 0]
        assert results["b"] == [0, 0]
        assert results["c"] == [0, 0]

    def test_parallel_test(self):
        qasm_string = get_qasm2("parallel_test.qasm")
        hardware = get_default_echo_hardware(10)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c0"] == [0, 0]

    def test_random_n5_d5(self):
        qasm_string = get_qasm2("random_n5_d5.qasm")
        hardware = get_default_echo_hardware(5)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c"] == [0, 0, 0, 0, 0]

    def test_metrics_filtered(self):
        metrics = CompilationMetrics(MetricsType.Empty)
        metrics.record_metric(MetricsType.OptimizedCircuit, "hello")
        assert metrics.get_metric(MetricsType.OptimizedCircuit) is None

    def test_metrics_add(self):
        metrics = CompilationMetrics()
        value = "hello"
        metrics.record_metric(MetricsType.OptimizedCircuit, value)
        assert metrics.get_metric(MetricsType.OptimizedCircuit) == value

    def test_parllel_execution(self):
        qasm_string = get_qasm2("parallel_test.qasm")

        opts = Qasm2Optimizations()
        opts.tket_optimizations = TketOptimizations.Empty
        config = CompilerConfig(
            repeats=300,
            repetition_period=1e-4,
            optimizations=opts,
            results_format=QuantumResultsFormat().binary_count(),
        )
        results = execute_qasm(
            qasm_string, hardware=get_default_echo_hardware(8), compiler_config=config
        )
        assert results is not None

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_basic_execution(self):
        qasm_string = get_qasm2("basic.qasm")
        results = execute_qasm(qasm_string)
        assert results is not None
        assert len(results) == 1
        assert len(results["c"]) == 2

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_binary_count_return(self):
        config = CompilerConfig(results_format=QuantumResultsFormat().binary_count())
        results = execute_qasm(get_qasm2("basic.qasm"), compiler_config=config)
        assert "c" in results
        assert len(results["c"]) == 4
        assert {"11", "01", "00", "10"} == set(results["c"].keys())

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_mid_circuit_measurement(self):
        qasm_string = get_qasm2("mid_circuit_measure.qasm")
        with pytest.raises(ValueError):
            execute_qasm(qasm_string)

    def test_too_many_qubits(self):
        with pytest.raises(ValueError):
            hw = get_default_echo_hardware()
            (
                get_builder(hw)
                .X(hw.get_qubit(5))
                .Y(hw.get_qubit(1))
                .parse()
                .parse_and_execute()
            )

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_basic_single_measures(self):
        qasm_string = get_qasm2("basic_single_measures.qasm")
        results = execute_qasm(qasm_string)

        # We're testing that individual assignments to a classical register get
        # correctly assigned, aka that measuring c[0] then c[1] results in c = [c0, c1].
        assert len(results["c"]) == 2

    @pytest.mark.parametrize(
        "use_experimental,frontend_mod",
        [
            (True, experimental_frontends),
            (False, core_frontends),
        ],
        ids=("Experimental", "Standard"),
    )
    def test_frontend_peek(self, use_experimental, frontend_mod):
        with pytest.raises(ValueError):
            fetch_frontend("", use_experimental=use_experimental)

        qasm2_string = get_qasm2("basic.qasm")
        frontend = fetch_frontend(qasm2_string, use_experimental=use_experimental)
        assert isinstance(frontend, frontend_mod.QASMFrontend)

        qasm3_string = get_qasm3("basic.qasm")
        frontend = fetch_frontend(qasm3_string, use_experimental=use_experimental)
        assert isinstance(frontend, frontend_mod.QASMFrontend)

        qir_string = get_test_file_path(ProgramFileType.QIR, "generator-bell.ll")
        frontend = fetch_frontend(qir_string, use_experimental=use_experimental)
        assert isinstance(frontend, frontend_mod.QIRFrontend)

    @pytest.mark.parametrize("use_experimental", [True, False])
    def test_separate_compilation_from_execution(self, use_experimental):
        hardware = get_default_echo_hardware()
        contents = get_qasm2("basic.qasm")
        frontend = fetch_frontend(contents, use_experimental=use_experimental)
        built, _ = frontend.parse(contents, hardware=hardware)
        assert isinstance(built, (InstructionBuilder, List))
        results = frontend.execute(instructions=built, hardware=hardware)
        assert results is not None

    def test_qasm_sim(self):
        model = get_default_qiskit_hardware(20)
        qasm = get_qasm2("basic.qasm")
        results = execute(qasm, model, CompilerConfig()).get("c")
        assert len(results) == 4
        assert results["11"] > 200
        assert results["01"] > 200
        assert results["10"] > 200
        assert results["00"] > 200

    def test_execute_different_qat_input_types(self):
        hw = get_default_echo_hardware(5)
        qubit = hw.get_qubit(0)
        phase_shift_1 = 0.2
        phase_shift_2 = 0.1
        builder = (
            get_builder(hw)
            .phase_shift(qubit, phase_shift_1)
            .X(qubit, np.pi / 2.0)
            .phase_shift(qubit, phase_shift_2)
            .X(qubit, np.pi / 2.0)
            .measure_mean_z(qubit)
        )

        with pytest.raises(TypeError):
            execute_qasm(qat_input=builder.instructions, hardware=hw)

    def test_serialized_references_persist(self):
        qasm_string = get_qasm2("serialize_orphan.qasm")
        hardware = get_default_echo_hardware(8)
        config = CompilerConfig()

        frontend = core_frontends.QASMFrontend()
        builder, metrics = frontend.parse(qasm_string, hardware, config)

        serialized_builder = builder.serialize()
        builder = QuantumInstructionBuilder.deserialize(serialized_builder)

        results_orig_hw, _ = frontend.execute(builder, hardware, config)
        results_rehy_hw, _ = frontend.execute(builder, builder.model, config)

        assert len(results_orig_hw) != 0
        assert len(results_rehy_hw) != 0
        assert results_orig_hw == results_rehy_hw
