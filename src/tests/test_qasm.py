# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from os import listdir
from os.path import dirname, isfile, join

import numpy as np
import pytest
from pytket.qasm import circuit_from_qasm_str
from qat.core import execute, execute_qasm
from qat.purr.backends.echo import EchoEngine, get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import (
    get_default_RTCS_hardware,
    qutip_available,
)
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.compiler.config import (
    CompilerConfig,
    MetricsType,
    Qasm2Optimizations,
    Qasm3Optimizations,
    QuantumResultsFormat,
    TketOptimizations,
)
from qat.purr.compiler.devices import PulseShapeType, QubitCoupling, Resonator
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QASMFrontend, fetch_frontend
from qat.purr.compiler.hardware_models import (
    QuantumHardwareModel,
    resolve_qb_pulse_channel,
)
from qat.purr.compiler.instructions import (
    Acquire,
    CrossResonancePulse,
    CustomPulse,
    Delay,
    MeasurePulse,
    Pulse,
    Return,
    SweepValue,
    Variable,
)
from qat.purr.compiler.metrics import CompilationMetrics
from qat.purr.compiler.runtime import execute_instructions, get_builder
from qat.purr.integrations.features import OpenPulseFeatures
from qat.purr.integrations.qasm import (
    DefaultQasm2Parser,
    Qasm2Parser,
    Qasm3Parser,
    RestrictedQasm2Parser,
    get_qasm_parser,
)
from qat.purr.integrations.tket import TketBuilder, TketQasmParser
from qiskit.qasm import QasmError

from .qasm_utils import (
    TestFileType,
    get_qasm2,
    get_qasm3,
    get_test_file_path,
    parse_and_apply_optimiziations,
)


class TestQASM3:
    """
    Tests for the parsing and instruction generation of OPENQASM3.0.

    In particular these tests currently only operate on a subset of OQ3
    and focus on the OpenPulse side of the language definition.
    """

    def test_qb_channel_resolution(self):
        hw = get_default_echo_hardware(8)
        qb = hw.get_qubit(1)

        res_qb, channel = resolve_qb_pulse_channel(qb)
        assert res_qb == qb
        assert channel == qb.get_default_pulse_channel()

        res_qb, channel = resolve_qb_pulse_channel(qb.get_default_pulse_channel())
        assert res_qb == qb
        assert channel == qb.get_default_pulse_channel()

    def test_named_defcal_arg(self):
        hw = get_default_echo_hardware(8)
        comp = CompilerConfig()
        comp.results_format.binary_count()
        results = execute_qasm(
            get_qasm3("named_defcal_arg.qasm"), hw, compiler_config=comp
        )
        results = next(iter(results.values()), dict())
        assert len(results) == 1
        assert results["00"] == 1000

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_basic(self):
        hw = get_default_RTCS_hardware()
        config = CompilerConfig()
        config.results_format.binary_count()
        results = execute_qasm(get_qasm3("basic.qasm"), hw, compiler_config=config)
        assert len(results["c"]) == 4

        # Assert the distribution is mostly correct. Don't care about absolute accuracy,
        # just that it's spread equally.
        assert not any([val for val in results["c"].values() if (val / 1000) < 0.15])

    def test_zmap(self):
        hw = get_default_echo_hardware(8)
        results = execute_qasm(get_qasm3("openpulse_tests/zmap.qasm"), hw)
        assert not isinstance(results, dict)

    @pytest.mark.parametrize(
        "arg_count",
        [1, 2, 3],  # 2 includes a generic qubit def, should be separate test
    )
    def test_expr_list_defcal(self, arg_count):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw),
            get_qasm3(f"openpulse_tests/expr_list_caldef_{arg_count}.qasm"),
        )
        instructions = result.instructions
        # There is a sync instruction from the implicit defcal barrier first
        assert len(instructions) == 2
        assert instructions[1].shape.value == PulseShapeType.SOFT_SQUARE.value

    def test_expr_list_defcal_different_arg(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw),
            get_qasm3("openpulse_tests/expr_list_defcal_different_arg.qasm"),
        )
        instructions = result.instructions
        assert len(instructions) > 2
        for instruction in instructions:
            if isinstance(instruction, Pulse):
                assert instruction.shape.value != PulseShapeType.SOFT_SQUARE.value

    def test_features(self):
        # Basic test that it dosen't exception.
        hw = get_default_echo_hardware(8)
        features = OpenPulseFeatures.from_hardware(hw)
        assert any(features.frames)
        assert any(features.ports)
        assert any(features.waveforms)
        assert features.constraints is not None

    def test_ghz(self):
        hw = get_default_echo_hardware(4)
        v3_qasm = get_qasm3("ghz.qasm")
        v2_qasm = get_qasm3("ghz_v2.qasm")

        v3_instructions = get_qasm_parser(v3_qasm).parse(get_builder(hw), v3_qasm)
        v2_instructions = get_qasm_parser(v2_qasm).parse(get_builder(hw), v2_qasm)

        assert len(v3_instructions.instructions) == len(v2_instructions.instructions)

    @pytest.mark.skip(reason="Need to be able to parse 'dt' correctly.")
    def test_complex_gates(self):
        hw = get_default_echo_hardware(8)
        execute_qasm(get_qasm3("complex_gates_test.qasm"), hw)

    def test_execution(self):
        hw = get_default_echo_hardware(8)
        execute_qasm(get_qasm3("lark_parsing_test.qasm"), hw)

    def test_invalid_qasm_version(self):
        hw = get_default_echo_hardware(8)
        with pytest.raises(ValueError) as context:
            execute_qasm(get_qasm3("invalid_version.qasm"), hw)

        # Assert we've errored and the language is mentioned
        message = str(context.value)
        assert "No valid parser could be found" in message
        assert "Qasm2" in message
        assert "Qasm3" in message

    def test_parsing(self):
        hw = get_default_echo_hardware(8)
        parser = Qasm3Parser()
        parser.parse(get_builder(hw), get_qasm3("lark_parsing_test.qasm"))

    def test_no_header(self):
        hw = get_default_echo_hardware(8)
        parser = Qasm3Parser()
        with pytest.raises(ValueError):
            parser.parse(get_builder(hw), get_qasm3("no_header.qasm"))

    def test_invalid_syntax(self):
        hw = get_default_echo_hardware(8)
        parser = Qasm3Parser()
        with pytest.raises(ValueError):
            parser.parse(get_builder(hw), get_qasm3("invalid_syntax.qasm"))

    @pytest.mark.skip("Test incomplete.")
    def test_wave_forms(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("waveform_test_sum.qasm"))
        instructions = result.instructions
        assert len(instructions) == 2
        assert instructions[0].shape.value == PulseShapeType.GAUSSIAN

    def test_ecr(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("ecr_test.qasm"))
        assert any(
            isinstance(inst, CrossResonancePulse) for inst in result.instructions
        )

    def test_cx_override(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("cx_override_test.qasm"))
        # assert that there are 2 rounded_square pulses, coming from custom def
        assert (
            len(
                [
                    inst
                    for inst in result.instructions
                    if hasattr(inst, "shape")
                    and (inst.shape is PulseShapeType.ROUNDED_SQUARE)
                ]
            )
            == 2
        )

    def test_invalid_frames(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        with pytest.raises(ValueError) as context:
            parser.parse(get_builder(hw), get_qasm3("invalid_frames.qasm"))
        assert "q42_drive" in str(context.value)

    def test_invalid_port(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        with pytest.raises(ValueError) as context:
            parser.parse(get_builder(hw), get_qasm3("invalid_port.qasm"))
        assert "channel_42" in str(context.value)

    def test_invalid_waveform(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        with pytest.raises(ValueError) as context:
            parser.parse(get_builder(hw), get_qasm3("invalid_waveform.qasm"))
        assert "waves_for_days" in str(context.value)

    def test_arb_waveform(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("arb_waveform.qasm"))
        pluses = [val for val in result.instructions if isinstance(val, CustomPulse)]
        pulse: CustomPulse = next(iter(pluses), None)

        assert len(pluses) == 1
        assert pulse is not None
        assert pulse.samples == [1, 1j]

    def test_delay(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("delay.qasm"))
        delays = [val for val in result.instructions if isinstance(val, Delay)]
        delay: Delay = next(iter(delays), None)

        assert len(delays) == 1
        assert delay is not None
        assert delay.time == 42e-9

    def test_redefine_cal(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("redefine_defcal.qasm"))
        pulses = [val for val in result.instructions if isinstance(val, Pulse)]
        assert len(pulses) == 2
        assert pulses[0].shape is PulseShapeType.GAUSSIAN_SQUARE
        assert pulses[1].shape is PulseShapeType.SQUARE

    def test_excessive_pulse_width_fails(self):
        hw = get_default_echo_hardware(8)
        with pytest.raises(ValueError):
            execute_qasm(get_qasm3("invalid_pulse_length.qasm"), hardware=hw)
        # pulse width within limits
        execute_qasm(get_qasm3("redefine_defcal.qasm"), hardware=hw)

    @pytest.mark.skip(
        "Double-check that you should be able to call gates this way. Seems dubious."
    )
    def test_simultaneous_frame_gates_fail(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openpulse_tests/frame_collision_test.qasm")
        )
        instruction = result.instructions
        assert len(instruction) > 0

    @pytest.mark.skip(
        "Timing isn't considered in pulse definition, only after scheduling. "
        "Fix or remove."
    )
    def test_pulse_timing(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openqasm_tests/gate_timing_test.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 1

    @pytest.mark.skip(
        "Timing isn't considered in pulse definition, only after scheduling. "
        "Fix or remove."
    )
    def test_barrier_timing(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openqasm_tests/barrier_timing_test.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 1

    @pytest.mark.skip("Test incomplete.")
    def test_phase_tracking(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openpulse_tests/phase_tracking_test.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 1
        assert isinstance(instruction[0], Delay)
        assert np.isclose(instruction[0].time, 42e-9)

    @pytest.mark.skip("Test incomplete.")
    def test_cross_res(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3("openpulse_tests/cross_resonance.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 1
        assert isinstance(instruction[0], Delay)
        assert np.isclose(instruction[0].time, 42e-9)

    def test_gaussians(self):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3("gaussians.qasm"))
        instruction = result.instructions
        assert len(instruction) > 1

    @pytest.mark.parametrize(
        "file_name,test_value",
        (
            ("sum", 0.425),
            ("mix", 0.04335),
        ),
    )
    def test_waveform_processing(self, file_name, test_value):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3(f"waveform_tests/waveform_test_{file_name}.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 2
        assert np.allclose(instruction[1].samples, test_value)

    @pytest.mark.parametrize(
        "file_name,attribute,test_value",
        (("scale", "scale_factor", 0.42), ("phase_shift", "phase", 0.4 + 0.2j)),
    )
    def test_waveform_processing_single_waveform(
        self, file_name, attribute, test_value
    ):
        hw = get_default_echo_hardware()
        parser = Qasm3Parser()
        result = parser.parse(
            get_builder(hw), get_qasm3(f"waveform_tests/waveform_test_{file_name}.qasm")
        )
        instruction = result.instructions
        assert len(instruction) == 2
        assert getattr(instruction[1], attribute) == test_value

    @pytest.mark.parametrize(
        "qasm_name",
        [
            "acquire",
            "constant_wf",
            "cross_ressonance",
            "detune_gate",
            "set_frequency",
            "shift_phase",
            "waveform_numerical_types",
        ],
    )
    def test_op(self, qasm_name):
        qasm_string = get_qasm3(f"openpulse_tests/{qasm_name}.qasm")
        hardware = get_default_echo_hardware(8)
        config = CompilerConfig(
            repeats=10,
            results_format=QuantumResultsFormat(),
            optimizations=Qasm3Optimizations(),
        )
        assert execute_qasm(qasm_string, hardware=hardware, compiler_config=config)


class TestExecutionFrontend:
    def test_invalid_paths(self):
        with pytest.raises(ValueError):
            execute("/very/wrong.qasm")

    def test_valid_qasm_path(self):
        hardware = get_default_echo_hardware(8)
        execute(get_test_file_path(TestFileType.QASM2, "basic.qasm"), hardware=hardware)

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
        assert results["c"]["00"] == 1000

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
        assert results["c"]["00"] == 1000

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

    def test_zero_repeats(self):
        config = CompilerConfig()
        config.repeats = 0
        qasm_string = get_qasm2("primitives.qasm")
        hardware = get_default_echo_hardware(8)
        with pytest.raises(ValueError):
            execute_qasm(qasm_string, compiler_config=config, hardware=hardware)

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_primitives(self):
        qasm_string = get_qasm2("primitives.qasm")
        hardware = get_default_RTCS_hardware()
        results = execute_qasm(qasm_string, hardware)

        assert len(results) == 1
        assert "c" in results
        assert results["c"]["11"] > 700

    def test_engine_as_model(self):
        qasm_string = get_qasm2("ghz.qasm")
        engine = EchoEngine(get_default_echo_hardware(5))
        results = execute_qasm(qasm_string, engine)

        assert len(results) == 1
        assert "b" in results
        assert results["b"]["0000"] == 1000

    def test_ghz(self):
        qasm_string = get_qasm2("ghz.qasm")
        hardware = get_default_echo_hardware(5)
        results = execute_qasm(qasm_string, hardware)
        assert len(results) == 1
        assert "b" in results
        assert results["b"]["0000"] == 1000

    def test_basic_binary(self):
        qasm_string = get_qasm2("basic_results_formats.qasm")
        hardware = get_default_echo_hardware(8)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 2
        assert "ab" in results
        assert "c" in results
        assert results["ab"]["00"] == 1000
        assert results["c"]["00"] == 1000

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_basic_binary_count(self):
        qasm_string = get_qasm2("basic_results_formats.qasm")
        config = CompilerConfig()
        config.results_format = QuantumResultsFormat().binary_count()
        hardware = get_default_RTCS_hardware()
        results = execute_qasm(qasm_string, hardware, config)
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
        assert results["meas"]["00"] == 1000

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
        assert results["meas"]["00"] == 1000

    def test_example(self):
        qasm_string = get_qasm2("example.qasm")
        hardware = get_default_echo_hardware(9)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 2
        assert results["c"]["000"] == 1000
        assert results["d"]["000"] == 1000

    def test_example_if(self):
        qasm_string = get_qasm2("example_if.qasm")
        hardware = get_default_echo_hardware(2)
        with pytest.raises(ValueError):
            execute_qasm(qasm_string, hardware=hardware)

    def test_valid_mid_circuit_measure_without_second_measure(self):
        qasm_string = get_qasm2("valid_mid_circuit_measure.qasm")
        hardware = get_default_echo_hardware(2)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c"]["00"] == 1000

    def test_mid_circuit_measure(self):
        qasm_string = get_qasm2("invalid_mid_circuit_measure.qasm")
        hardware = get_default_echo_hardware(3)
        with pytest.raises(ValueError):
            execute_qasm(qasm_string, hardware=hardware)

    def test_more_basic(self):
        qasm_string = get_qasm2("more_basic.qasm")
        hardware = get_default_echo_hardware(6)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c"]["00"] == 1000

    def test_move_measurements(self):
        qasm_string = get_qasm2("move_measurements.qasm")
        hardware = get_default_echo_hardware(12)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c"]["000"] == 1000

    def test_order_cregs(self):
        qasm_string = get_qasm2("ordered_cregs.qasm")
        hardware = get_default_echo_hardware(4)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 3
        assert results["a"]["00"] == 1000
        assert results["b"]["00"] == 1000
        assert results["c"]["00"] == 1000

    def test_parallel_test(self):
        qasm_string = get_qasm2("parallel_test.qasm")
        hardware = get_default_echo_hardware(10)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c0"]["00"] == 1000

    def test_random_n5_d5(self):
        qasm_string = get_qasm2("random_n5_d5.qasm")
        hardware = get_default_echo_hardware(5)
        results = execute_qasm(qasm_string, hardware=hardware)
        assert len(results) == 1
        assert results["c"]["00000"] == 1000

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
        assert results["c"]["00"] == 1000

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_binary_count_return(self):
        config = CompilerConfig(results_format=QuantumResultsFormat().binary_count())
        hardware = get_default_RTCS_hardware()
        results = execute_qasm(get_qasm2("basic.qasm"), hardware, config)
        assert "c" in results
        assert len(results["c"]) == 4
        assert {"11", "01", "00", "10"} == set(results["c"].keys())

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_mid_circuit_measurement(self):
        qasm_string = get_qasm2("invalid_mid_circuit_measure.qasm")
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
            )  # yapf: disable

    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_basic_single_measures(self):
        qasm_string = get_qasm2("basic_single_measures.qasm")
        results = execute_qasm(qasm_string)

        # We're testing that individual assignments to a classical register get
        # correctly assigned, aka that measuring c[0] then c[1] results in c = [c0, c1].
        assert results["c"]["00"] == 1000

    def test_frontend_peek(self):
        with pytest.raises(ValueError):
            fetch_frontend("")

        qasm2_string = get_qasm2("basic.qasm")
        frontend = fetch_frontend(qasm2_string)
        assert isinstance(frontend, QASMFrontend)

        qasm3_string = get_qasm3("basic.qasm")
        frontend = fetch_frontend(qasm3_string)
        assert isinstance(frontend, QASMFrontend)

    def test_separate_compilation_from_execution(self):
        config = CompilerConfig()
        hardware = get_default_echo_hardware()
        contents = get_qasm2("basic.qasm")
        frontend = fetch_frontend(contents)
        built, _ = frontend.parse(contents, hardware, config)
        results = frontend.execute(built, hardware, config)
        assert results is not None

    def test_qasm_sim(self):
        model = get_default_qiskit_hardware(20)
        qasm = get_qasm2("basic.qasm")
        results = execute(qasm, model, CompilerConfig())
        assert len(results) == 4
        assert results["11"] > 200
        assert results["01"] > 200
        assert results["10"] > 200
        assert results["00"] > 200


class TestParsing:
    echo = get_default_echo_hardware(6)

    def test_compare_tket_parser(self):
        qasm_folder = join(dirname(__file__), "files", "qasm")
        for file in [f for f in listdir(qasm_folder) if isfile(join(qasm_folder, f))]:
            qasm = get_qasm2(file)
            circ = None
            try:
                circ = circuit_from_qasm_str(qasm)
            except:
                pass

            tket_builder: TketBuilder = TketQasmParser().parse(TketBuilder(), qasm)
            if circ is not None:
                assert circ.n_gates <= tket_builder.circuit.n_gates

    def test_invalid_gates(self):
        with pytest.raises(ValueError):
            RestrictedQasm2Parser({"cx"}).parse(
                get_builder(self.echo), get_qasm2("example.qasm")
            )

    def test_example(self):
        builder = parse_and_apply_optimiziations("example.qasm")
        assert 349 == len(builder.instructions)

    def test_parallel(self):
        builder = parse_and_apply_optimiziations("parallel_test.qasm", qubit_count=8)
        assert 2117 == len(builder.instructions)

    def test_example_if(self):
        with pytest.raises(ValueError):
            parse_and_apply_optimiziations("example_if.qasm")

    def test_move_measurements(self):
        # We need quite a few more qubits for this test.
        builder = parse_and_apply_optimiziations(
            "move_measurements.qasm", qubit_count=12
        )
        assert len(builder.instructions) == 97469

    def test_random_n5_d5(self):
        builder = parse_and_apply_optimiziations("random_n5_d5.qasm")
        assert len(builder.instructions) == 4957

    def test_ordered_keys(self):
        builder = parse_and_apply_optimiziations(
            "ordered_cregs.qasm", parser=DefaultQasm2Parser()
        )
        ret_node: Return = builder.instructions[-1]
        assert isinstance(ret_node, Return)
        assert list(ret_node.variables) == ["a", "b", "c"]

    def test_restrict_if(self):
        with pytest.raises(ValueError):
            RestrictedQasm2Parser(disable_if=True).parse(
                get_builder(self.echo), get_qasm2("example_if.qasm")
            )

    def test_invalid_arbitrary_gate(self):
        with pytest.raises(QasmError) as e:
            Qasm2Parser().parse(
                get_builder(self.echo), get_qasm2("invalid_custom_gate.qasm")
            )
            assert e.value == "Invalid gate invocation inside gate definition."

    def test_valid_arbitrary_gate(self):
        parse_and_apply_optimiziations("valid_custom_gate.qasm")

    def test_ecr_intrinsic(self):
        builder = parse_and_apply_optimiziations("ecr.qasm")
        assert any(
            isinstance(inst, CrossResonancePulse) for inst in builder.instructions
        )
        assert len(builder.instructions) == 182

    def test_ecr_already_exists(self):
        Qasm2Parser().parse(get_builder(self.echo), get_qasm2("ecr_exists.qasm"))


class TestQatOptimization:
    def _measure_merge_timings(self, file, qubit_count, keys, expected):
        builder = parse_and_apply_optimiziations(file, qubit_count=qubit_count)
        qat_file = InstructionEmitter().emit(builder.instructions, builder.model)
        timeline = EchoEngine(builder.model).create_duration_timeline(qat_file.instructions)
        model: QuantumHardwareModel = builder.model

        def assert_times_match(res_key, start_end):
            resonator: Resonator = model.get_device(res_key)
            resonator.get_measure_channel()

            acquire_time = (
                (0, 0)
                if (
                    r1_m := next(
                        iter(
                            val
                            for val in timeline[resonator.get_acquire_channel()]
                            if isinstance(val.instruction, Acquire)
                        ),
                        None,
                    )
                )
                is None
                else (r1_m.start, r1_m.end)
            )
            assert acquire_time == start_end

            measure_time = (
                (0, 0)
                if (
                    r1_m := next(
                        iter(
                            val
                            for val in timeline[resonator.get_measure_channel()]
                            if isinstance(val.instruction, MeasurePulse)
                        ),
                        None,
                    )
                )
                is None
                else (r1_m.start, r1_m.end)
            )
            assert measure_time == start_end

        # We check that every measurement fires at the same time.
        for key, start_end in zip(keys, expected):
            assert_times_match(key, start_end)

    def test_measure_merge_example(self):
        self._measure_merge_timings(
            "example.qasm",
            6,
            ("R0", "R1", "R2", "R3", "R4", "R5"),
            (
                (750, 1750),
                (1750, 2750),
                (1750, 2750),
                (1750, 2750),
                (750, 1750),
                (1750, 2750),
            ),
        )

    def test_measure_merge_move_measurements(self):
        self._measure_merge_timings(
            "move_measurements.qasm",
            12,
            ("R5", "R6", "R9"),
            ((579800, 580800), (579800, 580800), (589800, 590800)),
        )
