# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from itertools import permutations
from os import listdir
from os.path import dirname, isfile, join
from typing import List

import networkx as nx
import numpy as np
import pytest
from compiler_config.config import (
    CompilerConfig,
    MetricsType,
    Qasm2Optimizations,
    Qasm3Optimizations,
    QuantumResultsFormat,
    TketOptimizations,
)
from docplex.mp.model import Model
from pytket.qasm import circuit_from_qasm_str
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator as QuantumInstance
from qiskit_algorithms import QAOA, VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (
    ADMMOptimizer,
    ADMMParameters,
    CobylaOptimizer,
    GroverOptimizer,
    MinimumEigenOptimizer,
)
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp

import qat.purr.compiler.experimental.frontends as experimental_frontends
import qat.purr.compiler.frontends as core_frontends
from qat.purr.backends.echo import EchoEngine, get_default_echo_hardware
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.backends.realtime_chip_simulator import (
    get_default_RTCS_hardware,
    qutip_available,
)
from qat.purr.compiler.builders import InstructionBuilder, QuantumInstructionBuilder
from qat.purr.compiler.devices import ChannelType, PulseShapeType, QubitCoupling
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.hardware_models import get_cl2qu_index_mapping
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
from qat.purr.integrations.qasm import (
    CloudQasmParser,
    Qasm2Parser,
    Qasm3Parser,
    RestrictedQasm2Parser,
    get_qasm_parser,
)
from qat.purr.integrations.qiskit import QatBackend
from qat.purr.integrations.tket import TketBuilder, TketQasmParser
from qat.qat import execute, execute_qasm, fetch_frontend

from tests.qat.qasm_utils import (
    ProgramFileType,
    get_qasm2,
    get_qasm3,
    get_test_file_path,
    parse_and_apply_optimiziations,
)
from tests.qat.utils.models import get_jagged_echo_hardware, update_qubit_indices


class TestQASM3:
    """
    Tests for the parsing and instruction generation of OPENQASM3.0.

    In particular these tests currently only operate on a subset of OQ3
    and focus on the OpenPulse side of the language definition.
    """

    def test_named_defcal_arg(self):
        hw = get_default_echo_hardware(8)
        comp = CompilerConfig()
        comp.results_format.binary_count()
        results = execute_qasm(get_qasm3("named_defcal_arg.qasm"), hw, compiler_config=comp)
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

    def test_frequency(self):
        hw = get_default_echo_hardware(8)
        execute_qasm(get_qasm3("openpulse_tests/freq.qasm"), hw)

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

    def test_ghz(self):
        hw = get_default_echo_hardware(4)
        v3_qasm = get_qasm3("ghz.qasm")
        v2_qasm = get_qasm3("ghz_v2.qasm")

        v3_instructions = get_qasm_parser(v3_qasm).parse(get_builder(hw), v3_qasm)
        v2_instructions = get_qasm_parser(v2_qasm).parse(get_builder(hw), v2_qasm)

        assert len(v3_instructions.instructions) == len(v2_instructions.instructions)

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
        assert any(isinstance(inst, CrossResonancePulse) for inst in result.instructions)

    # ("cx", "cnot") is intentional: qir parses cX as cnot, but echo engine does
    # not support cX.
    @pytest.mark.parametrize(
        "test, gate", [("cx", "cnot"), ("cnot", "cnot"), ("ecr", "ECR")]
    )
    def test_override(self, test, gate):
        # Tests overriding gates using openpulse: checks the overridden gate
        # yields the correct pulses, and that the unchanged gates are the same
        # as those created by the circuit builder.
        hw = get_default_echo_hardware(4)
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), get_qasm3(f"{test}_override_test.qasm"))
        qasm_inst = result.instructions
        qasm_inst_names = [str(inst) for inst in qasm_inst]

        # test the extra_soft_square pulses are as expected
        ess_pulses = [
            inst
            for inst in qasm_inst
            if hasattr(inst, "shape") and (inst.shape is PulseShapeType.EXTRA_SOFT_SQUARE)
        ]
        assert len(ess_pulses) == 2
        assert all([len(inst.quantum_targets) == 1 for inst in ess_pulses])
        assert ess_pulses[0].quantum_targets[0].partial_id() == "q1_frame"
        assert ess_pulses[1].quantum_targets[0].partial_id() == "q2_frame"

        # test the ecrs on (0, 1) and (2, 3) are unchanged by the override
        circuit = hw.create_builder()
        func = getattr(circuit, gate)
        func(hw.get_qubit(0), hw.get_qubit(1))
        circ_inst = circuit.instructions
        circ_inst_names = [str(inst) for inst in circ_inst]
        assert qasm_inst_names[0 : len(circ_inst_names)] == circ_inst_names

        circuit = hw.create_builder()
        func = getattr(circuit, gate)
        func(hw.get_qubit(2), hw.get_qubit(3))
        circ_inst = circuit.instructions
        circ_inst_names = [str(inst) for inst in circ_inst]
        assert (
            qasm_inst_names[len(qasm_inst_names) - len(circ_inst_names) :]
            == circ_inst_names
        )

    @pytest.mark.parametrize(
        "params",
        [
            ["pi", "2*pi", "-pi/2", "-7*pi/2", "0", "pi/4"],
            [np.pi, 2 * np.pi, -np.pi / 2, -7 * np.pi / 2, 0.0, np.pi / 4],
            np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=(6)),
            [0.0, 0.0, 0.0, 0.0, "pi/2", 0.0],
            ["2*pi", "-pi/2", 0.0, 0.0, "pi/2", "-2*pi"],
        ],
    )
    def test_u_gate(self, params):
        """
        Tests the validty of the U gate with OpenPulse by checking that the
        parsed circuit matches the same circuit created with the circuit builder.
        """
        hw = get_default_echo_hardware(2)

        # build the circuit from QASM
        qasm = get_qasm3("u_gate.qasm")
        for i in range(6):
            qasm = qasm.replace(f"param{i}", str(params[i]))
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), qasm)
        qasm_inst = result.instructions

        # create the circuit using the circuit builder
        builder = hw.create_builder()
        params = [eval(str(param).replace("pi", "np.pi")) for param in params]
        q1 = hw.get_qubit(0)
        q2 = hw.get_qubit(1)
        (
            builder.Z(q1, params[2])
            .Y(q1, params[0])
            .Z(q1, params[1])
            .Z(q2, params[5])
            .Y(q2, params[3])
            .Z(q2, params[4])
        )
        circ_inst = builder.instructions

        # validate that the circuits match
        assert len(qasm_inst) == len(circ_inst)
        for i in range(len(qasm_inst)):
            assert str(qasm_inst[i]) == str(circ_inst[i])

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
        assert pulses[0].shape is PulseShapeType.GAUSSIAN_ZERO_EDGE
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

    @pytest.mark.parametrize(
        "file_name,test_value",
        (
            ("sum", 5.0),
            ("mix", 6.0),
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
        (("scale", "scale_factor", 42), ("phase_shift", "phase", 4 + 2j)),
    )
    def test_waveform_processing_single_waveform(self, file_name, attribute, test_value):
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

    def test_internal_pulses(self):
        qasm_string = get_qasm3(f"waveform_tests/internal_waveform_tests.qasm")
        hardware = get_default_echo_hardware(8)
        config = CompilerConfig(
            repeats=10,
            results_format=QuantumResultsFormat(),
            optimizations=Qasm3Optimizations(),
        )
        assert execute_qasm(qasm_string, hardware=hardware, compiler_config=config)

    @pytest.mark.parametrize(
        "qasm_file",
        [
            "basic.qasm",
            "ghz.qasm",
        ],
    )
    def test_on_jagged_hardware(self, qasm_file):
        hw = get_jagged_echo_hardware(8)
        qasm_string = get_qasm3(qasm_file)
        parser = Qasm3Parser()
        result = parser.parse(get_builder(hw), qasm_string)
        assert len(result.instructions) > 0

    @pytest.mark.parametrize(
        "qasm_file",
        [
            "named_defcal_arg.qasm",
            "delay.qasm",
            "redefine_defcal.qasm",
            "lark_parsing_test.qasm",
            "arb_waveform.qasm",
            "tmp.qasm",
            "cx_override_test.qasm",
            "ecr_test.qasm",
            "openpulse_tests/acquire.qasm",
            "openpulse_tests/expr_list_defcal_different_arg.qasm",
            "openpulse_tests/set_frequency.qasm",
            "openpulse_tests/freq.qasm",
            "openpulse_tests/constant_wf.qasm",
            "openpulse_tests/detune_gate.qasm",
            "openpulse_tests/zmap.qasm",
            "openpulse_tests/expr_list_caldef_1.qasm",
            "openpulse_tests/expr_list_caldef_2.qasm",
            "openpulse_tests/expr_list_caldef_3.qasm",
            "openpulse_tests/waveform_numerical_types.qasm",
            "openpulse_tests/cross_ressonance.qasm",
            "openpulse_tests/shift_phase.qasm",
            "waveform_tests/waveform_test_scale.qasm",
            "waveform_tests/internal_waveform_tests.qasm",
            "waveform_tests/waveform_test_phase_shift.qasm",
            "waveform_tests/waveform_test_sum.qasm",
        ],
    )
    def test_dollar_on_jagged_hardware(self, qasm_file):
        hw = get_jagged_echo_hardware(8)
        parser = Qasm3Parser()
        qasm_string = get_qasm3(qasm_file)
        with pytest.raises(ValueError):
            parser.parse(get_builder(hw), qasm_string)
        qasm_string = update_qubit_indices(qasm_string, [q.index for q in hw.qubits])
        result = parser.parse(get_builder(hw), qasm_string)
        assert len(result.instructions) > 0

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

    @pytest.mark.parametrize(
        "hw", [get_default_echo_hardware(2), get_default_RTCS_hardware()]
    )
    def test_capture_with_delay(self, hw):
        # Tests that capture v2 in openpulse makes use of the qubit delay for an acquire channel.
        qubit = hw.get_qubit(0)
        parser = Qasm3Parser()
        builder = parser.parse(
            hw.create_builder(), get_qasm3("openpulse_tests/capture.qasm")
        )
        delay = [inst.delay for inst in builder.instructions if isinstance(inst, Acquire)]
        assert delay[0] == qubit.measure_acquire["delay"]

    def test_gaussian_square(self):
        # Checks the that Gaussian Square pulses parse correectly.
        hw = get_default_echo_hardware(2)
        qasm_string = get_qasm3("waveform_tests/gaussian_square.qasm")
        parser = Qasm3Parser()
        builder = parser.parse(get_builder(hw), qasm_string)
        pulses = [inst for inst in builder.instructions if isinstance(inst, Pulse)]
        # Check the properties of the first pulse
        assert np.isclose(pulses[0].width, 100e-9)
        assert np.isclose(pulses[0].amp, 1)
        assert np.isclose(pulses[0].square_width, 50e-9)
        assert pulses[0].zero_at_edges == True
        # Check the properties of the second pulse
        assert np.isclose(pulses[1].width, 200e-9)
        assert np.isclose(pulses[1].amp, 2.5)
        assert np.isclose(pulses[1].square_width, 50e-9)
        assert pulses[1].zero_at_edges == False


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
        builder = parse_and_apply_optimiziations("move_measurements.qasm", qubit_count=12)
        assert 97469 == len(builder.instructions)

    def test_random_n5_d5(self):
        builder = parse_and_apply_optimiziations("random_n5_d5.qasm")
        assert 4957 == len(builder.instructions)

    def test_ordered_keys(self):
        builder = parse_and_apply_optimiziations(
            "ordered_cregs.qasm", parser=CloudQasmParser()
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
        with pytest.raises(ValueError):
            Qasm2Parser().parse(
                get_builder(self.echo), get_qasm2("invalid_custom_gate.qasm")
            )

    def test_valid_arbitrary_gate(self):
        parse_and_apply_optimiziations("valid_custom_gate.qasm")

    def test_ecr_intrinsic(self):
        builder = parse_and_apply_optimiziations("ecr.qasm")
        assert any(isinstance(inst, CrossResonancePulse) for inst in builder.instructions)
        assert 64 == len(builder.instructions)

    def test_ecr_already_exists(self):
        Qasm2Parser().parse(get_builder(self.echo), get_qasm2("ecr_exists.qasm"))

    @pytest.mark.parametrize(
        "qasm_file",
        [
            "basic.qasm",
            "basic_results_formats.qasm",
            "basic_single_measures.qasm",
            "ecr_exists.qasm",
            "ecr.qasm",
            "invalid_mid_circuit_measure.qasm",
            "logic_example.qasm",
            "more_basic.qasm",
            "parallel_test.qasm",
            "primitives.qasm",
            "valid_custom_gate.qasm",
        ],
    )
    def test_on_jagged_hardware(self, qasm_file):
        hw = get_jagged_echo_hardware(8)
        qasm_string = get_qasm2(qasm_file)
        parser = Qasm2Parser()
        result = parser.parse(get_builder(hw), qasm_string)
        assert len(result.instructions) > 0


class TestQatOptimization:
    def _measure_merge_timings(self, file, qubit_count, keys, expected):
        builder = parse_and_apply_optimiziations(file, qubit_count=qubit_count)
        qat_file = InstructionEmitter().emit(builder.instructions, builder.model)
        timeline = EchoEngine(builder.model).create_duration_timeline(qat_file.instructions)

        def get_start_end(key, instruction, channel_type):
            pulse_channel = builder.model.get_pulse_channel_from_device(channel_type, key)
            return (
                (0, 0)
                if (
                    r1_m := next(
                        iter(
                            val
                            for val in timeline[pulse_channel]
                            if isinstance(val.instruction, instruction)
                        ),
                        None,
                    )
                )
                is None
                else (r1_m.start, r1_m.end)
            )

        # We check that every measurement fires at the same time.
        for key, start_end in zip(keys, expected):
            assert get_start_end(key, MeasurePulse, ChannelType.measure) == start_end
            assert get_start_end(key, Acquire, ChannelType.acquire) == start_end

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


class TestQiskitOptimization:
    @pytest.mark.skip("RTCS needs to handle 4 qubits.")
    def test_minimum_eigen_solver(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/03_minimum_eigen_optimizer.html
        qubo = QuadraticProgram()
        qubo.binary_var("x")
        qubo.binary_var("y")
        qubo.binary_var("z")
        qubo.minimize(
            linear=[1, -2, 3], quadratic={("x", "y"): 1, ("x", "z"): -1, ("y", "z"): 2}
        )

        op, offset = qubo.to_ising()
        qp = QuadraticProgram()
        qp.from_ising(op, offset, linear=True)

        algorithm_globals.random_seed = 10598
        quantum_instance = QuantumInstance(
            QatBackend(),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        qaoa_mes = QAOA(quantum_instance=quantum_instance, initial_point=[0.0, 0.0])
        exact_mes = NumPyMinimumEigensolver()

        qaoa = MinimumEigenOptimizer(qaoa_mes)  # using QAOA
        exact = MinimumEigenOptimizer(
            exact_mes
        )  # using the exact classical numpy minimum eigen solver

        exact_result = exact.solve(qubo)
        qaoa_result = qaoa.solve(qubo)

        assert exact_result == qaoa_result

    @pytest.mark.skip("RTCS needs to handle 10 qubits.")
    def test_grover_adaptive_search(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/04_grover_optimizer.html
        backend = QatBackend()
        model = Model()
        x0 = model.binary_var(name="x0")
        x1 = model.binary_var(name="x1")
        x2 = model.binary_var(name="x2")
        model.minimize(-x0 + 2 * x1 - 3 * x2 - 2 * x0 * x2 - 1 * x1 * x2)
        qp = from_docplex_mp(model)

        grover_optimizer = GroverOptimizer(6, num_iterations=10, quantum_instance=backend)
        results = grover_optimizer.solve(qp)

        exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        exact_result = exact_solver.solve(qp)

        # TODO: Assert these results are what we should check against.
        self.assertEqual(results.x, exact_result.x)
        self.assertEqual(results.fval, exact_result.fval)

    @pytest.mark.skip("RTCS needs to handle 3 qubits.")
    def test_ADMM(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/05_admm_optimizer.html
        mdl = Model("ex6")

        v = mdl.binary_var(name="v")
        w = mdl.binary_var(name="w")
        t = mdl.binary_var(name="t")
        u = mdl.continuous_var(name="u")

        mdl.minimize(v + w + t + 5 * (u - 2) ** 2)
        mdl.add_constraint(v + 2 * w + t + u <= 3, "cons1")
        mdl.add_constraint(v + w + t >= 1, "cons2")
        mdl.add_constraint(v + w == 1, "cons3")

        # load quadratic program from docplex model
        qp = from_docplex_mp(mdl)

        admm_params = ADMMParameters(
            rho_initial=1001,
            beta=1000,
            factor_c=900,
            maxiter=100,
            three_block=True,
            tol=1.0e-6,
        )

        # define COBYLA optimizer to handle convex continuous problems.
        cobyla = CobylaOptimizer()

        # initialize ADMM with classical QUBO and convex optimizer
        exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())  # to solve QUBOs
        admm = ADMMOptimizer(
            params=admm_params, qubo_optimizer=exact, continuous_optimizer=cobyla
        )

        # run ADMM to solve problem
        result = admm.solve(qp)

        # initialize ADMM with quantum QUBO optimizer and classical convex optimizer
        qaoa = MinimumEigenOptimizer(QAOA(quantum_instance=QatBackend()))
        admm_q = ADMMOptimizer(
            params=admm_params, qubo_optimizer=qaoa, continuous_optimizer=cobyla
        )

        # run ADMM to solve problem
        result_q = admm_q.solve(qp)

        self.assertEqual(result_q.x, result.x)
        self.assertEqual(result_q.fval, result.fval)

    @pytest.mark.skip("RTCS needs to handle 4 qubits.")
    def test_max_cut(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/06_examples_max_cut_and_tsp.html
        n = 4  # Number of nodes in graph
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, n, 1))
        elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]

        # tuple is (i,j,weight) where (i,j) is the edge
        G.add_weighted_edges_from(elist)

        w = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                temp = G.get_edge_data(i, j, default=0)
                if temp != 0:
                    w[i, j] = temp["weight"]

        best_cost_brute = 0
        for b in range(2**n):
            x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
            cost = 0
            for i in range(n):
                for j in range(n):
                    cost = cost + w[i, j] * x[i] * (1 - x[j])
            if best_cost_brute < cost:
                best_cost_brute = cost
                xbest_brute = x

        colors = ["r" if xbest_brute[i] == 0 else "c" for i in range(n)]

        max_cut = Maxcut(w)
        qp = max_cut.to_quadratic_program()

        qubitOp, offset = qp.to_ising()
        exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        result = exact.solve(qp)

        # Making the Hamiltonian in its full form and getting the lowest eigenvalue and
        # eigenvector
        ee = NumPyMinimumEigensolver()
        result = ee.compute_minimum_eigenvalue(qubitOp)

        x = max_cut.sample_most_likely(result.eigenstate)

        seed = 10598
        quantum_instance = QuantumInstance(
            QatBackend(), seed_simulator=seed, seed_transpiler=seed
        )

        # construct VQE
        spsa = SPSA(maxiter=300)
        ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
        vqe = VQE(ry, optimizer=spsa, quantum_instance=quantum_instance)

        # run VQE
        result = vqe.compute_minimum_eigenvalue(qubitOp)

        # print results
        x = max_cut.sample_most_likely(result.eigenstate)

        vqe_optimizer = MinimumEigenOptimizer(vqe)

        # solve quadratic program
        result = vqe_optimizer.solve(qp)

        # TODO: Work out what to assert.

    @pytest.mark.skip("RTCS needs to handle 9 qubits.")
    def test_travelling_salesman(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/06_examples_max_cut_and_tsp.html
        # Generating a graph of 3 nodes
        n = 3
        # num_qubits = n**2
        tsp = Tsp.create_random_instance(n, seed=123)
        adj_matrix = nx.to_numpy_matrix(tsp.graph)

        # Plotting args
        # colors = ["r" for node in tsp.graph.nodes]
        # pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]

        def brute_force_tsp(w, N):
            a = list(permutations(range(1, N)))
            last_best_distance = 1e10
            for i in a:
                distance = 0
                pre_j = 0
                for j in i:
                    distance = distance + w[j, pre_j]
                    pre_j = j
                distance = distance + w[pre_j, 0]
                order = (0,) + i
                if distance < last_best_distance:
                    best_order = order
                    last_best_distance = distance
            return last_best_distance, best_order

        best_distance, best_order = brute_force_tsp(adj_matrix, n)

        qp = tsp.to_quadratic_program()
        ee = NumPyMinimumEigensolver()
        # exact = MinimumEigenOptimizer(ee)
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        qubitOp, offset = qubo.to_ising()
        # exact_result = exact.solve(qubo)

        # Making the Hamiltonian in its full form and getting the lowest eigenvalue and
        # eigenvector
        result = ee.compute_minimum_eigenvalue(qubitOp)

        x = tsp.sample_most_likely(result.eigenstate)
        # z = tsp.interpret(x)

        algorithm_globals.random_seed = 123
        seed = 10598
        backend = QatBackend()
        quantum_instance = QuantumInstance(
            backend, seed_simulator=seed, seed_transpiler=seed
        )
        spsa = SPSA(maxiter=300)
        ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
        vqe = VQE(ry, optimizer=spsa, quantum_instance=quantum_instance)

        result = vqe.compute_minimum_eigenvalue(qubitOp)

        x = tsp.sample_most_likely(result.eigenstate)
        tsp.interpret(x)

        vqe_optimizer = MinimumEigenOptimizer(vqe)

        # solve quadratic program
        result = vqe_optimizer.solve(qp)
        # z = tsp.interpret(x)


class TestQiskitBackend:
    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_simple_circuit(self):
        backend = QatBackend()
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.measure_all()
        result = backend.run(circuit, shots=1000).result()
        counts = result.get_counts()
        assert counts["1"] > 900


mapping_setup1 = (
    """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg b[2];
        measure q[0] -> b[1];
        measure q[1] -> b[0];
        """,
    {"0": 1, "1": 0},
)
mapping_setup2 = (
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg b[3];
    measure q[0] -> b[2];
    measure q[1] -> b[1];
    measure q[2] -> b[0];
    """,
    {"0": 2, "1": 1, "2": 0},
)
mapping_setup3 = (
    """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg b[3];
    measure q -> b;
    """,
    {"0": 0, "1": 1, "2": 2},
)


@pytest.mark.parametrize(
    "qasm_string, expected_mapping", [mapping_setup1, mapping_setup2, mapping_setup3]
)
def test_cl2qu_index_mapping(qasm_string, expected_mapping):
    hw = get_default_echo_hardware(3)
    parser = Qasm2Parser()
    result = parser.parse(get_builder(hw), qasm_string)
    mapping = get_cl2qu_index_mapping(result.instructions, hw)
    assert mapping == expected_mapping

    blob = result.serialize()
    result2 = InstructionBuilder.deserialize(blob)
    mapping = get_cl2qu_index_mapping(result2.instructions, hw)
    assert mapping == expected_mapping
