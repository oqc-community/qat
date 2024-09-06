# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from dataclasses import dataclass, field
from enum import Enum, auto
from random import random
from typing import Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from qat.purr.compiler.devices import (
    Calibratable,
    ChannelType,
    PhysicalBaseband,
    PhysicalChannel,
    QuantumComponent,
    Qubit,
    Resonator,
)
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.instructions import (
    AcquireMode,
    Instruction,
    PostProcessing,
    Repeat,
    Reset,
)
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.utils.logger import get_default_logger

from ..compiler.execution import QuantumExecutionEngine, SweepIterator
from ..compiler.hardware_models import QuantumHardwareModel
from ..utils.logging_utils import log_duration
from .utilities import UPCONVERT_SIGN, PositionData, get_axis_map

logger = get_default_logger()

try:
    from qutip import Qobj, basis, create, destroy, mesolve, qeye, tensor

    qutip_available = True
except ModuleNotFoundError:
    qutip_available = False


class ControlType(Enum):
    MEASURE = "measure"
    RESET = "reset"

    def __repr__(self):
        return self.name


@dataclass
class OperatorInfo:
    operator: "Qobj"
    name: str
    qubit_id: int = None

    def __repr__(self):
        return f"Operator({self.name}, {self.qubit_id})"


@dataclass
class Section:
    indices: List[int]
    qubits: List[str]
    control_type: ControlType
    simulation_time: List[float] = None
    drive_hamiltonian: List["Qobj"] = field(default_factory=list)

    def __repr__(self):
        return f"Section({self.indices}, {self.qubits}, {str(self.control_type)})"


class MeasurementStatistics:
    def __init__(self, state, level=0, probability=1, qubits_measured=None, result=None):
        self.state: "Qobj" = state
        self.level: int = level
        self.probability: float = probability
        self.result: str = result
        self.qubits_measured: List[int] = qubits_measured
        self.measurement_statistics: List[MeasurementStatistics] = []
        self.branch_number: Union[int, None] = None
        self.dynamics_results = None
        self.hamiltonian: Union[List["Qobj", float], None] = None
        self.c_ops: Union[List["Qobj"], None] = None
        self.sim_t: Union[List[float], None] = None

    def build_tree(
        self,
        H0,
        c_ops,
        simulation_sections,
        qubit_states,
        reset_operators,
        options=None,
        branch_number=-1,
    ):
        """
        Simulate the dynamics of initial condition using the section of the Hamiltonian
        specified by the level. Perform a measurement on the specified qubits in the
        Hamiltonian section and for each result calculate the probability of it occuring
        and the resultant state. For each result perform a new simulation with the next
        Hamiltonian section to create a branching structure of all possible outcomes for
        a given initial condition and Hamiltonian sections dictionary. Continue until
        all Hamiltonian sections have been simulated.
        """
        if self.level == len(simulation_sections) or self.probability.real <= 0:
            branch_number += 1
            self.branch_number = branch_number
            return branch_number

        section = simulation_sections[self.level]
        control_type = section.control_type
        qubit_indices = section.qubits
        drive_hamiltonian = section.drive_hamiltonian
        sim_t = section.simulation_time

        H = [H0]
        H.extend(drive_hamiltonian)

        if len(drive_hamiltonian) == 0:
            self.dynamics_results = [self.state]
            final_state = self.state
        else:
            result = mesolve(H, self.state, sim_t, c_ops=c_ops, options=options)
            self.dynamics_results = result.states
            final_state = result.states[-1]
        self.hamiltonian = H
        self.c_ops = c_ops
        self.sim_t = sim_t

        if control_type == ControlType.MEASURE:
            empty_state = H0 - H0
            qubit_indices.sort()
            sub_space_size = len(qubit_indices)
            # Create list of sub states from qubits to be measured
            measurement_operators = {
                format(out_state, f"0{sub_space_size}b"): empty_state
                for out_state in range(2**sub_space_size)
            }

            # Calculate the projection operators for the qubits being measured
            for key, state in qubit_states.items():
                sub_state = "".join([key[index] for index in qubit_indices])
                measurement_operators[sub_state] += state

            for result, operator in measurement_operators.items():
                probability = (operator * final_state).tr()
                self.measurement_statistics.append(
                    MeasurementStatistics(
                        state=(
                            operator * final_state * operator.dag() / probability
                            if probability.real > 0
                            else empty_state
                        ),
                        level=self.level + 1,
                        probability=probability if probability.real > 0 else 0,
                        qubits_measured=qubit_indices,
                        result=result,
                    )
                )

        elif control_type == ControlType.RESET:
            for idx in qubit_indices:
                final_state = (
                    reset_operators[0][idx] * final_state * reset_operators[0][idx].dag()
                    + reset_operators[1][idx] * final_state * reset_operators[1][idx].dag()
                )

            self.measurement_statistics.append(
                MeasurementStatistics(
                    state=final_state,
                    level=self.level + 1,
                    probability=1,
                    qubits_measured=qubit_indices,
                    result=ControlType.RESET,
                )
            )

        for branch in self.measurement_statistics:
            branch_number = branch.build_tree(
                H0,
                c_ops,
                simulation_sections,
                qubit_states,
                reset_operators,
                options,
                branch_number,
            )

        return branch_number

    def extract_outcome_probabilities(self):
        def add_result(result_dict=None):
            if result_dict is None:
                result_dict = {}

            for i, qubit in enumerate(self.qubits_measured):
                result_list = result_dict.setdefault(qubit, [])
                result_list.insert(
                    0,
                    (
                        self.result
                        if self.result == ControlType.RESET
                        else int(self.result[i])
                    ),
                )

            return result_dict

        section = self.level - 1
        outcome_probabilities = {}
        if self.branch_number is not None:
            outcome_probabilities[self.branch_number] = [self.probability, add_result()]
        else:
            for measurement_statistic in self.measurement_statistics:
                outcomes_dict = measurement_statistic.extract_outcome_probabilities()
                if section > -1:
                    for branch_number, outcome in outcomes_dict.items():
                        outcome[0] *= self.probability
                        add_result(outcome[1])
                        outcome_probabilities[branch_number] = outcome
                else:
                    outcome_probabilities.update(outcomes_dict)

        return outcome_probabilities

    def get_branch_nodes(self):
        branch_nodes = {}
        if self.branch_number is not None:
            branch_nodes[self.branch_number] = []
        else:
            for i, measurement_statistic in enumerate(self.measurement_statistics):
                branch_dict = measurement_statistic.get_branch_nodes()
                for branch_number, nodes in branch_dict.items():
                    nodes.insert(0, i)
                    branch_nodes[branch_number] = nodes

        return branch_nodes

    def extract_branch_trajectory(
        self, operators: List["Qobj"], branch: int, step: int = 1, plot_end=False
    ):
        section_dynamics = [[] for i in range(len(operators))]
        sim_t = [[] for i in range(len(operators))]
        dynamics = self.dynamics_results
        measurement_statistic = self
        with log_duration("Branch trajectory extracted in {} seconds."):
            for i in self.get_branch_nodes()[branch]:
                for j in range(len(operators)):
                    section_dynamics[j].extend(
                        [
                            (operators[j] * dynamics[s]).tr()
                            for s in range(0, len(dynamics), step)
                        ]
                    )
                    if measurement_statistic.sim_t is not None:
                        sim_t[j].extend(measurement_statistic.sim_t[::step])
                    else:
                        # If the step doesn't have a time duration due to being a
                        # measurement at time zero or a reset then duplicate the
                        # previous time
                        sim_t[j].extend([0] if len(sim_t[j]) == 0 else [sim_t[j][-1]])
                measurement_statistic = measurement_statistic.measurement_statistics[i]
                if measurement_statistic.branch_number is None:
                    dynamics = measurement_statistic.dynamics_results
                elif plot_end:
                    for j in range(len(operators)):
                        # Append result of final measurement
                        section_dynamics[j].append(
                            (operators[j] * measurement_statistic.state).tr()
                        )
                        sim_t[j].append(sim_t[j][-1])

        return section_dynamics, sim_t


class RTCSQubit(Qubit):
    """Subclass to allow simulation-specific modelling of a qubit."""

    def __init__(
        self,
        index: int,
        resonator: Resonator,
        physical_channel: PhysicalChannel,
        frequency,
        *args,
        N=3,
        anharmonicity=None,
        decay_rate=0.0,
        absorb_rate=0.0,
        dephase_rate=0.0,
        rotating_frame_frequency=0.0,
        **kwargs,
    ):
        super().__init__(index, resonator, physical_channel, *args, **kwargs)
        self._N = 2
        self._rotating_frame_frequency = 0

        self.N = N  # number of qubit energy levels to be simulated
        self.frequency = frequency
        self.rotating_frame_frequency = rotating_frame_frequency
        self.anharmonicity = anharmonicity
        self.decay_rate = decay_rate
        self.absorb_rate = absorb_rate
        self.dephase_rate = dephase_rate
        self._set_qutip_operators()

    def _set_qutip_operators(self):
        # Qutip values / operators
        self.I = qeye(self.N)  # identity
        self.a = destroy(self.N)  # destruction
        self.qutip_N = create(self.N) * destroy(self.N)  # population number
        # Use Pauli if 2 energy levels
        if self.N == 2:
            self.Z = self.I - 2 * self.qutip_N
            self.H0 = -np.pi * self.frequency * self.Z
        else:
            # unitary evolution
            self.H0 = (
                2.0
                * np.pi
                * (self.frequency + 0.5 * (self.qutip_N - 1) * self.anharmonicity)
                * self.qutip_N
            )
        self.cops = [
            self.decay_rate**0.5 * self.a,
            self.absorb_rate**0.5 * self.a.dag(),
            self.dephase_rate**0.5 * self.qutip_N,
        ]  # collapse operators
        self.rho0 = basis(self.N, 0) * basis(self.N, 0).dag()  # Initial state

    @property
    def N(self):
        return self._N

    @property
    def rotating_frame_frequency(self):
        return self._rotating_frame_frequency

    @N.setter
    def N(self, value):
        if self.rotating_frame_frequency != 0 and value != 2:
            raise NotImplementedError(
                "Qubits with a non-zero rotating frame frequency are only implemented "
                "for qubits with 2 energy levels."
            )
        if value < 2:
            raise ValueError("N must be greater than 1")
        self._N = value

    @rotating_frame_frequency.setter
    def rotating_frame_frequency(self, value):
        if self.N != 2 and value != 0:
            raise NotImplementedError(
                "Qubits with more than 2 energy levels are only implemented for a zero "
                "rotating frame frequency"
            )
        self._rotating_frame_frequency = value

    def __getstate__(self):
        dictionary_basis = super().__getstate__()

        del dictionary_basis["I"]
        del dictionary_basis["a"]
        del dictionary_basis["qutip_N"]
        del dictionary_basis["Z"]
        del dictionary_basis["H0"]
        del dictionary_basis["cops"]
        del dictionary_basis["rho0"]

        return dictionary_basis

    def __setstate__(self, state):
        for key, value in state.items():
            self.__dict__[key] = value
        self._set_qutip_operators()


class RTCSResonator(Resonator):
    def __init__(
        self,
        index,
        physical_channel: PhysicalChannel,
        frequency,
        width,
        id_=None,
        *args,
        **kwargs,
    ):
        super().__init__(id_ or f"R{index}", physical_channel, *args, **kwargs)
        self.index = index
        self.frequency = frequency
        # width of lorentzian amplitude about resonator resonant frequency
        self.width = width


class CouplingType(Enum):
    CrossKerr = auto()  # Coupling between a resonator and a qubit
    Exchange = auto()  # Coupling between 2 qubits

    def __repr__(self):
        return self.name


class RTCSCoupling(QuantumComponent, Calibratable):
    """
    Resonators/Qubits are coupled in a very particular way in this model. This object
    holds information about those couplings.
    """

    def __init__(
        self,
        id_: str,
        device_one: QuantumComponent,
        device_two: QuantumComponent,
        frequency,
        coupling_type: CouplingType,
        *args,
        **kwargs,
    ):
        super().__init__(id_, *args, **kwargs)

        self.device_one = device_one
        self.device_two = device_two
        self.coupling_type: CouplingType = coupling_type
        self.frequency = frequency

    def targets(self):
        return [self.device_one, self.device_two]


def _get_highest_id(id_list, prefix, start_id=0):
    highest_id = start_id
    for id in id_list:
        if id.startswith(prefix):
            try:
                highest_id = (
                    int(id[len(prefix) :])
                    if int(id[len(prefix) :]) > highest_id
                    else highest_id
                )
            except:
                pass

    return highest_id


def add_qubit_stack(hw, frequency: float, anharmonicity: float, N: int):
    highest_channel_id = _get_highest_id(list(hw.physical_channels.keys()), "CH")
    highest_baseband_id = _get_highest_id(list(hw.basebands.keys()), "L")

    qubit_channel = f"CH{highest_channel_id+1}"
    qubit_baseband = f"L{highest_baseband_id+1}"
    resonator_channel = f"CH{highest_channel_id+2}"
    resonator_baseband = f"L{highest_baseband_id+2}"
    bb1 = PhysicalBaseband(qubit_baseband, frequency)
    bb2 = PhysicalBaseband(resonator_baseband, 8.0e9)
    hw.add_physical_baseband(bb1, bb2)
    ch1 = PhysicalChannel(qubit_channel, 0.5e-9, bb1, 1)
    ch2 = PhysicalChannel(resonator_channel, 1.0e-9, bb2, 1)
    hw.add_physical_channel(ch1, ch2)

    highest_resonator_id = _get_highest_id(list(hw.quantum_devices.keys()), "R", -1)
    # Currently the frequency of the resonator isn't used in the simulations so we can
    # have the same frequency for all of them without issue
    r = RTCSResonator(highest_resonator_id + 1, ch2, 8.0e9, 1.0e6)
    r.create_pulse_channel(ChannelType.measure, frequency=8.0e9)
    r.create_pulse_channel(ChannelType.acquire, frequency=8.0e9)

    highest_qubit_id = _get_highest_id(list(hw.quantum_devices.keys()), "Q", -1)
    q = RTCSQubit(
        highest_qubit_id + 1,
        r,
        ch1,
        frequency=frequency,
        anharmonicity=anharmonicity,
        N=N,
    )
    q.create_pulse_channel(ChannelType.drive, frequency=frequency)
    q.create_pulse_channel(ChannelType.second_state, frequency=frequency + anharmonicity)

    q_r_coupling = RTCSCoupling(
        f"R{highest_resonator_id+1}<->Q{highest_qubit_id+1}",
        r,
        q,
        10.0e6,
        CouplingType.CrossKerr,
    )

    hw.add_quantum_device(q, r)
    hw.add_couplings(q_r_coupling)

    return q


def add_qubit_coupling(hw, qubit1, qubit2, frequency):
    couple_id = f"{qubit1.id}<->{qubit2.id}"
    q1_q2_coupling = RTCSCoupling(
        couple_id, qubit1, qubit2, frequency, CouplingType.Exchange
    )
    hw.add_couplings(q1_q2_coupling)

    qubit2.add_coupled_qubit(qubit1)
    qubit1.add_coupled_qubit(qubit2)

    q1drive = qubit1.get_pulse_channel(ChannelType.drive)
    q2drive = qubit2.get_pulse_channel(ChannelType.drive)

    qubit1.create_pulse_channel(
        auxiliary_devices=[qubit2],
        channel_type=ChannelType.cross_resonance,
        frequency=q2drive.frequency,
    )

    qubit2.create_pulse_channel(
        auxiliary_devices=[qubit1],
        channel_type=ChannelType.cross_resonance,
        frequency=q1drive.frequency,
    )

    qubit1.create_pulse_channel(
        auxiliary_devices=[qubit2],
        channel_type=ChannelType.cross_resonance_cancellation,
        frequency=q1drive.frequency,
    )

    qubit2.create_pulse_channel(
        auxiliary_devices=[qubit1],
        channel_type=ChannelType.cross_resonance_cancellation,
        frequency=q2drive.frequency,
    )


def apply_setup_to_hardware(hw, rotating_frame=True):
    """Apply the default real-time chip sim hardware setup to the passed-in hardware."""
    bb1 = PhysicalBaseband("L1", 5.0e9)
    bb2 = PhysicalBaseband("L2", 8.0e9)
    bb3 = PhysicalBaseband("L3", 5.09e9)
    bb4 = PhysicalBaseband("L4", 8.5e9)
    hw.add_physical_baseband(bb1, bb2, bb3, bb4)

    ch1 = PhysicalChannel("CH1", 0.5e-9, bb1, 1)
    ch2 = PhysicalChannel("CH2", 1.0e-9, bb2, 1, acquire_allowed=True)
    ch3 = PhysicalChannel("CH3", 0.5e-9, bb3, 1)
    ch4 = PhysicalChannel("CH4", 1.0e-9, bb4, 1, acquire_allowed=True)
    hw.add_physical_channel(ch1, ch2, ch3, ch4)

    r0 = RTCSResonator(0, ch2, 8.0e9, 1.0e6)
    r0.create_pulse_channel(ChannelType.measure, frequency=8.0e9)
    r0.create_pulse_channel(ChannelType.acquire, frequency=8.0e9)

    q0_freq = 5.0e9
    q0 = RTCSQubit(
        0,
        r0,
        ch1,
        q0_freq,
        anharmonicity=-250.0e6,
        N=2 if rotating_frame else 3,
        rotating_frame_frequency=q0_freq if rotating_frame else 0,
    )
    q0.create_pulse_channel(ChannelType.drive, frequency=5.0e9, scale=1.0 + 0.0j)
    q0.create_pulse_channel(ChannelType.second_state, frequency=5.00e9 - 250.0e6)
    q0.mean_z_map_args = [2.136329226009086, -1.0254571168804927]
    q0.pulse_hw_x_pi_2["amp"] = 4368000.007595086

    r1 = RTCSResonator(1, ch4, 8.5e9, 1.0e6)
    r1.create_pulse_channel(ChannelType.measure, frequency=8.5e9)
    r1.create_pulse_channel(ChannelType.acquire, frequency=8.5e9)

    q1_freq = 5.09e9
    q1 = RTCSQubit(
        1,
        r1,
        ch3,
        q1_freq,
        anharmonicity=-250.0e6,
        N=2 if rotating_frame else 3,
        rotating_frame_frequency=q1_freq if rotating_frame else 0,
    )
    q1.create_pulse_channel(ChannelType.drive, frequency=5.09e9, scale=1.0 + 0.0j)
    q1.create_pulse_channel(ChannelType.second_state, frequency=5.09e9 - 250e6)
    q1.mean_z_map_args = [2.1330969344627744, -1.0223925761549326]
    q1.pulse_hw_x_pi_2["amp"] = 4368197.314536925

    q1.add_coupled_qubit(q0)
    q0.add_coupled_qubit(q1)
    q0_r0_coupling = RTCSCoupling("r0<->q0", r0, q0, 10.0e6, CouplingType.CrossKerr)
    q0_q1_coupling = RTCSCoupling("q0<->q1", q0, q1, 4.5e6, CouplingType.Exchange)
    q1_r1_coupling = RTCSCoupling("q1<->r1", q1, r1, 10.0e6, CouplingType.CrossKerr)
    hw.add_couplings(q0_r0_coupling, q0_q1_coupling, q1_r1_coupling)

    q1drive = q1.get_pulse_channel(ChannelType.drive)
    q0drive = q0.get_pulse_channel(ChannelType.drive)

    q0.create_pulse_channel(
        auxiliary_devices=[q1],
        channel_type=ChannelType.cross_resonance,
        frequency=q1drive.frequency,
        scale=20.43888303393216 + 1.9018783700527548j,
    )

    q1.create_pulse_channel(
        auxiliary_devices=[q0],
        channel_type=ChannelType.cross_resonance,
        frequency=q0drive.frequency,
        scale=-19.960910767291143 + 1.7755801175916415j,
    )

    q1.create_pulse_channel(
        auxiliary_devices=[q0],
        channel_type=ChannelType.cross_resonance_cancellation,
        frequency=q1drive.frequency,
        scale=-0.0006383792131791451 - 0.0018484650690712681j,
    )

    q0.create_pulse_channel(
        auxiliary_devices=[q1],
        channel_type=ChannelType.cross_resonance_cancellation,
        frequency=q0drive.frequency,
        scale=-0.002608579454171639 + 0.006639464852534781j,
    )

    hw.add_quantum_device(q0, r0, q1, r1)

    # Simulators don't need automatic calibration.
    hw.is_calibrated = True
    return hw


# noinspection PyPep8Naming
def get_default_RTCS_hardware(repeats=1000, rotating_frame=True):
    model = apply_setup_to_hardware(
        RealtimeSimHardwareModel(repeat_count=repeats), rotating_frame=rotating_frame
    )
    return model


class RealtimeSimHardwareModel(QuantumHardwareModel):
    couplings: List[RTCSCoupling] = []

    def create_engine(self):
        return RealtimeChipSimEngine(self)

    def add_couplings(self, *args):
        self.couplings.extend(list(args))


class RealtimeChipSimEngine(QuantumExecutionEngine):
    """
    Simulation that is built to be as close to our hardware as possible to allow for
    developers, quantum engineers and the fabrication team to be able to run simulations
    and be confident that the results are close to reality.
    """

    model: RealtimeSimHardwareModel

    def __init__(self, model=None, auto_plot=False, sim_qubit_dt=0.25e-10):
        super().__init__(model)
        if not qutip_available:
            raise RuntimeError("qutip unavailable")

        self.auto_plot = auto_plot
        self.sim_qubit_dt = sim_qubit_dt
        self.measurement_statistics: Optional[MeasurementStatistics] = None
        self.sim_t: Optional[List[int]] = None
        self.channel_pulse_data: Optional[dict] = None

    def process_reset(self, position: PositionData):
        """
        When the superclass process_reset is implemented, it should remain empty for the
        simulator.
        """
        pass

    def build_simulator_resets(self, position_map: Dict[str, List[PositionData]]):
        """
        Qubit resets are handled in a unqiue way for the simulator so require their own
        function.
        """
        resets = {}
        for pulse_channel_id, positions in position_map.items():
            for pos in positions:
                if isinstance(pos.instruction, Reset):
                    reset_indices = resets.setdefault(pulse_channel_id, [])
                    reset_indices.append(pos.start)

        return resets

    def optimize(self, instructions: List[Instruction]):
        with log_duration("Instructions optimized in {} seconds."):
            instructions = super().optimize(instructions)
            if not any(inst for inst in instructions if isinstance(inst, Repeat)):
                instructions.append(Repeat(self.model.default_repeat_count))

            for instruction in instructions:
                if isinstance(instruction, PostProcessing):
                    acq = instruction.acquire
                    if acq.mode == AcquireMode.INTEGRATOR:
                        acq.mode = self.model.default_acquire_mode

            return instructions

    def _execute_on_hardware(
        self,
        sweep_iterator: SweepIterator,
        package: QatFile,
        interrupt: Interrupt = NullInterrupt(),
    ) -> Dict[str, np.ndarray]:
        """
        Derivation of the mathematics behind this simulation can be found in the docs
        folder, "Realtime chip simulator mathematical derivation.pdf". Emulate the
        effects of the firmware and quantum hardware for a given input. Before
        instructions are passed to execute they are optimised for this specific function
        so that all measurements occur consecutively and never at the same time across
        all channels.

        Before a measurement is made all quantum channels executing instructions up to
        the measurement must have finished operating and only one one measurement
        channel executes at time. This is achieved by synchronising a measurement
        channel with qubit channels before measurement. This may not be how the hardware
        will handle measurements as it is an inefficent use of time but it makes
        splicing up the simulation into sections between measurements easier for the
        emulator.
        """
        results = {}
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)
            logger.debug(f"Starting sweep #{sweep_iterator.accumulated_sweep_iteration}")

            metadata = {"sweep_iteration": sweep_iterator.get_current_sweep_iteration()}
            interrupt.if_triggered(metadata, throw=True)

            position_map = self.create_duration_timeline(package.instructions)
            pulse_channel_buffers = self.build_pulse_channel_buffers(position_map, True)
            resets = self.build_simulator_resets(position_map)
            buffers = self.build_physical_channel_buffers(pulse_channel_buffers)
            aq_map = self.build_acquire_list(position_map)

            repeats = package.repeat.repeat_count

            sim_length = max(
                [
                    pc.sample_time * len(buffers[pc.id])
                    for pc in self.model.physical_channels.values()
                ]
            )

            # Get simulation time
            self.sim_t = np.arange(0.0, sim_length, self.sim_qubit_dt)

            response_buffers = {}
            resonator_buffers = {}
            for resonator in self.model.resonators:
                measure_channel = resonator.get_measure_channel()
                resonator_buffers[measure_channel.physical_channel_id] = buffers[
                    measure_channel.physical_channel_id
                ]
                response_buffers[measure_channel.physical_channel_id] = np.zeros(
                    shape=[
                        int(repeats),
                        *buffers[measure_channel.physical_channel_id].shape,
                    ],
                    dtype=buffers[measure_channel.physical_channel_id].dtype,
                )

            if all(len(buf) == 0 for buf in resonator_buffers.values()):
                raise NotImplementedError(
                    "There must be at least one qubit measurement to perform a "
                    "simulation"
                )

            # Get resonator channel time steps
            min_resonator_dt = 1.0
            for buffer_idx in resonator_buffers.keys():
                physical_channel = self.model.get_physical_channel(buffer_idx)
                resonator_dt = physical_channel.sample_time
                if min_resonator_dt > resonator_dt:
                    min_resonator_dt = resonator_dt

            # Conversion between resonator and qubit channel time steps
            # TODO: replace with something more stable (interpolation via cubicspline?)
            step = int(min_resonator_dt / self.sim_qubit_dt + 0.5)

            # Convert qubit indices to resonator indices for consistency and change
            # pulse channel to physical channel
            for key in list(resets.keys()):
                item = resets.pop(key)
                pulse_channel = self.model.get_pulse_channel_from_id(key)
                qubit, _ = self.model._resolve_qb_pulse_channel(pulse_channel)
                measure_channel = qubit.get_measure_channel()
                resets[measure_channel.id] = [
                    int(reset * pulse_channel.sample_time / measure_channel.sample_time)
                    for reset in item
                ]

            # Split resonator buffers into segments, one for each measurement channel
            buffer_segments = get_resonator_response_segments(resonator_buffers, resets)

            # find resonator qubit couplings
            target_devices = {}
            lorentzian_responses = {}
            qubit_states = {}
            qutip_size = len(self.model.qubits)
            for buffer_idx, buffer_segment in buffer_segments.items():
                target_device = next(
                    (
                        qb
                        for qb in self.model.qubits
                        if qb.get_measure_channel().physical_channel_id == buffer_idx
                    ),
                    None,
                )
                if target_device is None:
                    continue

                resonator: RTCSResonator = target_device.measure_device
                res_qubit_coupling = next(
                    (
                        coup
                        for coup in self.model.couplings
                        if target_device in coup.targets() and resonator in coup.targets()
                    ),
                    None,
                )
                if res_qubit_coupling is None:
                    continue

                phys_channel = self.model.get_physical_channel(buffer_idx)

                target_devices[buffer_idx] = target_device
                lo = phys_channel.baseband.frequency
                lorentzian_responses[buffer_idx] = {}
                # Generate expected resonator response for qubit in ground and excited
                # state.
                for i, segment in enumerate(buffer_segment):
                    if segment[0] == ControlType.MEASURE:
                        lorentzian_responses[buffer_idx][i] = (
                            get_resonator_response_signal_segment(
                                resonator_input=resonator_buffers[buffer_idx][
                                    segment[1] : segment[2]
                                ],
                                resonator_dt=phys_channel.sample_time,
                                width=resonator.width,
                                res_freq=resonator.frequency,
                                shift=res_qubit_coupling.frequency,
                                lo_freq=lo,
                            )
                        )

            # Generate projection operators for each possible system state
            for out_state in range(2**qutip_size):
                tensor_ops = {}
                output_state_str = format(out_state, f"0{qutip_size}b")
                for idx, device in enumerate(self.model.qubits):
                    if output_state_str[idx] == "1":
                        tensor_ops[idx] = basis(device.N, 1)
                    else:
                        tensor_ops[idx] = basis(device.N, 0)
                qubit_states[output_state_str] = (
                    self.get_tensor(tensor_ops) * self.get_tensor(tensor_ops).dag()
                )

            # Generate reset operators for each qubit
            reset_operators = {
                j: [0 for _ in range(len(self.model.qubits))] for j in range(2)
            }
            for idx, device in enumerate(self.model.qubits):
                P0 = self.get_tensor({idx: basis(device.N, 0) * basis(device.N, 0).dag()})
                P1 = self.get_tensor({idx: basis(device.N, 1) * basis(device.N, 1).dag()})
                Px = self.get_tensor(
                    {
                        idx: basis(device.N, 0) * basis(device.N, 1).dag()
                        + basis(device.N, 1) * basis(device.N, 0).dag()
                    }
                )
                reset_operators[0][idx] = P0
                reset_operators[1][idx] = Px * P1

            # Using the segments, split the simulation into sections, one for each
            # measurement. Consecutive measurements of different channels are considered
            # to be multi-qubit measurements. the first return is the simulation
            # sections and the second the channels measured at the end of each section.
            for key in list(buffer_segments.keys()):
                buffer_segments[target_devices[key].index] = buffer_segments.pop(key)
            simulation_sections: Dict[Section] = get_resonator_response_splicing_indices(
                buffer_segments
            )

            # Construct uncoupled Hamiltonian and find collapse operators
            H0 = 0.0 * self.get_tensor()
            c_ops = []
            for qubit in self.model.qubits:
                qubit: RTCSQubit
                H0 += self.get_tensor({qubit.index: qubit.H0}) - self.get_tensor(
                    {qubit.index: qubit.H0}
                ) * (qubit.rotating_frame_frequency / qubit.frequency)
                c_ops += [self.get_tensor({qubit.index: cop}) for cop in qubit.cops]

            # Find qubit couplings.
            rotation_func = lambda f, T: np.exp(2.0j * np.pi * f * T)
            for coupling in self.model.couplings:
                if coupling.coupling_type == CouplingType.Exchange:
                    allI = []
                    for qb in self.model.qubits:
                        if qb == coupling.device_one:
                            allI.append(qb.a)
                        elif qb == coupling.device_two:
                            allI.append(qb.a.dag())
                        else:
                            allI.append(qb.I)
                    c = 2.0 * np.pi * coupling.frequency * tensor(*allI)
                    delta = (
                        coupling.device_one.rotating_frame_frequency
                        - coupling.device_two.rotating_frame_frequency
                    )

                    previous_index = 0
                    for section_num, section in simulation_sections.items():
                        index = section.indices[0] * step
                        if index != previous_index:
                            section_time = self.sim_t[previous_index : index + 1]
                            section.simulation_time = section_time

                            section.drive_hamiltonian.append(
                                [c, rotation_func(-delta, section_time)]
                            )
                            section.drive_hamiltonian.append(
                                [c.dag(), rotation_func(delta, section_time)]
                            )

                        previous_index = index

            # Upconvert the buffers and use them to create the time dependent
            # Hamiltonian operators
            self.channel_pulse_data = {}
            for device in self.model.quantum_devices.values():
                # Make sure buffers are the same length
                if isinstance(device, Qubit):
                    device: RTCSQubit
                    channel = device.get_drive_channel().physical_channel
                    baseband = channel.baseband
                    buf = buffers[channel.full_id()]
                    dt = channel.sample_time
                    if len(buf) < 2:
                        continue

                    d = spline_time(dt, self.sim_t, buf)
                    self.channel_pulse_data[channel] = d

                    bb = self.model.basebands[baseband.full_id()]
                    LO = bb.frequency
                    d = np.pi * np.exp(UPCONVERT_SIGN * 2.0j * np.pi * LO * self.sim_t) * d

                    # Split the Hamiltonian into measurement sections, specifying which
                    # qubits are to be measured at the end of each section
                    previous_index = 0
                    for section_num, section in simulation_sections.items():
                        index = section.indices[0] * step
                        if index != previous_index:
                            section_time = self.sim_t[previous_index : index + 1]
                            section_buffer = d[previous_index : index + 1]
                            section.simulation_time = section_time

                            section.drive_hamiltonian.append(
                                [
                                    self.get_tensor({device.index: device.a}),
                                    rotation_func(
                                        -device.rotating_frame_frequency, section_time
                                    )
                                    * section_buffer,
                                ]
                            )
                            section.drive_hamiltonian.append(
                                [
                                    self.get_tensor({device.index: device.a.dag()}),
                                    rotation_func(
                                        device.rotating_frame_frequency, section_time
                                    )
                                    * np.conjugate(section_buffer),
                                ]
                            )

                        previous_index = index
                else:
                    channel = device.get_default_pulse_channel().physical_channel
                    dt = self.model.physical_channels[channel.full_id()].sample_time
                    buf = resonator_buffers[channel.full_id()]
                    if len(buf) < 2:
                        continue
                    self.channel_pulse_data[channel.full_id()] = spline_time(
                        dt, self.sim_t, buf
                    )

            options = {"max_step": 1e-11}

            # Initial state.
            rho0 = tensor(*[qb.rho0 for qb in self.model.qubits])

            # Simulate the circuit
            self.measurement_statistics = MeasurementStatistics(rho0)

            # Create a branching structure of all possible outcomes then extract the
            # results of each branch and the probability of it occurring
            self.measurement_statistics.build_tree(
                H0, c_ops, simulation_sections, qubit_states, reset_operators, options
            )
            outcome_probabilities = (
                self.measurement_statistics.extract_outcome_probabilities()
            )

            logger.info("Sim complete.")
            cumulative_probabilities = {}
            counter = 0

            # Cumulative probabilities do not add to 1 as leakage states aren't
            # considered
            for branch, probabilities in outcome_probabilities.items():
                if probabilities[0] != 0:
                    counter += probabilities[0]
                    cumulative_probabilities[counter] = branch

            get_default_logger().debug(str(outcome_probabilities))

            # Generate a random number to choose a possible outcome branch
            for i in range(int(repeats)):
                success = False

                # If a leakage state is chosen try again
                while not success:
                    # Measure qubit state to determine resonator response
                    prob = random()
                    branch = next(
                        (
                            branch
                            for val, branch in cumulative_probabilities.items()
                            if isinstance(val, complex)
                            and prob < val.real
                            or not isinstance(val, complex)
                            and prob < val
                        ),
                        None,
                    )

                    # Just keep trying until we get a value.
                    if branch is not None:
                        success = True

                measurement_results = outcome_probabilities[branch][1]
                for qubit, measurement_result in measurement_results.items():
                    buffer_idx = (
                        self.model.get_qubit(qubit)
                        .get_measure_channel()
                        .physical_channel_id
                    )
                    for j, measurement in enumerate(measurement_result):
                        segment = buffer_segments[qubit][j]
                        if measurement != ControlType.RESET:
                            response_buffers[buffer_idx][i][segment[1] : segment[2]] = (
                                lorentzian_responses[buffer_idx][j][measurement]
                            )

            # Perform post processing to generate a single complex number result for
            # each measurement.
            # Map the result onto the classical registers using
            for channel, aqs in aq_map.items():
                for aq in aqs:
                    response = response_buffers[aq.physical_channel.full_id()][
                        :, aq.start : aq.start + aq.samples
                    ]
                    response_axis = get_axis_map(aq.mode, response)
                    for pp in package.get_pp_for_variable(aq.output_variable):
                        response, response_axis = self.run_post_processing(
                            pp, response, response_axis
                        )

                    var_result = results.setdefault(
                        aq.output_variable,
                        np.empty(
                            sweep_iterator.get_results_shape(response.shape),
                            response.dtype,
                        ),
                    )
                    sweep_iterator.insert_result_at_sweep_position(var_result, response)

            if self.auto_plot:
                self.plot_pulses()
                self.plot_dynamics()

        return results

    def plot_pulses(self, channels: List[str] = None):
        """
        Plot pulses used to drive the system. Pulses are shown before being transformed
        by the baseband frequnecy.

        :param channels: List of channel ids to plot. If None then all channels are plotted.
            Defaults to None.
        """
        if channels is None:
            channels = list(self.channel_pulse_data.keys())
        fig, axes = create_subplots(len(channels))
        axes = iter(axes)

        for channel in channels:
            pulse = self.channel_pulse_data[channel]
            ax = next(axes)

            ax.set_title(channel)
            ax.plot(self.sim_t, pulse.real, label="I")
            ax.plot(self.sim_t, pulse.imag, label="Q")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amp")
            ax.legend(shadow=True, fancybox=True)

        fig.suptitle("Pulses", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_dynamics(
        self,
        operator_info: List[Union[OperatorInfo, int]] = None,
        branches: List[int] = None,
        step: int = 1,
    ):
        """
        Plot the dynamics of operator expectation values for a simulation run on this
        hardware. By default the pauli operators will be plot for each qubit but custom
        operators can be specified.

        :param operator_info: Use the OperatorInfo object to specify an operator to
            plot, the id of the qubit the operator should act on and the name of the
            operator. Alternatively qubit indices (int) can be specified in which case
            the Pauli x, y and z expectations will be plotted for these qubits. If None
            then the Pauli operators for every qubit will be plotted. Defaults to None.
        :param branches: Specify a list of indices for the branches you wish to plot.
            This is only really relevent if the circuit includes mid-circuit
            measurements, in which case there will be a different trajectory for each
            possible measurement outcome. If None then the first trajectory will be
            plotted which is identical to all other trajectories for circuits without
            mid-circuit measurements. Defaults to None.
        :param step: The step length between expectation calculations along the
            trajectory. 1 gives the highest resolution but takes longer to calculate.
            Defaults to 1.
        """
        if branches is None:
            branches = [0]
            plot_end = False
        else:
            plot_end = True

        # we separate out operators and their names as it is more efficient to process
        # all operators at once
        operators = []
        operator_names = []
        # Plot all qubits if not specified
        if operator_info is None:
            operator_info = list(range(len(self.model.qubits)))
        for i, op in enumerate(operator_info):
            # if an int is given, plot pauli x, y and z for the qubit with the
            # corresponding id
            if isinstance(op, int):
                device = self.model.qubits[op]
                operators.append(
                    self.get_tensor(
                        {
                            op: basis(device.N, 0) * basis(device.N, 1).dag()
                            + basis(device.N, 1) * basis(device.N, 0).dag()
                        }
                    )
                )
                operator_names.append((op, "Pauli X"))

                operators.append(
                    self.get_tensor(
                        {
                            op: 1.0j
                            * (
                                -basis(device.N, 0) * basis(device.N, 1).dag()
                                + basis(device.N, 1) * basis(device.N, 0).dag()
                            )
                        }
                    )
                )
                operator_names.append((op, "Pauli Y"))

                operators.append(
                    self.get_tensor(
                        {
                            op: basis(device.N, 0) * basis(device.N, 0).dag()
                            - basis(device.N, 1) * basis(device.N, 1).dag()
                        }
                    )
                )
                operator_names.append((op, "Pauli Z"))
            elif isinstance(op.qubit_id, int):
                operators.append(self.get_tensor({op.qubit_id: op.operator}))
                operator_names.append((op.qubit_id, op.name))
            else:
                operators.append(op.operator)
                operator_names.append((f"not specified {i}", op.name))

        for branch in branches:
            (
                branch_trajectories,
                branch_time,
            ) = self.measurement_statistics.extract_branch_trajectory(
                operators, branch, step, plot_end
            )
            # Sort by qubit id
            trajectory_dict = {}
            for name, trajectory, time in zip(
                operator_names, branch_trajectories, branch_time
            ):
                item = trajectory_dict.setdefault(name[0], [])
                item.append((name[1], trajectory, time))

            for id, operator_info in trajectory_dict.items():
                fig, axes = create_subplots(len(operator_info))
                fig.suptitle(f"Branch {branch}, Qubit {id}")
                axes = iter(axes)
                for name, trajectory, t in operator_info:
                    ax = next(axes)
                    ax.set_title(name)
                    ax.set_xlabel("Time")
                    ax.set_ylabel("<op>")
                    ax.plot(t, np.array(trajectory).real)
                fig.tight_layout()

        plt.show()

    def get_branches(self):
        """Return the branch ids along with the measurements associated with them.

        :Example:

        ``{branch_id: [branch_probability, {qubit: [resets_and_measurement_outcomes]}]}``

          Where:

          - branch_id: id to call when plotting specific branches
          - branch_probability: probability branch occured
          - qubit: qubit number
          - resets_and_measurement_outcomes: the outcome of measurements for this branch
            in order. Also includes when a reset occured.

        :returns: A dictionary mapping branch ids to their associated measurements.
        :rtype: dict
        """
        return self.measurement_statistics.extract_outcome_probabilities()

    def get_tensor(self, ops: dict = None):
        if ops is None:
            ops = {}
        allI = [qb.I for qb in self.model.qubits]
        for _idx, op in ops.items():
            allI[_idx] = op
        return tensor(*allI)


def spline_time(dt, sim_t, buffer):
    """
    Perform a cubic spline on the buffer to make the number of points the same as the
    number of points in the simulation time
    """
    t = np.linspace(0.0, (len(buffer) - 1) * dt, len(buffer))
    if t[-1] < sim_t[-1]:
        extra_t = np.arange(t[-1] + dt, sim_t[-1], dt)
        extra_d = np.zeros(len(extra_t), dtype=np.complex128)

        t = np.concatenate((t, extra_t))
        buffer = np.concatenate((buffer, extra_d))

    buffer = CubicSpline(t, buffer)(sim_t)

    return buffer


def tukey_window(N, alpha=0.1):
    n = np.linspace(0.0, 1.0, N)
    x = 0.5 * (1 - np.cos(2.0 * np.pi * n / alpha))
    start = int(N * alpha * 0.5)
    x[start:].fill(1.0)
    y = x + np.flip(x, axis=0) - 1.0
    return y


def get_resonator_response_segments(buffers, resets):
    zero_tolerance = 1e-6
    zero_tolerance_sq = zero_tolerance**2

    max_len = max([len(buffer) for buffer in buffers.values()])
    buffer_inside = {k: False for k, v in buffers.items()}
    buffer_segments = {}
    i = 0
    while i < max_len:
        for buffer_idx, buffer in buffers.items():
            reset_indices = resets.get(buffer_idx, [])
            if i < len(buffer):
                if buffer[i] ** 2 > zero_tolerance_sq:
                    if not buffer_inside[buffer_idx]:
                        segment = buffer_segments.setdefault(buffer_idx, [])
                        segment.append([ControlType.MEASURE, i])
                        buffer_inside[buffer_idx] = True
                elif buffer_inside[buffer_idx]:
                    buffer_segments[buffer_idx][-1].append(i)
                    buffer_inside[buffer_idx] = False
                if i in reset_indices:
                    segment = buffer_segments.setdefault(buffer_idx, [])

                    # If currently constructing a measurement section then reset
                    # must be before this measurement
                    if buffer_inside[buffer_idx]:
                        segment.insert(-1, [ControlType.RESET, i, i])
                    else:
                        segment.append([ControlType.RESET, i, i])
            elif buffer_inside[buffer_idx] == True:
                buffer_segments[buffer_idx][-1].append(i)
                buffer_inside[buffer_idx] = False
        i += 1
    for buffer_idx, buffer_segment in buffer_segments.items():
        if buffer_inside[buffer_idx]:
            buffer_segment[-1].append(i - 1)

    return buffer_segments


def get_resonator_response_splicing_indices(buffer_segments):
    """
    Take the buffer segments and parse them to find where the qubit buffer needs to be
    split in order to perform mid-circuit measurements.
    """
    previous_start_indice = 0
    section_num = -1
    simulation_sections: Dict[Section] = {}
    ordered_segments = []
    for qubit, segments in buffer_segments.items():  # Order segments from first to last
        for segment in segments:
            ordered_segments.append([qubit, *segment])
    # sort by last index if first indices are the same to make sure reset are before
    # measures
    ordered_segments.sort(key=lambda x: (x[2], x[3]))

    previous_control_type = ordered_segments[0][1]
    for segment in ordered_segments:
        indices = [segment[2], segment[3]]
        if indices[0] == previous_start_indice and segment[1] == previous_control_type:
            if section_num == -1:  # Catch case measurement occurs at time 0
                section_num += 1
                simulation_sections[section_num] = Section(
                    indices, [segment[0]], segment[1]
                )
            else:
                simulation_sections[section_num].qubits.append(segment[0])
        else:
            section_num += 1
            simulation_sections[section_num] = Section(indices, [segment[0]], segment[1])

        previous_start_indice = indices[0]
        previous_control_type = segment[1]

    return simulation_sections


def get_resonator_response_signal_segment(
    resonator_input, resonator_dt, width, res_freq, shift, lo_freq
):
    # Peak of resonator resonant frequency relative to LO freq
    centre0 = res_freq - lo_freq
    # Peak of resonator resonant frequency relative to LO freq with excited qubit
    centre1 = centre0 + shift

    sig0 = tukey_window(len(resonator_input)) * resonator_input

    o = np.fft.fft(sig0)
    f = np.fft.fftfreq(len(o), d=resonator_dt)

    def lorentzian(u, centre, width):
        return 1 / (1 + ((u - centre) / (width / 2)) ** 2)

    # Create Lorentzian curves in the Fourier transform of the buffer segment
    # representing the resonator spectrum
    resonator_output_0 = np.fft.ifft(
        o * lorentzian(f, centre0, width)
    )  # Response if qubit in ground state
    resonator_output_1 = np.fft.ifft(
        o * lorentzian(f, centre1, width)
    )  # Response if qubit in excited state

    return [resonator_output_0, resonator_output_1]


def get_simple_resonator_response(
    qubit_dt, qubit_z, resonator_dt, resonator_iq, width, res_freq, shift, lo_freq
):
    centre0 = res_freq - lo_freq
    centre1 = centre0 + shift

    iq_t = np.linspace(
        0.0, resonator_dt * len(resonator_iq), len(resonator_iq), endpoint=False
    )  # Create array of sample times
    resonator_iq *= np.exp(-UPCONVERT_SIGN * 2.0j * np.pi * (res_freq - lo_freq) * iq_t)

    ZERO_TOLERANCE = 1e-6

    abssq = np.abs(resonator_iq) ** 2
    ztsq = ZERO_TOLERANCE**2

    inside = False
    start = 0
    i = 0
    segments = []
    while i < len(abssq):
        if inside:
            if abssq[i] < ztsq:
                inside = False
                segments.append((start, i))
        else:
            if abssq[i] > ztsq:
                inside = True
                start = i
        i += 1
    if inside:
        segments.append((start, i))

    simulate_continuous_measurement = (
        len(segments) == 1
        and segments[0][0] == 0
        and segments[0][1] == len(resonator_iq) - 1
    )

    # segments is now a list of start and stop index of non-zero segments for the
    # resonator response

    # for now assume qubit_t and resonator_t are linearly spaced arrays starting at zero

    # qubit_dt = qubit_t[1] - qubit_t[0]
    # resonator_dt = resonator_t[1] - resonator_t[0]

    factor = qubit_dt / resonator_dt

    qsegments = [(int(round(factor * s[0])), int(round(factor * s[1]))) for s in segments]

    qzsegments = []
    for start, stop in qsegments:
        z = np.mean(qubit_z[start:stop])
        if np.isnan(z):
            z = qubit_z[len(qubit_z) - 1]
        qzsegments.append(z)

    input = resonator_iq

    output = np.zeros(shape=input.shape, dtype=input.dtype)

    def flat_top_window(N):
        n = np.linspace(0.0, 1.0, N)
        a0 = 0.35875
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168
        return (
            a0
            - a1 * np.cos(2.0 * np.pi * n)
            + a2 * np.cos(4.0 * np.pi * n)
            - a3 * np.cos(6.0 * np.pi * n)
        )

    for s, qz in zip(segments, qzsegments):
        p0 = 0.5 * (qz + 1)
        p1 = 1 - p0

        i, j = s
        x = input[i:j]
        # if simulate_continuous_measurement:
        #     sig0 = x # flat_top_window(len(x))*x
        # else:
        sig0 = tukey_window(len(x)) * x

        o = np.fft.fft(sig0)
        f = np.fft.fftfreq(len(o), d=resonator_dt)

        def L(u, centre, width):
            return 1 / (1 + ((u - centre) / (width / 2)) ** 2)

        S = p0 * L(f, centre0, width) + p1 * L(f, centre1, width)
        o *= S
        o = np.fft.ifft(o)

        # plt.subplots()
        # plt.plot(sig0.real)
        # plt.plot(sig0.imag)
        # plt.plot(o.real)
        # plt.plot(o.imag)
        # plt.show()

        # todo: maybe method to force signal continuity should be here, but this method
        #   messes up sometimes
        # diff = o[0] - sig0[0]
        # o -= diff

        output[i:j] = o

        if np.isnan(output[i]):
            raise Exception()

    return output


def create_subplots(plot_num):
    square = int(np.sqrt(plot_num))
    extra_rows = int(np.ceil((plot_num - square**2) / square))
    fig, axes = plt.subplots(nrows=square + extra_rows, ncols=square, squeeze=True)
    if isinstance(axes, Iterable):
        axes = list(axes.flatten())
    else:
        axes = [axes]
    for i in range(len(axes) - plot_num):
        fig.delaxes(axes.pop())

    return fig, axes
