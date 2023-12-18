# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from typing import List, Optional

from qat.purr.backends.live import LiveHardwareModel, apply_setup_to_hardware
from qat.purr.backends.live_devices import ControlHardware
from qat.purr.backends.utilities import get_axis_map, UPCONVERT_SIGN
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import CompilerConfig
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import QuantumExecutionEngine, SweepIterator
from qat.purr.compiler.instructions import AcquireMode
from qat.qat import execute
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QPUVersion:
    def __init__(self, make: str, version: str = None):
        self.make = make
        self.version = version

    def __repr__(self):
        return f"{self.make}-{self.version if self.version is not None else 'latest'}"

    @staticmethod
    def with_version(version: str = None):
        """Creates a QPU version with an empty QPU make. Only used in very special circumstances."""
        return QPUVersion("", version)


def inject_name(cls):
    """
    Decorator to inject class name into the QPU make/version fields
    in our static make/model naming objects.
    """
    for value in vars(cls).values():
        if isinstance(value, QPUVersion):
            if value.make is None or value.make == "":
                value.make = cls.__name__

    return cls


@inject_name
class Lucy:
    Latest = QPUVersion.with_version()


class VerificationModel(LiveHardwareModel):
    def __init__(
        self,
        qpu_version,
        verification_engine_type: type,
        control_hardware: ControlHardware = ControlHardware(),
    ):
        super().__init__(control_hardware, [verification_engine_type], None)
        self.version = qpu_version


def verify_instructions(builder: InstructionBuilder, qpu_type: QPUVersion):
    """
    Runs instruction verification for the instructions in this builder.

    Only run this on instructions that will go through no more transformations
    before being sent to the driver, otherwise you cannot rely upon its results as
    being accurate. In most situations this will only be fully truthful on instructions that
    have already gone through the entire pipeline.
    """
    model = get_verification_model(qpu_type)
    if model is None:
        raise ValueError(
            f"Cannot verify instructions, {qpu_type} isn't a valid QPU version."
        )

    engine: VerificationEngine = model.get_engine()
    engine._verify_timeline(builder.instructions)


def get_verification_model(qpu_type: QPUVersion) -> Optional[VerificationModel]:
    """
    Get verification model for a particular QPU make and model. Each make has its own class,
    which has a field that is each individual version available for verification.

    For example, if you wanted to verify our Lucy machine, that'd be done with:
    ``
    get_verification_model(Lucy.Latest)
    ``

    Or with a specific version:
    ``
    get_verification_model(Lucy.XY)
    ``
    """
    if not isinstance(qpu_type, QPUVersion):
        raise ValueError(
            f"{qpu_type} is not a QPU version, can't find verification engine."
        )

    if qpu_type.make == Lucy.__name__:
        # TODO: Should have an apply_setup_to_hardware in live which people can use to build our own architecture.
        model = VerificationModel(qpu_type, LucyVerificationEngine)
        return apply_setup_to_hardware(
            model,
            qubit_count=8,
            pulse_hw_x_pi_2_width=25e-9,
            pulse_hw_zx_pi_4_width=3.18e-07,
            pulse_measure_width=2.41e-06,
        )  # type hinted incorrectly - needs to return Union[VerificationModel, QuantumHardwareModel]
    return None


class VerificationEngine(QuantumExecutionEngine, ABC):
    @abstractmethod
    def _validate_instructions(self, package: QatFile):
        pass

    def _execute_on_hardware(self, sweep_iterator: SweepIterator, package: QatFile):
        self._validate_instructions(package)
        results = {}
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            position_map = self.create_duration_timeline(package)
            pulse_channel_buffers = self.build_pulse_channel_buffers(position_map, True)
            buffers = self.build_physical_channel_buffers(pulse_channel_buffers)
            aq_map = self.build_acquire_list(position_map)

            repeats = package.repeat.repeat_count
            for channel_id, aqs in aq_map.items():
                for aq in aqs:
                    # just echo the output pulse back for now
                    response = buffers[aq.physical_channel.full_id()][
                        aq.start : aq.start + aq.samples
                    ]
                    if aq.mode != AcquireMode.SCOPE:
                        if repeats > 0:
                            response = np.tile(response, repeats).reshape((repeats, -1))

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

        return results


class LucyVerificationEngine(VerificationEngine):
    """
    1) Integrate verification as part of the execution, in which case called right after Sweep iteration changes
        This means any call to execute() will also call verify()
        implementation is empty by default, but can be overriden in an specific Engine.
    2) Define verification separately in an engine.
        a) intergate the verification in execution
            This means overriding execute(), execute_on_hw(), process_results(), process_assigns()
        b) leave it separete from execution
            Call verify directly to test ...
    """

    max_circuit_duration = 90000e-9

    def _validate_instructions(self, package: QatFile):
        if self._get_circuit_duration(package) > self.max_circuit_duration:
            return False

        return True

    # def _get_circuit_duration(self, package:QatFile):
    #     position_map = self.create_duration_timeline(package)
    #     pc2samples = {pc: positions[-1].end for pc, positions in position_map.items()}
    #     durations = {pc: samples * pc.sample_time for pc, samples in pc2samples.items()}
    #
    #     if any([duration > self.max_circuit_duration for duration in durations.values()]):
    #         lo(f"Exceeds the circuit duration limit of {self.max_circuit_duration} microseconds.")

    def _get_circuit_duration(self, package: QatFile):
        position_map = self.create_duration_timeline(package)
        pc2samples = {pc: positions[-1].end for pc, positions in position_map.items()}
        durations = {pc: samples * pc.sample_time for pc, samples in pc2samples.items()}

        circuit_duration = max([duration for duration in durations.values()])

        log.info(
            f"Exceeds the circuit duration limit of {circuit_duration} microseconds."
        )

        return circuit_duration

    def _execute_on_hardware(self, sweep_iterator: SweepIterator, package: QatFile):
        if self.model.control_hardware is None:
            raise ValueError("Please add a control hardware first!")

        results = {}
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            position_map = self.create_duration_timeline(package)
            #  self._verify_timeline(package)
            #  self._verify_timeline(position_map) # much more efficient <------

            pulse_channel_buffers = self.build_pulse_channel_buffers(position_map, True)
            buffers = self.build_physical_channel_buffers(pulse_channel_buffers)
            baseband_freqs = self.build_baseband_frequencies(pulse_channel_buffers)
            aq_map = self.build_acquire_list(position_map)

            data = {}
            for ch, buffer in buffers.items():
                if len(buffer) > 0:
                    data[ch] = buffer
                    if ch in baseband_freqs:
                        physical_channel = self.model.get_physical_channel(ch)
                        physical_channel.baseband.set_frequency(int(baseband_freqs[ch]))

            self.model.control_hardware.set_data(data)

            repetitions = package.repeat.repeat_count
            repetition_time = package.repeat.repetition_period

            for aqs in aq_map.values():
                if len(aqs) > 1:
                    raise ValueError(
                        "Multiple acquisitions are not supported on the same channel "
                        "in one sweep step."
                    )
                for aq in aqs:
                    physical_channel = aq.physical_channel
                    dt = physical_channel.sample_time
                    physical_channel.readout_start = aq.start * dt + aq.delay
                    physical_channel.readout_length = aq.samples * dt
                    physical_channel.acquire_mode_integrator = (
                        aq.mode == AcquireMode.INTEGRATOR
                    )

            playback_results = self.model.control_hardware.start_playback(
                repetitions=repetitions, repetition_time=repetition_time
            )

            for channel, aqs in aq_map.items():
                if len(aqs) > 1:
                    raise ValueError(
                        "Multiple acquisitions are not supported on the same channel "
                        "in one sweep step."
                    )
                for aq in aqs:
                    response = playback_results[aq.physical_channel.id]
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
        return results

    def build_baseband_frequencies(
        self, pulse_channel_buffers: Dict[PulseChannel, np.ndarray]
    ):
        """Find fixed intermediate frequencies for physical channels if they exist."""
        baseband_freqs = {}
        baseband_freqs_fixed_if = {}
        for pulse_channel in pulse_channel_buffers.keys():
            if pulse_channel.fixed_if:
                if (
                    baseband_freqs_fixed_if.get(
                        pulse_channel.physical_channel_id, False
                    )
                    and baseband_freqs[pulse_channel.physical_channel_id]
                    != pulse_channel.frequency
                    - UPCONVERT_SIGN * pulse_channel.baseband_if_frequency
                ):
                    raise ValueError(
                        "Cannot fix the frequency for two pulse channels of different "
                        "frequencies on the same physical channel!"
                    )
                baseband_freqs[pulse_channel.physical_channel_id] = (
                    pulse_channel.frequency
                    - UPCONVERT_SIGN * pulse_channel.baseband_if_frequency
                )
                baseband_freqs_fixed_if[
                    pulse_channel.physical_channel_id
                ] = pulse_channel.fixed_if
            else:
                if (
                    pulse_channel.physical_channel_id not in baseband_freqs_fixed_if
                    or not baseband_freqs_fixed_if[pulse_channel.physical_channel_id]
                ):
                    baseband_freqs_fixed_if[
                        pulse_channel.physical_channel_id
                    ] = pulse_channel.fixed_if

        return baseband_freqs


def verify_program(
    program: str, compiler_config: CompilerConfig, qpu_version: QPUVersion
):
    model = get_verification_model(qpu_version)
    return any(execute(program, model, compiler_config))
