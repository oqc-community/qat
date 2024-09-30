from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.live import build_lucy_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.devices import (
    PhysicalBaseband,
    PhysicalChannel,
    add_cross_resonance,
    build_qubit,
    build_resonator,
)
from qat.purr.compiler.hardware_models import QuantumHardwareModel


def echo_hardware(num_qubits: int):
    """
    Returns a hardware model for the echo simulator.
    """
    return get_default_echo_hardware(
        num_qubits, connectivity=[(q, q + 1) for q in range(num_qubits - 1)]
    )


def rtcs_hardware():
    """
    Returns a two qubit model that simulates a real time chip.
    """
    return get_default_RTCS_hardware()


def physical_two_qubit_hardware():
    """
    A different two qubit model, pulled from the tests of QAT.
    """
    # Create the hardware model
    model = QuantumHardwareModel()
    bb1 = PhysicalBaseband("AP1-L1", 4.024e9, 250e6)
    bb2 = PhysicalBaseband("AP1-L2", 8.43135e9, 250e6)
    bb3 = PhysicalBaseband("AP1-L3", 3.6704e9, 250e6)
    bb4 = PhysicalBaseband("AP1-L4", 7.8891e9, 250e6)

    ch1 = PhysicalChannel("Ch1", 0.5e-9, bb1, 1)
    ch2 = PhysicalChannel("Ch2", 1e-09, bb2, 1, acquire_allowed=True)
    ch3 = PhysicalChannel("Ch3", 0.5e-9, bb3, 1)
    ch4 = PhysicalChannel("Ch4", 1e-09, bb4, 1, acquire_allowed=True)

    r0 = build_resonator("R0", ch2, frequency=8.68135e9, measure_fixed_if=True)
    q0 = build_qubit(
        0,
        r0,
        ch1,
        drive_freq=4.274e9,
        second_state_freq=4.085e9,
        measure_amp=10e-3,
        fixed_drive_if=True,
    )

    r1 = build_resonator("R1", ch4, frequency=8.68135e9, measure_fixed_if=True)
    q1 = build_qubit(
        1,
        r1,
        ch3,
        drive_freq=3.9204e9,
        second_state_freq=3.7234e9,
        measure_amp=10e-3,
        fixed_drive_if=True,
    )

    add_cross_resonance(q0, q1)

    model.add_physical_baseband(bb1, bb2, bb3, bb4)
    model.add_physical_channel(ch1, ch2, ch3, ch4)
    model.add_quantum_device(r0, q0, r1, q1)
    return model


def lucy_hardware():
    """
    A two-qubit hardware model based on Lucy.
    """

    return build_lucy_hardware(QuantumHardwareModel())
