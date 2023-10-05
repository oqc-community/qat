import numpy as np
from qat.purr.backends.realtime_chip_simulator import (
    RealtimeChipSimEngine,
    get_default_RTCS_hardware,
)
from qat.purr.compiler.runtime import QuantumRuntime, get_builder


def conv(x):
    return 0 if x > 0.0 else 1


hw = get_default_RTCS_hardware()

control_q = hw.get_qubit(0)
target_q = hw.get_qubit(1)
cr_channel = control_q.get_cross_resonance_channel(target_q)
sync_channels = [cr_channel, control_q.get_drive_channel(), target_q.get_drive_channel()]
prep = np.linspace(0.0, np.pi, 2)

engine = RealtimeChipSimEngine(hw, auto_plot=True)
runtime = QuantumRuntime(engine)
builder = get_builder(hw).had(control_q, ).synchronize(sync_channels).X(
    target_q, np.pi
).ECR(control_q, target_q).X(control_q).X(target_q, -np.pi / 2.0).Z(
    control_q, -np.pi / 2.0
).measure_mean_z(control_q).measure_mean_z(target_q)

runtime.execute(builder)
