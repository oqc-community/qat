import numpy as np
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.runtime import get_builder

hw = get_default_echo_hardware()
control_q = hw.get_qubit(0)
target_q = hw.get_qubit(1)
cr_channel = control_q.get_cross_resonance_channel(target_q)
sync_channels = [
    cr_channel,
    control_q.get_drive_channel(),
    target_q.get_drive_channel(),
]
prep = np.linspace(0.0, np.pi, 2)

# sweep over a time range
builder = (
    get_builder(hw).X(control_q,
                      np.pi).synchronize(sync_channels).X(target_q,
                                                          np.pi).ECR(control_q,
                                                                     target_q).X(control_q).
    X(target_q,
      -np.pi / 2.0).Z(control_q,
                      -np.pi / 2.0).measure_mean_z(control_q).measure_mean_z(target_q)
)

for instruction in builder.instructions:
    print(instruction)
