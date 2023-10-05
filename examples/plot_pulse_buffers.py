from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from qat.purr.backends.echo import EchoEngine, get_default_echo_hardware
from qat.purr.backends.live import LiveDeviceEngine
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.runtime import get_builder
from qat.purr.integrations.qasm import Qasm3Parser

np.set_printoptions(
    edgeitems=3000000, linewidth=100000000, formatter=dict(float=lambda x: "%.3g" % x)
)

import logging

logging.disable()


def plot_physical_buffers(
    builder,
    engine: LiveDeviceEngine = None,
    channels=None,
    upconvert=True,
    figsize=None,
    name=None,
):
    """
    Plot the physical channel buffers generated from a qat instruction builder
    Parameters
    ----------
    builder: the qat builder object to be plotted
    engine: qat execution engine to use to generate buffers. Defaults to Echo hardware engine
    channels: physical channel ids for the channels to be plotted. If None, plots all channels.
    """
    hw = builder.model
    engine = engine or EchoEngine(hw)

    qat_file = InstructionEmitter().emit(builder.instructions, builder.model)
    position_map = engine.create_duration_timeline(qat_file)
    pulse_buffers = engine.build_pulse_channel_buffers(position_map, do_upconvert=upconvert)
    buffers = engine.build_physical_channel_buffers(pulse_buffers)
    channels = channels or buffers.keys()

    if not isinstance(channels, Iterable):
        channels = [channels]
    if figsize is None:
        figsize = (8, 3 * len(channels))
    fig, ax = plt.subplots(len(channels), 1, figsize=figsize)
    for i, channel in enumerate(channels):
        dt = engine.model.physical_channels[channel].sample_time
        t = np.arange(len(buffers[channel])) * dt
        ax[i].plot(t, buffers[channel].real, label="Real")
        ax[i].plot(t, buffers[channel].imag, label="Imag")
        ax[i].legend()
        ax[i].set_title(channel)
        ax[i].set_xlabel("Time")
        ax[i].xaxis.set_major_formatter(ticker.EngFormatter("s"))
        ax[i].set_ylabel("Pulse amplitude")
        ax[i].yaxis.set_major_formatter(ticker.EngFormatter("V"))
    fig.set_tight_layout(True)
    if name:
        plt.savefig(name)

    res_bufs = []
    for res in hw.resonators:
        res_bufs.append(res.physical_channel.id)
    for key, value in buffers.items():
        if key in res_bufs:
            tmp_buf = []
            for val in value:
                tmp_buf.append(val)
                tmp_buf.append(val)
            buffers[key] = tmp_buf
    return buffers


if __name__ == "__main__":
    qasm = """
    OPENQASM 3;
    include "qelib1.inc";
    defcalgrammar "openpulse";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q -> c;
    """

    hw = get_default_echo_hardware(2)
    parser = Qasm3Parser()
    builder = parser.parse(get_builder(hw), qasm)

    plot_physical_buffers(builder)
    plt.show()
