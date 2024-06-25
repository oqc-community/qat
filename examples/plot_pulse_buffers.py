from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker

from qat.purr.backends.echo import EchoEngine, get_default_echo_hardware
from qat.purr.backends.live import LiveDeviceEngine
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import SweepIterator
from qat.purr.compiler.runtime import QuantumRuntime, get_builder
from qat.purr.integrations.qasm import Qasm3Parser

np.set_printoptions(
    edgeitems=3000000, linewidth=100000000, formatter=dict(float=lambda x: "%.3g" % x)
)

import logging

logging.disable()


class PhysicalBufferPlotEngine(EchoEngine):
    def __init__(
        self,
        model,
        target_engine=None,
        channels=None,
        upconvert=True,
        figsize=None,
        name=None,
    ):
        super().__init__(model)
        self.target = target_engine or self
        self.channels = channels
        self.upconvert = upconvert
        self.figsize = figsize
        self.name = name

    def _execute_on_hardware(self, sweep_iterator: SweepIterator, package: QatFile):
        position_map = self.target.create_duration_timeline(package.instructions)
        pulse_buffers = self.target.build_pulse_channel_buffers(
            position_map, do_upconvert=self.upconvert
        )
        buffers = self.target.build_physical_channel_buffers(pulse_buffers)
        channels = self.channels or buffers.keys()

        if not isinstance(channels, Iterable):
            channels = [channels]

        figsize = self.figsize
        if figsize is None:
            figsize = (8, 3 * len(channels))

        fig, ax = plt.subplots(len(channels), 1, figsize=figsize)
        for i, channel in enumerate(channels):
            dt = self.model.physical_channels[channel].sample_time
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
        if self.name:
            plt.savefig(self.name)

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

        # As we're processing a QASM file the runtime assumes results are needed, so we
        # just call the default echo for this. We're immediately throwing them away anyway.
        return super()._execute_on_hardware(sweep_iterator, package)


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
    engine = PhysicalBufferPlotEngine(
        builder.model, engine, channels, upconvert, figsize, name
    )
    runtime = QuantumRuntime(engine)
    runtime.execute(builder)


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
