# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import itertools
import numpy as np
import re

from typing import Dict, List
from enum import Enum, auto

from qat.purr.utils.logger import get_default_logger

from matplotlib import pyplot as plt

log = get_default_logger()


class ResultType(Enum):
    WAVEFORM = auto()
    ACQUISITION = auto()
    SEQUENCE_FILE = auto()


class Oscilloscope:
    def __init__(self, result_type=None, filter=None, sample_start=0, sample_end=-1):
        self.result_type = result_type
        self.filter = filter or "(.*)"
        self.sample_start = sample_start
        self.sample_end = sample_end

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            return_value = func(*args, **kwargs)
            if self.result_type == ResultType.WAVEFORM:
                for context, waveforms in return_value.items():
                    self.plot_waveforms(waveforms, context)
            elif self.result_type == ResultType.ACQUISITION:
                for context, acquisitions in return_value.items():
                    self.plot_acquisitions(
                        acquisitions, context, self.sample_start, self.sample_end
                    )
            elif self.result_type == ResultType.SEQUENCE_FILE:
                self.plot_sequence_files(return_value, self.filter)
            else:
                raise ValueError(f"Invalid result type {self.result_type}")
            return return_value

        return wrapper

    def extend(self, y, new_size: int):
        if y.size < new_size:
            return np.append(y, np.zeros(new_size - y.size))
        return y

    def plot_waveforms(self, waveforms: Dict, context):
        if waveforms is None or not any(waveforms):
            return
        amps = {name: np.array(wf["data"]) for name, wf in waveforms.items()}
        max_size = max([y.size for y in amps.values()])
        amps = {name: self.extend(y, max_size) for name, y in amps.items()}
        time = np.linspace(0, max_size, max_size)
        self.single_display(
            time, amps, title=f"Waveform plots ({context})", y_label="Amplitude"
        )

    def plot_acquisitions(self, acquisitions: Dict, context, sample_start, sample_end):
        if acquisitions is None or not any(acquisitions):
            return

        bins = acquisitions["acquisition"]["bins"]
        scope = acquisitions["acquisition"]["scope"]

        amp_data = {}
        # phase_data = {}
        i = np.array(scope["path0"]["data"][sample_start:sample_end])
        q = np.array(scope["path1"]["data"][sample_start:sample_end])

        offset_I, offset_Q = np.mean(i[-1000:]), np.mean(q[-1000:])
        i -= offset_I
        q -= offset_Q

        # amp_data[f"{name}_i"] = i
        # amp_data[f"{name}_q"] = q
        amp_data[f"|i + 1j*q|"] = np.abs(i + 1j * q)
        # phase_data[f"{name}_phase"] = np.arctan2(q, i) * 180 / np.pi

        # time = np.linspace(0, i.size, i.size)
        # self.single_display(time, {f"{name}_i": i}, f"I ({context})", y_label="I")
        # self.single_display(time, {f"{name}_q": q}, f"Q ({context})", y_label="Q")
        # self.single_display(time, {f"{name}_amp": np.abs(i + 1j * q)}, f"ABS ({context})", y_label="Amplitude")
        # self.single_display(
        # time, {f"{name}_phase": np.arctan2(q, i) * 180 / np.pi}, f"PHASE ({context})", y_label="Phase"
        # )

        size = max([a.size for a in amp_data.values()], default=0)
        time = np.linspace(sample_start, sample_start + size, size)
        self.single_display(
            time, amp_data, f"Amplitude plots ({context})", y_label="Amplitude"
        )
        # self.single_display_twin_scale(
        #     time, amp_data, phase_data, title=f"Amplitude and Phase plots ({context})"
        # )

    def plot_sequence_files(self, seq_files: Dict, filter_regex: str):
        data = {
            pc: {
                name: np.array(wf["data"]) for name,
                wf in seq_file.waveforms.items() if re.match(filter_regex, name)
            } for pc,
            seq_file in seq_files.items()
        }
        data = {pc: amps for pc, amps in data.items() if any(amps)}
        if any(data):
            max_size = max([y.size for wf in data.values() for y in wf.values()], default=0)
            data = {
                pc: {name: self.extend(y, max_size) for name, y in amps.items()} for pc,
                amps in data.items()
            }
            time = np.linspace(0, max_size, max_size)
            self.multi_display(time, data, "Multiple channel display")

    def single_display(self, x, amps: Dict, title: str, y_label, x_label="time (ns)"):
        plt.title(title)

        for name, y in amps.items():
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.plot(x, y, label=name)

        plt.legend()
        plt.show()

    def single_display_twin_scale(self, x, amp_data: Dict, phase_data: Dict, title: str):
        fig, ax0 = plt.subplots()
        fig.suptitle(title)
        color_cycle = itertools.cycle(["tab:blue", "tab:orange", "tab:red", "tab:green"])

        ax0.set_xlabel("time (ns)")

        ax0.set_ylabel("Amplitude")
        for name, y in amp_data.items():
            ax0.plot(x, y, color=next(color_cycle), label=name)
            ax0.tick_params(axis='y')
            ax0.legend()

        ax1 = ax0.twinx()

        ax1.set_ylabel('Phase')
        for name, y in phase_data.items():
            ax1.plot(x, y, color=next(color_cycle), label=name)
            ax1.tick_params(axis='y')
            ax1.legend()

        fig.tight_layout()
        plt.show()

    def multi_display(self, x, data: Dict[str, Dict], title: str):
        fig, axs = plt.subplots(len(data))

        if not isinstance(
            axs, (List, np.ndarray)
        ):  # Case where only one subplot (len(data) == 1)
            axs = [axs]

        fig.suptitle(title)
        plt.xlabel("time (ns)")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        for i, (sub_title, amps) in enumerate(data.items()):
            for name, y in amps.items():
                axs[i].plot(x, y, label=name)
                axs[i].legend()
                axs[i].set_title(sub_title)

        plt.show()
