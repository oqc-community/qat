from qat.purr.compiler.builders import InstructionBuilder

from qat.purr.compiler.instructions import Label
from qat.purr.backends.protobuf_stuff.builder_pb2 import (
    Instructions, 
    ResultsProcessing, 
    Measure,
    X,
    Y,
    Z,
    Swap,
    Had,
    PostProcessing,
    Pulse,
    Acquire,
    Delay,
    Syncronise,
    PhaseShift,
    SX,
    SXdg,
    S,
    Sdg,
    T,
    Tdg,
    cX,
    cY,
    cZ,
    cnot,
    ccnot,
    cswap,
    ECR,
    repeat,
    assign,
    returns,
    device_assign,
    CreateLabel,
    ExistingNames,
    reset
)
import itertools
import math
from collections import defaultdict
from enum import Enum, auto
from typing import List, Set, Union

import jsonpickle
import numpy as np

from qat.purr.compiler.config import InlineResultsProcessing
from qat.purr.compiler.devices import (
    ChannelType,
    CyclicRefPickler,
    CyclicRefUnpickler,
    PulseChannel,
    PulseChannelView,
    Qubit,
    Resonator,
)


class ProtoBuilder(InstructionBuilder):
    def __init__(self):
        self._instructions = Instructions()
        self.existing_names = ExistingNames()
    
    def results_processing(self, variable: str, res_format: InlineResultsProcessing):
        results_processing = ResultsProcessing()
        results_processing.variable = variable
        results_processing.res_format = res_format
        return self.add(results_processing)
    
    def measure_single_shot_z(self, target: Qubit, axis = None, output_variable: str = None):
        meas = Measure()
        meas.target = target
        meas.axis = axis
        meas.output_variable = output_variable
        return self.add(meas)
    
    def measure_mean_z(
        self, target: Qubit, axis = None, output_variable: str = None
    ):
        meas = Measure()
        meas.target = target
        meas.axis = axis
        meas.output_variable = output_variable
        return self.add(meas)

    def X(self, target: Union[Qubit, PulseChannel], radii=None):
        x = X()
        x.target = target
        x.radii = radii
        return self.add(x)
    
    def Y(self, target: Union[Qubit, PulseChannel], radii=None):
        x = Y()
        x.target = target
        x.radii = radii
        return self.add(x)
    
    def Z(self, target: Union[Qubit, PulseChannel], radii=None):
        x = Z()
        x.target = target
        x.radii = radii
        return self.add(x)
    
    def swap(self, target: Qubit, destination: Qubit):
        swap = Swap()
        swap.target = target
        swap.destination = destination
        return self.add(swap)
    
    def had(self, qubit: Qubit):
        h = Had()
        h.qubit = qubit
        return self.add(qubit)
    
    def post_processing(
        self, acq: Acquire, process, axes=None, target: Qubit = None, args=None
    ):
        pp = PostProcessing()
        pp.acquire = acq 
        pp.process = process
        pp.axes = axes
        pp.target = target
        pp.args = args
        return self.add(pp)
    
    def pulse(self, *args, **kwargs):
        pp = Pulse()
        pp.quantum_target = args[0]
        pp.shape = args[1]
        pp.width = args[2]
        pp.amp =  args[2]
        pp.phase =  args[3]
        pp.drag =  args[4]
        pp.rise = args[5]
        pp.amp_setup =  args[6]
        pp.scale_factor =  args[7]
        pp.zero_at_edges =  args[8]
        pp.beta =  args[9]
        pp.frequency =  args[10]
        pp.internal_phase =  args[11]
        pp.std_dev =  args[12]
        pp.square_width =  args[13]
        pp.ignore_channel_scale =  args[14]

    def acquire(self, *args, **kwargs):
        pass

    def delay(self, target: Union[Qubit, PulseChannel], time: float):
        d= Delay()
        d.target = target
        d.time = time

    def synchronize(self, targets: Union[Qubit, List[Qubit]]):
        s = Syncronise()
        s.targets = targets

    def phase_shift(self, target: PulseChannel, phase):
        ps = PhaseShift()
        ps.target = target
        ps.phase = phase

    def SX(self, target):
        return self.X(target, np.pi / 2)

    def SXdg(self, target):
        return self.X(target, -(np.pi / 2))

    def S(self, target):
        return self.Z(target, np.pi / 2)

    def Sdg(self, target):
        return self.Z(target, -(np.pi / 2))

    def T(self, target):
        return self.Z(target, np.pi / 4)

    def Tdg(self, target):
        return self.Z(target, -(np.pi / 4))

    def cnot(self, control: Union[Qubit, List[Qubit]], target_qubit: Qubit):
        cn = cnot()
        cn.target = target_qubit
        cn.control = control
        self.add(cn)

    def ECR(self, control: Qubit, target: Qubit):
        ecr = ECR()
        ecr.control = control
        ecr.target = target
        self.add(ecr)

    def repeat(self, repeat_value: int, repetition_period=None):
        rep = repeat()
        rep.repeat_value = repeat_value
        rep.repetition_period = repetition_period
        self.add(rep)

    def assign(self, name, value):
        ass = assign()
        ass.name = name
        ass.value = value

    def returns(self, variables=None):
        ret = returns()
        ret.variables = variables
        self.add(variables)

    def reset(self, qubits):
        res = reset()
        res.qubits = qubits
        self.add(qubits)

    def device_assign(self, target, attribute, value):
        """
        Special node that allows manipulation of device attributes during execution.
        """
        da = device_assign()
        da.target = target
        da.attribute = attribute
        da.value = value
        self.add(da)

    def create_label(self, name=None):
        """
        Creates and returns a label. Generates a non-clashing name if none is provided.
        """
        if name is None:
            name = Label.generate_name(self.existing_names)
        elif name in self.existing_names:
            new_name = Label.generate_name(self.existing_names)

        return Label(name)