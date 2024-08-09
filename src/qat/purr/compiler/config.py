# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from __future__ import annotations

import inspect
import re
import sys
from enum import Enum, Flag, IntEnum, auto
from typing import List, Optional

from qat.purr.utils.serializer import json_dumps, json_loads


class InlineResultsProcessing(Flag):
    """
    Results transforms applied directly to the read-out value on the QPU. In most
    situations applied post-execution, but can also be interwoven.
    """

    # Raw readout from the QPU. Normally Z-axis values for each shot.
    Raw = auto()

    # Shot results averaged out to get a single 0/1 value for each qubit.
    Binary = auto()

    # Return the values in numpy arrays not Python lists.
    NumpyArrays = auto()

    Experiment = Raw | NumpyArrays
    Program = Binary

    def __repr__(self):
        return self.name


class ResultsFormatting(Flag):
    # Transforms each shot into binary then counts the instances.
    # Example for two qubits: { '00': 15, '01': 2524, '10': 250, '11': 730 }
    BinaryCount = auto()

    # Change results value based on conditions for ease-of-use. Set as a flag because
    # it means that return values from execution may change format unexpectedly, so
    # should have a way to disable it for certain uses.
    DynamicStructureReturn = auto()

    # If your qubit results are lists of binary, squash to one string representation.
    # Changes 1: [1, 0, 0, 1] to 1: '1001'.
    # Only works when Binary format is returned.
    SquashBinaryResultArrays = auto()

    def __repr__(self):
        return self.name


class QuantumResultsFormat:
    def __init__(self):
        self.format: Optional[InlineResultsProcessing] = None
        self.transforms: Optional[ResultsFormatting] = (
            ResultsFormatting.DynamicStructureReturn
        )

    def raw(self) -> QuantumResultsFormat:
        self.format = InlineResultsProcessing.Raw
        return self

    def binary(self) -> QuantumResultsFormat:
        self.format = InlineResultsProcessing.Binary
        return self

    def binary_count(self):
        """
        Returns a count of each instance of measured qubit registers.
        Switches result format to raw.
        """
        self.transforms = (
            ResultsFormatting.BinaryCount | ResultsFormatting.DynamicStructureReturn
        )
        self.format = InlineResultsProcessing.Raw
        return self

    def squash_binary_result_arrays(self):
        """
        Squashes binary result list into a singular bit string. Switches results to
        binary.
        """
        self.transforms = (
            ResultsFormatting.SquashBinaryResultArrays
            | ResultsFormatting.DynamicStructureReturn
        )
        self.format = InlineResultsProcessing.Binary
        return self

    def __contains__(self, other):
        if isinstance(other, ResultsFormatting):
            return self.transforms.__contains__(other)
        elif isinstance(other, InlineResultsProcessing):
            return self.format.__contains__(other)
        return False

    def __or__(self, other):
        if isinstance(other, ResultsFormatting):
            self.transforms = self.transforms.__or__(other)
        elif isinstance(other, InlineResultsProcessing):
            self.format = self.format.__or__(other)
        return self

    def __and__(self, other):
        if isinstance(other, ResultsFormatting):
            self.transforms = self.transforms.__and__(other)
        elif isinstance(other, InlineResultsProcessing):
            self.format = self.format.__and__(other)
        return self

    def __xor__(self, other):
        if isinstance(other, ResultsFormatting):
            self.transforms = self.transforms.__xor__(other)
        elif isinstance(other, InlineResultsProcessing):
            self.format = self.format.__xor__(other)
        return self

    def __repr__(self):
        return f"Format: {str(self.format)}. Transforms: {str(self.transforms)}."

    def __eq__(self, other):
        return self.format == other.format and self.transforms == other.transforms


class TketOptimizations(Flag):
    """Flags for the various Tket optimizations we can apply."""

    Empty = auto()
    DefaultMappingPass = auto()
    FullPeepholeOptimise = auto()
    ContextSimp = auto()
    DirectionalCXGates = auto()
    CliffordSimp = auto()
    DecomposeArbitrarilyControlledGates = auto()
    # EulerAngleReduction = auto()
    GlobalisePhasedX = auto()
    # GuidedPauliSimp = auto()
    KAKDecomposition = auto()
    # OptimisePhaseGadgets = auto()
    # PauliSimp = auto()
    # PauliSquash = auto()
    PeepholeOptimise2Q = auto()
    RemoveDiscarded = auto()
    RemoveBarriers = auto()
    RemoveRedundancies = auto()
    ThreeQubitSquash = auto()
    SimplifyMeasured = auto()

    One = DefaultMappingPass | DirectionalCXGates
    Two = One | FullPeepholeOptimise | ContextSimp


class QiskitOptimizations(Flag):
    """Flags for the various Qiskit optimizations we can apply."""

    Empty = auto()


class QatOptimizations(Flag):
    """Flags for the various Qat optimizations we can apply."""

    Empty = auto()


class MetricsType(Flag):
    Empty = auto()

    # Returns circuit after optimizations have been run.
    OptimizedCircuit = auto()

    # Count of transformed instructions after all forms of optimizations have
    # been performed.
    OptimizedInstructionCount = auto()

    # Set of basic metrics that should be returned at all times.
    Default = OptimizedCircuit | OptimizedInstructionCount

    def is_composite(self):
        """
        Any flags that are only composed of other ones should be signaled here. This
        is used for automatic metric generation and whether to build/validate this
        particular value.
        """
        return self == self.Default or self == self.Empty

    def snake_case_name(self):
        """
        Generate the Python field name that'll be used to hold the results of this
        metric.
        """
        name = self.name
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


class ErrorMitigationConfig(Flag):
    Empty = auto()
    MatrixMitigation = auto()
    LinearMitigation = auto()


class ExperimentalFeatures:
    error_mitigation = ErrorMitigationConfig.Empty


class CompilerConfig:
    """
    Full settings for the compiler. All values are defaulted on initialization.

    If no explicit optimizations are passed then the default set of optimization for the
    language you're attempting to compile will be applied.
    """

    def __init__(
        self,
        repeats=None,
        repetition_period=None,
        results_format: QuantumResultsFormat = None,
        metrics=MetricsType.Default,
        active_calibrations=None,
        optimizations: "OptimizationConfig" = None,
        error_mitigation: ErrorMitigationConfig = None,
    ):
        self.repeats: Optional[int] = repeats
        self.repetition_period: Optional = repetition_period
        self.results_format: QuantumResultsFormat = results_format or QuantumResultsFormat()
        self.metrics: MetricsType = metrics
        self.active_calibrations: List[CalibrationArguments] = active_calibrations or []
        self.optimizations: Optional[OptimizationConfig] = optimizations
        self.error_mitigation: Optional[ErrorMitigationConfig] = error_mitigation

    def to_json(self):
        return json_dumps(self, serializable_types=get_serializable_types())

    def from_json(self, json: str):
        vars(self).update(
            vars(json_loads(json, serializable_types=get_serializable_types()))
        )
        return self

    @classmethod
    def create_from_json(cls, json: str):
        return CompilerConfig().from_json(json)

    def validate(self, hardware):
        from qat.purr.compiler.hardware_models import QuantumHardwareModel

        if (
            self.error_mitigation is not None
            and self.error_mitigation != ErrorMitigationConfig.Empty
        ):
            if isinstance(hardware, QuantumHardwareModel) and (
                hardware.error_mitigation is None
                or not hardware.error_mitigation.readout_mitigation
            ):
                raise ValueError("Error mitigation not calibrated on this device.")
            if ResultsFormatting.BinaryCount not in self.results_format:
                raise ValueError(
                    "Binary Count format required for readout error mitigation"
                )


class CalibrationArguments:
    """Base class for individual calibration arguments."""

    def to_json(self):
        return json_dumps(self)

    def from_json(self, json: str):
        self.from_dict(json_loads(json))

    def _get_field_names(self):
        """Get existing field names for all attributes and properties"""
        existing_field_names = set(vars(self).keys())
        # Include class properties
        class_type = type(self)
        for prop in dir(class_type):
            if isinstance(getattr(class_type, prop), property):
                existing_field_names.add(prop)

        return existing_field_names

    def from_dict(self, dict_values):
        """
        Loads this dictionary into the arguments. Throws if key dosen't exist on the
        object.
        """
        valid_names = self._get_field_names()
        invalid_fields = [val for val in dict_values.keys() if val not in valid_names]
        if any(invalid_fields):
            raise ValueError(
                f"Field(s) {','.join(invalid_fields)} are not valid for "
                f"{self.__class__.__name__}."
            )

        vars(self).update(dict_values)


class Languages(IntEnum):
    Empty, Qasm2, Qasm3, QIR = range(4)

    def __repr__(self):
        return self.name


class OptimizationConfig:
    """
    Base class for instantiated optimizations as well as mix-in classes. Built this way
    so we can mix and match optimization objects across multiple setups and languages
    without duplication.
    """

    def __init__(self):
        super().__init__()

    def default(self):
        """Apply default set of optimizations to the current set."""
        return self

    def disable(self):
        """Disable all optimizations."""
        return self

    def minimum(self):
        """
        Apply minimum working set for current optimizations.
        """
        return self

    def __contains__(self, item):
        return False


class Tket(OptimizationConfig):
    def __init__(self, tket_optimization=None):
        super().__init__()
        self.tket_optimizations: TketOptimizations = TketOptimizations.Empty
        if tket_optimization is not None:
            self.tket_optimizations = tket_optimization

    def default(self):
        self.tket_optimizations = TketOptimizations.One
        return self

    def disable(self):
        self.tket_optimizations = TketOptimizations.Empty
        return self

    def minimum(self):
        self.tket_optimizations = TketOptimizations.DefaultMappingPass
        return self

    def __contains__(self, item):
        if isinstance(item, TketOptimizations) and item in self.tket_optimizations:
            return True
        return super().__contains__(item)


class Qiskit(OptimizationConfig):
    def __init__(self):
        super().__init__()
        self.qiskit_optimizations: QiskitOptimizations = QiskitOptimizations.Empty

    def default(self):
        self.qiskit_optimizations = QiskitOptimizations.Empty
        return self

    def __contains__(self, item):
        if isinstance(item, QiskitOptimizations) and item in self.qiskit_optimizations:
            return True
        return super().__contains__(item)


class Qasm2Optimizations(Tket, Qiskit):
    def __init__(self):
        super().__init__()
        self.default()

    def __repr__(self):
        return f"Qiskit: {self.qiskit_optimizations}. Tket: {self.tket_optimizations}."


class Qasm3Optimizations(OptimizationConfig):
    pass


class QIROptimizations(OptimizationConfig):
    pass


def get_optimizer_config(lang: Languages) -> Optional[OptimizationConfig]:
    """
    Returns the optimization config for this particular language. None if no valid ones
    found.
    """
    if lang == Languages.Qasm2:
        return Qasm2Optimizations()
    elif lang == Languages.Qasm3:
        return Qasm3Optimizations()
    elif lang == Languages.QIR:
        return QIROptimizations()
    return None


def get_config(lang: Languages, **kwargs):
    """
    Helper method to build a compiler config for a particular language. Forwards
    keywords to the CompilerConfig constructor.
    """
    config = CompilerConfig(**kwargs)
    config.optimizations = get_optimizer_config(lang)
    return config


serializable_types_dict = None


def get_serializable_types():
    global serializable_types_dict

    if serializable_types_dict is not None:
        return serializable_types_dict

    serializable_types = {}

    def update_dict(type):
        if issubclass(type, Enum):
            serializable_types.update({f"<enum '{type.__module__}.{type.__name__}'>": type})
        else:
            serializable_types.update({str(type): type})

    def get_serializable_types_dict(type):
        update_dict(type)

        for subclass in type.__subclasses__():
            get_serializable_types_dict(subclass)

    types_in_module = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    for name, typ in types_in_module:
        get_serializable_types_dict(typ)

    serializable_types_dict = serializable_types

    return serializable_types_dict
