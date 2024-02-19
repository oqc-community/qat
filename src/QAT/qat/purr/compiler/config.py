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

    # Returns the most probably binary string from your result. Meant to work in conjunction
    # with BinaryCount
    ReturnMostProbable = auto()

    def __repr__(self):
        return self.name


class QuantumResultsFormat:
    """
    There are numerous types of results people want from a quantum computer, from low-level experimentation
    to circuit and quantum program execution. Usually these all start at the raw readout and then progressively get
    simplified and condensed into whatever format you want.

    We mirror this approach and all of our transformations are layered upon one another allowing quite a bit of
    flexibility in the results you get back. This does also mean that some transformations are meant to work together,
    or after one another, or have various conditions that are more subtle.

    To that end we've provided combinations of flags that allow for the most current results formats to be targeted
    in this object.

    Note: binary_count() is a results distribution, and for most circuit runs you'll want to use that.

    Here's an example for results distribution:
    ::
        config = CompilerConfig()
        config.results_format.binary_count()

    And here's for the raw results:
    ::
        config = CompilerConfig()
        config.results_format.raw()
    """

    def __init__(self):
        self.format: Optional[InlineResultsProcessing] = None
        self.transforms: Optional[
            ResultsFormatting
        ] = ResultsFormatting.DynamicStructureReturn

    def raw(self) -> QuantumResultsFormat:
        """Raw QPU output."""
        self.format = InlineResultsProcessing.Raw
        return self

    def binary(self) -> QuantumResultsFormat:
        """
        Returns a definitive 0/1 for each qubit instead of raw measurement results * shots.
        """
        self.format = InlineResultsProcessing.Binary
        return self

    def binary_count(self):
        """
        Returns the count of each time a particular bitstring was seen in a result.
        {'00': 40, '01': 150, ...}
        """
        self.transforms = (
            ResultsFormatting.BinaryCount | ResultsFormatting.DynamicStructureReturn
        )
        self.format = InlineResultsProcessing.Raw
        return self

    def most_probable_bitstring(self):
        """
        Returns the most probable bitstring from your execution. Clusters the results then
        picks the one with the highest incident.
        """
        self.transforms = (
            ResultsFormatting.BinaryCount
            | ResultsFormatting.ReturnMostProbable
            | ResultsFormatting.DynamicStructureReturn
        )
        self.format = InlineResultsProcessing.Raw
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
    """
    A variety of metrics can be returned about compilation and execution. The default metrics are a selection of
    information that people generally want to know such as what the post-optimization circuit/IR looks like before we
    start executing.

    Default metrics will only have a minute impact on runtime, so are fine to leave enabled.
    """

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


class CompilerConfig:
    """
    Main configuration for many compiler passes and features. All defaults are chosen based upon target
    machine and IR used (QASM, QIR, etc) if not explicitly passed in.

    Look at the documentation on each object for more detailed uses.

    Usage Example:
    ::
        conf = CompilerConfig()
        conf.results_format.binary_count()
        conf.optimizations = QIR2Optimizations()
        results = execute(..., ..., conf)

    You can also use `get_default_config` to build a default configuration for a particular language, which
    is usually the easier option.
    ::
        conf = get_default_config(Languages.Qasm2, repeats=5000)

    Note: 'repeats' are more commonly known as shots.
    """

    def __init__(
        self,
        repeats=None,
        repetition_period=None,
        results_format: QuantumResultsFormat = None,
        metrics=MetricsType.Default,
        active_calibrations=None,
        optimizations: "OptimizationConfig" = None,
    ):
        self.repeats: Optional[int] = repeats
        self.repetition_period: Optional = repetition_period
        self.results_format: QuantumResultsFormat = results_format or QuantumResultsFormat()
        self.metrics: MetricsType = metrics
        self.active_calibrations: List[CalibrationArguments] = active_calibrations or []
        self.optimizations: Optional[OptimizationConfig] = optimizations

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
    We have a variety of third-party libraries that can be run before your program reaches our
    main compilation and runtime pipeline. Each of these will have an associated set of flags (like `TketOptimizations`)
    where you can tell us which passes you want run.

    Each flag is associated with a particular run/feature in the various libraries. We try and keep a thin wrapper
    around them to keep maintenance/usage easy. Please look for a flags name in the associated library for specific
    information about that pass. For example, for `TketOptimizations.ThreeQubitSquash` just look up ThreeQubitSquash
    in Tkets documentation.

    There are helper composites for various languages such as `Qasm2Optimizations` and `QIROptimizations`. These
    allow you to set multiple optimizers settings in one object and filter to only the ones that work for that language.

    It's recommended you use these over the raw optimization flags, but you can also use those as well.

    Example:
    ::
        conf = CompilerConfig(optimizations = Qasm2Optimizations())
        conf.optimizations = conf.optimizations
            | TketOptimizations.ThreeQubitSquash
            | TketOptimizations.RemoveDiscarded

    You can also use `get_optimizer_config` to return an optimization config for a particular language.
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


class QIROptimizations(Tket):
    def __init__(self):
        super().__init__()
        self.default()


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


_default_results_format = QuantumResultsFormat()


def default_language_options(lang: Languages, config: CompilerConfig):
    """Applies default language-specific options to an existing configuration."""
    if lang == Languages.Qasm2:
        if config.optimizations is None:
            config.optimizations = Qasm2Optimizations()
        if config.results_format == _default_results_format:
            config.results_format.binary_count()

    elif lang == Languages.Qasm3:
        if config.optimizations is None:
            config.optimizations = Qasm3Optimizations()

    elif lang == Languages.QIR:
        if config.optimizations is None:
            config.optimizations = QIROptimizations()
        if config.results_format == _default_results_format:
            config.results_format.binary_count()


def get_default_config(lang: Languages, **kwargs):
    """
    Helper method to build a compiler config for a particular language. Forwards
    keywords to the CompilerConfig constructor.
    """
    config = CompilerConfig(**kwargs)
    default_language_options(lang, config)
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
