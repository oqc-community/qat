# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import ClassVar, Generic

from xdsl.dialects.builtin import (
    AnyFloat,
    BoolAttr,
    ComplexType,
    FloatAttr,
    IntegerType,
    StringAttr,
    i1,
)
from xdsl.interfaces import HasFolderInterface
from xdsl.irdl import (
    AnyOf,
    AtLeast,
    Attribute,
    BaseAttr,
    IRDLOperation,
    Operation,
    RangeOf,
    SSAValue,
    VarConstraint,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import Commutative, ConstantLike, Pure
from xdsl.utils.exceptions import VerifyException

from qat.ir.waveforms import (
    BlackmanWaveform,
    CosWaveform,
    DragGaussianWaveform,
    ExtraSoftSquareWaveform,
    GaussianSquareWaveform,
    GaussianWaveform,
    GaussianZeroEdgeWaveform,
    RoundedSquareWaveform,
    SechWaveform,
    SetupHoldWaveform,
    SinWaveform,
    SofterGaussianWaveform,
    SofterSquareWaveform,
    SoftSquareWaveform,
    SquareWaveform,
)

from .attributes import (
    AmplitudeAttr,
    DiscriminatorPolicyAttr,
    EqualiseAttr,
    FrequencyAttr,
    PhaseAttr,
    PulseNumericTypedAttr,
    SampledWaveformAttr,
    StateMapDictAttr,
    TimeAttr,
    WeightsAttr,
)
from .interfaces import IsAnalyticalWaveformInterface, extract_constant_scalar
from .traits import (
    AdvancesTimeTrait,
    FrameCanonicalizationPatternsTrait,
    PulseTypesCanonicalizationPatternsTrait,
)
from .types import (
    PULSE_VAR_TYPE,
    AcquisitionType,
    AmplitudeType,
    FrameType,
    FrequencyType,
    IQResultType,
    PhaseType,
    StateKeyType,
    TimeType,
    WaveformType,
)

_CONSTANT_OP_TYPES = (FrequencyType, PhaseType, TimeType, AmplitudeType, WaveformType)
_CONSTANT_OP_ATTRS = (
    FrequencyAttr,
    PhaseAttr,
    TimeAttr,
    AmplitudeAttr,
    SampledWaveformAttr,
)
_ARITH_OP_TYPES = _CONSTANT_OP_TYPES + (AcquisitionType,)


@irdl_op_definition
class ConstantOp(IRDLOperation, HasFolderInterface, Generic[PULSE_VAR_TYPE]):
    """Represents a constant value of a given type. This is used to represent constant
    frequencies, phases, durations, amplitudes and waveforms in the IR.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frequency = pulse.constant<5e9> : !pulse.frequency
    """

    _T: ClassVar = VarConstraint("T", AnyOf(_CONSTANT_OP_TYPES))
    _A: ClassVar = VarConstraint("A", AnyOf(_CONSTANT_OP_ATTRS))

    name = "pulse.constant"
    traits = traits_def(ConstantLike(), Pure())
    value = prop_def(_A)
    result = result_def(_T)

    def __init__(
        self,
        value: PulseNumericTypedAttr[PULSE_VAR_TYPE],
        result_type: PULSE_VAR_TYPE | None = None,
    ):
        """
        :param value: The value of the constant, which is a PulseNumericTypedAttr such as
            FrequencyAttr, PhaseAttr, TimeAttr or AmplitudeAttr.
        :param result_type: The type of the result SSA value. If not provided, it will be
            inferred from the type of the value.
        """

        if result_type is None:
            result_type = value.associated_type()

        return super().__init__(
            properties={"value": value},
            result_types=[result_type],
        )

    def fold(self) -> Sequence[SSAValue | Attribute] | None:
        """Returns the constant value, used within constant operation folding."""
        return (self.value,)

    def verify(self):
        """Verifies that the result type is consistent with the attribute provided."""
        super().verify()
        if self.result.type != self.value.associated_type():
            raise VerifyException(
                f"Type of value attribute and type of result must be compatible, but got "
                f"{type(self.value).__name__} and {self.result.type}."
            )


class BinaryOp(IRDLOperation, ABC):
    """Abstract base class for binary operations in the pulse dialect.

    This is used to represent operations that take two operands of the same type and produce
    a result of the same type, such as addition and multiplication.
    """

    @property
    @abstractmethod
    def lhs(self) -> SSAValue: ...

    @property
    @abstractmethod
    def rhs(self) -> SSAValue: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @staticmethod
    @abstractmethod
    def py_operation(lhs, rhs):
        """Hooks in the associated Python operation to be used as part of transforms and
        optimizations."""
        ...


class InternalBinaryOp(BinaryOp, Generic[PULSE_VAR_TYPE], ABC):
    """Abstract base class for operations that take two operands of a type within the pulse
    dialect and return a result of the same type, such as addition and subtraction."""

    lhs = operand_def(AnyOf(_ARITH_OP_TYPES))
    rhs = operand_def(AnyOf(_ARITH_OP_TYPES))
    result = result_def(AnyOf(_ARITH_OP_TYPES))

    def __init__(
        self,
        lhs: SSAValue[PULSE_VAR_TYPE] | Operation,
        rhs: SSAValue[PULSE_VAR_TYPE] | Operation,
        result_type: PULSE_VAR_TYPE,
    ):
        """
        :param lhs: The left-hand side operand of the binary operation, which must be of a
            type within the pulse dialect.
        :param rhs: The right-hand side operand of the binary operation, which must be of a
            type within the pulse dialect.
        :param result_type: The type of the result SSA value, which must be the same as the
            types of the operands.
        """
        return super().__init__(operands=[lhs, rhs], result_types=[result_type])

    def verify(self):
        """Ensures that the types of the operands and result are consistent."""

        super().verify()

        if self.lhs.type != self.rhs.type:
            raise VerifyException(
                f"Types of lhs and rhs must be the same, but got {self.lhs.type} and "
                f"{self.rhs.type}."
            )

        if self.lhs.type != self.result.type:
            raise VerifyException(
                f"Type of result must be the same as type of operands, but got "
                f"{self.result.type} and {self.lhs.type}."
            )


@irdl_op_definition
class AddOp(InternalBinaryOp[PULSE_VAR_TYPE], Generic[PULSE_VAR_TYPE]):
    """Represents addition of two values of the same type, including addition of
    frequencies, phases, durations, amplitudes and waveforms.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frequency1 = pulse.constant<5e9> : !pulse.frequency
        %frequency2 = pulse.constant<1e9> : !pulse.frequency
        %result = pulse.add(%frequency1, %frequency2) : !pulse.frequency

    :ivar lhs: The left-hand side operand of the addition operation.
    :ivar rhs: The right-hand side operand of the addition operation.
    :ivar result: The SSA value representing the result of the addition operation, which can
        be used as an operand in later operations.
    """

    name = "pulse.add"
    traits = traits_def(Pure(), Commutative(), PulseTypesCanonicalizationPatternsTrait())

    @staticmethod
    def py_operation(lhs, rhs):
        """Performs the addition operation on given literals.

        This is used for constant folding.
        """
        return lhs + rhs


@irdl_op_definition
class SubOp(InternalBinaryOp[PULSE_VAR_TYPE], Generic[PULSE_VAR_TYPE]):
    """Represents subtraction of two values of the same types, including frequencies,
    phases, durations, amplitudes and waveforms.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frequency1 = pulse.constant<5e9> : !pulse.frequency
        %frequency2 = pulse.constant<1e9> : !pulse.frequency
        %result = pulse.sub(%frequency1, %frequency2) : !pulse.frequency

    :ivar lhs: The left-hand side operand of the subtraction operation.
    :ivar rhs: The right-hand side operand of the subtraction operation.
    :ivar result: The SSA value representing the result of the subtraction operation, which
        can be used as an operand in later operations.
    """

    name = "pulse.sub"
    traits = traits_def(Pure(), PulseTypesCanonicalizationPatternsTrait())

    @staticmethod
    def py_operation(lhs, rhs):
        """Performs the subtraction operation on given literals.

        This is used for constant folding.
        """
        return lhs - rhs


@irdl_op_definition
class MixOp(InternalBinaryOp[WaveformType]):
    """Represents the element-wise mixing of one waveform envelope with another.

    Mixing two waveforms does a pointwise multiplication of the two waveform envelopes,
    resulting in a new waveform that has the same duration as the input waveforms.

    .. code-block:: mlir

        %duration = pulse.constant<128e-9> : !pulse.time
        %amplitude1 = pulse.constant<0.5> : !pulse.amplitude
        %waveform1 = pulse.square_waveform(%duration, %amplitude1) : !pulse.waveform
        %amplitude2 = pulse.constant<0.25> : !pulse.amplitude
        %waveform2 = pulse.square_waveform(%duration, %amplitude2) : !pulse.waveform
        %result = pulse.mix(%waveform1, %waveform2) : !pulse.waveform

    :ivar lhs: The left-hand side operand of the mixing operation, which is a waveform.
    :ivar rhs: The right-hand side operand of the mixing operation, which is a waveform.
    :ivar result: The SSA value representing the waveform result of the mixing operation,
        which can be used as an operand in later operations.
    """

    name = "pulse.mix"
    traits = traits_def(Pure(), PulseTypesCanonicalizationPatternsTrait())

    lhs = operand_def(WaveformType)
    rhs = operand_def(WaveformType)
    result = result_def(WaveformType)

    def __init__(
        self,
        lhs: SSAValue[WaveformType] | Operation,
        rhs: SSAValue[WaveformType] | Operation,
    ):
        """
        :param lhs: The left-hand side operand of the mixing operation, which is a
            waveform.
        :param rhs: The right-hand side operand of the mixing operation, which is a
            waveform.
        """
        return super().__init__(lhs, rhs, WaveformType())

    @staticmethod
    def py_operation(lhs, rhs):
        """Performs the mixing operation on given literals.

        This is used for constant folding.
        """
        return lhs * rhs


@irdl_op_definition
class ScaleOp(BinaryOp, Generic[PULSE_VAR_TYPE]):
    """Represents the scaling of a type in the pulse dialect by some dimensionless quantity,
    represented by a floating point or complex number. This is used for scaling operations
    that are not necessarily multiplication, e.g. scaling the duration of a waveform by some
    factor.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %duration = pulse.constant<128e-9> : !pulse.time
        %scale = arith.constant<0.5> : !f64
        %result = pulse.scale(%scale, %duration) : !pulse.time

    :ivar lhs: The scaling factor, which is a dimensionless quantity represented by a
        floating point or complex number.
    :ivar rhs: The operand to be scaled, which can be a frequency, phase, duration,
        amplitude or waveform.
    :ivar result: The SSA value representing the result of the scaling operation, which can
        be used as an operand in later operations.
    """

    name = "pulse.scale"
    traits = traits_def(Pure(), PulseTypesCanonicalizationPatternsTrait())

    lhs = operand_def(AnyOf((IntegerType, AnyFloat, ComplexType)))
    rhs = operand_def(AnyOf(_ARITH_OP_TYPES))
    result = result_def(AnyOf(_ARITH_OP_TYPES))

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue[PULSE_VAR_TYPE] | Operation,
        result_type: PULSE_VAR_TYPE,
    ):
        """
        :param lhs: The scaling factor, which is a dimensionless quantity represented by a
            standard type such as integer, float or complex.
        :param rhs: The operand of the operation, which must be of a type within the pulse
            dialect.
        :param result_type: The type of the result SSA value, which must be the same as the
            type of the rhs operand.
        """
        return super().__init__(operands=[lhs, rhs], result_types=[result_type])

    def verify(self):
        """Ensures that the type of the operand and result are consistent."""

        super().verify()
        if self.rhs.type != self.result.type:
            raise VerifyException(
                f"Type of result must be the same as type of operand, but got "
                f"{self.result.type} and {self.rhs.type}."
            )

        if isinstance(self.lhs.type, ComplexType) and self.rhs.type not in (
            AmplitudeType(),
            WaveformType(),
        ):
            raise VerifyException(
                f"Complex scaling is only supported for amplitude and waveform types, but "
                f"got {self.rhs.type}."
            )

    @staticmethod
    def py_operation(lhs, rhs):
        """Performs the scaling operation on given literals.

        This is used for constant folding.
        """
        return lhs * rhs


@irdl_op_definition
class MaxTimeOp(IRDLOperation):
    """Finds the maximum of a variable number of time operands, returning a time result.

    This is used to resolve the maximum duration of a set of operations, which is
    particularly relevant for resolving the duration of synchronizations between multiple
    frames.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %time1 = pulse.constant<128e-9> : !pulse.time
        %time2 = pulse.constant<256e-9> : !pulse.time
        %time3 = pulse.constant<64e-9> : !pulse.time
        %max_time = pulse.max_time(%time1, %time2, %time3) : !pulse.time

    :ivar times: A variable number of SSA values representing time operands, which must all
        be of type pulse.time. At least one operand is required.
    :ivar result: The SSA value representing the maximum of the time operands, which can be
        used as an operand in later operations.
    """

    name = "pulse.max_time"
    traits = traits_def(Pure(), PulseTypesCanonicalizationPatternsTrait())

    times = var_operand_def(RangeOf(TimeType).of_length(AtLeast(1)))
    result = result_def(TimeType)

    def __init__(
        self,
        *times: SSAValue[TimeType] | Operation,
    ):
        """
        :param times: A variable number of SSA values representing time operands, which must
            all be of type pulse.time.
        """
        return super().__init__(operands=[times], result_types=[TimeType()])


@irdl_op_definition
class ModuloOp(InternalBinaryOp[PhaseType]):
    """Represents the modulo operation on two phases.

    Example of how this looks in textual MLIR:

        %phase1 = pulse.constant<3.5> : !pulse.phase
        %phase2 = pulse.constant<1.0> : !pulse.phase
        %result = pulse.modulo(%phase1, %phase2) : !pulse.phase

    :ivar lhs: The left-hand side operand of the modulo operation, which must be phase type.
    :ivar rhs: The right-hand side operand of the modulo operation, which must be phase type.
    """

    name = "pulse.modulo"
    traits = traits_def(Pure(), PulseTypesCanonicalizationPatternsTrait())

    lhs = operand_def(PhaseType)
    rhs = operand_def(PhaseType)
    result = result_def(PhaseType)

    @staticmethod
    def py_operation(lhs, rhs):
        """Performs the modulo operation on given literals.

        This is used for constant folding.
        """
        return lhs % rhs


@irdl_op_definition
class SquareWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """Represents a square waveform, defined by its duration and amplitude.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %duration = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %waveform = pulse.square_waveform(%duration, %amplitude) : !pulse.waveform

    :ivar width: The duration of the square waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the square waveform, represented as a SSA value of
        type pulse.amplitude.
    :ivar result: The SSA value representing the resulting square waveform, which can be
        used as an operand in later operations.
    """

    name = "pulse.square_waveform"
    WAVEFORM_NAME: ClassVar[str] = "square"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    result = result_def(WaveformType)

    def __init__(self, width: SSAValue | Operation, amplitude: SSAValue | Operation):
        """
        :param width: The duration of the square waveform, represented as a SSA value of
            type pulse.time.
        :param amplitude: The amplitude of the square waveform, represented as a SSA value
            of type pulse.amplitude.
        """
        return super().__init__(operands=[width, amplitude], result_types=[WaveformType()])

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        if width is None or amp is None:
            return None
        return SquareWaveform(width=width, amp=amp)


@irdl_op_definition
class SoftSquareWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A square pulse with smooth ``tanh``-shaped rise and fall edges.

    The envelope uses two hyperbolic-tangent steps to produce rounded transitions while
    maintaining a flat top.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %rise = arith.constant<0.1> : !f64
        %waveform = pulse.soft_square_waveform(%width, %amplitude, %rise) : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar rise: Controls the steepness of the tanh transition. Larger values produce a more
        gradual edge; smaller values approach a sharp step.
    :ivar result: The SSA value representing the resulting softened square waveform, which
        can be used as an operand in later operations.
    """

    name = "pulse.soft_square_waveform"
    WAVEFORM_NAME: ClassVar[str] = "soft_square"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    rise = operand_def(AnyFloat)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        rise: SSAValue | Operation,
    ):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        :param rise: The rise of the waveform.
        """
        return super().__init__(
            operands=[width, amplitude, rise], result_types=[WaveformType()]
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        rise = extract_constant_scalar(self.rise)
        if width is None or amp is None or rise is None:
            return None
        return SoftSquareWaveform(width=width, amp=amp, rise=rise)


@irdl_op_definition
class SofterSquareWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A normalised double-tanh square pulse with extra edge softening.

    Similar to :class:`SoftSquareWaveform` but the envelope is normalised so that the peak
    is always one and the edges are pulled further toward zero by offsetting with the rise
    parameter on both sides.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %std_dev = pulse.constant<32e-9> : !pulse.time
        %rise = arith.constant<0.1> : !f64
        %waveform = pulse.softer_square_waveform(%width, %amplitude, %std_dev, %rise)
            : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar std_dev: Half-width parameter governing the flat-top duration.
    :ivar rise: Edge rise/fall scale.  Larger values give a softer slope than
        :class:`SoftSquareWaveform` because the tanh transitions are shifted inward by
        exactly one ``rise`` step on each side.
    :ivar result: The SSA value representing the resulting softened square waveform, which
        can be used as an operand in later operations.
    """

    name = "pulse.softer_square_waveform"
    WAVEFORM_NAME: ClassVar[str] = "softer_square"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    std_dev = operand_def(TimeType)
    rise = operand_def(AnyFloat)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        std_dev: SSAValue | Operation,
        rise: SSAValue | Operation,
    ):
        """
        :param width: The duration of the square waveform, represented as a SSA value of
            type pulse.time.
        :param amplitude: The amplitude of the square waveform, represented as a SSA value
            of type pulse.amplitude.
        :param std_dev: Half-width parameter governing the flat-top duration.
        :param rise: The SSA value representation of the rise parameter.
        """
        return super().__init__(
            operands=[width, amplitude, std_dev, rise], result_types=[WaveformType()]
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        std_dev = extract_constant_scalar(self.std_dev)
        rise = extract_constant_scalar(self.rise)
        if width is None or amp is None or std_dev is None or rise is None:
            return None
        return SofterSquareWaveform(width=width, amp=amp, std_dev=std_dev, rise=rise)


@irdl_op_definition
class ExtraSoftSquareWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A square pulse with heavily softened edges.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %std_dev = pulse.constant<32e-9> : !pulse.time
        %rise = arith.constant<0.1> : !f64
        %waveform = pulse.extra_soft_square_waveform(%width, %amplitude, %std_dev, %rise)
            : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar std_dev: Half-width parameter governing the flat-top duration.
    :ivar rise: Edge rise/fall scale.  Larger values give a softer slope than
        :class:`SofterSquareWaveform` because the tanh transitions are shifted inward by
        more than one ``rise`` step on each side.
    :ivar result: The SSA value representing the resulting softened square waveform, which
        can be used as an operand in later operations.
    """

    name = "pulse.extra_soft_square_waveform"
    WAVEFORM_NAME: ClassVar[str] = "extra_soft_square"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    std_dev = operand_def(TimeType)
    rise = operand_def(AnyFloat)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        std_dev: SSAValue | Operation,
        rise: SSAValue | Operation,
    ):
        """
        :param width: The duration of the square waveform, represented as a SSA value of
            type pulse.time.
        :param amplitude: The amplitude of the square waveform, represented as a SSA value
            of type pulse.amplitude.
        :param std_dev: The SSA value representation of the standard deviation parameter
            governing the flat-top duration.
        :param rise: The SSA value representation of the rise parameter.
        """
        return super().__init__(
            operands=[width, amplitude, std_dev, rise], result_types=[WaveformType()]
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        std_dev = extract_constant_scalar(self.std_dev)
        rise = extract_constant_scalar(self.rise)
        if width is None or amp is None or std_dev is None or rise is None:
            return None
        return ExtraSoftSquareWaveform(width=width, amp=amp, std_dev=std_dev, rise=rise)


@irdl_op_definition
class GaussianSquareWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A square pulse with a Gaussian rise and fall at the edges, i.e a flat-top pulse with
    Gaussian-shaped rise and fall flanks. The envelope is flat (value = 1) over the inner
    square_width and decays as a Gaussian outside that region.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %std_dev = pulse.constant<32e-9> : !pulse.time
        %square_width = pulse.constant<64e-9> : !pulse.time
        %waveform = pulse.gaussian_square_waveform<true>(%width, %amplitude, %std_dev,%square_width)
            : !pulse.waveform


    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar square_width: Duration of the central flat-top section.
    :ivar std_dev: Standard deviation of the Gaussian flanks.  Larger values produce more
        gradual rise/fall; smaller values produce steeper flanks.
    :ivar zero_at_edges: If True, the envelope is offset and rescaled so that it is exactly
        zero at the outermost sample points.
    :ivar result: The SSA value representing the resulting Gaussian-square waveform, which
        can be used as an operand in later operations.
    """

    name = "pulse.gaussian_square_waveform"
    WAVEFORM_NAME: ClassVar[str] = "gaussian_square"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    std_dev = operand_def(TimeType)
    square_width = operand_def(TimeType)
    zero_at_edges = prop_def(BoolAttr)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        std_dev: SSAValue | Operation,
        square_width: SSAValue | Operation,
        zero_at_edges: BoolAttr,
    ):
        return super().__init__(
            operands=[width, amplitude, std_dev, square_width],
            properties={"zero_at_edges": zero_at_edges},
            result_types=[WaveformType()],
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        std_dev = extract_constant_scalar(self.std_dev)
        square_width = extract_constant_scalar(self.square_width)
        if width is None or amp is None or std_dev is None or square_width is None:
            return None
        zero_at_edges = bool(self.zero_at_edges.value.data)
        return GaussianSquareWaveform(
            width=width,
            amp=amp,
            std_dev=std_dev,
            square_width=square_width,
            zero_at_edges=zero_at_edges,
        )


@irdl_op_definition
class GaussianWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """Represents a Gaussian waveform, defined by its duration, amplitude, and standard
    deviation.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %rise = arith.constant<0.1> : !f64
        %waveform = pulse.gaussian_waveform(%width, %amplitude, %rise) : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar rise: Dimensionless shape parameter contributing to the effective width
        k = width * rise.  A larger rise spreads the Gaussian; a smaller rise narrows
        it.
    :ivar result: The SSA value representing the resulting Gaussian waveform, which can be
        used as an operand in later operations.
    """

    name = "pulse.gaussian_waveform"
    WAVEFORM_NAME: ClassVar[str] = "gaussian"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    rise = operand_def(AnyFloat)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        rise: SSAValue | Operation,
    ):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        :param rise: The SSA value representation of the rise parameter.
        """
        return super().__init__(
            operands=[width, amplitude, rise], result_types=[WaveformType()]
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        rise = extract_constant_scalar(self.rise)
        if width is None or amp is None or rise is None:
            return None
        return GaussianWaveform(width=width, amp=amp, rise=rise)


@irdl_op_definition
class SofterGaussianWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A Gaussian envelope normalised so the minimum is zero and peak is one.

    Uses the same underlying :class:`GaussianFunction` as :class:`GaussianWaveform` but
    subtracts the edge value and rescales, ensuring the pulse is exactly zero at
    ±width/2 (approximately) and peaks at 1.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %rise = arith.constant<0.1> : !f64
        %waveform = pulse.softer_gaussian_waveform(%width, %amplitude, %rise)
            : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar rise: Shape parameter contributing to the effective width.
    :ivar result: The SSA value representing the resulting softened Gaussian waveform, which
        can be used as an operand in later operations.
    """

    name = "pulse.softer_gaussian_waveform"
    WAVEFORM_NAME: ClassVar[str] = "softer_gaussian"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    rise = operand_def(AnyFloat)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        rise: SSAValue | Operation,
    ):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        :param rise: The SSA value representation of the rise parameter.
        """
        return super().__init__(
            operands=[width, amplitude, rise], result_types=[WaveformType()]
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        rise = extract_constant_scalar(self.rise)
        if width is None or amp is None or rise is None:
            return None
        return SofterGaussianWaveform(width=width, amp=amp, rise=rise)


@irdl_op_definition
class BlackmanWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A Blackman-window shaped pulse, offering excellent spectral leakage suppression.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir
        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %waveform = pulse.blackman_waveform(%width, %amplitude) : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar result: The SSA value representing the resulting Blackman waveform, which can be
        used as an operand in later operations.
    """

    name = "pulse.blackman_waveform"
    WAVEFORM_NAME: ClassVar[str] = "blackman"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    result = result_def(WaveformType)

    def __init__(self, width: SSAValue | Operation, amplitude: SSAValue | Operation):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        """
        return super().__init__(operands=[width, amplitude], result_types=[WaveformType()])

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        if width is None or amp is None:
            return None
        return BlackmanWaveform(width=width, amp=amp)


@irdl_op_definition
class SetupHoldWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A two-level rectangular pulse with a high-amplitude setup portion followed by a
    lower-amplitude hold portion.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %rise = pulse.constant<32e-9> : !pulse.time
        %amp_setup = pulse.constant<0.5> : !pulse.amplitude
        %waveform = pulse.setup_hold_waveform(%width, %amplitude, %amp_setup, %rise)
            : !pulse.waveform

    :ivar width: The total duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the hold portion of the waveform, represented as a SSA
        value of type pulse.amplitude.
    :ivar amp_setup: The amplitude of the setup portion of the waveform, represented as a
        SSA value of type pulse.amplitude.
    :ivar rise: The duration of the setup portion of the waveform, represented as a SSA
        value of type pulse.time.
    :ivar result: The SSA value representing the resulting setup-hold waveform, which can be
        used as an operand in later operations.
    """

    name = "pulse.setup_hold_waveform"
    WAVEFORM_NAME: ClassVar[str] = "setup_hold"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    amp_setup = operand_def(AmplitudeType)
    rise = operand_def(TimeType)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        amp_setup: SSAValue | Operation,
        rise: SSAValue | Operation,
    ):
        """
        :param width: The total duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the hold portion of the waveform, represented as
            a SSA value of type pulse.amplitude.
        :param amp_setup: The amplitude of the setup portion of the waveform, represented as
            a SSA value of type pulse.amplitude.
        :param rise: The duration of the setup portion of the waveform, represented as a SSA
            value of type pulse.time.
        """
        return super().__init__(
            operands=[width, amplitude, amp_setup, rise], result_types=[WaveformType()]
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        amp_setup = extract_constant_scalar(self.amp_setup)
        rise = extract_constant_scalar(self.rise)
        if width is None or amp is None or amp_setup is None or rise is None:
            return None
        return SetupHoldWaveform(width=width, amp=amp, amp_setup=amp_setup, rise=rise)


@irdl_op_definition
class RoundedSquareWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A square pulse with smooth erf-shaped (S-curve) rise and fall.

    .. code-block:: text

             ____
            /    \
        ___|      |___

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %rise = arith.constant<0.1> : !f64
        %std_dev = pulse.constant<32e-9> : !pulse.time
        %waveform = pulse.rounded_square_waveform(%width, %amplitude, %rise, %std_dev)
        : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar rise: The SSA value representation of the rise parameter.
    :ivar std_dev: The SSA value representation of the standard deviation parameter
        governing the flat-top duration.
    :ivar result: The SSA value representing the resulting rounded square waveform that can
        be used as an operand in later operations.
    """

    name = "pulse.rounded_square_waveform"
    WAVEFORM_NAME: ClassVar[str] = "rounded_square"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    rise = operand_def(AnyFloat)
    std_dev = operand_def(TimeType)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        rise: SSAValue | Operation,
        std_dev: SSAValue | Operation,
    ):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        :param rise: The SSA value representation of the rise parameter.
        :param std_dev: The SSA value representation of the standard deviation parameter
            governing the flat-top duration.
        """
        return super().__init__(
            operands=[width, amplitude, rise, std_dev], result_types=[WaveformType()]
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        rise = extract_constant_scalar(self.rise)
        std_dev = extract_constant_scalar(self.std_dev)
        if width is None or amp is None or rise is None or std_dev is None:
            return None
        return RoundedSquareWaveform(width=width, amp=amp, rise=rise, std_dev=std_dev)


@irdl_op_definition
class DragGaussianWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """Drag Gaussian, tighter on one side and long tail on the other.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %std_dev = pulse.constant<32e-9> : !pulse.time
        %beta = arith.constant<0.1> : !f64
        %waveform = pulse.drag_gaussian_waveform<true>(%width, %amplitude, %std_dev, %beta)
            : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar std_dev: The SSA value representation of the standard deviation parameter
        governing the width of the Gaussian.
    :ivar beta: The SSA value representation of the DRAG coefficient, controlling the
        magnitude of the imaginary component of the waveform.
    :ivar zero_at_edges: A boolean property indicating whether to normalise the envelope to
        zero at its edges.
    :ivar result: The SSA value representing the resulting DRAG Gaussian waveform, which can
        be used as an operand in later operations.
    """

    name = "pulse.drag_gaussian_waveform"
    WAVEFORM_NAME: ClassVar[str] = "drag_gaussian"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    std_dev = operand_def(TimeType)
    beta = operand_def(AnyFloat)
    zero_at_edges = prop_def(BoolAttr)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        std_dev: SSAValue | Operation,
        beta: SSAValue | Operation,
        zero_at_edges: BoolAttr,
    ):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        :param std_dev: The SSA value representation of the standard deviation parameter
            governing the width of the Gaussian.
        :param beta: The SSA value representation of the DRAG coefficient, controlling the
            magnitude of the imaginary component of the waveform.
        :param zero_at_edges: The boolean property indicating whether to normalise the
            envelope to zero at its edges.
        """
        return super().__init__(
            operands=[width, amplitude, std_dev, beta],
            properties={"zero_at_edges": zero_at_edges},
            result_types=[WaveformType()],
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        std_dev = extract_constant_scalar(self.std_dev)
        beta = extract_constant_scalar(self.beta)
        if width is None or amp is None or std_dev is None or beta is None:
            return None
        zero_at_edges = bool(self.zero_at_edges.value.data)
        return DragGaussianWaveform(
            width=width, amp=amp, std_dev=std_dev, beta=beta, zero_at_edges=zero_at_edges
        )


@irdl_op_definition
class GaussianZeroEdgeWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A Gaussian pulse that can be normalized to be zero at the edges.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %std_dev = pulse.constant<32e-9> : !pulse.time
        %waveform = pulse.gaussian_zero_edge_waveform<true>(%width, %amplitude, %std_dev)
            : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar std_dev: The SSA value representation of the standard deviation parameter
        governing the width of the Gaussian.
    :ivar zero_at_edges: A boolean property indicating whether to normalise the envelope to
        zero at its edges.
    :ivar result: The SSA value representing the resulting Gaussian waveform, which can be
        used as an operand in later operations.
    """

    name = "pulse.gaussian_zero_edge_waveform"
    WAVEFORM_NAME: ClassVar[str] = "gaussian_zero_edge"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    std_dev = operand_def(TimeType)
    zero_at_edges = prop_def(BoolAttr)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        std_dev: SSAValue | Operation,
        zero_at_edges: BoolAttr,
    ):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        :param std_dev: The SSA value representation of the standard deviation parameter
            governing the width of the Gaussian.
        :param zero_at_edges: The boolean property indicating whether to normalise the
            envelope to zero at its edges.
        """
        return super().__init__(
            operands=[width, amplitude, std_dev],
            properties={"zero_at_edges": zero_at_edges},
            result_types=[WaveformType()],
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        std_dev = extract_constant_scalar(self.std_dev)
        if width is None or amp is None or std_dev is None:
            return None
        zero_at_edges = bool(self.zero_at_edges.value.data)
        return GaussianZeroEdgeWaveform(
            width=width, amp=amp, std_dev=std_dev, zero_at_edges=zero_at_edges
        )


@irdl_op_definition
class CosWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A cosine-oscillating envelope.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %frequency = pulse.constant<5e9> : !pulse.frequency
        %internal_phase = pulse.constant<1.5708> : !pulse.phase
        %waveform = pulse.cos_waveform(%width, %amplitude, %frequency, %internal_phase)
            : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar frequency: The oscillation frequency of the waveform, represented as a SSA value
        of type pulse.frequency.
    :ivar internal_phase: The internal phase offset of the waveform, represented as a SSA
        value of type pulse.phase.
    :ivar result: The SSA value representing the resulting cosine waveform.
    """

    name = "pulse.cos_waveform"
    WAVEFORM_NAME: ClassVar[str] = "cos"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    frequency = operand_def(FrequencyType)
    internal_phase = operand_def(PhaseType)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        frequency: SSAValue | Operation,
        internal_phase: SSAValue | Operation,
    ):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        :param frequency: The oscillation frequency of the waveform, represented as a SSA
            value of type pulse.frequency.
        :param internal_phase: The internal phase offset of the waveform, represented as a
            SSA value of type pulse.phase.
        """
        return super().__init__(
            operands=[width, amplitude, frequency, internal_phase],
            result_types=[WaveformType()],
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        frequency = extract_constant_scalar(self.frequency)
        internal_phase = extract_constant_scalar(self.internal_phase)
        if width is None or amp is None or frequency is None or internal_phase is None:
            return None
        return CosWaveform(
            width=width, amp=amp, frequency=frequency, internal_phase=internal_phase
        )


@irdl_op_definition
class SinWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A sine-oscillating envelope.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %frequency = pulse.constant<5e9> : !pulse.frequency
        %internal_phase = pulse.constant<1.5708> : !pulse.phase
        %waveform = pulse.sin_waveform(%width, %amplitude, %frequency, %internal_phase)
            : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar frequency: The oscillation frequency of the waveform, represented as a SSA value
        of type pulse.frequency.
    :ivar internal_phase: The internal phase offset of the waveform, represented as a SSA
        value of type pulse.phase.
    :ivar result: The SSA value representing the resulting sine waveform.
    """

    name = "pulse.sin_waveform"
    WAVEFORM_NAME: ClassVar[str] = "sin"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    frequency = operand_def(FrequencyType)
    internal_phase = operand_def(PhaseType)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        frequency: SSAValue | Operation,
        internal_phase: SSAValue | Operation,
    ):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        :param frequency: The oscillation frequency of the waveform, represented as a SSA
            value of type pulse.frequency.
        :param internal_phase: The internal phase offset of the waveform, represented as a
            SSA value of type pulse.phase.
        """
        return super().__init__(
            operands=[width, amplitude, frequency, internal_phase],
            result_types=[WaveformType()],
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        frequency = extract_constant_scalar(self.frequency)
        internal_phase = extract_constant_scalar(self.internal_phase)
        if width is None or amp is None or frequency is None or internal_phase is None:
            return None
        return SinWaveform(
            width=width, amp=amp, frequency=frequency, internal_phase=internal_phase
        )


@irdl_op_definition
class SechWaveformOp(IRDLOperation, IsAnalyticalWaveformInterface):
    """A hyperbolic-secant (sech) pulse envelope.

    Implements a sech pulse defined by sech(x / width). Note that it is not normalized to be
    zero at the edges. The sech pulse has the desirable property of being its own Fourier
    transform (up to scaling), making it self-similar in time and frequency.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %width = pulse.constant<128e-9> : !pulse.time
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %std_dev = pulse.constant<32e-9> : !pulse.time
        %waveform = pulse.sech_waveform(%width, %amplitude, %std_dev) : !pulse.waveform

    :ivar width: The duration of the waveform, represented as a SSA value of type
        pulse.time.
    :ivar amplitude: The amplitude of the waveform, represented as a SSA value of type
        pulse.amplitude.
    :ivar std_dev: The SSA value representation of the width parameter `sigma` of the sech
        pulse.
    :ivar result: The SSA value representing the resulting sech waveform, which can be
        used as an operand in later operations.
    """

    name = "pulse.sech_waveform"
    WAVEFORM_NAME: ClassVar[str] = "sech"

    width = operand_def(TimeType)
    amplitude = operand_def(AmplitudeType)
    std_dev = operand_def(TimeType)
    result = result_def(WaveformType)

    def __init__(
        self,
        width: SSAValue | Operation,
        amplitude: SSAValue | Operation,
        std_dev: SSAValue | Operation,
    ):
        """
        :param width: The duration of the waveform, represented as a SSA value of type
            pulse.time.
        :param amplitude: The amplitude of the waveform, represented as a SSA value of type
            pulse.amplitude.
        :param std_dev: The SSA value representation of the width parameter `sigma` of the
            sech pulse.
        """
        return super().__init__(
            operands=[width, amplitude, std_dev], result_types=[WaveformType()]
        )

    def build_waveform(self):
        width = extract_constant_scalar(self.width)
        amp = extract_constant_scalar(self.amplitude)
        std_dev = extract_constant_scalar(self.std_dev)
        if width is None or amp is None or std_dev is None:
            return None
        return SechWaveform(width=width, amp=amp, std_dev=std_dev)


@irdl_op_definition
class CreateFrameOp(IRDLOperation):
    """Creates a frame, which is a medium for waveforms to be played at a given frequency,
    and tracks any phase manipulations.

    Frames are associated with a port that the pulses will be played on. They
    can have many-to-one association, allowing multiple frames to act concurrently on a
    single port.

    They are defined by a static frequency, and optionally take attributes associated with
    the control hardware calibrated for that frame.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) : !pulse.frame<"channel_1">

    :ivar frequency: The frequency of the frame.
    :ivar imbalance: An optional attribute that stores the imbalance between I and Q paths,
        obtained from mixer calibrations.
    :ivar phase_offset: An optional attribute that stores the phase offset between I and Q
        paths, obtained from mixer calibrations.
    :ivar acquire_allowed: An optional boolean attribute that states if the frame is allowed
        to do acquisitions. This annotation is motivated by the fact not all IO channels
        might allow acquisition, and also simplifies allocation logic. Defaults to True.
    :ivar pulse_allowed: An optional boolean attribute that states if the frame is allowed
        to play pulses. This annotation is motivated by optimization logic for allocation
        on the hardware. Defaults to True.
    :ivar track_phase: An optional boolean attribute that states if phase strictly needs to
        be tracked when frame swapping on hardware. If False, this highly simplifies
        allocation logic, allowing us to make more efficient use of hardware. This should
        be used carefully. Defaults to True.
    :ivar port: The string attribute containing the port identifier.
    :ivar result: The SSA value representing the Frame. Can only be consumed by a single
        operation.
    """

    name = "pulse.create_frame"

    frequency = operand_def(FrequencyType)
    imbalance = opt_attr_def(FloatAttr)
    phase_offset = opt_attr_def(FloatAttr)
    acquire_allowed = attr_def(BoolAttr, default_value=BoolAttr(True, value_type=1))
    pulse_allowed = attr_def(BoolAttr, default_value=BoolAttr(True, value_type=1))
    track_phase = attr_def(BoolAttr, default_value=BoolAttr(True, value_type=1))

    result = result_def(FrameType)

    def __init__(
        self,
        frequency: SSAValue | Operation,
        port: StringAttr,
        imbalance: FloatAttr | None = None,
        phase_offset: FloatAttr | None = None,
        acquire_allowed: BoolAttr | None = None,
        pulse_allowed: BoolAttr | None = None,
        track_phase: BoolAttr | None = None,
    ):
        """
        :param frequency: The SSA value representing the frequency of the frame.
        :param port: The string attribute containing the port identifier.
        :param imbalance: The float attribute representing the imbalance between I
            and Q paths, obtained from mixer calibrations. Optional.
        :param phase_offset: The float attribute representing the phase offset between I
            and Q paths, obtained from mixer calibrations. Optional.
        :param acquire_allowed: The boolean attribute stating if the frame is allowed to do
            acquisitions. Defaults to an attribute with True.
        :param pulse_allowed: The boolean attribute stating if the frame is allowed to play
            pulses. Defaults to an attribute with True.
        :param track_phase: The boolean attribute stating if phase strictly needs to be
            tracked when frame swapping on hardware. Defaults to an attribute with True.
        """
        attributes = {}
        if imbalance is not None:
            attributes["imbalance"] = imbalance
        if phase_offset is not None:
            attributes["phase_offset"] = phase_offset
        if acquire_allowed is not None:
            attributes["acquire_allowed"] = acquire_allowed
        if pulse_allowed is not None:
            attributes["pulse_allowed"] = pulse_allowed
        if track_phase is not None:
            attributes["track_phase"] = track_phase

        return super().__init__(
            operands=[frequency],
            attributes=attributes,
            result_types=[FrameType(port)],
        )

    @property
    def port(self) -> StringAttr:
        """Returns the port that the frame plays on as a string attribute."""
        return self.result.type.port


class PhaseOp(IRDLOperation, ABC):
    """Abstract base class for operations that manipulate the phase of a frame."""

    frame = operand_def(FrameType)
    phase = operand_def(PhaseType)
    result = result_def(FrameType)

    @property
    @abstractmethod
    def name(self) -> str:
        """To be specified by subclasses to define the operation name in MLIR."""

    def __init__(self, frame: SSAValue | Operation, phase: SSAValue | Operation):
        """
        :param frame: The SSA value representing the frame whose phase is being manipulated.
        :param phase: The SSA value representing the phase operand, which specifies the
            amount by which to manipulate the phase.
        """
        frame_ssa = SSAValue.get(frame, type=FrameType)
        return super().__init__(operands=[frame, phase], result_types=[frame_ssa.type])


@irdl_op_definition
class PhaseShiftOp(PhaseOp):
    """Changes the phase of a frame by a given amount. The resulting phase is relative to
    the current phase of the frame.

    Phase shifts are used to create phase differences in superpositions of quantum states.
    They are how we implement virtual-Z gates.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) : !pulse.frame<"channel_1">
        %phase = pulse.constant<1.5708> : !pulse.phase
        %frame2 = pulse.phase_shift(%frame, %phase) : !pulse.frame<"channel_1">

    :ivar frame: The SSA value representing the frame whose phase is being shifted.
    :ivar phase: The SSA value representing the phase operand, which specifies the amount by
        which to shift the phase.
    :ivar result: The SSA value representing the resulting frame with the shifted phase,
        which can be used as an operand in later operations.
    """

    name = "pulse.phase_shift"
    traits = traits_def(FrameCanonicalizationPatternsTrait())


@irdl_op_definition
class PhaseSetOp(PhaseOp):
    """Resets the accumulated phase of a frame to a given value.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) : !pulse.frame<"channel_1">
        %phase = pulse.constant<1.5708> : !pulse.phase
        %frame2 = pulse.phase_set(%frame, %phase) : !pulse.frame<"channel_1">

    :ivar frame: The SSA value representing the frame whose phase is being set.
    :ivar phase: The SSA value representing the phase operand, which specifies the value to
        which to set the phase.
    :ivar result: The SSA value representing the resulting frame with the set phase, which
        can be used as an operand in later operations.
    """

    name = "pulse.phase_set"


@irdl_op_definition
class WaitOp(IRDLOperation):
    """Progresses time on a given frame by a specified amount, without playing any waveform.

    This is used to ensure waveforms are played at the correct time.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) : !pulse.frame<"channel_1">
        %frame2 = pulse.wait(%frame, %duration) : !pulse.frame<"channel_1">

    .. note::

        In older versions of QAT-IR, this operation was called "Delay".

    :ivar frame: The SSA value representing the frame on which to wait.
    :ivar duration: The SSA value representing the amount of time to wait, of type
        pulse.time.
    :ivar result: The SSA value representing the resulting frame after waiting, which can be
        used as an operand in later operations.
    """

    name = "pulse.wait"
    traits = traits_def(AdvancesTimeTrait(), FrameCanonicalizationPatternsTrait())

    frame = operand_def(FrameType)
    duration = operand_def(TimeType)
    result = result_def(FrameType)

    def __init__(self, frame: SSAValue | Operation, duration: SSAValue | Operation):
        """
        :param frame: The SSA value representing the frame on which to wait.
        :param duration: The SSA value representing the amount of time to wait, of type
            pulse.time.
        """
        frame_ssa = SSAValue.get(frame, type=FrameType)
        return super().__init__(operands=[frame, duration], result_types=[frame_ssa.type])


@irdl_op_definition
class SynchronizeOp(IRDLOperation):
    """Synchronizes a set of frames, ensuring they all progress to the same time.

    This is used to ensure operations on different frames are correctly synchronized in
    time.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frame1 = pulse.create_frame(%frequency1) : !pulse.frame<"channel_1">
        %frame2 = pulse.create_frame(%frequency2) : !pulse.frame<"channel_2">
        %frame3, %frame4 = pulse.sync(%frame1, %frame2)
            : (!pulse.frame<"channel_1">, !pulse.frame<"channel_2">)

    :ivar frames: A list of SSA values representing the frames to be synchronized.
    :ivar result: A list of SSA values representing the resulting synchronized frames, which
        can be used as operands in later operations. The order of the results corresponds to
        the order of the input frames.
    """

    name = "pulse.sync"
    traits = traits_def(AdvancesTimeTrait())

    frames = var_operand_def(FrameType)
    result = var_result_def(FrameType)

    def __init__(self, *frames: SSAValue | Operation):
        """
        :param frames: A variable number of SSA values representing the frames to be
            synchronized.
        """
        frame_types = [SSAValue.get(frame, type=FrameType).type for frame in frames]
        return super().__init__(operands=[frames], result_types=[frame_types])

    def verify(self):
        """Verifies that at least two frames are being synchronized, and that the number of
        results matches the number of operands."""

        super().verify()

        if len(self.frames) < 2:
            raise VerifyException(
                f"At least two frames must be synchronized, but got {len(self.frames)}."
            )


@irdl_op_definition
class PulseOp(IRDLOperation):
    """Represents a pulse, which is a waveform played on a frame at a given frequency, and
    with a given phase.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) : !pulse.frame<"channel_1">
        %duration = arith.constant<128e-9> : !pulse.time
        %amplitude = arith.constant<0.5> : !pulse.amplitude
        %waveform = pulse.square_waveform(%duration, %amplitude) : !pulse.waveform
        %frame2 = pulse.pulse(%frame, %waveform) : !pulse.frame<"channel_1">

    :ivar frame: The SSA value representing the frame on which to play the pulse.
    :ivar waveform: The SSA value representing the waveform to be played, of type
        pulse.waveform.
    :ivar result: The SSA value representing the resulting frame after playing the pulse,
        which can be used as an operand in later operations.
    """

    name = "pulse.pulse"
    traits = traits_def(AdvancesTimeTrait())

    frame = operand_def(FrameType)
    waveform = operand_def(WaveformType)
    result = result_def(FrameType)

    def __init__(self, frame: SSAValue | Operation, waveform: SSAValue | Operation):
        """
        :param frame: The SSA value representing the frame on which to play the pulse.
        :param waveform: The SSA value representing the waveform to be played, of type
            pulse.waveform.
        """
        frame_ssa = SSAValue.get(frame, type=FrameType)
        return super().__init__(operands=[frame, waveform], result_types=[frame_ssa.type])


@irdl_op_definition
class StartContinuousWaveformOp(IRDLOperation):
    """Represents the start of a continuous waveform, which is a waveform that is played
    indefinitely until a corresponding stop operation is reached.

    Example of how this looks in textual MLIR, paired with
    :class:`StopContinuousWaveformOp`:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) : !pulse.frame<"channel_1">
        %amplitude = pulse.constant<0.5> : !pulse.amplitude
        %frame2 = pulse.start_continuous_waveform(%frame, %amplitude) : !pulse.frame<"channel_1">
        %duration = pulse.constant<800e-9> : !pulse.time
        %frame3 = pulse.wait(%frame2, %duration) : !pulse.frame<"channel_1">
        %frame4 = pulse.stop_continuous_waveform(%frame3) : !pulse.frame<"channel_1">

    :ivar frame: The SSA value representing the frame on which to start the continuous
        waveform.
    :ivar amplitude: The SSA value representing the amplitude of the continuous waveform,
        of type pulse.amplitude.
    :ivar result: The SSA value representing the resulting frame after starting the
        continuous waveform, which can be used as an operand in later operations.
    """

    name = "pulse.start_continuous_waveform"

    frame = operand_def(FrameType)
    amplitude = operand_def(AmplitudeType)
    result = result_def(FrameType)

    def __init__(self, frame: SSAValue | Operation, amplitude: SSAValue | Operation):
        """
        :param frame: The SSA value representing the frame on which to start the continuous
            waveform.
        :param amplitude: The SSA value representing the amplitude of the continuous
            waveform, of type pulse.amplitude.
        """
        frame_ssa = SSAValue.get(frame, type=FrameType)
        return super().__init__(operands=[frame, amplitude], result_types=[frame_ssa.type])


@irdl_op_definition
class StopContinuousWaveformOp(IRDLOperation):
    """Represents stopping a continuous waveform, which is a waveform that is played
    indefinitely until a corresponding stop operation is reached. Paired with
    :class:`StartContinuousWaveformOp`.

    :ivar frame: The SSA value representing the frame on which to stop the continuous
        waveform.
    :ivar result: The SSA value representing the resulting frame after stopping the
        continuous waveform, which can be used as an operand in later operations.
    """

    name = "pulse.stop_continuous_waveform"

    frame = operand_def(FrameType)
    result = result_def(FrameType)

    def __init__(self, frame: SSAValue | Operation):
        """
        :param frame: The SSA value representing the frame on which to stop the continuous
            waveform.
        """
        frame_ssa = SSAValue.get(frame, type=FrameType)
        return super().__init__(operands=[frame], result_types=[frame_ssa.type])


@irdl_op_definition
class AcquireOp(IRDLOperation):
    """Represents an acquisition operation, which listens to the waveform input to the
    channel within the reference frame.

    Acquisition is used within qubit readout. Often, the backend can support weighted
    acquisitions, where a custom array of real or complex numbers is used for demodulation.
    This can optionally be attached as an attribute to the acquisition.
    No validation is done to enforce length checks, as weights can be backend-specific.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) : !pulse.frame<"channel_1">
        %duration = pulse.constant<800e-9> : !pulse.time
        %frame_result, %acquire_result = pulse.acquire(%frame, %duration)
            : (!pulse.frame<"channel_1">, !pulse.acquisition)


    :ivar frame: The SSA value representing the frame on which to perform the acquisition.
    :ivar duration: The SSA value representing the duration of the acquisition, of type
        pulse.time.
    :ivar frame_result: The SSA value representing the resulting frame after the
        acquisition, which can be used as an operand in later operations.
    :ivar acquisition_result: The SSA value representing the resulting acquisition obtained
        from the acquisition, which can be used as an operand in later operations.
    :ivar weights: Optional weights attribute for the acquisition.
    """

    name = "pulse.acquire"
    traits = traits_def(AdvancesTimeTrait())

    frame = operand_def(FrameType)
    duration = operand_def(TimeType)
    frame_result = result_def(FrameType)
    acquisition_result = result_def(AcquisitionType)
    weights = opt_attr_def(WeightsAttr)

    def __init__(
        self,
        frame: SSAValue | Operation,
        duration: SSAValue | Operation,
        weights: WeightsAttr | None = None,
    ):
        """
        :param frame: The SSA value representing the frame on which to perform the
            acquisition.
        :param duration: The SSA value representing the duration of the acquisition, of type
            pulse.time.
        :param weights: Optional weights attribute for the acquisition.
        """
        frame_ssa = SSAValue.get(frame, type=FrameType)

        attributes = {} if weights is None else {"weights": weights}

        return super().__init__(
            operands=[frame, duration],
            result_types=[frame_ssa.type, AcquisitionType()],
            attributes=attributes,
        )


@irdl_op_definition
class IntegrateOp(IRDLOperation):
    """Represents the integration of an acquisition result into a single IQ point.

    Example of how this looks in textual MLIR:

    .. code-block:: mlir

        %frame = pulse.create_frame(%frequency) {physical_channel = "channel_1"}
            : !pulse.frame<"output">
        %duration = pulse.constant<800e-9> : !pulse.time
        %frame_result, %acquisition_result = pulse.acquire(%frame, %duration)
            : (!pulse.frame<"output">, !pulse.acquisition)
        %integration_result = pulse.integrate(%acquisition_result) : !pulse.iq_result

    :ivar acquisition: The SSA value representing the acquisition result to be integrated.
    :ivar result: The SSA value representing the resulting IQ result obtained from the
        integration, which can be used as an operand in later operations.
    """

    name = "pulse.integrate"
    traits = traits_def(Pure())

    acquisition = operand_def(AcquisitionType)
    result = result_def(IQResultType)

    def __init__(self, acquisition: SSAValue):
        """
        :param acquisition: The SSA value representing the acquisition result to be
            integrated.
        """
        return super().__init__(operands=[acquisition], result_types=[IQResultType()])


@irdl_op_definition
class EqualiseOp(IRDLOperation):
    """Apply an affine transformation to an IQ result from a readout.

    This is expected to be the first step in the post-processing pipeline of a result, which
    is used to transform the IQ results into a standardized form for state discrimination.

    In superconducting qubit readout, the downconverted IQ signals can be distorted by
    hardware imperfections:

    * **Phase imbalance**: The I and Q channels may not be perfectly orthogonal, leading to
      a rotation of the IQ plane.
    * **Gain imbalance**: The I and Q channels may have different gains due to unequal
      amplifier chains.
    * **DC offsets**: Mixer leakage and biases in the amplifier chains can introduce DC
      offsets in the I and Q channels.

    The result is that the ``(I, Q)`` samples cluster on a distorted, offset ellipse rather
    than a compact point, degrading any downstream discriminator.

    The ``Equalise`` instruction corrects all three imperfections in a single real affine
    transform with calibrated values:

    .. math::

        \\begin{pmatrix} I' \\\\ Q' \\end{pmatrix}
        = A \\begin{pmatrix} I \\\\ Q \\end{pmatrix} + \\begin{pmatrix} b_I \\\\ b_Q \\end{pmatrix}

    where ``A`` is a **real** 2×2 matrix (``transform``) and ``[b_I, b_Q]`` is the real
    offset vector (``offset``).  The output is returned as a complex value ``I' + j Q'``.


    The affine transformation is represented by a property and not an operand because it is
    expected to be a constant value, and not a value that is computed at runtime. This
    allows for more efficient compilation and optimization of the pulse program.

    :ivar value: The SSA value representing the IQ result to be equalized.
    :ivar affine_transform: The :class:`EqualiseAttr` property that defines the
        correction to be applied to the IQ result.
    :ivar result: The SSA value representing the resulting equalized IQ result, which can be
        used as an operand in later operations.
    """

    # TODO: to reach high TRL, we should add canonicalization hooks to this to squash
    # consecutive equalise operations into a single one, and to remove identity transforms.
    # In practice, we're not too concerned about seeing multiple, but if it did somehow
    # happen, our post-processing pipeline would be slowed down, or might fail entirely.
    # COMPILER-1351

    name = "pulse.equalise"
    traits = traits_def(Pure())

    value = operand_def(IQResultType)
    affine_transform = prop_def(EqualiseAttr)
    result = result_def(IQResultType)

    def __init__(
        self,
        value: SSAValue | Operation,
        affine_transform: EqualiseAttr,
    ):
        """
        :param value: The SSA value representing the IQ result to be equalized.
        :param affine_transform: The :class:`EqualiseAttr` property that defines the
            correction to be applied to the IQ result.
        """
        return super().__init__(
            operands=[SSAValue.get(value, type=IQResultType)],
            properties={"affine_transform": affine_transform},
            result_types=[IQResultType()],
        )


@irdl_op_definition
class DiscriminateOp(IRDLOperation):
    """Discriminate equalised values to integer state keys.

    State discrimination is the mechanism of mapping an IQ value into a discrete state key,
    which can be used to classify the qubit state. In the most simple situation, this maps
    to a binary outcome, but in general, can map to many integer keys, with each revealing
    different information about the qubit state, or uncertainty in the qubit state.

    :ivar value: The SSA value representing the IQ value operand subjected to state
        discrimination.
    :ivar policy: The state discrimination policy to apply as an attribute.
    :ivar result: The discriminated integer state result as an SSA value.
    """

    name = "pulse.discriminate"
    traits = traits_def(Pure())

    value = operand_def(IQResultType)
    policy = prop_def(BaseAttr(DiscriminatorPolicyAttr))
    result = result_def(StateKeyType)

    def __init__(
        self, value: SSAValue[IQResultType] | Operation, policy: DiscriminatorPolicyAttr
    ):
        """
        :param value: The IQ value result, or an operation producing the IQ value as a
            result.
        :param policy: The state discrimination policy to be used.
        """
        # Taking the min and max states from the policy structurally enforces that the
        # result type is aligned with the policy.
        min_state, max_state = policy.state_range
        return super().__init__(
            operands=[SSAValue.get(value, type=IQResultType)],
            properties={"policy": policy},
            result_types=[StateKeyType(min_state, max_state)],
        )

    def verify_(self):
        """Verifies that the result state type matches the attached policy."""

        expected_state_range = self.policy.state_range
        if self.result.type.state_range != expected_state_range:
            raise VerifyException(
                "The result state type must match the discrimination policy state range, "
                f"expected {expected_state_range}, got {self.result.type.state_range}."
            )


@irdl_op_definition
class StateMapOp(IRDLOperation):
    """Maps a state key to a binary value.

    The state key is a value determined from state discrimination, which reveals information
    about the qubit state after a readout, but might not directly tell you the exact state
    of the qubit. In the circuit model of quantum computing, we work in the language of
    binary values, which represent the logical states of a given qubit basis. The mapping
    operator acts as a bridge between an arbitrary state discrimination to a binary value.

    :ivar value: The operand that carries the discriminated state type.
    :ivar mapping: The attribute that carries the mapping from discriminated state type to a
        binary value.
    :ivar result: The mapped binary result.
    """

    name = "pulse.state_map"
    traits = traits_def(Pure())

    value = operand_def(StateKeyType)
    mapping = prop_def(StateMapDictAttr)
    result = result_def(i1)

    def __init__(
        self,
        value: SSAValue[StateKeyType] | Operation,
        mapping: StateMapDictAttr | Mapping[int, int],
    ):
        """
        :param value: The SSA value representing the state key to be mapped.
        :param mapping: The state mapping attribute that defines the mapping from state keys
            to binary values.
        """

        if not isinstance(mapping, StateMapDictAttr):
            mapping = StateMapDictAttr(mapping)

        return super().__init__(
            operands=[SSAValue.get(value, type=StateKeyType)],
            properties={"mapping": mapping},
            result_types=[i1],
        )

    def verify_(self):
        """Verifies that each state in the state type is represented in the state map."""

        state_type_min, state_type_max = self.value.type.state_range
        state_map_keys = self.mapping.data.keys()

        if not state_map_keys:
            raise VerifyException(
                f"The state map cannot be empty. Expected a map in the range "
                f"({state_type_min}, {state_type_max})."
            )

        state_map_min, state_map_max = min(state_map_keys), max(state_map_keys)

        if (
            state_map_min != state_type_min
            or state_map_max != state_type_max
            or (state_map_max - state_map_min + 1) != len(state_map_keys)
        ):
            raise VerifyException(
                f"The state map does not contain a mapping for every allowed state, "
                f"expected a map in the range ({state_type_min}, {state_type_max}), got "
                f"mapping keys {tuple(state_map_keys)}."
            )

        for val in self.mapping.data.values():
            if val.data not in (0, 1):
                raise VerifyException(
                    f"State map values must be binary (i1: 0 or 1), but got {val.data}."
                )
