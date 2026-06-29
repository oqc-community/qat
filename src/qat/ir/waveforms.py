# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2026 Oxford Quantum Circuits Ltd
"""Contains elementary IR units for waveforms, including waveform types and pulses."""

from __future__ import annotations

import numpy as np
from pydantic import Field, PositiveFloat, model_validator

from qat.ir.instructions import QuantumInstruction
from qat.utils.pydantic import (
    AllowExtraFieldsModel,
    ComplexNDArray,
    FloatNDArray,
    find_all_subclasses,
)
from qat.utils.waveform import (
    BlackmanFunction,
    ComplexFunction,
    Cos,
    DragGaussianFunction,
    ExtraSoftSquareFunction,
    GaussianFunction,
    GaussianSquareFunction,
    GaussianZeroEdgeFunction,
    RoundedSquareFunction,
    SechFunction,
    SetupHoldFunction,
    Sin,
    SofterGaussianFunction,
    SofterSquareFunction,
    SoftSquareFunction,
    SquareFunction,
)


class AbstractWaveform(AllowExtraFieldsModel):
    @classmethod
    def name(cls) -> str:
        return cls.__name__.replace("Waveform", "")


class Waveform(AbstractWaveform):
    """A time-dependent complex signal with a specific shape.

    The envelope shape of the waveform will be calibrated to
    implement desired operations on the qubit such as gates or
    readout. The :class:`Waveform` class stores the attributes
    which define the envelope shape. Calling :meth:`sample`
    (or the convenience wrapper :func:`sample_waveform`)
    converts the waveform into discrete in-phase (I) and
    quadrature (Q) samples — see :meth:`sample` for the
    mathematical formula.

    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
        When the hardware channel is correctly calibrated
        (via ``PulseChannel.scale``), constant ``amp`` drives Rabi
        oscillations at ``amp`` Hz.
    :param phase: Phase rotation applied to the waveform in
        radians. Combined with the external ``phase_offset``
        to rotate the IQ vector in the complex plane.
    :param drag: A coefficient which scales a quadrature
        component to the waveform envelope consisting of the
        time derivative of the in-phase component of the
        waveform envelope, see for example
        `<https://arxiv.org/abs/0901.0534>`_.
    :param scale_factor: Additional multiplicative scaling
        applied independently of ``amp``, defaults to ``1.0``.

    See :meth:`sample` for the full mathematical formula.
    """

    width: float = Field(ge=0, default=0)
    amp: float | complex = 0.0
    phase: float | complex = 0.0

    drag: float = 0.0
    rise: float = 0.0
    amp_setup: float = 0.0
    scale_factor: float | complex = 1.0
    zero_at_edges: bool = False
    beta: float = 0.0
    frequency: float = 0.0
    internal_phase: float = 0.0
    std_dev: float = 0.0
    square_width: float = 0.0

    shape_function_type: type[ComplexFunction] | None = None

    @property
    def duration(self):
        return self.width

    def __repr__(self):
        return f"{self.__class__.__name__}(width={self.width}, amp={self.amp}, phase={self.phase})"

    def sample(self, t: np.ndarray, phase_offset: float = 0.0) -> SampledWaveform:
        r"""Evaluate the waveform at discrete times *t* and return a
        :class:`SampledWaveform`.

        The output samples are computed as:

        .. math::

           s(t) = \text{scale\_factor} \cdot \text{amp}
           \cdot e^{\,i\,(\text{phase} + \text{phase\_offset})}
           \cdot \bigl[\,f(t)
           + i \cdot \text{drag} \cdot f'(t)\,\bigr]

        where :math:`f(t)` is the shape function determined by the
        subclass (via ``shape_function_type``).

        Correct calibration of the hardware channel
        (``PulseChannel.scale``) means that using constant amplitude
        :attr:`amp` will drive the qubit in Rabi oscillations
        of ``amp`` Hz.  For example setting only ``amp=10e6``
        and ``PulseChannel.scale=1e-8`` means
        ``10e6 * 1e-8 = 0.1`` peak amplitude of the waveform
        sent to the control hardware.

        :param t: 1-D array of discrete sample times in seconds.
        :param phase_offset: Accumulated channel phase in radians,
            passed by the compiler at sample time. Added to
            :attr:`phase` before computing the complex exponential.
            Defaults to ``0.0``.
        :returns: A :class:`SampledWaveform` containing one complex
            sample per element of *t*.
        :raises AttributeError: If :attr:`shape_function_type` is
            ``None``.
        """
        if self.shape_function_type is None:
            raise AttributeError(
                f"Waveform of type `{self.__class__.__name__}` cannot be evaluated, please provide a valid shape function type."
            )

        # Generate a shape function based on the given type. The shape function
        # will only use whatever member attributes from this class it needs.
        shape_function = self.shape_function_type(**self.model_dump())

        amplitude = shape_function(t)
        phase_offset = self.phase + phase_offset
        samples = self.scale_factor * self.amp * np.exp(1.0j * phase_offset) * amplitude

        if self.drag:
            amplitude_differential = shape_function.derivative(t, amplitude)
            if len(amplitude_differential) < len(samples):
                amplitude_differential = np.pad(
                    amplitude_differential,
                    (0, len(samples) - len(amplitude_differential)),
                    "edge",
                )
            samples += (
                self.drag
                * 1.0j
                * self.amp
                * self.scale_factor
                * np.exp(1.0j * phase_offset)
                * amplitude_differential
            )

        return SampledWaveform(samples=samples)

    def __hash__(self):
        return hash(
            (
                self.__class__.__name__,
                self.width,
                self.amp,
                self.phase,
                self.drag,
                self.rise,
                self.amp_setup,
                self.scale_factor,
                self.zero_at_edges,
                self.beta,
                self.frequency,
                self.internal_phase,
                self.std_dev,
                self.square_width,
            )
        )


class SampledWaveform(AbstractWaveform):
    """A waveform defined by an arbitrary complex array rather than an analytic function.

    Use this class when the pulse envelope is provided as pre-computed
    samples, e.g. from an optimisation routine or when the compiler
    lowers an analytical :class:`Waveform` into discrete samples for
    hardware that does not support analytic descriptions natively.

    :param samples: Complex amplitude values, one per time step.
    :param sample_time: Time between consecutive samples in seconds.
        Required to compute :attr:`duration`.
    """

    samples: ComplexNDArray | FloatNDArray
    sample_time: float | None = None  # Time between samples, in seconds

    @property
    def duration(self):
        if self.sample_time is None:
            # TODO: COMPILER-723 -- Do we want to raise an error here, or return NAN or None or 0?
            raise ValueError(
                "Cannot determine duration of SampledWaveform without sample_time being set."
            )
        return self.sample_time * len(self.samples)

    def __repr__(self):
        return "sampled waveform"

    def __eq__(self, other: SampledWaveform):
        return np.array_equal(self.samples, other.samples)


class SquareWaveform(Waveform):
    """A flat (rectangular) pulse.

    The envelope is uniform across the entire ``width``.

    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[SquareFunction] = SquareFunction


class SoftSquareWaveform(Waveform):
    r"""A square pulse with ``tanh``-shaped rise and fall edges.

    The envelope is:

    .. math::

       f(t) = \tfrac{1}{2}\bigl[
       \tanh\!\bigl(\tfrac{t + (w-r)/2}{r}\bigr)
       - \tanh\!\bigl(\tfrac{t - (w-r)/2}{r}\bigr)
       \bigr]

    where :math:`w` = ``width`` and :math:`r` = ``rise``.

    Edge sharpness depends on ``rise``; very small values can
    introduce discontinuities at the truncation boundary.

    :param rise: Scale of the ``tanh`` transition in seconds.
        Larger values produce a more gradual edge; smaller values
        approach a sharp step.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[SoftSquareFunction] = SoftSquareFunction


class SofterSquareWaveform(Waveform):
    r"""A normalised double-``tanh`` square pulse with extra edge softening.

    The raw envelope is:

    .. math::

       g(t) = \tanh\!\bigl(\tfrac{t + \sigma/2 - r}{r}\bigr)
            - \tanh\!\bigl(\tfrac{t - \sigma/2 + r}{r}\bigr)

    which is then min/max normalised to :math:`[0, 1]`.
    Here :math:`\sigma` = ``std_dev`` and :math:`r` = ``rise``.

    :param std_dev: Width of the flat-top region in seconds. The
        ``tanh`` transitions are placed at ``±std_dev / 2``.
    :param rise: Edge rise/fall scale in seconds. The ``tanh``
        transitions are shifted inward by one ``rise`` step on
        each side.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[SofterSquareFunction] = SofterSquareFunction


class ExtraSoftSquareWaveform(Waveform):
    r"""A heavily softened normalised square pulse with a ``2×rise`` inset.

    The raw envelope is:

    .. math::

       g(t) = \tanh\!\bigl(\tfrac{t + \sigma/2 - 2r}{r}\bigr)
            - \tanh\!\bigl(\tfrac{t - \sigma/2 + 2r}{r}\bigr)

    which is then min/max normalised to :math:`[0, 1]`.
    Identical to :class:`SofterSquareWaveform` but with a
    ``2×rise`` inset instead of ``1×rise``.

    :param std_dev: Width of the flat-top region in seconds. The
        ``tanh`` transitions are placed at ``±std_dev / 2``.
    :param rise: Edge rise/fall scale in seconds. A larger value
        relative to ``std_dev`` causes the flat-top to shrink.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[ExtraSoftSquareFunction] = ExtraSoftSquareFunction


class GaussianWaveform(Waveform):
    r"""A standard Gaussian envelope pulse.

    The envelope is:

    .. math::

       f(t) = e^{-(t / k)^{2}}

    where :math:`k = \text{width} \times \text{rise}`.

    :param rise: Dimensionless shape parameter. A larger ``rise``
        spreads the Gaussian; a smaller ``rise`` narrows it.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[GaussianFunction] = GaussianFunction


class SofterGaussianWaveform(Waveform):
    r"""A Gaussian envelope normalised so the minimum is zero and peak is one.

    Uses the same underlying
    :class:`~qat.utils.waveform.GaussianFunction` as
    :class:`GaussianWaveform` (with
    :math:`k = \text{width} \times \text{rise}`) but subtracts the
    edge value and rescales, ensuring the pulse is exactly zero at
    the edges and peaks at 1.

    :param rise: Dimensionless shape parameter.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[SofterGaussianFunction] = SofterGaussianFunction


class BlackmanWaveform(Waveform):
    r"""A Blackman-window shaped pulse.

    The envelope is:

    .. math::

       f(t) = a_0 - a_1\cos\!\bigl(2\pi(t/w + 0.5)\bigr)
            + a_2\cos\!\bigl(4\pi(t/w + 0.5)\bigr)

    with :math:`a_0 = 7938/18608`, :math:`a_1 = 9240/18608`,
    :math:`a_2 = 1430/18608`, and :math:`w` = ``width``.
    These are the *exact Blackman* coefficients that minimise
    the sidelobe level.

    The Blackman window offers excellent spectral leakage
    suppression. Only ``width`` (inherited) controls the shape.
    See original paper `Blackman & Tukey (1958)
    <https://doi.org/10.1002/j.1538-7305.1958.tb03874.x>`_.

    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[BlackmanFunction] = BlackmanFunction


class SetupHoldWaveform(Waveform):
    """A two-level rectangular pulse: a high-amplitude *setup* portion
    followed by a lower-amplitude *hold* portion.

    The setup section occupies the first ``rise`` seconds of the pulse;
    the hold section occupies the remainder (``width - rise`` seconds).

    :param rise: Duration of the high-amplitude setup section in
        seconds.
    :param amp_setup: Amplitude of the setup section, dimensionless
        (same scale as ``amp``). The hold section uses ``amp`` directly.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[SetupHoldFunction] = SetupHoldFunction


class RoundedSquareWaveform(Waveform):
    r"""A square pulse with ``erf``-shaped (S-curve) rise and fall.

    The envelope uses a pair of error-function steps:

    .. math::

       S(v) = \tfrac{1}{2}\bigl[\operatorname{erf}(v) + 1\bigr]

    .. math::

       f(t) = c\,\bigl[S\!\bigl(\tfrac{t - x_1}{r}\bigr)
            + S\!\bigl(\tfrac{-(t - x_2)}{r}\bigr) - 1\bigr]

    where :math:`x_1 = (w - \sigma)/2`,
    :math:`x_2 = (w + \sigma)/2`, :math:`r` = ``rise``,
    :math:`w` = ``width``, and :math:`c` is a normalisation
    constant.

    :param rise: Steepness of the ``erf`` transition in seconds.
        Smaller values give a sharper step; larger values give a
        more gradual transition.
    :param std_dev: Controls the flat-top width relative to the
        total ``width``, in seconds.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[RoundedSquareFunction] = RoundedSquareFunction


class GaussianSquareWaveform(Waveform):
    r"""A flat-top pulse with Gaussian-shaped rise and fall flanks.

    The envelope is:

    .. math::

       f(t) = \begin{cases}
       e^{-(t + s/2)^{2}/(2\sigma^{2})} & t < -s/2 \\
       1 & |t| \le s/2 \\
       e^{-(t - s/2)^{2}/(2\sigma^{2})} & t > s/2
       \end{cases}

    where :math:`s` = ``square_width`` and :math:`\sigma` =
    ``std_dev``. When ``zero_at_edges`` is ``True`` the envelope
    is offset and rescaled to be zero at the outermost samples.
    See `Sheldon et al. (2016)
    <https://doi.org/10.1103/PhysRevA.93.060302>`_ for use in
    cross-resonance gates.

    :param square_width: Duration of the central flat-top section
        in seconds.
    :param std_dev: Standard deviation of the Gaussian flanks in
        seconds.
    :param zero_at_edges: If ``True``, the envelope is offset and
        rescaled to be exactly zero at the outermost sample
        points.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[GaussianSquareFunction] = GaussianSquareFunction


class DragGaussianWaveform(Waveform):
    r"""A complex Gaussian envelope with an in-built first-order DRAG correction.

    The envelope is (when ``zero_at_edges`` is ``False``):

    .. math::

       f(t) = \bigl(1 - i\,\beta\,t / \sigma^{2}\bigr)
              \;e^{-t^{2}/(8\sigma^{2})}

    When ``zero_at_edges`` is ``True`` the Gaussian is offset
    and rescaled so it approaches zero at the pulse boundaries.
    See `<https://arxiv.org/abs/0901.0534>`_ for background on
    the DRAG technique.

    :param std_dev: Standard deviation controlling the width of
        the Gaussian in seconds.
    :param beta: DRAG in-pulse correction coefficient. Controls
        the magnitude of the quadrature (Q) component;
        ``beta = 0`` gives a real-valued Gaussian.
    :param zero_at_edges: If ``True``, subtracts a constant to
        make the envelope approach zero at the pulse edges.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.

    .. note::
        This class implements the DRAG correction inside the
        shape function itself (via the complex envelope). The
        top-level ``drag`` parameter on :class:`Waveform` applies
        an additional derivative-based correction; normally only
        one of the two mechanisms is activated at a time.
    """

    shape_function_type: type[DragGaussianFunction] = DragGaussianFunction


class GaussianZeroEdgeWaveform(Waveform):
    r"""A Gaussian pulse optionally normalised to be exactly zero at its edges.

    The envelope is:

    .. math::

       g(t) = e^{-t^{2} / (2\sigma^{2})}

    When ``zero_at_edges`` is ``True``:

    .. math::

       f(t) = \frac{g(t) - g(w/2)}{1 - g(w/2)}

    ensuring :math:`f(\pm w/2) = 0` and :math:`f(0) = 1`.

    :param std_dev: Standard deviation of the Gaussian in seconds.
    :param zero_at_edges: When ``True``, the Gaussian is offset
        and rescaled so that it is exactly zero at
        ``t = ±width/2`` and peaks at 1. When ``False``, a
        standard Gaussian is returned.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[GaussianZeroEdgeFunction] = GaussianZeroEdgeFunction


class CosWaveform(Waveform):
    r"""A cosine-oscillating envelope.

    .. math::

       f(t) = \cos(2\pi\,f\,t + \varphi)

    where :math:`f` = ``frequency`` and :math:`\varphi` =
    ``internal_phase``.

    :param frequency: Oscillation frequency in Hz.
    :param internal_phase: Phase offset applied inside the cosine
        argument in radians. Shifts the cosine in phase without
        rotating the full IQ vector.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.
    """

    shape_function_type: type[Cos] = Cos


class SinWaveform(Waveform):
    r"""A sine-oscillating envelope.

    .. math::

       f(t) = \sin(2\pi\,f\,t + \varphi)

    where :math:`f` = ``frequency`` and :math:`\varphi` =
    ``internal_phase``.

    :param frequency: Oscillation frequency in Hz.
    :param internal_phase: Internal phase offset in radians.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.

    .. note::
        :class:`SinWaveform` and :class:`CosWaveform` are related
        by a 90° phase shift.
    """

    shape_function_type: type[Sin] = Sin


class SechWaveform(Waveform):
    r"""A hyperbolic-secant (sech) pulse envelope.

    The envelope is:

    .. math::

       f(t) = \operatorname{sech}(t / \sigma)
            = \frac{1}{\cosh(t / \sigma)}

    The sech pulse has the desirable property of being its own
    Fourier transform (up to scaling), making it self-similar in
    time and frequency. See `Rosen & Zener (1932)
    <https://doi.org/10.1103/PhysRev.40.502>`_.

    :param std_dev: Width parameter :math:`\sigma` of the sech
        pulse in seconds.
    :param width: Duration of the waveform in seconds.
    :param amp: Amplitude pre-factor to the envelope in Hz.
    :param phase: Phase rotation in radians.
    :param drag: DRAG derivative scaling coefficient.
    :param scale_factor: Additional multiplicative scaling,
        defaults to ``1.0``.

    .. note::
        The sech pulse does not reach zero at finite times; it
        decays exponentially.
    """

    shape_function_type: type[SechFunction] = SechFunction


waveform_classes = tuple(find_all_subclasses(Waveform) + [SampledWaveform])


class Pulse(QuantumInstruction):
    """Instructs a pulse channel to send a waveform.

    The intention of the waveform (e.g. a drive or measure pulse) can be specified using the
    type.
    """

    ignore_channel_scale: bool = False
    waveform: Waveform | SampledWaveform

    @model_validator(mode="before")
    def validate_duration(cls, data):
        # TODO: Review with COMPILER-723
        if isinstance(data, dict) and isinstance(data.get("waveform"), Waveform):
            data["duration"] = data["waveform"].duration
        return data

    def __repr__(self):
        return f"{self.__class__.__name__} on targets {set(self.targets)} with {self.waveform}."

    def update_duration(self, duration: float, sample_time: float | None = None):
        if isinstance(self.waveform, Waveform):
            self.duration = duration
            self.waveform.width = duration
        elif isinstance(self.waveform, SampledWaveform):
            if sample_time is None:
                sample_time = self.waveform.sample_time
            else:
                self.waveform.sample_time = sample_time

            current_duration = self.waveform.duration
            if current_duration > duration and not np.isclose(current_duration, duration):
                raise NotImplementedError(
                    "Cannot update the duration of a SampledWaveform to a smaller value. "
                    f"{current_duration} > {duration}\n"
                    "This would require removing samples, which is not supported."
                )
            # If the new duration is larger, we can pad the samples with zeros.
            padding = int(np.round((duration - current_duration) / sample_time, 0))
            self.waveform.samples = np.pad(
                self.waveform.samples,
                (0, padding),
                mode="constant",
                constant_values=0,
            )
            self.duration = self.waveform.duration
        else:
            raise ValueError(
                f"{type(self.waveform)} does not support updating duration. "
                "Can only apply with a Waveform or SampledWaveform."
            )

    @property
    def target(self):
        return next(iter(self.targets))

    @property
    def pulse_channel(self):
        return self.target


def sample_waveform(waveform: Waveform, sample_time: PositiveFloat) -> SampledWaveform:
    """Utility function to sample a waveform at a given time per sample (sample rate).

    :param waveform: The analytical waveform to sample.
    :param sample_time: The time between samples, in seconds.
    :return: A SampledWaveform containing the sampled values.
    """

    edge = (waveform.duration - sample_time) / 2.0
    num_samples = int(np.ceil(waveform.duration / sample_time - 1e-10))
    t = np.linspace(start=-edge, stop=edge, num=num_samples)
    samples = waveform.sample(t)
    return SampledWaveform(samples=samples.samples, sample_time=sample_time)
