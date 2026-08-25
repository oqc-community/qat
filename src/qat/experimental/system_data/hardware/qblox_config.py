# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Configuration dataclasses for the Qblox RF chain.

These immutable dataclasses group the QCoDeS parameters exposed by the Qblox Instruments
driver (see the `Qblox Instruments API reference
<https://docs.qblox.com/en/main/products/qblox_instruments/api/>`_) by the hardware block
they configure. They are a slimmed-down, hardware-view-oriented successor to the
``qat.backend.qblox.config.specification`` models.

Two families of configuration are distinguished:

* :class:`SequencerConfig` subclasses configure the *digital* side of the chain (per
  sequencer): the NCO, the AWG, acquisition and marker/trigger behaviour.
* :class:`GeneratorConfig` subclasses configure the *analogue* side of the chain (per
  generator / output path): mixer correction, pulse-shaping filters and output/input signal
  conditioning.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SequencerConfig:
    """Base class for the digital (per-sequencer) configuration components.

    A sequencer is the digital side of a Qblox module's RF chain. Concrete subclasses group
    the QCoDeS ``sequencer.*`` parameters by the hardware block they configure (NCO, AWG,
    acquisition, markers, ...).
    """


@dataclass(frozen=True, slots=True)
class NcoConfig(SequencerConfig):
    """Configuration components related to the sequencer's NCO.

    :ivar phase_offs: Phase offset of the NCO in degrees with a resolution of 3.6e-7
        degrees.
    :ivar prop_delay_comp: Delay that compensates the NCO phase to the input path with
        respect to the instrument's combined output and input propagation delay. This delays
        the frequency update as well.
    :ivar prop_delay_comp_en: Flag to enable/disable compensation of propagation delay.
    """

    phase_offs: float = 0.0
    prop_delay_comp: int = 0
    prop_delay_comp_en: bool = False


@dataclass(frozen=True, slots=True)
class AwgConfig(SequencerConfig):
    """Configuration components related to the sequencer's AWG.

    :ivar gain_path0: Gain for AWG path 0.
    :ivar gain_path1: Gain for AWG path 1.
    :ivar offset_path0: Offset for AWG path 0.
    :ivar offset_path1: Offset for AWG path 1.
    :ivar mod_en: Flag to enable/disable modulation for AWG.
    """

    gain_path0: float = 0.0
    gain_path1: float = 0.0

    offset_path0: float = 0.0
    offset_path1: float = 0.0

    mod_en: bool = True


@dataclass(frozen=True, slots=True)
class SquareWeightAcq(SequencerConfig):
    """Configuration components for non-weighed acquisition.

    :ivar integration_length: Integration length in number of samples for non-weighed
        acquisitions on paths 0 and 1. Must be a multiple of 4. Default value is 1024.
    """

    integration_length: int = 1024


@dataclass(frozen=True, slots=True)
class ThresholdedAcqConfig(SequencerConfig):
    """Configuration components for thresholded acquisition.

    :ivar rotation: Phase rotation (in degrees) for the integration result.
    :ivar threshold: Threshold for discretizing the phase-rotated result (see
        :attr:`rotation`). Discretization is done by comparing the threshold to the rotated
        integration result of path 0. This comparison is applied before normalization (i.e.
        division) of the rotated value with the integration length and therefore the
        threshold needs to be compensated (i.e. multiplied) with this length for the
        discretization to function properly.
    """

    rotation: float = 1.0
    threshold: float = 0.0


@dataclass(frozen=True, slots=True)
class AcquireConfig(SequencerConfig):
    """Configuration components for the acquisition path of a readout sequencer.

    :ivar auto_bin_incr_en: Flag to enable/disable whether the bin index is automatically
        incremented when acquiring multiple triggers. Disabling the TTL trigger acquisition
        path resets the bin index.
    :ivar demod_en_acq: Flag to enable/disable demodulation on the acquisition path.
    """

    auto_bin_incr_en: bool | None = None
    demod_en_acq: bool | None = None


@dataclass(frozen=True, slots=True)
class MarkerSwitchConfig(SequencerConfig):
    """Configuration for marker overriding on a sequencer.

    When enabled, the override takes priority over the ``set_mrk`` instruction. Used to
    control marker outputs (digital triggers) and RF switches, which need to be enabled
    for RF operation.

    :ivar marker_ovr_en: Flag to enable/disable the marker overriding feature.
    :ivar marker_ovr_value: Marker override value. Its binary representation codifies the
        On/Off flags for the marker channels.
    """

    marker_ovr_en: bool | None = None
    marker_ovr_value: int | None = None


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Base class for the analogue (per-generator) configuration components.

    A generator is a single tone-producing subsection of a module's analogue RF chain.
    Concrete subclasses group the QCoDeS module-level parameters that shape the analogue
    signal on a generator's output/input path (mixer correction, pulse-shaping filters,
    output/input signal conditioning and scope acquisition).
    """


@dataclass(frozen=True, slots=True)
class MixerConfig(GeneratorConfig):
    """Configuration for the generator's mixer correction component.

    Mirrors the QCoDeS ``sequencer.mixer_corr_phase_offset_degree`` and
    ``sequencer.mixer_corr_gain_ratio`` parameters used to correct IQ mixer imbalance on the
    AWG path.

    :ivar phase_offset: Mixer phase imbalance correction for the AWG, in degrees.
    :ivar gain_ratio: Mixer gain imbalance correction for the AWG.
    """

    phase_offset: float = 0.0
    gain_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class PulseShapingConfig(GeneratorConfig):
    """Configuration of the output pulse-shaping filters.

    Groups the finite-impulse-response (FIR) filter and the four exponential-overshoot
    filters applied to a generator's output (see the QCoDeS ``out*_fir_config`` and
    ``out*_exp*_config`` parameters). Each filter accepts ``'bypassed'`` to disable it, or
    ``'delay_comp'`` to bypass it while delaying the output as if it were applied.

    :ivar fir_out: Configuration of the finite-impulse-response filter for the output.
    :ivar exp_overshoot_0_out: Configuration of exponential-overshoot filter 0 for the
        output.
    :ivar exp_overshoot_1_out: Configuration of exponential-overshoot filter 1 for the
        output.
    :ivar exp_overshoot_2_out: Configuration of exponential-overshoot filter 2 for the
        output.
    :ivar exp_overshoot_3_out: Configuration of exponential-overshoot filter 3 for the
        output.
    """

    fir_out: str | None = None
    exp_overshoot_0_out: str | None = None
    exp_overshoot_1_out: str | None = None
    exp_overshoot_2_out: str | None = None
    exp_overshoot_3_out: str | None = None


@dataclass(frozen=True, slots=True)
class OutputSignalConfig(GeneratorConfig):
    """Output signal conditioning for a generator's output path.

    Groups the QCoDeS output attenuation (``out*_att``) and DC offset
    (``out*_offset_path*``) parameters used to level and de-bias the output signal.

    :ivar attenuation: Output attenuation in dB.
    :ivar offset_path_0: Offset (in mV) for output path 0 (I) in QCM-RF/QRM-RF.
    :ivar offset_path_1: Offset (in mV) for output path 1 (Q) in QCM-RF/QRM-RF.
    """

    attenuation: float = 0.0
    offset_path_0: float = 0.0
    offset_path_1: float = 0.0


@dataclass(frozen=True, slots=True)
class InputSignalConfig(GeneratorConfig):
    """Input signal conditioning for a readout generator's input path.

    Groups the QCoDeS input attenuation (``in*_att``), input gain (``in*_gain``) and DC
    offset (``in*_offset_path*``) parameters used to level and de-bias the acquired signal.

    :ivar attenuation: Input attenuation in dB.
    :ivar gain: Input gain in dB.
    :ivar offset_path_0: Offset (in mV) for input path 0 (I) in QRM-RF.
    :ivar offset_path_1: Offset (in mV) for input path 1 (Q) in QRM-RF.
    """

    attenuation: float = 0.0
    gain: float = 0.0
    offset_path_0: float = 0.0
    offset_path_1: float = 0.0


@dataclass(frozen=True, slots=True)
class ScopeAcquireConfig(GeneratorConfig):
    """Scope (trace) acquisition configuration for a readout generator.

    Groups the QCoDeS ``scope_acq_*`` parameters that control the scope acquisition path.

    :ivar select_sequencer_to_scope_memory: Whether this sequencer's acquisition is written
        to the module's scope memory.
    :ivar enable_average_mode: Flag to enable/disable scope acquisition averaging mode.
    """

    select_sequencer_to_scope_memory: bool = True
    enable_average_mode: bool = True
