# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Canonical dataclass schema for experimental quantum system data.

This module defines the immutable, schema-first representation used by the experimental
system-data layer to describe calibration metadata, resources, signal paths, qubits,
operations, and related top-level attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True, kw_only=True)
class AttributeEntry:
    """A generic key/value entry carried on canonical records.

    This is an intentional extension point for legacy or target-specific details that are
    not promoted to typed top-level fields.

    .. note:: A tuple of these is preferable to a dict of attributes to preserve
       immutability and ordering, and to allow for duplicate keys if necessary.

    :ivar key: Attribute key.
    :ivar value: Attribute value.
    """

    key: str
    value: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalResourceData:
    """Canonical descriptor for an external hardware resource.

    These resources are typically instruments, controllers, or other backend-facing entities
    that are referenced by signal-path objects. Attributes are intentionally unstructured as
    different backends may have widely varying resource data requirements.

    :ivar id: Resource identifier that remains consistent across references and calibration
        snapshots for the same underlying external device.
    :ivar object_type: Optional descriptive label from the source system. E.g. in a
        Qblox-backed setup this might be values such as ``Cluster``, ``ClusterModule``,
        ``QCM-RF``, ``QRM-RF``, or ``LocalOscillator``.
    :ivar attributes: Additional unstructured metadata.
    """

    id: str
    object_type: str | None = None
    attributes: tuple[AttributeEntry, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class OscillatorData:
    """Canonical oscillator configuration for a signal path.

    An oscillator captures a reusable frequency reference that may be shared by one or more
    channels and optionally links to the external resource that implements it.

    :ivar id: Oscillator identifier.
    :ivar frequency: Oscillator frequency in Hz.
    :ivar external_resource_id: Optional linked external resource identifier.
    """

    id: str
    frequency: int
    external_resource_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PortData:
    """Canonical physical port data with ID-based references.

    A port models a physical input/output connection to one or more quantum devices. It
    captures acquisition capability and timing granularity constraints. ``block_size`` is
    the canonical replacement for the legacy ``samples_per_clock_cycle`` concept.

    :ivar id: Port identifier.
    :ivar sample_time: Sample period in picoseconds.
    :ivar block_size: Hardware block granularity in samples. Together with
        ``sample_time``, this defines the timing granularity for the port.
    :ivar min_blocks: Minimum number of blocks required for a valid operation on this port.
        This is a hardware constraint that may be used to validate operation definitions.
    :ivar max_blocks: Maximum number of blocks allowed for a valid operation on this port.
        This is a hardware constraint that may be used to validate operation definitions,
        or ``-1`` to indicate no maximum.
    :ivar acquire_allowed: Whether acquisition is allowed on this port.
    :ivar external_resource_id: Optional linked external resource identifier.
    """

    id: str
    sample_time: int
    block_size: int = 1
    min_blocks: int = 1
    max_blocks: int = -1
    acquire_allowed: bool = False
    external_resource_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelData:
    """Canonical logical channel calibration bound to a physical port.

    Channel records represent the calibrated operating point used by waveform generation and
    execution layers.

    :ivar id: Channel identifier.
    :ivar port_id: Referenced physical port identifier.
    :ivar frequency: Target channel frequency in Hz.
    :ivar oscillator_reference: Optional referenced oscillator identifier used as the
        frequency anchor for derived mixing calculations.
    :ivar scale: Complex scaling factor.
    :ivar imbalance: IQ gain imbalance factor.
    :ivar phase_offset: IQ phase offset in radians.
    """

    id: str
    port_id: str
    frequency: int
    oscillator_reference: str | None = None
    scale: complex = 1.0 + 0.0j
    imbalance: float = 1.0
    phase_offset: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class WaveformData:
    """Parameterized waveform definition used by operations and modes.

    This captures reusable waveform parameters independently from where or when the pulse is
    applied.

    :ivar id: Waveform definition identifier.
    :ivar shape: Named waveform shape.
    :ivar width: Waveform width in picoseconds.
    :ivar rise: Rise fraction or rise time in picoseconds, depending on shape semantics.
        Time values must be defined as integer picoseconds, while fractional values are
        defined as floats in the range [0.0, 1.0].
    :ivar amp: Waveform amplitude.
    :ivar drag: DRAG coefficient.
    :ivar phase: Waveform phase in radians.
    :ivar amp_setup: Optional setup/plateau amplitude.
    """

    # TODO: Sort out common parameters from shape specific ones and potentially make the shape
    # specific ones attribute entries - COMPILER-1218
    id: str
    shape: str | None = None
    width: int | None = None
    rise: int | float | None = None
    amp: float | None = None
    drag: float | None = None
    phase: float | None = None
    amp_setup: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AcquireDefinitionData:
    """Acquisition timing and weighting configuration for a mode or operation.

    This models signal-capture settings. State-interpretation probabilities are modeled
    separately so capture and interpretation remain decoupled.

    :ivar id: Acquisition definition identifier.
    :ivar delay: Acquisition start delay in picoseconds.
    :ivar sync: Whether acquisition timing is coupled to the measurement window; if
        ``True``, the acquire window is derived from the measurement pulse timing
        (after applying ``delay``), and if ``False`` an independent acquire ``width``
        is used.
    :ivar width: Acquisition window width in picoseconds when ``sync`` is ``False``.
    :ivar weights: Optional integration weights as per-sample coefficients.
        Values are typically absent (``None``) or a tuple of float/complex samples.
    """

    id: str
    delay: int | None = None
    sync: bool | None = None
    width: int | None = None
    weights: tuple[complex | float, ...] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class LinearMapToRealMethodData:
    """Threshold discriminator using a calibrated complex-to-real projection.

    This canonical record mirrors the ``linear_map_complex_to_real`` method in the
    runtime model. Each complex IQ readout is projected with two complex coefficients
    ``a`` and ``b`` from ``mean_z_map_args`` using ``Re(a * IQ + b)`` and then
    thresholded by sign into integer keys ``0`` and ``1``.

    :ivar method: Discriminator identifying this post-processing method variant.
    :ivar mean_z_map_args: Two complex coefficients used by the linear projection.
    """

    method: Literal["linear_map_complex_to_real"] = "linear_map_complex_to_real"
    mean_z_map_args: tuple[complex, complex] = (1 + 0j, 0j)


@dataclass(frozen=True, slots=True, kw_only=True)
class MaxLikelihoodDiscriminateParams:
    """IQ-plane centroid descriptor for a single integer discriminate key.

    Each entry represents one measurement outcome in a maximum-likelihood classifier.
    The integer output value and disallowed behavior are encoded by the key in
    ``MaxLikelihoodMethodData.states``:

    - Non-negative keys (``>= 0``) represent allowed states.
    - Negative keys (``< 0``) represent disallowed/background states.

    :ivar location: Complex IQ-plane centroid associated with this state.
    :ivar label: Optional human-readable state label for display and diagnostics.
    """

    location: complex
    label: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MaxLikelihoodMethodData:
    """Maximum-likelihood discriminator for IQ measurement results.

    This canonical record mirrors the ``max_likelihood`` method in the runtime model.
    A shot is assigned to the state with highest normalised likelihood, computed from
    state centroids in ``states`` and the noise variance ``noise_est``.

    ``p_min`` enables outlier rejection: if the winning normalised likelihood is below
    the threshold, the shot is treated as background and should be discarded by
    post-selection.

    ``transform`` and ``offset`` optionally describe an affine IQ pre-transform
    applied before likelihood evaluation.

    :ivar method: Discriminator identifying this post-processing method variant.
    :ivar states: Entries mapping integer discriminate keys to state centroids.
    :ivar noise_est: Global Gaussian noise variance used in likelihood evaluation.
    :ivar p_min: Minimum normalised likelihood required for acceptance.
    :ivar transform: Optional real (2, 2) affine IQ transform matrix.
    :ivar offset: Optional real length-2 IQ offset vector.
    """

    method: Literal["max_likelihood"] = "max_likelihood"
    states: tuple[tuple[int, MaxLikelihoodDiscriminateParams], ...]
    noise_est: float = 1.0
    p_min: float = 0.0
    transform: tuple[tuple[float, float], tuple[float, float]] | None = None
    offset: tuple[float, float] | None = None


PostProcessMethodData = LinearMapToRealMethodData | MaxLikelihoodMethodData


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeData:
    """A named mode that packages data for a targeted transition or operation.

    Modes typically represent a specific transition or operation at a lattice site, such as
    ``f01`` drive, standard transmon readout, or cross resonance.

    A mode binds that intent to a concrete channel (which may be shared by multiple modes)
    and stores the calibration context needed to generate instructions through that channel.

    Modes are not restricted to single-transition behavior; they can also represent richer
    operations (for example Raman-style gates) while following the same principles: channel
    binding plus waveform and acquisition definitions for the intended action.

    :ivar id: Unique mode identifier.
    :ivar channel_id: Referenced channel identifier used for mode signal delivery
        and/or acquisition.
    :ivar waveform_definitions: Waveform definitions available in this mode (see
        :class:`WaveformData`), for example ``pi/2`` and ``pi`` pulses for an ``f01``
        mode.
    :ivar acquire_definitions: Acquisition definitions available in this mode (see
        :class:`AcquireDefinitionData`).
    :ivar post_process_method: Optional canonical post-processing method.
    :ivar preselect_disallowed_states: Integer state keys discarded during
        pre-selection.
    """

    id: str
    channel_id: str
    waveform_definitions: tuple[WaveformData, ...] = ()
    acquire_definitions: tuple[AcquireDefinitionData, ...] | None = None
    post_process_method: PostProcessMethodData | None = None
    preselect_disallowed_states: frozenset[int] | None = None


# This is still under design and subject to change as we determine how we want to
# "serialise" and "restore" "functions" like sequences or gates.
@dataclass(frozen=True, slots=True, kw_only=True)
class AcquireOperationStepData:
    """Single acquisition step in an operation expansion.

    A step links an operation to a mode and an acquisition definition to apply on that mode.

    :ivar mode_id: Referenced mode identifier.
    :ivar acquire_definition: Acquisition definition identifier or inline definition.
    """

    mode_id: str
    acquire_definition: str | AcquireDefinitionData


# This is still under design and subject to change as we determine how we want to
# "serialise" and "restore" "functions" like sequences or gates.
@dataclass(frozen=True, slots=True, kw_only=True)
class DelayOperationStepData:
    """Single delay step in an operation expansion.

    A step links an operation to a mode and a delay duration to apply on that mode.

    :ivar mode_id: Referenced mode identifier.
    :ivar duration: Delay duration in picoseconds.
    """

    mode_id: str
    duration: int


# This is still under design and subject to change as we determine how we want to
# "serialise" and "restore" "functions" like sequences or gates.
@dataclass(frozen=True, slots=True, kw_only=True)
class PulseOperationStepData:
    """Single step in an operation expansion.

    A step links an operation to a mode and a pulse definition to apply on that mode.

    :ivar mode_id: Referenced mode identifier.
    :ivar waveform_definition: Waveform definition identifier or inline definition.
    """

    mode_id: str
    waveform_definition: str | WaveformData


# This is still under design and subject to change as we determine how we want to
# "serialise" and "restore" "functions" like sequences or gates.
@dataclass(frozen=True, slots=True, kw_only=True)
class SyncOperationStepData:
    """Single synchronization step in an operation expansion.

    A step links an operation to a set of modes for synchronization.

    :ivar mode_ids: Referenced mode identifiers.
    """

    mode_ids: frozenset[str]


OperationStepData = (
    PulseOperationStepData
    | AcquireOperationStepData
    | DelayOperationStepData
    | SyncOperationStepData
)


# This is still under design and subject to change as we determine how we want to
# "serialise" and "restore" "functions" like sequences or gates.
@dataclass(frozen=True, slots=True, kw_only=True)
class OperationData:
    """Canonical operation definition composed of ordered steps.

    Operations may represent native hardware primitives or composite virtual instructions
    expanded into nested steps.

    :ivar id: Operation identifier.
    :ivar operation_steps: Ordered operation steps or nested operations.
    """

    id: str
    operation_steps: tuple[OperationStepData | OperationData, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbabilityEntry:
    """A single probability entry for a given state preparation and measurement outcome.

    :ivar prepared_state: The state that was prepared.
    :ivar measured_state: The state that was measured.
    :ivar probability: The conditional probability of observing measured_state given that
        prepared_state was prepared.
    """

    prepared_state: int
    measured_state: int
    probability: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadoutProbabilityData:
    """Readout confusion probabilities stored as explicit state-pair entries.

    The entries capture conditional probabilities p(measured|prepared) for a prepared and
    measured state pair, rather than requiring a fixed binary confusion-matrix layout.

    :ivar probability_entries: Probability entries for prepared/measured state pairs.
    """

    probability_entries: tuple[ProbabilityEntry, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class QubitData:
    """Canonical calibration data for a device-level qubit abstraction.

    This qubit represents the implementation-level unit exposed by the system data. It may
    be realized by multiple underlying physical elements, but is treated as one qubit
    resource at higher abstraction layers. It is therefore distinct from both microscopic
    physical components and QEC logical qubits.

    :ivar id: Qubit identifier.
    :ivar index: Qubit index.
    :ivar modes: Modes supported by this qubit.
    :ivar operations: Operation definitions available on this qubit.
    :ivar readout_probability: Optional readout confusion probabilities.
    """

    id: str
    index: int
    modes: tuple[ModeData, ...] = ()
    operations: tuple[OperationData, ...] = ()
    readout_probability: ReadoutProbabilityData | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TwoQubitGateFidelityData:
    """Two-qubit gate fidelity data for a qubit pair.

    :ivar gate: Two-qubit gate identifier.
    :ivar fidelity: Fidelity of the two-qubit gate.
    """

    gate: str
    fidelity: float


@dataclass(frozen=True, slots=True, kw_only=True)
class QubitCouplingData:
    """Directional coupling metadata between two qubits.

    This is intentionally lightweight and stores only connectivity direction plus per-gate
    fidelity estimates.

    :ivar source_qubit_id: Source qubit identifier.
    :ivar target_qubit_id: Target qubit identifier.
    :ivar gate_fidelities: Two-qubit gate fidelity entries for this directed qubit pair.
    """

    source_qubit_id: str
    target_qubit_id: str
    gate_fidelities: tuple[TwoQubitGateFidelityData, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResetData:
    """Supported reset-strategy descriptor for hardware capabilities.

    Hardware backends may expose different reset strategies. This record captures
    one supported strategy and optional metadata about its behavior.

    :ivar type: Reset strategy type (for example ``passive``).
    :ivar attributes: Additional reset strategy metadata. (for example, a ``passive`` reset
        might include a ``duration`` attribute for the expected T1 decay time).
    """

    type: str
    attributes: tuple[AttributeEntry, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class AcquireModeData:
    """Supported acquisition mode descriptor for hardware capabilities.

    Hardware backends may expose different acquisition modes. This record captures one
    supported mode and optional metadata about its behavior.

    :ivar type: Acquisition mode type.
    :ivar attributes: Additional mode metadata.
    """

    type: str
    attributes: tuple[AttributeEntry, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalSystemData:
    """Canonical system data used by the experimental stack.

    :ivar calibration_id: Calibration identifier.
    :ivar acquire_limit: Maximum allowed acquisitions for a single execution batch.
    :ivar acquire_modes: Supported acquisition mode descriptors for hardware capabilities.
    :ivar default_acquire_mode: Optional default acquisition mode type selected from
        ``acquire_modes``.
    :ivar reset_methods: Supported reset-strategy descriptors for hardware capabilities.
    :ivar default_reset_method: Optional default reset strategy type selected from
        ``reset_methods``.
    :ivar oscillators: Canonical oscillator configurations used as frequency references.
    :ivar ports: Canonical physical port descriptors with ID-based references.
    :ivar channels: Canonical logical channel calibrations bound to physical ports.
    :ivar qubits: Canonical calibration records for device-level qubit abstractions.
    :ivar couplings: Directional coupling metadata between qubit pairs.
    :ivar external_resources: Canonical descriptors for external hardware resources.
    :ivar metadata: Additional top-level system metadata entries.
    """

    calibration_id: str = ""
    acquire_limit: int = -1
    acquire_modes: tuple[AcquireModeData, ...] = ()
    default_acquire_mode: str | None = None
    reset_methods: tuple[ResetData, ...] = ()
    default_reset_method: str | None = None
    oscillators: tuple[OscillatorData, ...] = ()
    ports: tuple[PortData, ...] = ()
    channels: tuple[ChannelData, ...] = ()
    qubits: tuple[QubitData, ...] = ()
    couplings: tuple[QubitCouplingData, ...] = ()
    external_resources: tuple[ExternalResourceData, ...] = ()
    metadata: tuple[AttributeEntry, ...] = ()
