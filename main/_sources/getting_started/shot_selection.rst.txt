.. _shot_selection:
.. _post_selection:

Shot Selection (Pre & Post)
===========================

Shot selection discards individual shots whose discriminated integer state keys
indicate the qubit was in an undesirable state.  QAT supports two flavours:

* **Post-selection** — filters shots based on the *end-of-circuit*
  measurement result (e.g. discard leakage states).
* **Pre-selection** — filters shots based on a measurement injected
  *before* the circuit begins, verifying each qubit starts in its ground
  state.

Both mechanisms produce per-output boolean validity masks.  The runtime
ANDs all masks together in
:func:`~qat.runtime.passes.transform._build_and_apply_global_mask`
so that a shot is retained only if it passes **every** check.

.. contents::
   :local:
   :depth: 2


Post-selection
--------------

Post-selection discards shots where a qubit's end-of-circuit measurement
lands in a disallowed state (e.g. a leakage state in a multi-level
system).


Configuring disallowed states
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Disallowed states are declared on :class:`~qat.model.post_processing.MaxLikelihoodMethod`
using **negative integer keys** in the ``states`` dict.  Shots classified to a state with
a negative key are filtered by :class:`~qat.ir.measure.PostSelect`.

.. code-block:: python

    from qat.model.post_processing import MaxLikelihoodMethod, MLDiscriminateParams

    # Qutrit: states 0 and 1 are allowed; state at key -2 is disallowed (leakage).
    method = MaxLikelihoodMethod(
        states={
            0: MLDiscriminateParams(location=1+0j),
            1: MLDiscriminateParams(location=-1+0j),
            -2: MLDiscriminateParams(location=0+1j),
        },
    )

:class:`~qat.model.post_processing.LinearMapToRealMethod` always outputs the non-negative
integers ``0`` and ``1`` and does **not** support post-selection on shot outcomes.
Use :class:`~qat.model.post_processing.MaxLikelihoodMethod` with negative keys to
configure post-selection.


The PostSelect instruction
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qat.ir.measure.PostSelect
   :members:
   :show-inheritance:



IR encoding
^^^^^^^^^^^

Frontends that require post-selection call
:meth:`~qat.ir.instruction_builder.QuantumInstructionBuilder.emit_post_select`,
which inserts a complete ``PostSelect`` instruction block immediately after
the corresponding ``Discriminate`` instruction.


Runtime mask logic
^^^^^^^^^^^^^^^^^^

:class:`~qat.runtime.passes.transform.AcquisitionPostprocessing` collects
per-output masks from every ``PostSelect`` instruction (both end-of-circuit
post-selection and ahead-of-circuit pre-selection), computes a single global
mask via ``AND``, filters all result arrays, and stores metadata:

.. autoclass:: qat.runtime.passes.analysis.PostSelectionResult
   :members:


Pre-selection
-------------

Pre-selection verifies that every flagged qubit starts in its ground state
**before** the quantum algorithm begins.  The compiler injects a
measurement-and-discriminate step at the start of the circuit; shots in
which a qubit is found in a disallowed state are discarded at runtime.


Enabling pre-selection
^^^^^^^^^^^^^^^^^^^^^^

Pre-selection is controlled by two settings:

1. **Global flag** — set ``pre_selection=True`` on
   :class:`~compiler_config.config.CompilerConfig`.
2. **Per-qubit field** — set
   :attr:`~qat.model.device.Qubit.preselect_disallowed_states` to a set
   of state **indices** that should cause the shot to be discarded.  The
   default is ``{1}`` (discard the excited state); set ``set()`` to disable
   pre-selection for an individual qubit.

Only qubits where *both* settings are active receive pre-selection
measurements.

Example:

.. code-block:: python

    from compiler_config.config import CompilerConfig
    from qat.model.loaders.lucy import LucyModelLoader

    model = LucyModelLoader(qubit_count=1).load()

    # Configure qubit 0 for pre-selection.
    qubit = model.qubits[0]
    qubit.preselect_disallowed_states = {1}  # discard excited state (index 1)

    config = CompilerConfig(pre_selection=True, repeats=1000)

For multi-state hardware (e.g. transmon with four explicit states), specify all
state indices that are not the ground state:

.. code-block:: python

    qubit.preselect_disallowed_states = {1, 2, 3}  # discard all non-ground states

The integers in ``preselect_disallowed_states`` are **integer state keys** — they
correspond directly to the dict keys in the qubit's
:attr:`~qat.model.post_processing.MaxLikelihoodMethod.states` mapping.  For example,
if a transmon is calibrated with four states::

    MaxLikelihoodMethod(states={
        0: MLDiscriminateParams(location=1+0j,   label="|0⟩"),          # ground
        1: MLDiscriminateParams(location=-1+0j,  label="|1⟩"),          # first excited
        2: MLDiscriminateParams(location=0+1j,   label="|2⟩"),          # second excited
        3: MLDiscriminateParams(location=0-1j,   label="|3⟩"),          # third excited
    })

then ``preselect_disallowed_states = {1, 2, 3}`` discards any shot where the
pre-selection measurement returns key ``1``, ``2``, or ``3`` — i.e. any shot not in
the ground state.

For :class:`~qat.model.post_processing.LinearMapToRealMethod` the two states are
always the implicit pair ``{0: above-threshold, 1: at-or-below-threshold}``; labels
are not configurable.


How it works
^^^^^^^^^^^^

The :class:`~qat.middleend.passes.transform.InsertPreSelectionMeasurement`
pass runs early in the middleend pipeline. For each qualifying qubit it
injects the following instructions immediately after the ``Repeat``
instruction, before the circuit body:

**Per-qubit block** (output variable ``"presel_{qubit_index}"``):

* :class:`~qat.ir.waveforms.Pulse` — readout tone on the measure channel.
* :class:`~qat.ir.instructions.Delay` (``acquire.delay``) — ring-up offset
  on the acquire channel; waits for the resonator to respond before
  sampling starts.
* :class:`~qat.ir.measure.Acquire` with
  :attr:`~qat.ir.instruction_basetypes.AcquirePurpose.PRE_SELECTION`.
* :class:`~qat.ir.instructions.Delay`
  (``resonator.relaxation_delay``) — ring-down settle on the acquire
  channel; waits for the resonator to drain before the next operation.
  This delay is typically a few microseconds, allowing the readout
  resonator to return to its equilibrium state.
* :class:`~qat.ir.measure.Equalise` (when calibrated; omitted if equalisation
  calibration data is unavailable).
* :class:`~qat.ir.measure.Discriminate`.
* :class:`~qat.ir.measure.PostSelect` with ``additional_disallowed`` set to
  the qubit's ``preselect_disallowed_states``.

**Global synchronisation:** a single
:class:`~qat.ir.instructions.Synchronize` covering all qubit channels is
emitted *before* the first qubit block and again *after* the last.  The
trailing sync simultaneously realigns drive/measure/acquire channels (which
diverge during the readout window) and provides the cross-qubit alignment
barrier before the circuit begins.

The ``presel_*`` output variable holds integer state keys after
``Discriminate`` and is for internal runtime use only — it drives the
validity mask but is never returned to the user. For per-shot debug
access, inspect the
:class:`~qat.runtime.passes.analysis.DiscriminateResult` stored in the
:class:`~qat.core.result_base.ResultManager` after execution.
Pre-selection statistics (for example, shots requested/retained counts)
are recorded in post-selection metadata such as
:class:`~qat.runtime.passes.analysis.PostSelectionResult`, available via
the :class:`~qat.core.result_base.ResultManager`.

Because the pre-selection acquire has ``purpose=PRE_SELECTION``,
:class:`~qat.middleend.passes.validation.NoMidCircuitMeasurementValidation`
skips it when checking for mid-circuit measurements (the validation pass
focuses on user-defined measurements; compiler-inserted pre-selection
measurements are intentionally allowed).

**Channel timeline for a two-qubit circuit:**

.. mermaid::

    sequenceDiagram
        participant Q0D as Q0 Drive
        participant Q0M as Q0 Measure
        participant Q0A as Q0 Acquire
        participant Q1D as Q1 Drive
        participant Q1M as Q1 Measure
        participant Q1A as Q1 Acquire

        Note over Q0D,Q1A: Synchronize(all_qubit_channels) — t=T0

        Q0M->>Q0M: Pulse (readout tone)
        Q0A->>Q0A: Delay (acquire.delay)
        Q0A->>Q0A: Acquire [PRE_SELECTION]
        Q0A->>Q0A: Delay (relaxation_delay)
        Note over Q0D,Q0A: drive@T0, measure@T0+W, acquire@T0+acq_delay+D+acq_dur
        Note over Q0D,Q0A: Equalise → Discriminate → PostSelect

        Q1M->>Q1M: Pulse (readout tone)
        Q1A->>Q1A: Delay (acquire.delay)
        Q1A->>Q1A: Acquire [PRE_SELECTION]
        Q1A->>Q1A: Delay (relaxation_delay)
        Note over Q1D,Q1A: same pattern for Q1
        Note over Q1D,Q1A: Equalise → Discriminate → PostSelect

        Note over Q0D,Q1A: Synchronize(all_qubit_channels)
        Note over Q0D,Q1A: all channels realigned — circuit begins


Combined behaviour
------------------

When both pre-selection and post-selection are active, each produces its
own per-output validity mask.  The runtime ANDs all masks together so that
a shot is retained only if it passes **both** checks.

.. note::

   **Two separate disallowed-states settings** exist and serve different
   purposes:

   * **Negative keys** in :class:`~qat.model.post_processing.MaxLikelihoodMethod`
     ``states`` — controls **post-selection** on end-of-circuit measurements.
     Frontends emit ``PostSelect`` when ``CompilerConfig.post_selection=True``;
     the runtime then filters shots whose discriminated key is negative.

   * ``preselect_disallowed_states`` on
     :class:`~qat.model.device.Qubit` — controls **pre-selection**.
     The middleend pass uses these for the measurement injected before
     the circuit begins.

   These are independent: neither overrides the other.  A shot must pass
   both checks to be retained.

**Results format impact** — the semantic matrix in
:ref:`results_format_semantics` applies identically: filtered shots are
removed from ``raw()``, ``binary()``, and ``binary_count()`` outputs,
and ``shots_retained`` reflects the combined surviving count.

**Error mitigation impact** — shot selection removes data before error
mitigation techniques are applied. If using mitigation methods that rely
on shot statistics (e.g. readout error mitigation), ensure the calibration
data and the filtered dataset are consistent in their shot selection
criteria to avoid biasing the mitigation matrix.


API reference
-------------

* :class:`~qat.middleend.passes.transform.InsertPreSelectionMeasurement`
  — middleend pass that injects pre-selection measurements.
* :class:`~qat.ir.measure.PostSelect` — IR instruction for shot
  filtering.
* :class:`~qat.ir.instruction_basetypes.AcquirePurpose` — enum
  distinguishing measurement vs pre-selection acquires. This is independent
  of :class:`~qat.ir.instruction_basetypes.AcquireMode`; ``AcquirePurpose``
  indicates *why* the acquisition is happening (normal measurement,
  pre-selection, etc.), while ``AcquireMode`` controls *how* the hardware
  performs the acquisition (scope, integrator, etc.).
* :attr:`~qat.model.device.Qubit.preselect_disallowed_states` — per-qubit
  pre-selection configuration.
* :class:`~qat.runtime.passes.analysis.PostSelectionResult` — runtime
  metadata for filtered shots. Post-selection is configured via negative
  keys in :class:`~qat.model.post_processing.MaxLikelihoodMethod`; pre-selection
  uses the per-qubit ``preselect_disallowed_states`` attribute.


See also
--------

* :ref:`post_processing_pipeline` — readout signal processing
  (Equalise → Discriminate for end-of-circuit measurements).
* :mod:`qat.ir.measure` — instruction model docs.
* :mod:`qat.runtime.post_processing` — runtime helper docs.
* :ref:`execution`
