.. _post_processing_pipeline:

Readout Post-Processing
=======================

This page explains how QAT's granular readout pipeline transforms raw IQ
acquisition data into state labels and numeric values.  For shot-level
filtering (pre-selection and post-selection), see :ref:`shot_selection`.

Pipeline order:

``Equalise`` → ``Discriminate`` → ``PostSelect`` → ``Demap``

.. contents::
   :local:
   :depth: 2


Overview
--------

The data flow spans three layers:

1. **Hardware model** —
   Qubits carry a configured discrimination method
   (``LinearMapToRealMethod`` or ``MaxLikelihoodMethod``).
2. **IR builder** —
   Frontends emit granular readout instructions from that method.
3. **Runtime pass** —
   Runtime applies instructions, builds masks, filters invalid shots,
   and formats final outputs.

.. code-block:: text

    Qubit.post_process_method
      └─ LinearMapToRealMethod | MaxLikelihoodMethod
               │
               ▼
    measure_with_granular_post_processing()       ← frontend parsers call this
      ├─ MeasureBlock
      ├─ emit_granular_post_processing()
      │    └─ Equalise → Discriminate → Demap
      └─ emit_post_select()  (if disallowed states configured)
           └─ PostSelect (inserted before Demap)  → see shot_selection
               │
               ▼
    AcquisitionPostprocessing pass (runtime)
      ├─ apply_equalise()
      ├─ apply_discriminate_instruction()
      ├─ apply_post_select() → validity_mask      → see shot_selection
      └─ apply_demap_instruction()
      │
      ├─ global_mask = AND of all per-output masks
      ├─ filter all result arrays by global_mask
      └─ store PostSelectionResult(shots_requested, shots_retained, mask)
               │
               ▼
    ResultTransform
      └─ uses shots_retained as denominator for binary_count


Pipeline steps
--------------


Step 1 — Equalise
^^^^^^^^^^^^^^^^^

.. autoclass:: qat.ir.measure.Equalise
   :members:
   :show-inheritance:

Runtime implementation: :func:`~qat.runtime.post_processing.apply_equalise`.


Step 2 — Discriminate
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: qat.ir.measure.Discriminate
   :members:
   :show-inheritance:

Runtime implementation: :func:`~qat.runtime.post_processing.apply_discriminate_instruction`.


Configuring classification methods
"""""""""""""""""""""""""""""""""""

Configuration lives on :class:`~qat.model.device.Qubit` via
``post_process_method``.

.. note::

    ``post_process_method`` and legacy ``mean_z_map_args`` are mutually
    exclusive.

.. autoclass:: qat.model.post_processing.LinearMapToRealMethod
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qat.model.post_processing.MaxLikelihoodMethod
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qat.model.post_processing.MLStateMap
   :members:
   :undoc-members:

**Example** — attaching a method to a qubit:

.. code-block:: python

    from qat.model.device import Qubit
    from qat.model.post_processing import LinearMapToRealMethod

    qubit = Qubit(
        ...,
        mean_z_map_args=None,
        post_process_method=LinearMapToRealMethod(disallowed_states=["1"]),
    )


Step 3 — PostSelect
^^^^^^^^^^^^^^^^^^^^

``PostSelect`` sits between ``Discriminate`` and ``Demap`` and marks shots
for filtering based on discriminated state labels.  For full details on
post-selection and pre-selection, see :ref:`shot_selection`.


Step 4 — Demap
^^^^^^^^^^^^^^

.. autoclass:: qat.ir.measure.Demap
   :members:
   :show-inheritance:

Runtime implementation: :func:`~qat.runtime.post_processing.apply_demap_instruction`.


Step 5 — Results format
^^^^^^^^^^^^^^^^^^^^^^^

.. _results_format_semantics:

:class:`~qat.runtime.passes.transform.ResultTransform` formats final results by
:attr:`~compiler_config.config.CompilerConfig.results_format`.

``raw()``
    Complex IQ arrays (equalised path output).

``binary()``
    Per-shot mapped int values.

``binary_count()``
    ``{label: count}`` from discriminated string labels (retained shots only).

**Semantic matrix**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Format
     - Selection **off**
     - Selection **on** (e.g. 3 of 10 filtered)
   * - ``raw()``
     - 10 complex IQ values
     - 7 complex IQ values
   * - ``binary()``
     - 10 mapped int values
     - 7 mapped int values
   * - ``binary_count()``
     - ``{"0": 6, "1": 4}`` over 10 shots
     - ``{"0": 4, "1": 3}`` over 7 retained shots

When ``results_format`` is ``None``, dynamic structure return is used.


IR encoding
-----------

:meth:`~qat.ir.instruction_builder.QuantumInstructionBuilder.measure_with_granular_post_processing`
    Frontend path (QASM2/QASM3/QIR/tket). Emits ``MeasureBlock``, optional
    ``PostProcessing(MEAN, TIME)`` (SCOPE only), then granular instructions via
    :meth:`~qat.ir.instruction_builder.QuantumInstructionBuilder.emit_granular_post_processing`.

:meth:`~qat.ir.instruction_builder.QuantumInstructionBuilder.measure_single_shot_z`
    Customer-facing path. Emits legacy
    ``PostProcessing(LINEAR_MAP_COMPLEX_TO_REAL)`` and does not emit granular
    instructions.

Legacy :class:`~qat.ir.measure.PostProcessing` instructions remain supported for
backward compatibility.


Runtime execution
-----------------

:class:`~qat.runtime.passes.transform.AcquisitionPostprocessing` applies
the full ``Equalise``/``Discriminate``/``PostSelect``/``Demap`` chain.
See :ref:`shot_selection` for details on mask construction and filtering.


See also
--------

* :ref:`shot_selection` — shot filtering (pre-selection & post-selection).
* :mod:`qat.ir.measure` — instruction model docs (canonical API semantics).
* :mod:`qat.runtime.post_processing` — runtime helper docs.
* :mod:`qat.model.post_processing` — classification model docs.
* :class:`qat.runtime.passes.transform.AcquisitionPostprocessing`
* :class:`qat.runtime.passes.transform.ResultTransform`
* :ref:`execution`
