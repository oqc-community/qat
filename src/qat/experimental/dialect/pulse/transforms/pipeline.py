# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Centralised builder for the default pulse-level IR preprocessing pipeline.

The :class:`PulsePipelineManager` is the single point of entry for constructing a
pulse-level pass pipeline from a hardware view.  It accepts a
:class:`~qat.experimental.system_data.pulse.constraints.PulseLevelConstraints` instance
and exposes :meth:`~PulsePipelineManager.build_default_pipeline`, which returns a
validated :class:`~qat.experimental.passes.pass_ordering.OrderedPassPipeline`.

The default pipeline runs the following passes, in order:

1. :class:`~qat.experimental.dialect.pulse.transforms.constants.OrderedCanonicalizePass`
   — fold all constant arithmetic so that subsequent passes only see canonical
   :class:`~qat.experimental.dialect.pulse.ir.ops.ConstantOp` operands.
1. :class:`.ApplyGranularitySanitisation`
   — round constant durations and sampled-waveform widths up to the hardware timing
   granularity, so later passes only see realisable times.
1. :class:`.EvaluateWaveformsAsSamples`
   — convert analytical waveform ops into sampled
   :class:`~qat.experimental.dialect.pulse.ir.ops.ConstantOp` payloads, using the
   per-port sample times from *constraints*.
1. :class:`.TimelineNormalization`
   — resolve :class:`~qat.experimental.dialect.pulse.ir.ops.SynchronizeOp` operations
   into explicit :class:`~qat.experimental.dialect.pulse.ir.ops.WaitOp` chains using
   symbolic time-expression analysis.
1. :class:`.DeadFrameEliminationPass`
   — eliminate frame creation and manipulation operations that do not contribute to
   any hardware-visible pulse operations, reducing IR complexity.
1. :class:`.ApplySquashContiguousOptimizations`
   — merge adjacent wait and phase operations.
1. :class:`.LowerKernelsToResultsArrays`
   — rewrite each kernel signature from ``ResultsCollectionType`` values to
   ``ResultsArrayType`` values and update all call sites to match the expanded signature.
1. :class:`~qat.experimental.dialect.pulse.transforms.constants.OrderedCanonicalizePass`
   — a final canonicalization pass that folds any constants left behind by the earlier
   passes and removes the resulting no-ops.

The :class:`~qat.experimental.passes.pass_ordering.OrderedPassPipeline`
validates any declared :class:`~qat.experimental.passes.pass_ordering.OrderedPass`
constraints when it is constructed, so a mis-ordered schedule fails eagerly at build
time.
"""

from __future__ import annotations

from qat.experimental.conversion.pulse.lower_kernels_to_arrays import (
    LowerKernelsToResultsArrays,
)
from qat.experimental.dialect.pulse.transforms.constants import OrderedCanonicalizePass
from qat.experimental.dialect.pulse.transforms.dead_frame_elimination import (
    DeadFrameEliminationPass,
)
from qat.experimental.dialect.pulse.transforms.granularity_sanitisation import (
    ApplyGranularitySanitisation,
)
from qat.experimental.dialect.pulse.transforms.optimize_contiguous_squashable_instructions import (  # noqa: E501
    ApplySquashContiguousOptimizations,
)
from qat.experimental.dialect.pulse.transforms.timeline_normalization import (
    TimelineNormalization,
)
from qat.experimental.dialect.pulse.transforms.waveform_evaluation import (
    EvaluateWaveformsAsSamples,
)
from qat.experimental.passes.pass_ordering import OrderedPassPipeline
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.pulse.constraints import PulseLevelConstraints


class PulsePipelineManager:
    """Builds pulse-level IR pipelines from a hardware constraint view.

    :param constraints: Hardware-derived pulse-level constraints (timing granularity, per-
        port sample times, native waveform shapes).
    """

    def __init__(self, constraints: PulseLevelConstraints) -> None:
        self.constraints = constraints

    @classmethod
    def from_canonical_data(cls, data: CanonicalSystemData) -> PulsePipelineManager:
        """Construct a manager by deriving constraints from canonical system data.

        :param data: The canonical hardware description.
        :returns: A new :class:`PulsePipelineManager` whose constraints are derived from
            *data*.
        """
        return cls(constraints=PulseLevelConstraints.derive(data))

    def build_default_pipeline(self) -> OrderedPassPipeline:
        """Build the default pulse-level preprocessing pipeline.

        :returns: An :class:`~qat.experimental.passes.pass_ordering.OrderedPassPipeline`
            containing the standard pulse-level passes in conflict-free order, ending
            with a final constant-propagation cleanup.
        :raises ~xdsl.utils.exceptions.VerifyException: If the declared ordering
            constraints are violated (should not occur with the default configuration).
        """
        return OrderedPassPipeline(
            (
                OrderedCanonicalizePass(),
                ApplyGranularitySanitisation(constraints=self.constraints),
                EvaluateWaveformsAsSamples(constraints=self.constraints),
                TimelineNormalization(),
                DeadFrameEliminationPass(),
                ApplySquashContiguousOptimizations(),
                LowerKernelsToResultsArrays(),
                OrderedCanonicalizePass(),
            )
        )
