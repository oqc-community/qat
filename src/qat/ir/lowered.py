from collections import defaultdict
from dataclasses import dataclass, field

from qat.ir.instructions import Assign, Instruction, ResultsProcessing, Return
from qat.ir.measure import Acquire, PostProcessing
from qat.model.device import PulseChannel


@dataclass
class PartitionedIR:
    """Stores the results of the :class:`PartitionByPulseChannel`."""

    # TODO: When refactoring into Pydantic instructions, we should replace this object with
    # a Pydantic base model. Ideally, we would unify the different IR levels by making a
    # general "QatIR` flexible enough to support them all, but that might be too optimistic. (COMPILER-382)

    # TODO: Remove `PulseChannel` as keys once we fully migrated to pydantic passes 581
    target_map: dict[PulseChannel | str, list[Instruction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # TODO: Remove Repeat type option: COMPILER-451
    shots: int | None = field(default_factory=lambda: None)
    compiled_shots: int | None = field(default=None)
    returns: list[Return] = field(default_factory=list)
    assigns: list[Assign] = field(default_factory=list)
    acquire_map: dict[PulseChannel | str, list[Acquire]] = field(
        default_factory=lambda: defaultdict(list)
    )
    pp_map: dict[str, list[PostProcessing]] = field(
        default_factory=lambda: defaultdict(list)
    )
    rp_map: dict[str, ResultsProcessing] = field(default_factory=dict)
