# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Declarative ordering constraints for xDSL pass pipelines.

xDSL's :class:`~xdsl.passes.PassPipeline` runs its passes in the order they are
listed, but offers no way for a pass to declare that it must run before (or after)
another. This module adds a lightweight, opt-in mix-in: a :class:`ModulePass`
subclass that also inherits :class:`OrderedPass` may declare two class-level
constraints,

* :meth:`OrderedPass.runs_before` — pass classes that, *if present* in the same
  pipeline, must appear **after** this pass; and
* :meth:`OrderedPass.required_predecessors` — pass classes that **must be
  present** in the same pipeline and appear **before** this pass.

Both are instance methods that return the referenced pass classes directly rather than
their string names, so a typo or a renamed pass is caught at import time. An override
may import the referenced pass lazily to avoid a circular import, and should memoise
its result in a class field so the (trivial) computation only happens once.

:class:`OrderedPassPipeline` validates these constraints when it is constructed, so
an official pass grouping enforces its invariants up front rather than failing
part-way through execution.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.passes import ModulePass, PassPipeline
from xdsl.utils.exceptions import VerifyException


class OrderedPass:
    """Mix-in that lets a :class:`~xdsl.passes.ModulePass` declare ordering constraints.

    Combine it with :class:`~xdsl.passes.ModulePass` on any pass that participates in
    ordering, for example ``class MyPass(OrderedPass, ModulePass)``. Override either
    method to declare a constraint; both return the **pass classes** they constrain
    against, so the reference is a direct class rather than a loose string. An override
    may import the referenced pass lazily inside the method when a module-level import
    would be circular, and should memoise its result in a class field. The defaults
    impose no constraints.
    """

    def runs_before(self) -> frozenset[type[ModulePass]]:
        """Pass classes that, if present in the same pipeline, must run **after** this
        pass."""
        return frozenset()

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        """Pass classes that **must** be present in the same pipeline and run **before**
        this pass."""
        return frozenset()


def validate_pass_ordering(passes: Sequence[ModulePass]) -> None:
    """Verify that the ordering constraints declared by ``passes`` are satisfied.

    Only passes that inherit :class:`OrderedPass` carry constraints; every other pass
    is skipped. Constraints are evaluated against the **first occurrence** of each pass
    type, so a pass may appear again later (for example as a trailing clean-up stage)
    without violating its own :meth:`~OrderedPass.runs_before` constraint. Two families
    of constraint are checked:

    * For every pass declaring :meth:`OrderedPass.runs_before`, each referenced pass
      class that also appears in ``passes`` must first appear at a later position.
    * For every pass declaring :meth:`OrderedPass.required_predecessors`, each
      referenced pass class must first appear in ``passes`` at an earlier position.

    :param passes: The passes in the exact order they will be applied.
    :raises VerifyException: If any declared constraint is violated. The message
        identifies the offending pass and constraint so the pipeline can be fixed.
    """

    positions: dict[type[ModulePass], int] = {}
    first_occurrence: dict[type[ModulePass], ModulePass] = {}
    for index, pass_ in enumerate(passes):
        pass_type = type(pass_)
        if pass_type not in positions:
            positions[pass_type] = index
            first_occurrence[pass_type] = pass_

    for pass_type, index in positions.items():
        pass_ = first_occurrence[pass_type]
        if not isinstance(pass_, OrderedPass):
            continue

        name = pass_.name

        for later_pass in pass_.runs_before():
            other = positions.get(later_pass)
            if other is not None and other <= index:
                raise VerifyException(
                    f"Pass ordering violation: '{name}' declares it must run before "
                    f"'{later_pass.name}', but '{later_pass.name}' is scheduled at "
                    f"position {other} which is not after '{name}' at position {index}."
                )

        for predecessor in pass_.required_predecessors():
            other = positions.get(predecessor)
            if other is None:
                raise VerifyException(
                    f"Pass ordering violation: '{name}' requires '{predecessor.name}' "
                    f"to run before it, but '{predecessor.name}' is not present in the "
                    f"pipeline."
                )
            if other >= index:
                raise VerifyException(
                    f"Pass ordering violation: '{name}' requires '{predecessor.name}' "
                    f"to run before it, but '{predecessor.name}' is scheduled at "
                    f"position {other} which is not before '{name}' at position "
                    f"{index}."
                )


@dataclass(frozen=True)
class OrderedPassPipeline(PassPipeline):
    """A :class:`~xdsl.passes.PassPipeline` that validates ordering on construction.

    Official pass groupings should return this pipeline so that a mis-ordered or
    incomplete schedule fails as soon as it is built, rather than part-way through
    execution.
    """

    def __post_init__(self) -> None:
        """Validate the declared ordering constraints once the pipeline is built.

        :raises VerifyException: If :func:`validate_pass_ordering` finds a
            violation.
        """

        validate_pass_ordering(self.passes)
