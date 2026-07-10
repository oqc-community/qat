# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from typing import TypeVar

from qat.experimental.system_data.canonical.schema import CanonicalSystemData

_TDerivedView = TypeVar("_TDerivedView", bound="DerivedViewInterface")


class DerivedViewInterface(ABC):
    """A view of a canonical system data object derived from the canonical data.

    This is an abstract base class for derived views of canonical system data. They provide
    a representation of relevant data in a form that is convenient to the given application,
    such as a particular layer of abstraction, or a particular pass.
    """

    @classmethod
    @abstractmethod
    def from_canonical(
        cls: type[_TDerivedView], canonical_data: CanonicalSystemData
    ) -> _TDerivedView:
        """Construct a derived view from canonical system data.

        :param canonical_data: The canonical system data to derive from.
        :returns: A derived view of the canonical data.
        """
        ...
