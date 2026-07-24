# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

from qat.experimental.system_data.canonical.schema import CanonicalSystemData

_TDerivedView = TypeVar("_TDerivedView", bound="DerivedViewInterface")
_TParent = TypeVar("_TParent", bound=Union[CanonicalSystemData, "DerivedViewInterface"])


class DerivedViewInterface(ABC, Generic[_TParent]):
    """A view derived from an upstream data source.

    The type parameter ``_TParent`` declares what this view is derived from. That may be
    :class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData` for views
    that sit directly on the canonical data, or another :class:`DerivedViewInterface`
    subclass for views that sit further down the derivation chain.

    Subclasses declare their parent type in the class header::

        class QubitView(DerivedViewInterface[CanonicalSystemData]): ...
        class TopologyView(DerivedViewInterface[QubitView]): ...
        class ScipyTopologyView(DerivedViewInterface[TopologyView]): ...

    and must implement :meth:`derive`.
    """

    @classmethod
    @abstractmethod
    def derive(cls: type[_TDerivedView], parent: _TParent, **kwargs) -> _TDerivedView:
        """Construct this view from its upstream parent.

        :param parent: The upstream data source to derive from.
        :returns: A new derived view built from ``parent``.
        """
        ...
