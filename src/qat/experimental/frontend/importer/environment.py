# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from typing import Generic, TypeVar

from bidict import bidict
from xdsl.ir import Attribute, SSAValue

_AttributeType = TypeVar("_AttributeType", bound=Attribute)


class EnvironmentTracker(Generic[_AttributeType]):
    """Models an environment to track named variables and their current SSA values.

    This is not designed to support stacked environments or deal with phi nodes; it is
    intended to support simple linear programs that translate a linear list of operations,
    but has scope for generalisations in the future.

    This is a simple wrapper around a :class:`bidict` that provides a more convenient
    interface for looking up and setting variables by name. It encapsulates the
    responsibility of tracking SSA values without exposing the underlying implementation.
    """

    def __init__(self) -> None:
        self._env: bidict[str, SSAValue[_AttributeType]] = bidict()

    def get_by_name(
        self, name: str, default: SSAValue[_AttributeType] | None = None
    ) -> SSAValue[_AttributeType] | None:
        """Get the current SSA value for a variable by name.

        :param name: The name of the variable to look up.
        :param default: The default value to return if the variable is not found.
        :returns: The current SSA value for the variable, or *default* when the name is not
            present.
        """
        return self._env.get(name, default)

    def set_by_name(self, name: str, value: SSAValue[_AttributeType]) -> None:
        """Set the current SSA value for a variable by name.

        :param name: The name of the variable to set.
        :param value: The SSA value to associate with the variable.
        """
        self._env.forceput(name, value)

    def set_by_value(
        self, value: SSAValue[_AttributeType], new_value: SSAValue[_AttributeType]
    ) -> None:
        """Set the current SSA value for a variable by value.

        :param value: The SSA value to associate with the variable.
        :param new_value: The new SSA value to associate with the variable.
        """
        self._env.forceput(self._env.inverse[value], new_value)

    def items(self):
        """Returns name and value pairs."""
        return self._env.items()
