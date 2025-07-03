# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from typing import Dict

from pydantic import BaseModel, Field, IPvAnyAddress

from qat.engines import ConnectionMixin
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class InstrumentModel(BaseModel):
    """
    Used to parse JSON/CSV entries. An instrument has an id, name, and IP address
    """

    id: str = Field(frozen=True, alias="ID")
    name: str = Field(frozen=True, alias="NAME")
    address: IPvAnyAddress = Field(frozen=True, alias="ADDRESS")


class InstrumentMixin(ConnectionMixin):
    """
    Basic APIs through which an instrument must be able to interact with
    """

    def setup(self, *args, **kwargs):
        pass

    def playback(self, *args, **kwargs):
        pass

    def collect(self, *args, **kwargs):
        pass


class LeafInstrument(InstrumentModel, InstrumentMixin):
    def __repr__(self):
        return f"{self.id}_{self.name}_{self.address}"

    def __str__(self):
        return self.__repr__()


class CompositeInstrument(InstrumentMixin):
    def __init__(self):
        self._components: Dict[str, LeafInstrument] = {}

    def connect(self):
        for comp in self._components.values():
            comp.connect()

    def disconnect(self):
        for comp in self._components.values():
            comp.disconnect()

    def add(self, component: LeafInstrument):
        comp = next(
            (comp for comp in self._components.values() if comp.id == component.id),
            None,
        )
        if comp:
            raise ValueError(f"Instrument {component} already exists")
        else:
            self._components[component.id] = component

    def remove(self, component: LeafInstrument):
        if component.id in self._components:
            del self._components[component.id]
        else:
            log.warning(f"Instrument {component} could not be found, nothing to do")

    def lookup_by_name(self, name: str) -> LeafInstrument:
        comp = next((comp for comp in self._components.values() if comp.name == name), None)
        if not comp:
            raise ValueError(f"Could not find instrument with name {name}")
        return comp
