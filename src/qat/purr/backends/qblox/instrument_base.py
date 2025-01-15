# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from abc import ABC
from typing import Dict

from pydantic import BaseModel, Field, IPvAnyAddress

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class InstrumentModel(BaseModel):
    """
    Used to parse JSON/CSV entries. An instrument has an id, name, and IP address
    """

    id: str = Field(frozen=True, alias="ID")
    name: str = Field(frozen=True, alias="NAME")
    address: IPvAnyAddress = Field(frozen=True, alias="ADDRESS")


class InstrumentConcept(ABC):
    """
    Basic APIs through which an instrument must be able to interact with
    """

    def connect(self):
        pass

    def disconnect(self):
        pass

    def close(self):
        pass

    def upload(self, *args, **kwargs):
        pass

    def playback(self, *args, **kwargs):
        pass

    def collect(self, *args, **kwargs):
        pass


class LeafInstrument(InstrumentConcept):
    def __init__(self, parse_model: InstrumentModel):
        self.id: str = parse_model.id
        self.name: str = parse_model.name
        self.address: str = str(parse_model.address)

        self.is_connected = False

    def __repr__(self):
        return f"{self.id}_{self.name}_{self.address}"

    def __str__(self):
        return self.__repr__()


class CompositeInstrument(InstrumentConcept):
    def __init__(self):
        self.components: Dict[str, LeafInstrument] = {}

    def connect(self):
        for comp in self.components.values():
            comp.connect()

    def disconnect(self):
        for comp in self.components.values():
            comp.disconnect()

    def close(self):
        for comp in self.components.values():
            comp.close()

    def add(self, component: LeafInstrument):
        comp = next(
            (comp for comp in self.components.values() if comp.id == component.id),
            None,
        )
        if comp:
            log.warning(f"Instrument {component} already exists")
        else:
            self.components[component.id] = component

    def remove(self, component: LeafInstrument):
        if component.id in self.components:
            del self.components[component.id]

    def lookup_by_name(self, name: str) -> LeafInstrument:
        comp = next((comp for comp in self.components.values() if comp.name == name), None)
        if not comp:
            raise ValueError(f"Could not find instrument with name {name}")
        return comp
