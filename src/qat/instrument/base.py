# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import csv
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress

from qat.engines import ConnectionMixin
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class InstrumentModel(BaseModel):
    """
    Used to parse JSON/CSV entries. An instrument has an id, name, and IP address
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(frozen=True, alias="ID")
    name: str = Field(frozen=True, alias="NAME")
    address: IPvAnyAddress = Field(frozen=True, alias="ADDRESS")


class InstrumentConcept(ConnectionMixin):
    """
    The component interface, it defines what an instrument is and what APIs through which
    customer code can interact with.
    """

    def connect(self):
        pass

    def disconnect(self):
        pass

    def setup(self, *args, **kwargs):
        pass

    def playback(self, *args, **kwargs):
        pass


class LeafInstrument(InstrumentConcept):
    def __init__(self, id: str, name: str, address: str):
        self.id = id
        self.name = name
        self.address = address

        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool):
        self._is_connected = value

    def __repr__(self):
        return f"{self.id}_{self.name}_{self.address}"

    def __str__(self):
        return self.__repr__()


LInstr = TypeVar("LInstr", bound=LeafInstrument)


class CompositeInstrument(Generic[LInstr], InstrumentConcept):
    def __init__(self):
        self._components: dict[str, LInstr] = {}

    @property
    def components(self) -> dict[str, LInstr]:
        return self._components

    @property
    def is_connected(self) -> bool:
        return all((component.is_connected for component in self._components.values()))

    def connect(self):
        for comp in self._components.values():
            comp.connect()

    def disconnect(self):
        for comp in self._components.values():
            comp.disconnect()

    def add(self, component: LInstr):
        if next(
            (comp for id, comp in self._components.items() if id == component.id),
            None,
        ):
            raise ValueError(f"Instrument {component} already exists")
        else:
            self._components[component.id] = component

    def setup(self, *args, **kwargs):
        for comp in self._components.values():
            comp.setup(*args, **kwargs)

    def playback(self, *args, **kwargs):
        component_results = [
            comp.playback(*args, **kwargs) for comp in self._components.values()
        ]

        results = {}
        for comp_res in component_results:
            if comp_res.keys() & results.keys():
                raise ValueError("Component results have conflicting keys")
            results.update(comp_res)

        return results

    def __repr__(self):
        return "\n".join(repr(comp) for comp in self._components.values())

    def __str__(self):
        return "\n".join(str(comp) for comp in self._components.values())


CInstr = TypeVar("CInstr", bound=CompositeInstrument)


class InstrumentBuilder(ABC):
    @abstractmethod
    def build(self, *args, **kwargs) -> CInstr: ...


class CsvInstrumentBuilder(InstrumentBuilder):
    def __init__(
        self, file_path: str | Path, cinstr_type: type = None, linstr_type: type = None
    ):
        self.file_path = file_path
        self.cinstr_type = cinstr_type or CompositeInstrument
        self.linstr_type = linstr_type or LeafInstrument

    def build(self) -> CInstr:
        """
        A generic function that builds an InstrumentConcept object representing
        an arbitrary fleet of leaf instruments defined as CSV.

        :param cinstr_type: type of Composite Instrument to build
        :param linstr_type: type of Leaf Instrument to build
        """

        if not os.path.exists(self.file_path):
            raise ValueError(f"File '{self.file_path}' not found!")

        composite = self.cinstr_type()
        with open(self.file_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                instr_model = InstrumentModel.model_validate(row)
                composite.add(
                    self.linstr_type(
                        id=instr_model.id,
                        name=instr_model.name,
                        address=str(instr_model.address),
                    )
                )

        return composite


class ConfigInstrumentBuilder(InstrumentBuilder):
    """
    Builds an InstrumentConcept object representing an arbitrary fleet of leaf instruments
    defined as a list of config dicts.

    :param cinstr_type: type of Composite Instrument to build
    :param linstr_type: type of Leaf Instrument to build
    """

    def __init__(
        self, configs: list[dict], cinstr_type: type = None, linstr_type: type = None
    ):
        self.configs = configs
        self.cinstr_type = cinstr_type or CompositeInstrument
        self.linstr_type = linstr_type or LeafInstrument

    def build(self) -> CInstr:
        composite = self.cinstr_type()
        for config in self.configs:
            instr_model = InstrumentModel.model_validate(config)
            composite.add(
                self.linstr_type(
                    id=instr_model.id,
                    name=instr_model.name,
                    address=str(instr_model.address),
                )
            )

        return composite
