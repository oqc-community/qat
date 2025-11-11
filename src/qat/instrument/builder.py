import csv
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

from qat.instrument.base import (
    CInstr,
    CompositeInstrument,
    InstrumentModel,
    LeafInstrument,
    LInstr,
)


class InstrumentBuilder(ABC):
    @abstractmethod
    def build(self, *args, **kwargs) -> CInstr: ...


class CsvInstrumentBuilder(InstrumentBuilder):
    def __init__(self, file_path: str | Path):
        self.file_path = file_path

    def build(
        self, cinstr_type: Type[CInstr] = None, linstr_type: Type[LInstr] = None
    ) -> CInstr:
        """
        A generic function that builds an InstrumentConcept object representing
        an arbitrary fleet of leaf instruments defined as CSV.

        :param cinstr_type: type of Composite Instrument to build
        :param linstr_type: type of Leaf Instrument to build
        """

        cinstr_type = cinstr_type or CompositeInstrument
        linstr_type = linstr_type or LeafInstrument

        if not os.path.exists(self.file_path):
            raise ValueError(f"File '{self.file_path}' not found!")

        composite = cinstr_type()
        with open(self.file_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                instr_model = InstrumentModel.model_validate(row)
                composite.add(
                    linstr_type(
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

    def __init__(self, configs: list[dict]):
        self.configs = configs

    def build(
        self, cinstr_type: Type[CInstr] = None, linstr_type: Type[LInstr] = None
    ) -> CInstr:
        cinstr_type = cinstr_type or CompositeInstrument
        linstr_type = linstr_type or LeafInstrument

        composite = cinstr_type()
        for config in self.configs:
            instr_model = InstrumentModel.model_validate(config)
            composite.add(
                linstr_type(
                    id=instr_model.id,
                    name=instr_model.name,
                    address=str(instr_model.address),
                )
            )

        return composite
