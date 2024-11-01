# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.frontends import QASMFrontend
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.qatconfig import QatConfig
from qat.qat import QATInput, fetch_frontend


class QAT:
    def __init__(self, hardware_model: Optional[QuantumHardwareModel] = None):
        self.config = QatConfig()
        self.hardware_model = hardware_model

    def compile(self, program: QATInput, compiler_config: Optional[CompilerConfig] = None):
        # TODO: Replace frontend.parse with pass manager pipeline
        frontend = fetch_frontend(program)
        return frontend.parse(
            program, hardware=self.hardware_model, compiler_config=compiler_config
        )

    def execute(
        self,
        instructions: InstructionBuilder,
        compiler_config: Optional[CompilerConfig] = None,
    ):
        # TODO: Replace frontend.execute with pass manager pipeline
        frontend = QASMFrontend()
        return frontend.execute(instructions, self.hardware_model, compiler_config)

    @property
    def hardware_model(self):
        return self._hardware_model

    @hardware_model.setter
    def hardware_model(self, model: Optional[QuantumHardwareModel]):
        if model is None:
            model = get_default_RTCS_hardware()
        elif not isinstance(model, QuantumHardwareModel):
            raise ValueError(
                f"Expected value of type 'QuanutmHardwareModel', got type '{type(model)}'"
            )
        self._hardware_model = model
