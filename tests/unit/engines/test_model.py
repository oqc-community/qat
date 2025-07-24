# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.engines import ZeroEngine
from qat.engines.model import requires_hardware_model
from qat.model.loaders.purr import EchoModelLoader

from tests.unit.utils.engines import MockEngineWithModel
from tests.unit.utils.loaders import MockModelLoader


class TestRequiresHardwareModelMixin:
    def test_update_with_new_model(self):
        loader = MockModelLoader()
        model = loader.load()
        engine = MockEngineWithModel(model)
        assert engine.model == model
        new_model = loader.load()
        assert new_model != model
        engine.model = new_model
        assert engine.model == new_model
        assert engine.num_changes == 1

    def test_update_with_same_model(self):
        loader = MockModelLoader()
        model = loader.load()
        engine = MockEngineWithModel(model)
        assert engine.model == model
        engine.model = model
        assert engine.num_changes == 0


class TestRequiresHardwareModel:
    # TODO: delete with function (COMPILER-XXX)

    def test_requires_hardware_model_with_legacy(self):
        model = EchoModelLoader().load()
        engine = model.create_engine()
        assert requires_hardware_model(engine)

    def test_requires_hardware_model_with_mixin(self):
        model = EchoModelLoader().load()
        engine = MockEngineWithModel(model)
        assert requires_hardware_model(engine)

    def test_requires_hardware_model_without_mixin(self):
        EchoModelLoader().load()
        engine = ZeroEngine()
        assert not requires_hardware_model(engine)
