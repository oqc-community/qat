# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.model.loaders.purr import QbloxModelLoader
from qat.model.loaders.purr.qblox import DEFAULT_DUMMY_CONFIG, DEFAULT_QUBIT_COUNT
from qat.model.loaders.qblox import create_instrument
from qat.utils.uuid import temporary_uuid_seed, uuid4


@pytest.fixture(scope="module")
def qblox_instrument(request, module_seed):
    with temporary_uuid_seed(module_seed):
        id = f"test_{uuid4()}".replace("-", "_")

    if not request.param:
        request.param = {}
    address = request.param.get("address", None)
    dummy_config = request.param.get("dummy_config", DEFAULT_DUMMY_CONFIG)

    instrument = create_instrument(id, id, address, dummy_config)
    instrument.connect()
    yield instrument
    instrument.disconnect()


@pytest.fixture(scope="module")
def qblox_model(request, module_seed):
    with temporary_uuid_seed(module_seed):
        id = f"test_{uuid4()}".replace("-", "_")

    if not request.param:
        request.param = {}
    address = request.param.get("address", None)
    dummy_config = request.param.get("dummy_config", DEFAULT_DUMMY_CONFIG)
    qubit_count = request.param.get("qubit_count", DEFAULT_QUBIT_COUNT)

    hw_model = QbloxModelLoader(
        id=id,
        name=id,
        address=address,
        dummy_config=dummy_config,
        qubit_count=qubit_count,
    ).load()
    yield hw_model
