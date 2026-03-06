# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import random
from hashlib import sha256

import numpy as np
import pytest


def _generate_seed(test_name: str, offset: int = 0) -> int:
    digest = sha256(test_name.encode("utf-8")).digest()
    base_seed = int.from_bytes(digest[:4], byteorder="big")
    return (base_seed + offset) % (2**32)


@pytest.fixture(scope="function")
def function_seed(request):
    return _generate_seed(request.node.nodeid)


@pytest.fixture(scope="class")
def class_seed(request):
    module_name = request.module.__name__
    class_name = request.cls.__qualname__ if request.cls is not None else "<no-class>"
    return _generate_seed(f"{module_name}::{class_name}", offset=1)


@pytest.fixture(scope="module")
def module_seed(request):
    return _generate_seed(request.module.__name__)


@pytest.fixture(autouse=True)
def deterministic_rng(function_seed):
    # optional: auto-seed Python + NumPy per test
    random.seed(function_seed)
    np.random.seed(function_seed)
