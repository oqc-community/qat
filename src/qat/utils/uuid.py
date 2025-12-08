# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import uuid as py_uuid
from random import Random

uuid_randomiser = Random()


def uuid():
    return py_uuid.UUID(int=uuid_randomiser.getrandbits(128), version=4)


uuid4 = uuid
