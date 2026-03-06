# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import uuid as py_uuid
from contextlib import contextmanager
from contextvars import ContextVar
from random import Random

uuid_randomiser = Random()
_randomiser_stack: ContextVar[tuple[Random, ...]] = ContextVar(
    "_randomiser_stack", default=()
)
SeedType = int | float | str | bytes | bytearray


def _active_randomiser() -> Random:
    randomiser_stack = _randomiser_stack.get()
    return randomiser_stack[-1] if randomiser_stack else uuid_randomiser


def uuid():
    return py_uuid.UUID(int=_active_randomiser().getrandbits(128), version=4)


uuid4 = uuid


# Context manager for temporary seeding
@contextmanager
def temporary_uuid_seed(seed: SeedType | None = None):
    """
    Context manager to temporarily seed the uuid generation for reproducibility.
    Only affects uuid() and uuid4() in this module, for the current context.

    This is re-entrant and thread-safe: nested contexts are supported and random
    state is isolated between concurrent contexts.
    """
    randomiser_stack = _randomiser_stack.get()
    token = _randomiser_stack.set((*randomiser_stack, Random(seed)))
    try:
        yield
    finally:
        _randomiser_stack.reset(token)
