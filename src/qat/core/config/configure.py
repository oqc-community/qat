# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from contextlib import contextmanager
from contextvars import ContextVar

from qat.core.config.session import QatSessionConfig
from qat.purr.qatconfig import QatConfig

_global_config: QatSessionConfig | None = None
_session_config: ContextVar[QatSessionConfig | None] = ContextVar(
    "session_config", default=None
)


def get_qatconfig() -> QatConfig:
    """Returns the global QatConfig"""
    global _global_config
    if _global_config is None:
        _global_config = QatConfig()
    return _global_config


def get_config() -> QatSessionConfig | QatConfig:
    """Returns the session config if set or the global QatConfig if not"""
    if (session_config := _session_config.get()) is not None:
        return session_config

    return get_qatconfig()


@contextmanager
def override_config(config: QatSessionConfig):
    token = _session_config.set(config)
    try:
        yield
    finally:
        _session_config.reset(token)
