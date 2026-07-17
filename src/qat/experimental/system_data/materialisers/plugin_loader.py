# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Built-in plugin module loader for materialiser boundary bootstrap."""

import importlib

_BUILTIN_PLUGIN_MODULES = ("qat.experimental.system_data.materialisers.purr.plugin",)


def load_builtin_plugins() -> None:
    """Import built-in plugin modules for import-side registration."""

    for module_path in _BUILTIN_PLUGIN_MODULES:
        importlib.import_module(module_path)
