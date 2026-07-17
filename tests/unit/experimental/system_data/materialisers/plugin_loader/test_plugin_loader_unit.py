# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers import plugin_loader


def test_load_builtin_plugins_imports_all_configured_paths(monkeypatch):
    monkeypatch.setattr(
        plugin_loader,
        "_BUILTIN_PLUGIN_MODULES",
        ("module.one", "module.two"),
    )

    imported_paths = []

    def _fake_import(module_path):
        imported_paths.append(module_path)
        return object()

    monkeypatch.setattr(plugin_loader.importlib, "import_module", _fake_import)

    plugin_loader.load_builtin_plugins()

    assert imported_paths == ["module.one", "module.two"]


def test_load_builtin_plugins_propagates_import_errors(monkeypatch):
    monkeypatch.setattr(plugin_loader, "_BUILTIN_PLUGIN_MODULES", ("missing.module",))

    def _failing_import(_module_path):
        raise ModuleNotFoundError("missing.module")

    monkeypatch.setattr(plugin_loader.importlib, "import_module", _failing_import)

    with pytest.raises(ModuleNotFoundError, match="missing.module"):
        plugin_loader.load_builtin_plugins()
