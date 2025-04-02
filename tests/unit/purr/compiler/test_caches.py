# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd
from pathlib import Path

from qat.purr.compiler.caches import QatCache


class TestInstructions:
    def test_create_delete(self, testpath):
        cache = QatCache(testpath)
        cache.create_cache_folders()
        assert Path(cache.ll_cache).exists()
        assert Path(cache.qs_cache).exists()
        assert Path(cache.qat_cache).exists()
        cache.delete_cache_folders()
        assert not Path(cache.ll_cache).exists()
        assert not Path(cache.qs_cache).exists()
        assert not Path(cache.qat_cache).exists()
