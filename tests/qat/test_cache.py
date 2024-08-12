# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import os
import unittest
from pathlib import Path

from qat.purr.compiler.caches import QatCache


class TestInstructions(unittest.TestCase):
    def test_create_delete(self):
        root = os.path.realpath(os.path.join(__file__, ".."))
        cache = QatCache(root)
        cache.create_cache_folders()
        assert Path(cache.ll_cache).exists()
        assert Path(cache.qs_cache).exists()
        assert Path(cache.qat_cache).exists()
        cache.delete_cache_folders()
        assert not Path(cache.ll_cache).exists()
        assert not Path(cache.qs_cache).exists()
        assert not Path(cache.qat_cache).exists()
