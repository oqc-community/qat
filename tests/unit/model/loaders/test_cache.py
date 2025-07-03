# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.model.loaders.cache import CacheAccessLoader


class TestCacheAccessLoader:
    def test_load(self):
        """Test that the CacheAccessLoader can load a model from the cache."""
        cache = {"test_model": "This is a test model"}
        loader = CacheAccessLoader(cache, "test_model")
        assert loader.load() == "This is a test model"

        cache["test_model"] = "This is an updated test model"
        assert loader.load() == "This is an updated test model"
