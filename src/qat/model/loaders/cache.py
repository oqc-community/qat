# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.loaders.base import BaseModelLoader


class CacheAccessLoader(BaseModelLoader):
    """A loader that loads a hardware model from a cache of hardware models.

    While primarily used for qatconfig, this loader can be applied to any cache that can be
    indexed by a hardware model name."""

    def __init__(self, cache, name: str):
        """
        :param cache: An object that can be indexed by a hardware model name, such as a
            dict.
        :param name: The name of the hardware model to load from the cache.
        """
        self._cache = cache
        self._name = name

    def load(self) -> BaseModelLoader:
        """Loads a hardware model from the cache."""
        return self._cache[self._name]
