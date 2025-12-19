# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.extensions import QatExtension

loaded_extensions = set()


class SomeExtension(QatExtension):
    @staticmethod
    def load():
        loaded_extensions.add("SomeExtension")


class AnotherExtension(QatExtension):
    @staticmethod
    def load():
        loaded_extensions.add("AnotherExtension")


class NotAnExtension:
    @staticmethod
    def load(): ...
