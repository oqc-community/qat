# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import abc


class QatExtension:
    """Seperately installed QatExtensions can be registered in qatconfig and will be loaded automatically by calling the load method

    These are used internally at OQC for proprietary extensions.
    """

    @staticmethod
    @abc.abstractmethod
    def load():
        pass
