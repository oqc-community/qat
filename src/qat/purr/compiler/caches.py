# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import os
import shutil
from pathlib import Path

from qat.purr.utils.logger import get_default_logger


class QatCache:
    """
    Folder control object for the various caches of the toolchain. Consolidates
    creation/deletion and validation.
    """

    def __init__(self, root_folder=None):
        if root_folder is None:
            root_folder = os.path.realpath(os.path.join(__file__, "..", "..", "..", ".."))

        self.qat_root = os.path.join(root_folder, ".qat")
        self.ll_cache = os.path.realpath(os.path.join(self.qat_root, "ll"))
        self.qs_cache = os.path.realpath(os.path.join(self.qat_root, "qs"))
        self.qat_cache = os.path.realpath(os.path.join(self.qat_root, "qat"))

    def create_cache_folders(self):
        """Creates folders if they don't exist."""
        Path(self.ll_cache).mkdir(parents=True, exist_ok=True)
        Path(self.qs_cache).mkdir(parents=True, exist_ok=True)
        Path(self.qat_cache).mkdir(parents=True, exist_ok=True)

    def delete_cache_folders(self):
        try:
            shutil.rmtree(self.qat_root)
        except OSError as e:
            get_default_logger().warn(
                f"Attempted to delete cache folders at {self.qat_root}, "
                f"threw exception. {str(e)}"
            )
