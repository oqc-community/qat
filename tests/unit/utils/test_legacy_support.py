# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from qat.purr.compiler.caches import QatCache
from qat.purr.utils.serializer import (
    CustomQatJsonDecoder,
    json_dump,
    json_dumps,
    json_load,
    json_loads,
)


def test_qat_cache_creates_and_deletes_directories(tmp_path):
    cache = QatCache(tmp_path)

    cache.create_cache_folders()

    assert Path(cache.ll_cache).is_dir()
    assert Path(cache.qs_cache).is_dir()
    assert Path(cache.qat_cache).is_dir()

    cache.delete_cache_folders()

    assert not Path(cache.qat_root).exists()


def test_qat_cache_reports_delete_errors(tmp_path, mocker):
    warning = mocker.patch("qat.purr.compiler.caches.get_default_logger").return_value.warn
    mocker.patch(
        "qat.purr.compiler.caches.shutil.rmtree",
        side_effect=OSError("unavailable"),
    )

    QatCache(tmp_path).delete_cache_folders()

    warning.assert_called_once()


def test_serializer_round_trips_numpy_arrays():
    array = np.array([[1, 2], [3, 4]], dtype=np.int16)

    encoded = json_dumps(array)
    decoded = json_loads(encoded)

    np.testing.assert_array_equal(decoded, array)
    assert decoded.dtype == array.dtype


def test_serializer_supports_files_and_legacy_array_lists():
    stream = StringIO()
    json_dump({"values": [1, 2]}, stream)
    stream.seek(0)

    assert json_load(stream) == {"values": [1, 2]}
    np.testing.assert_array_equal(
        json_loads('{"type": "numpyarray", "list": [1, 2]}'),
        [1, 2],
    )


def test_serializer_relinks_hardware_components():
    model = type(
        "Model",
        (),
        {"get_device": lambda self, component_id: f"device:{component_id}"},
    )()

    assert json_loads('{"$component_id": "q0"}', model=model) == "device:q0"

    decoder = CustomQatJsonDecoder()
    assert decoder.default(1) == 1
    with pytest.raises(ValueError, match="requires re-linking"):
        decoder.default({"$component_id": "q0"})
