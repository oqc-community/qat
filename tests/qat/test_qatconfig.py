# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest

from qat import qatconfig


@pytest.mark.parametrize("repeat_limits", [10_000, 50_000, 16_874, 100_000])
def test_changed_max_repeats_limit(repeat_limits):
    # Default value is 100_000. Test manual change of repeats limit.
    qatconfig.MAX_REPEATS_LIMIT = repeat_limits
    assert qatconfig.MAX_REPEATS_LIMIT == repeat_limits
