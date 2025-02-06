# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.compiler.analysis_passes import BatchedShots, BatchedShotsResult
from qat.passes.result_base import ResultManager
from qat.purr.backends.echo import get_default_echo_hardware


class TestBatchedShots:

    @pytest.mark.parametrize("shots", [0, 1000, 10000])
    def test_shots_less_than_equal_to_max_gives_expected(self, shots):
        model = get_default_echo_hardware()
        model.repeat_limit = 10000
        builder = model.create_builder()
        builder.repeat(shots)
        batch_pass = BatchedShots(model)
        res_mgr = ResultManager()
        batch_pass.run(builder, res_mgr)
        batch_res = res_mgr.lookup_by_type(BatchedShotsResult)
        assert batch_res.total_shots == shots
        assert batch_res.batched_shots == shots

    def test_no_repeat_instruction(self):
        model = get_default_echo_hardware()
        builder = model.create_builder()
        batch_pass = BatchedShots(model)
        res_mgr = ResultManager()
        batch_pass.run(builder, res_mgr)
        batch_res = res_mgr.lookup_by_type(BatchedShotsResult)
        assert batch_res.total_shots == model.default_repeat_count
        assert batch_res.batched_shots == model.default_repeat_count

    @pytest.mark.parametrize("shots", [10001, 20000, 29999])
    def test_shots_greater_than_max_gives_appropiate_batches(self, shots):
        model = get_default_echo_hardware()
        model.repeat_limit = 10000
        builder = model.create_builder()
        builder.repeat(shots)
        batch_pass = BatchedShots(model)
        res_mgr = ResultManager()
        batch_pass.run(builder, res_mgr)
        batch_res = res_mgr.lookup_by_type(BatchedShotsResult)
        assert batch_res.total_shots == shots
        assert batch_res.batched_shots <= shots
        assert batch_res.batched_shots * np.ceil(shots / model.repeat_limit) >= shots
