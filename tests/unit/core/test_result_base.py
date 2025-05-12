# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from dataclasses import dataclass

from qat.core.result_base import ResultInfoMixin, ResultManager


@dataclass
class MockResult(ResultInfoMixin):
    val: str


class TestResultsManager:
    def test_result_overwrites(self):
        res = ResultManager()
        res.add(MockResult("test1"))
        res.add(MockResult("test2"))
        assert res.lookup_by_type(MockResult).val == "test2"
