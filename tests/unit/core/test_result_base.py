# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd
from dataclasses import dataclass
from uuid import UUID

import pytest

from qat.core.result_base import ResultInfoMixin, ResultManager, ResultModel


@dataclass
class MockResult(ResultInfoMixin):
    val: str


@dataclass
class OtherResult(ResultInfoMixin):
    val: str


class TestResultsManager:
    def test_result_overwrites(self):
        res = ResultManager()
        res.add(MockResult("test1"))
        res.add(MockResult("test2"))
        assert res.lookup_by_type(MockResult).val == "test2"

    def test_result_lifecycle(self):
        result = MockResult("result")
        manager = ResultManager()

        manager.add(result)

        wrapped = next(iter(manager.results))
        assert wrapped.value is result
        assert isinstance(wrapped.id, UUID)
        assert hash(wrapped) == hash(wrapped.id)
        assert manager.check_for_type(MockResult)

        manager.mark_as_dirty(result)
        manager.cleanup()

        assert not manager.results
        assert not manager.check_for_type(MockResult)

    def test_remove_result_object(self):
        result = MockResult("result")
        manager = ResultManager()
        manager.add(result)

        manager._remove(result)

        assert not manager.results

    def test_update_combines_result_managers(self):
        first = ResultManager()
        second = ResultManager()
        first.add(MockResult("first"))
        second.add(OtherResult("second"))

        first.update(second)

        assert first.lookup_by_type(MockResult).val == "first"
        assert first.lookup_by_type(OtherResult).val == "second"

    def test_update_rejects_incompatible_type(self):
        with pytest.raises(ValueError, match="Invalid type"):
            ResultManager().update(object())

    @pytest.mark.parametrize("operation", ["lookup_by_type", "remove_by_type"])
    def test_type_operation_rejects_missing_result(self, operation):
        manager = ResultManager()

        with pytest.raises(ValueError, match="Could not find any results"):
            getattr(manager, operation)(MockResult)

    @pytest.mark.parametrize("operation", ["lookup_by_type", "remove_by_type"])
    def test_type_operation_rejects_ambiguous_results(self, operation):
        manager = ResultManager()
        manager.results.update(
            {
                ResultModel(MockResult("first")),
                ResultModel(MockResult("second")),
            }
        )

        with pytest.raises(ValueError, match="Found multiple results"):
            getattr(manager, operation)(MockResult)

    def test_remove_by_type(self):
        manager = ResultManager()
        manager.add(MockResult("result"))

        manager.remove_by_type(MockResult)

        assert not manager.results

    @pytest.mark.parametrize(
        "results",
        [
            pytest.param([], id="missing"),
            pytest.param(
                [
                    ResultModel(MockResult("first")),
                    ResultModel(MockResult("first")),
                ],
                id="ambiguous",
            ),
        ],
    )
    def test_remove_result_object_rejects_invalid_match_count(self, results):
        manager = ResultManager()
        manager.results.update(results)

        expected = "Could not find result" if not results else "Found multiple results"
        with pytest.raises(ValueError, match=expected):
            manager._remove(MockResult("first"))
