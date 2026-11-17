# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import itertools
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qat.backend.base import AllocatingBackend


class DummyAllocatingBackend(AllocatingBackend):
    def emit(self, ir, res_mgr=None, met_mgr=None, **kwargs):
        raise NotImplementedError


def make_target(target_id: str, slot_idx: int, sequencers: list[int]):
    config = SimpleNamespace(sequencers=dict.fromkeys(sequencers))
    physical_channel = SimpleNamespace(slot_idx=slot_idx, config=config)
    return SimpleNamespace(full_id=lambda: target_id, physical_channel=physical_channel)


@pytest.fixture
def backend():
    return DummyAllocatingBackend(model=None)


class TestAllocateBipartite:
    @pytest.mark.parametrize("flexible_first", [True, False])
    def test_restricted_target_gets_its_only_sequencer(self, backend, flexible_first):
        flexible = make_target("flexible", 1, [0, 1])
        restricted = make_target("restricted", 1, [0])
        targets = [flexible, restricted] if flexible_first else [restricted, flexible]

        allocations = backend.allocate_bipartite(targets)

        assert allocations == {1: {"flexible": 1, "restricted": 0}}

    def test_slots_are_allocated_independently(self, backend):
        targets = [make_target("a", 1, [0]), make_target("b", 2, [0])]

        allocations = backend.allocate_bipartite(targets)

        assert allocations == {1: {"a": 0}, 2: {"b": 0}}

    def test_no_targets(self, backend):
        assert backend.allocate_bipartite([]) == {}

    @pytest.mark.parametrize(
        "sequencers",
        [
            pytest.param([[0], [0]], id="more-targets-than-sequencers"),
            pytest.param([[0], []], id="target-without-sequencers"),
        ],
    )
    def test_unallocatable_targets_raise(self, backend, sequencers):
        targets = [make_target(f"t{i}", 1, seqs) for i, seqs in enumerate(sequencers)]

        with pytest.raises(ValueError, match="Could not allocate a sequencer"):
            backend.allocate_bipartite(targets)

    def test_error_names_every_unplaced_target(self, backend):
        targets = [make_target(f"t{i}", 3, [0, 1]) for i in range(4)]

        with pytest.raises(ValueError, match=r"2 of 4") as exc_info:
            backend.allocate_bipartite(targets)

        message = str(exc_info.value)
        assert sum(f"t{i} (slot 3" in message for i in range(4)) == 2

    @pytest.mark.parametrize("order", itertools.permutations(range(4)))
    def test_assignment_is_independent_of_target_order(self, backend, order):
        candidates = [[0, 1, 2, 3], [1, 2, 3], [2, 3, 4], [0, 4]]
        targets = [make_target(f"t{i}", 1, seqs) for i, seqs in enumerate(candidates)]

        allocations = backend.allocate_bipartite([targets[i] for i in order])

        assert allocations == backend.allocate_bipartite(targets)

    def test_assignment_is_independent_of_hash_seed(self):
        script = (
            "from tests.unit.backend.test_base import DummyAllocatingBackend, make_target\n"
            "targets = [make_target(name, 1, [0, 1, 2, 3]) for name in 'abcd']\n"
            "print(dict(DummyAllocatingBackend(model=None).allocate_bipartite(targets)))\n"
        )
        outputs = {
            subprocess.run(  # noqa: S603
                [sys.executable, "-c", script],
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for seed in range(8)
        }
        assert len(outputs) == 1
