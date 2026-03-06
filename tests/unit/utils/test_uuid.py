# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import threading
import uuid as py_uuid
from random import Random

from qat.utils.uuid import temporary_uuid_seed, uuid4

THREAD_WAIT_TIMEOUT_SECONDS = 15


def _expected_uuids(seed, count):
    randomiser = Random(seed)
    return [py_uuid.UUID(int=randomiser.getrandbits(128), version=4) for _ in range(count)]


def test_temporary_uuid_seed_is_deterministic():
    expected = _expected_uuids(123, 3)

    with temporary_uuid_seed(123):
        actual = [uuid4() for _ in range(3)]

    assert actual == expected


def test_temporary_uuid_seed_is_reentrant():
    expected_outer = _expected_uuids(100, 2)
    expected_inner = _expected_uuids(200, 1)

    with temporary_uuid_seed(100):
        first_outer = uuid4()
        with temporary_uuid_seed(200):
            inner = uuid4()
        second_outer = uuid4()

    assert first_outer == expected_outer[0]
    assert inner == expected_inner[0]
    assert second_outer == expected_outer[1]


def test_temporary_uuid_seed_is_thread_isolated_for_overlapping_contexts():
    expected_thread_a = _expected_uuids(111, 1)[0]
    expected_thread_b = _expected_uuids(222, 1)[0]

    thread_a_entered = threading.Event()
    thread_b_entered = threading.Event()
    thread_a_generated = threading.Event()
    results = {}

    def run_thread_a():
        with temporary_uuid_seed(111):
            thread_a_entered.set()
            if not thread_b_entered.wait(timeout=THREAD_WAIT_TIMEOUT_SECONDS):
                results["a_error"] = (
                    "thread A timed out waiting for thread B to enter seeded context"
                )
                return
            results["a"] = uuid4()
            thread_a_generated.set()

    def run_thread_b():
        if not thread_a_entered.wait(timeout=THREAD_WAIT_TIMEOUT_SECONDS):
            results["b_error"] = "thread B timed out waiting for thread A to start"
            return
        with temporary_uuid_seed(222):
            thread_b_entered.set()
            if not thread_a_generated.wait(timeout=THREAD_WAIT_TIMEOUT_SECONDS):
                results["b_error"] = (
                    "thread B timed out waiting for thread A to generate a UUID"
                )
                return
            results["b"] = uuid4()

    a = threading.Thread(target=run_thread_a)
    b = threading.Thread(target=run_thread_b)
    a.start()
    b.start()
    a.join(timeout=THREAD_WAIT_TIMEOUT_SECONDS)
    b.join(timeout=THREAD_WAIT_TIMEOUT_SECONDS)

    assert not a.is_alive(), (
        f"thread A did not finish within {THREAD_WAIT_TIMEOUT_SECONDS}s"
    )
    assert not b.is_alive(), (
        f"thread B did not finish within {THREAD_WAIT_TIMEOUT_SECONDS}s"
    )
    assert "a_error" not in results, results.get("a_error")
    assert "b_error" not in results, results.get("b_error")
    assert results["a"] == expected_thread_a
    assert results["b"] == expected_thread_b
