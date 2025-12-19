# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "a-b_tests" in item.path.parts:
            item.add_marker(pytest.mark.ab_test)


def pytest_addoption(parser):
    parser.addoption(
        "--ab-enable", action="store_true", default=False, help="enable A/B tests"
    )
    parser.addoption(
        "--ab-skip", action="store_true", default=False, help="skip A/B tests, default"
    )
    parser.addoption("--ab-only", action="store_true", default=False, help="only A/B tests")
    parser.addoption(
        "--ab-grouping",
        action="store_true",
        default=False,
        help="group benchmark results to show A/B comparison",
    )


def pytest_configure(config):
    setattr(config.option, "markexpr", "not ab_test")
    if config.option.ab_enable:
        setattr(config.option, "markexpr", "ab_test or not ab_test")
    if config.option.ab_only:
        setattr(config.option, "markexpr", "ab_test")
    if config.option.ab_grouping:
        setattr(config.option, "benchmark_sort", "name")
        setattr(config.option, "benchmark_group_by", "group,func,param:key")
        setattr(config.option, "benchmark_enable", True)
