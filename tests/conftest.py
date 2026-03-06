# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import pytest

pytest_plugins = ("tests.plugins.seeding", "tests.plugins.qblox")


@pytest.fixture(scope="session")
def testpath(pytestconfig):
    return pytestconfig.rootpath / "tests"


@pytest.fixture
def tmp_cwd(monkeypatch, tmp_path):
    """Use a unique temporary directory for each test. Allows concurrent runs that rely on
    the same local files, e.g. tmp.hex and tmp.q1asm as used in qblox tests."""
    monkeypatch.chdir(tmp_path)


tests_dir = None


def pytest_addoption(parser):
    parser.addoption(
        "--experimental-enable",
        action="store_const",
        const=0,
        dest="experimental",
        default=-1,
        help="run experimental tests",
    )
    parser.addoption(
        "--experimental-only",
        action="store_const",
        const=1,
        dest="experimental",
        help="run only experimental tests",
    )
    parser.addoption(
        "--legacy-enable",
        action="store_const",
        const=0,
        dest="legacy",
        default=-1,
        help="run legacy tests",
    )
    parser.addoption(
        "--legacy-only",
        action="store_const",
        const=1,
        dest="legacy",
        help="run only legacy tests",
    )
    parser.addoption(
        "--qblox-enable",
        action="store_const",
        const=0,
        dest="qblox",
        default=0,
        help="run qblox tests",
    )
    parser.addoption(
        "--qblox-disable",
        action="store_const",
        const=-1,
        dest="qblox",
        help="skip qblox tests",
    )
    parser.addoption(
        "--qblox-only",
        action="store_const",
        const=1,
        dest="qblox",
        help="run only qblox tests",
    )


def pytest_configure(config):
    # Set global tests_dir path
    global tests_dir
    tests_dir = config.rootpath / "tests"
    mark_string = config.option.markexpr
    mark_list = [mark_string] if len(mark_string) > 0 else []
    for marker in ["experimental", "legacy", "qblox"]:
        if marker in mark_string:
            continue
        val = getattr(config.option, marker)
        if val == 1:
            mark_list.append(marker)
        elif val == -1:
            mark_list.append(f"not {marker}")
    setattr(config.option, "markexpr", " and ".join(mark_list))


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "qblox" in item.path.parts:
            item.add_marker(pytest.mark.qblox)
        if "legacy" in item.path.parts or item.path.is_relative_to(
            config.rootpath / "tests" / "unit" / "purr"
        ):
            item.add_marker(pytest.mark.legacy)


def pytest_sessionfinish(session, exitstatus):
    if session.config.option.experimental == 1 and exitstatus == 5:
        # No tests collected due to experimental-only flag
        session.exitstatus = 0
