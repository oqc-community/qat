import pytest


@pytest.fixture(scope="session")
def testpath(pytestconfig):
    return pytestconfig.rootpath / "tests"


tests_dir = None


def pytest_addoption(parser):
    parser.addoption(
        "--experimental-enable",
        action="store_const",
        const=0,
        dest="experimental",
        default=None,
        help="run experimental tests",
    )
    parser.addoption(
        "--experimental-only",
        action="store_const",
        const=1,
        dest="experimental",
        help="run only integration tests",
    )
    parser.addoption(
        "--legacy-enable",
        action="store_const",
        const=1,
        dest="legacy",
        default=1,
        help="run legacy tests",
    )
    parser.addoption(
        "--legacy-only",
        action="store_const",
        const=1,
        dest="legacy",
        default=-1,
        help="run legacy tests",
    )


def pytest_configure(config):
    # Set global tests_dir path
    global tests_dir
    tests_dir = config.rootpath / "tests"
    mark_string = config.option.markexpr
    if config.option.experimental is None:
        config.option.experimental = -1
    mark_list = [mark_string] if len(mark_string) > 0 else []
    for marker in ["experimental", "legacy"]:
        if marker in mark_string:
            continue
        val = getattr(config.option, marker)
        if val == 1:
            mark_list.append(marker)
        elif val == -1:
            mark_list.append(f"not {marker}")
    setattr(config.option, "markexpr", " and ".join(mark_list))
