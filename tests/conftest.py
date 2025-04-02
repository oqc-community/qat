import pytest


@pytest.fixture(scope="session")
def testpath(pytestconfig):
    return pytestconfig.rootpath / "tests"


tests_dir = None


def pytest_configure(config):
    # Set global tests_dir path
    global tests_dir
    tests_dir = config.rootpath / "tests"
