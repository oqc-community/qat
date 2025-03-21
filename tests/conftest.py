import pytest


@pytest.fixture
def testpath(pytestconfig):
    return pytestconfig.rootpath / "tests"
