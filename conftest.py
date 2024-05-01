import pytest
import glymur


@pytest.fixture(autouse=True)
def add_glymur(doctest_namespace):
    doctest_namespace['glymur'] = glymur
