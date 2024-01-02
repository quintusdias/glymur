"""
Doc tests
"""
# Standard library imports ...
import doctest
import os

# Third party library imports ...

# Local imports
import glymur


def docTearDown(doctest_obj):  # pragma: no cover
    glymur.set_option('parse.full_codestream', False)


# Doc tests should be run as well.
def load_tests(loader, tests, ignore):  # pragma: no cover
    """Should run doc tests as well"""
    if os.name == "nt":
        # Can't do it on windows, temporary file issue.
        return tests
    if glymur.lib.openjp2.OPENJP2 is not None:
        tests.addTests(
            doctest.DocTestSuite('glymur.jp2k', tearDown=docTearDown)
        )
        tests.addTests(
            doctest.DocTestSuite('glymur.jp2kr', tearDown=docTearDown)
        )
    return tests
