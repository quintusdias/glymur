"""glymur - read, write, and interrogate JPEG 2000 files
"""

from .jp2k import Jp2k
from .jp2dump import jp2dump

from . import data
from . import test


def runtests():
    """Discover and run all tests for the glymur package.
    """
    import unittest
    suite = unittest.defaultTestLoader.discover(__path__[0])
    unittest.TextTestRunner(verbosity=2).run(suite)
