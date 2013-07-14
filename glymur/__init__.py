"""glymur - read, write, and interrogate JPEG 2000 files
"""
import sys

from .jp2k import Jp2k
from .jp2dump import jp2dump

from . import data
from . import test


def runtests():
    """Discover and run all tests for the glymur package.
    """
    if sys.hexversion <= 0x02070000:
        import unittest2 as unittest
    else:
        import unittest
    suite = unittest.defaultTestLoader.discover(__path__[0])
    unittest.TextTestRunner(verbosity=2).run(suite)
