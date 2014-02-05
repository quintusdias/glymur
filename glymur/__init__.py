"""glymur - read, write, and interrogate JPEG 2000 files
"""
import sys
import unittest

from glymur import version
__version__ = version.version

from .jp2k import Jp2k
from .jp2dump import jp2dump

from . import data

def runtests():
    """Discover and run all tests for the glymur package.
    """
    suite = unittest.defaultTestLoader.discover(__path__[0])
    unittest.TextTestRunner(verbosity=2).run(suite)
