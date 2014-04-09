"""glymur - read, write, and interrogate JPEG 2000 files
"""
import sys
import unittest

from glymur import version
__version__ = version.version

from .jp2k import Jp2k
from .jp2dump import jp2dump
from .jp2box import get_printoptions, set_printoptions
from .jp2box import get_parseoptions, set_parseoptions

from . import data

def runtests():
    """Discover and run all tests for the glymur package.
    """
    suite = unittest.defaultTestLoader.discover(__path__[0])
    unittest.TextTestRunner(verbosity=2).run(suite)
