"""glymur - read, write, and interrogate JPEG 2000 files
"""
import sys

from glymur import version
__version__ = version.version

from .jp2k import Jp2k
from .jp2dump import jp2dump

from . import data


# unittest2 only in python-2.6 (pylint/python2.7 issue)
# pylint: disable=F0401
def runtests():
    """Discover and run all tests for the glymur package.
    """
    if sys.hexversion <= 0x02070000:
        import unittest2 as unittest
    else:
        import unittest
    suite = unittest.defaultTestLoader.discover(__path__[0])
    unittest.TextTestRunner(verbosity=2).run(suite)
