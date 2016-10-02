"""glymur - read, write, and interrogate JPEG 2000 files
"""
# Standard library imports ...
import unittest

# Local imports
from glymur import version
from .jp2k import Jp2k
from .config import (get_option, set_option, reset_option,
                     get_printoptions, set_printoptions,
                     get_parseoptions, set_parseoptions)
from . import data

__version__ = version.version


def runtests():
    """Discover and run all tests for the glymur package.
    """
    suite = unittest.defaultTestLoader.discover(__path__[0])
    unittest.TextTestRunner(verbosity=2).run(suite)


__all__ = [__version__, Jp2k, get_printoptions, set_printoptions,
           get_parseoptions, set_parseoptions, get_option, set_option,
           reset_option, data, runtests]
