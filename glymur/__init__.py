"""glymur - read, write, and interrogate JPEG 2000 files
"""
# Local imports
from glymur import version
from .options import (get_option, set_option, reset_option,
                      get_printoptions, set_printoptions,
                      get_parseoptions, set_parseoptions)
from .jp2k import Jp2k
from .tiff import Tiff2Jp2k
from . import data

__version__ = version.version


__all__ = [
    __version__, Jp2k, Tiff2Jp2k, get_printoptions, set_printoptions,
    get_parseoptions, set_parseoptions, get_option, set_option, reset_option,
    data
]
