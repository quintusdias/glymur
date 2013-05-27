"""glymur - read, write, and interrogate JPEG 2000 files
"""


def _glymurrc_fname():
    """Return the path to the configuration file.

    Search order:
        1) current working directory
        2) environ var GLYMURCONFIGDIR
        3) HOME/.glymur/glymurrc
    """

    # Current directory.
    fname = os.path.join(os.getcwd(), 'glymurrc')
    if os.path.exists(fname):
        return fname

    # environ var GLYMURCONFIGDIR
    if 'GLYMURCONFIGDIR' in os.environ:
        path = os.environ['GLYMURCONFIGDIR']
        if os.path.exists(path):
            fname = os.path.join(path, 'glymurrc')
            if os.path.exists(fname):
                return fname

    # HOME/.glymur/jptoolrsc
    confdir = _get_configdir()
    if confdir is not None:
        fname = os.path.join(_get_configdir(), 'glymurrc')
        if os.path.exists(fname):
            return fname

    # didn't find a configuration file.
    return None


def _config():
    """Read configuration file.

    Based on matplotlib.
    """
    filename = _glymurrc_fname()
    if filename is not None:
        # Read the configuration file for the library location.
        parser = ConfigParser()
        parser.read(filename)
        try:
            libopenjp2_path = parser.get('library', 'openjp2')
            if not os.path.exists(libopenjp2_path):
                msg = 'OpenJP2 library path specified by configuration file '
                msg += 'does not exist.'
                raise ImportError(msg)
        except NoOptionError as e:
            msg = "Error parsing configuration file '{0}':  {1}"
            msg = msg.format(fname, e.message)
            warnings.warn(msg, UserWarning)
        except:
            raise
    else:
        # No help from the config file, try to find it ourselves.
        from ctypes.util import find_library
        libopenjp2_path = find_library('openjp2')
        if libopenjp2_path is None:
            raise ImportError('OpenJP2 library not found.')

    _OPENJP2 = ctypes.CDLL(libopenjp2_path)
    return _OPENJP2


def _get_configdir():
    """Return string representing the configuration directory.

    Default is HOME/.glymur.  You can override this with the GLYMURCONFIGDIR
    environment variable.
    """

    if 'GLYMURCONFIGDIR' in os.environ:
        return os.environ['GLYMURCONFIGDIR']

    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.glymur')

import sys
if sys.hexversion <= 0x03000000:
    from ConfigParser import SafeConfigParser as ConfigParser
    from ConfigParser import NoOptionError
else:
    from configparser import ConfigParser
    from configparser import NoOptionError
import ctypes
import os

_OPENJP2 = _config()

from .jp2k import Jp2k
from .jp2dump import jp2dump

from . import test
