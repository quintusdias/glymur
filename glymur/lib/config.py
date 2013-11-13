"""
Configure glymur to use installed libraries if possible.
"""
# configparser is new in python3 (pylint/python-2.7)
# pylint: disable=F0401

import ctypes
from ctypes.util import find_library
import os
import platform
import warnings

import sys
if sys.hexversion <= 0x03000000:
    from ConfigParser import SafeConfigParser as ConfigParser
    from ConfigParser import NoOptionError
else:
    from configparser import ConfigParser
    from configparser import NoOptionError


def glymurrc_fname():
    """Return the path to the configuration file.

    Search order:
        1) current working directory
        2) environ var XDG_CONFIG_HOME
        3) $HOME/.config/glymur/glymurrc
    """

    # Current directory.
    fname = os.path.join(os.getcwd(), 'glymurrc')
    if os.path.exists(fname):
        return fname

    confdir = get_configdir()
    if confdir is not None:
        fname = os.path.join(confdir, 'glymurrc')
        if os.path.exists(fname):
            return fname

    # didn't find a configuration file.
    return None


def load_openjpeg(path):
    """Load the openjpeg library, falling back on defaults if necessary.

    Parameters
    ----------
    path : str
        Path to openjpeg 1.5 library as specified by configuration file.  Will
        be None if no configuration file specified.
    """
    if path is None:
        # Let ctypes try to find it.
        path = find_library('openjpeg')

    # If we could not find it, then look in some likely locations on mac
    # and win.
    if path is None:
        # Could not find a library via ctypes
        if platform.system() == 'Darwin':
            # MacPorts
            path = '/opt/local/lib/libopenjpeg.dylib'
        elif os.name == 'nt':
            path = os.path.join('C:\\', 'Program files', 'OpenJPEG 1.5',
                                'bin', 'openjpeg.dll')

        if path is not None and not os.path.exists(path):
            # the mac/win default location does not exist.
            return None

    return load_library_handle(path)


def load_openjp2(path):
    """Load the openjp2 library, falling back on defaults if necessary.
    """
    if path is None:
        # No help from the config file, try to find it via ctypes.
        path = find_library('openjp2')

    if path is None:
        # Could not find a library via ctypes
        if platform.system() == 'Darwin':
            # MacPorts
            path = '/opt/local/lib/libopenjp2.dylib'
        elif os.name == 'nt':
            path = os.path.join('C:\\', 'Program files', 'OpenJPEG 2.0',
                                'bin', 'openjp2.dll')

        if path is not None and not os.path.exists(path):
            # the mac/win default location does not exist.
            return None

    return load_library_handle(path)


def load_library_handle(path):
    """Load the library, return the ctypes handle."""

    if path is None:
        # Either could not find a library via ctypes or user-configuration-file,
        # or we could not find it in any of the default locations.
        # This is probably a very old linux.
        return None

    try:
        if os.name == "nt":
            opj_lib = ctypes.windll.LoadLibrary(path)
        else:
            opj_lib = ctypes.CDLL(path)
    except (TypeError, OSError):
       msg = '"Library {0}" could not be loaded.  Operating in degraded mode.'
       msg = msg.format(path)
       warnings.warn(msg, UserWarning)
       opj_lib = None

    return opj_lib


def read_config_file():
    """
    We must use a configuration file that the user must write.
    """
    lib = {'openjp2':  None, 'openjpeg':  None}
    filename = glymurrc_fname()
    if filename is not None:
        # Read the configuration file for the library location.
        parser = ConfigParser()
        parser.read(filename)
        try:
            lib['openjp2'] = parser.get('library', 'openjp2')
        except NoOptionError:
            pass
        try:
            lib['openjpeg'] = parser.get('library', 'openjpeg')
        except NoOptionError:
            pass

    return lib


def glymur_config():
    """Try to ascertain locations of openjp2, openjpeg libraries.
    """
    libs = read_config_file()
    libopenjp2_handle = load_openjp2(libs['openjp2'])
    libopenjpeg_handle = load_openjpeg(libs['openjpeg'])
    if libopenjp2_handle is None and libopenjpeg_handle is None:
        msg = "Neither the openjp2 nor the openjpeg library could be loaded.  "
        msg += "Operating in severely degraded mode."
        warnings.warn(msg, UserWarning)
    return libopenjp2_handle, libopenjpeg_handle


def get_configdir():
    """Return string representing the configuration directory.

    Default is $HOME/.config/glymur.  You can override this with the
    XDG_CONFIG_HOME environment variable.
    """
    if 'XDG_CONFIG_HOME' in os.environ:
        return os.path.join(os.environ['XDG_CONFIG_HOME'], 'glymur')

    if 'HOME' in os.environ and os.name != 'nt':
        # HOME is set by WinPython to something unusual, so we don't
        # necessarily want that.
        return os.path.join(os.environ['HOME'], '.config', 'glymur')

    # Last stand.  Should handle windows... others?
    return os.path.join(os.path.expanduser('~'), 'glymur')
