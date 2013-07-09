"""
Configure glymur to use installed libraries if possible.
"""
import ctypes
from ctypes.util import find_library
import os
import platform

import sys
if sys.hexversion <= 0x03000000:
    from ConfigParser import SafeConfigParser as ConfigParser
else:
    from configparser import ConfigParser


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

    # Either GLYMURCONFIGDIR/glymurrc or $HOME/.glymur/glymurrc
    confdir = get_configdir()
    if confdir is not None:
        fname = os.path.join(confdir, 'glymurrc')
        if os.path.exists(fname):
            return fname

    # didn't find a configuration file.
    return None


def get_openjpeg_config():
    libopenjpeg_path = find_library('openjpeg')
    
    # If we could not find it, then look in some likely locations.
    if libopenjpeg_path is None:
        if platform.system() == 'Darwin':
            path = '/opt/local/lib/libopenjpeg.dylib'
            if os.path.exists(path):
                libopenjpeg_path = path
        elif os.name == 'nt':
            path = os.path.join('C:\\', 'Program files', 'OpenJPEG 1.5',
                                'bin', 'openjpeg.dll')
            if os.path.exists(path):
                libopenjpeg_path = path

    try:
        if os.name == "nt":
            openjpeg_lib = ctypes.windll.LoadLibrary(libopenjpeg_path)
        else:    
            openjpeg_lib = ctypes.CDLL(libopenjpeg_path)
    except OSError:
        openjpeg_lib = None
        
    if openjpeg_lib is not None:
        # Must be at least 1.5.0
        openjpeg_lib.opj_version.restype = ctypes.c_char_p
        v = openjpeg_lib.opj_version()
        v = v.decode('utf-8')
        major, minor, patch = v.split('.')
        if minor != '5':
            openjpeg_lib = None
    return openjpeg_lib


def get_openjp2_config():
    filename = glymurrc_fname()
    if filename is not None:
        # Read the configuration file for the library location.
        parser = ConfigParser()
        parser.read(filename)
        libopenjp2_path = parser.get('library', 'openjp2')
    else:
        # No help from the config file, try to find it ourselves.
        libopenjp2_path = find_library('openjp2')

    if libopenjp2_path is None:
        return None

    try:
        if os.name == "nt":
            openjp2_lib = ctypes.windll.LoadLibrary(libopenjp2_path)
        else:
            openjp2_lib = ctypes.CDLL(libopenjp2_path)
    except OSError:
        msg = '"Library {0}" could not be loaded.  Operating in degraded mode.'
        msg = msg.format(libopenjp2_path)
        warnings.warn(msg, UserWarning)
        openjp2_lib = None
    return openjp2_lib


def glymur_config():
    """Try to ascertain locations of openjp2, openjpeg libraries.
    """
    openjp2_lib = get_openjp2_config()
    openjpeg_lib = get_openjpeg_config()
    return openjp2_lib, openjpeg_lib

def get_configdir():
    """Return string representing the configuration directory.

    Default is $HOME/.config/glymur.  You can override this with the
    XDG_CONFIG_HOME environment variable.
    """

    if 'XDG_CONFIG_HOME' in os.environ:
        return os.path.join(os.environ['XDG_CONFIG_HOME'], 'glymur')

    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.config', 'glymur')

    if 'USERPROFILE' in os.environ:
        # Windows?
        return os.path.join(os.environ['USERPROFILE'], 'Application Data', 'glymur')
