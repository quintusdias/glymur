"""
Configure glymur to use installed libraries if possible.
"""
from configparser import ConfigParser, NoOptionError, NoSectionError
import ctypes
from ctypes.util import find_library
import os
import pathlib
import platform
import warnings


def glymurrc_fname():
    """Return the path to the configuration file.

    Search order:
        1) current working directory
        2) environ var XDG_CONFIG_HOME
        3) $HOME/.config/glymur/glymurrc
    """

    # Current directory.
    path = pathlib.Path.cwd() / 'glymurrc'
    if path.exists():
        return path

    confdir_path = get_configdir()
    if confdir_path is not None:
        path = confdir_path / 'glymurrc'
        if path.exists():
            return path

    # didn't find a configuration file.
    return None


def _determine_full_path(libname):
    """
    Try to determine the path to the library.

    Parameters
    ----------
    libname : str
        short name for library (openjp2)

    Returns
    -------
    path to openjp2 library or None if openjp2 library not found
    """

    # A location specified by the glymur configuration file has precedence.
    path = read_config_file(libname)
    if path is not None:
        return path

    # No joy on config file.  Cygwin?  Cygwin is a bit of an odd case.
    if platform.system().startswith('CYGWIN'):
        g = pathlib.Path('/usr/bin').glob('cygopenjp2*.dll')
        try:
            path = list(g)[0]
        except IndexError:
            # openjpeg possibly not installed
            pass
        else:
            if path.exists():
                return path

    # No joy on config file and not Cygwin.  Can ctypes find it anyway?
    path = find_library(libname)
    if path is not None:
        return pathlib.Path(path)
    else:
        return None


def read_config_file(libname):
    """
    Extract library locations from a configuration file.

    Parameters
    ----------
    libname : str
        One of either 'openjp2' or 'openjpeg'

    Returns
    -------
    path : None or path
        None if no location is specified, otherwise a path to the library
    """
    filename = glymurrc_fname()
    if filename is None:
        # There's no library file path to return in this case.
        return None

    # Read the configuration file for the library location.
    parser = ConfigParser()
    parser.read(filename)
    try:
        path = parser.get('library', libname)
    except (NoOptionError, NoSectionError):
        path = None
    else:
        # Turn it into a pathlib object.
        path = pathlib.Path(path)
    return path


def glymur_config(libname):
    """
    Try to ascertain locations of openjp2 library.

    Parameters
    ----------
    libname : str
        Currently either 'openjp2' or 'tiff'

    Returns
    -------
    loaded shared library
    """
    if platform.system().startswith('Windows') and libname == 'c':
        return ctypes.cdll.msvcrt

    path = _determine_full_path(libname)

    if path is None or path in ['None', 'none']:
        # Either could not find a library via ctypes or
        # user-configuration-file, or we could not find it in any of the
        # default locations, or possibly the user intentionally does not want
        # one of the libraries to load.
        return None

    loader = ctypes.windll.LoadLibrary if os.name == 'nt' else ctypes.CDLL
    try:
        opj_lib = loader(path)
    except TypeError:
        # This can happen on Windows.  Apparently ctypes.windll.LoadLibrary
        # is no longer taking a WindowsPath
        path = str(path)
        opj_lib = loader(path)
    except OSError:
        msg = f'The {libname} library at {path} could not be loaded.'
        warnings.warn(msg, UserWarning)
        opj_lib = None

    return opj_lib


def get_configdir():
    """Return string representing the configuration directory.

    Default is $HOME/.config/glymur.  You can override this with the
    XDG_CONFIG_HOME environment variable.
    """
    if 'XDG_CONFIG_HOME' in os.environ:
        return pathlib.Path(os.environ['XDG_CONFIG_HOME']) / 'glymur'

    if 'HOME' in os.environ and platform.system() != 'Windows':
        # HOME is set by WinPython to something unusual, so we don't
        # necessarily want that.
        return pathlib.Path(os.environ['HOME']) / '.config' / 'glymur'

    # Last stand.  Should handle windows... others?
    return pathlib.Path.home() / 'glymur'
