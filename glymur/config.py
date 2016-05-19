"""
Configure glymur to use installed libraries if possible.
"""
import copy
import ctypes
from ctypes.util import find_library
import os
import platform
import sys
import warnings

if sys.hexversion <= 0x03000000:
    from ConfigParser import SafeConfigParser as ConfigParser
    from ConfigParser import NoOptionError, NoSectionError
else:
    from configparser import ConfigParser
    from configparser import NoOptionError, NoSectionError

# default library locations for MacPorts
_macports_default_location = {'openjp2': '/opt/local/lib/libopenjp2.dylib',
                              'openjpeg': '/opt/local/lib/libopenjpeg.dylib'}

# default library locations on Windows
_windows_default_location = {'openjp2': os.path.join('C:\\',
                                                     'Program files',
                                                     'OpenJPEG 2.0',
                                                     'bin',
                                                     'openjp2.dll'),
                             'openjpeg': os.path.join('C:\\',
                                                      'Program files',
                                                      'OpenJPEG 1.5',
                                                      'bin',
                                                      'openjpeg.dll')}


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


def load_openjpeg_library(libname):

    # A location specified by the glymur configuration file has precedence.
    path = read_config_file(libname)
    if path is not None:
        return load_library_handle(libname, path)

    # Attempt to locate in the usual location in Anaconda.
    if 'Anaconda' in sys.version:
        if platform.system() == 'Linux':
            suffix = '.so'
            basedir = os.path.dirname(os.path.dirname(sys.executable))
            lib = os.path.join(basedir, 'lib', 'lib' + libname + suffix)
        elif platform.system() == 'Darwin':
            suffix = '.dylib'
            basedir = os.path.dirname(os.path.dirname(sys.executable))
            lib = os.path.join(basedir, 'lib', 'lib' + libname + suffix)
        elif platform.system() == 'Windows':
            suffix = '.dll'
            basedir = os.path.dirname(sys.executable)
            lib = os.path.join(basedir, 'Library', 'bin', libname + suffix)

        if os.path.exists(lib):
            path = lib

    # No location specified by the configuration file, must look for it
    # elsewhere.  Here we attempt to locate it in the usual system-dependent
    # locations.  This works in Anaconda/windows, but not Anaconda/{mac,linux}.
    if path is None:
        path = find_library(libname)

    # Last gasp.
    if path is None:
        if platform.system() == 'Darwin':
            path = _macports_default_location[libname]
        elif platform.system == 'Windows':
            path = _windows_default_location[libname]

        if path is not None and not os.path.exists(path):
            # the mac/win default location does not exist.
            return None
    return load_library_handle(libname, path)


def load_library_handle(libname, path):
    """Load the library, return the ctypes handle."""

    if path is None or path in ['None', 'none']:
        # Either could not find a library via ctypes or
        # user-configuration-file, or we could not find it in any of the
        # default locations, or possibly the user intentionally does not want
        # one of the libraries to load.
        return None

    try:
        if os.name == "nt":
            opj_lib = ctypes.windll.LoadLibrary(path)
        else:
            opj_lib = ctypes.CDLL(path)
    except (TypeError, OSError):
        msg = 'The {libname} library at {path} could not be loaded.'
        msg = msg.format(path=path, libname=libname)
        warnings.warn(msg, UserWarning)
        opj_lib = None

    return opj_lib


def read_config_file(libname):
    """
    Extract library locations from a configuration file.

    Parameters
    ----------
    libname : str
        One of either 'openjp2' or 'openjpeg'

    Returns
    -------
    path : None or str
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
    return path


def glymur_config():
    """
    Try to ascertain locations of openjp2, openjpeg libraries.

    Returns
    -------
    tuple
        tuple of library handles
    """
    handles = (load_openjpeg_library(x) for x in ['openjp2', 'openjpeg'])
    handles = tuple(handles)

    if all(handle is None for handle in handles):
        msg = "Neither the openjp2 nor the openjpeg library could be loaded.  "
        warnings.warn(msg)
    return handles


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


_original_options = {
    'parse.full_codestream': False,
    'print.xml': True,
    'print.codestream': True,
    'print.short': False,
}
_options = copy.deepcopy(_original_options)


def set_option(key, value):
    """Set the value of the specified option.

    Available options:

        parse.full_codestream
        print.xml
        print.codestream
        print.short

    Parameters
    ----------
    key : str
        Name of a single option.
    value :
        New value of option.

    Option Descriptions
    -------------------
    parse.full_codestream : bool
        When False, only the codestream header is parsed for metadata.  This
        can results in faster JP2/JPX parsing.  When True, the entire
        codestream is parsed. [default: False]
    print.codestream : bool
        When False, the codestream segments are not printed.  Otherwise the
        segments are printed depending on the value of the
        parse.full_codestream option. [default: True]
    print.short : bool
        When True, only the box ID, offset, and length are displayed.  Useful
        for displaying only the basic structure or skeleton of a JPEG 2000
        file. [default: False]
    print.xml : bool
        When False, printing of the XML contents of any XML boxes or UUID XMP
        boxes is suppressed. [default: True]

    See also
    --------
    get_option
    """
    if key not in _options.keys():
        raise KeyError('{key} not valid.'.format(key=key))
    _options[key] = value


def get_option(key):
    """Return the value of the specified option

    Available options:

        parse.full_codestream
        print.xml
        print.codestream
        print.short

    Parameter
    ---------
    key : str
        Name of a single option.

    Returns
    -------
    result : the value of the option.

    See also
    --------
    set_option
    """
    return _options[key]


def reset_option(key):
    """
    Reset one or more options to their default value.

    Pass "all" as argument to reset all options.

    Available options:

        parse.full_codestream
        print.xml
        print.codestream
        print.short

    Parameter
    ---------
    key : str
        Name of a single option.
    """
    global _options
    if key == 'all':
        _options = copy.deepcopy(_original_options)
    else:
        if key not in _options.keys():
            raise KeyError('{key} not valid.'.format(key=key))
        _options[key] = _original_options[key]


def set_parseoptions(full_codestream=True):
    """Set parsing options.

    These options determine the way JPEG 2000 boxes are parsed.

    Parameters
    ----------
    full_codestream : bool, defaults to True
        When False, only the codestream header is parsed for metadata.  This
        can results in faster JP2/JPX parsing.  When True, the entire
        codestream is parsed for metadata.

    See also
    --------
    get_parseoptions

    Examples
    --------
    To put back the default options, you can use:

    >>> import glymur
    >>> glymur.set_parseoptions(full_codestream=True)
    """
    warnings.warn('Use set_option instead of set_parseoptions.',
                  DeprecationWarning)
    set_option('parse.full_codestream', full_codestream)


def get_parseoptions():
    """Return the current parsing options.

    Returns
    -------
    dict
        Dictionary of current print options with keys

          - codestream : bool

        For a full description of these options, see `set_parseoptions`.

    See also
    --------
    set_parseoptions
    """
    warnings.warn('Use set_option instead of set_parseoptions.',
                  DeprecationWarning)
    return {'full_codestream': get_option('parse.full_codestream')}


def set_printoptions(**kwargs):
    """Set printing options.

    These options determine the way JPEG 2000 boxes are displayed.

    Parameters
    ----------
    short : bool, optional
        When True, only the box ID, offset, and length are displayed.  Useful
        for displaying only the basic structure or skeleton of a JPEG 2000
        file.
    xml : bool, optional
        When False, printing of the XML contents of any XML boxes or UUID XMP
        boxes is suppressed.
    codestream : bool, optional
        When False, the codestream segments are not printed.  Otherwise the
        segments are printed depending on how set_parseoptions has been used.

    See also
    --------
    get_printoptions

    Examples
    --------
    To put back the default options, you can use:

    >>> import glymur
    >>> glymur.set_printoptions(short=False, xml=True, codestream=True)
    """
    warnings.warn('Use set_option instead of set_printoptions.',
                  DeprecationWarning)
    for key, value in kwargs.items():
        if key not in ['short', 'xml', 'codestream']:
            raise TypeError('"{0}" not a valid keyword parameter.'.format(key))
        set_option('print.' + key, value)


def get_printoptions():
    """Return the current print options.

    Returns
    -------
    dict
        Dictionary of current print options with keys

          - short : bool
          - xml : bool
          - codestream : bool

        For a full description of these options, see `set_printoptions`.

    See also
    --------
    set_printoptions
    """
    warnings.warn('Use get_option instead of get_printoptions.',
                  DeprecationWarning)
    d = {}
    for key in ['short', 'xml', 'codestream']:
        d[key] = _options['print.' + key]
    return d
