"""Manage glymur configuration settings."""
# Standard library imports
import copy
import warnings

# Local imports
from . import version
from .lib import openjp2 as opj2


_original_options = {
    'lib.num_threads': 1,
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
        lib.num_threads

    Parameters
    ----------
    key : str
        Name of a single option.
    value :
        New value of option.

    Option Descriptions
    -------------------
    lib.num_threads : int
        Set the number of threads used to decode an image.  This option is only
        available with OpenJPEG 2.2.0 or higher.
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

    Examples
    --------
    >>> glymur.set_option('print.short', True)
    >>> glymur.reset_option('all')

    See also
    --------
    get_option
    """
    if key not in _options.keys():
        raise KeyError(f'{key} not valid.')

    if key == 'lib.num_threads':
        if version.openjpeg_version < '2.2.0':
            msg = (
                f'Thread support is not available on versions of OpenJPEG '
                f'prior to 2.2.0.  Your version is '
                f'{version.openjpeg_version}.'
            )
            raise RuntimeError(msg)
        if not opj2.has_thread_support():
            msg = 'The OpenJPEG library is not configured with thread support.'
            raise RuntimeError(msg)

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
    """Reset one or more options to their default value.

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
            raise KeyError(f'{key} not valid.')
        _options[key] = _original_options[key]


def set_parseoptions(full_codestream=True):
    warnings.warn('Use set_option instead of set_parseoptions.',
                  DeprecationWarning)
    set_option('parse.full_codestream', full_codestream)


def get_parseoptions():
    warnings.warn('Use set_option instead of set_parseoptions.',
                  DeprecationWarning)
    return {'full_codestream': get_option('parse.full_codestream')}


def set_printoptions(**kwargs):
    warnings.warn('Use set_option instead of set_printoptions.',
                  DeprecationWarning)
    for key, value in kwargs.items():
        if key not in ['short', 'xml', 'codestream']:
            raise KeyError(f'"{key}" not a valid keyword parameter.')
        set_option('print.' + key, value)


def get_printoptions():
    warnings.warn('Use get_option instead of get_printoptions.',
                  DeprecationWarning)
    d = {}
    for key in ['short', 'xml', 'codestream']:
        d[key] = _options['print.' + key]
    return d
