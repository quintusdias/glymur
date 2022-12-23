"""Shipping JPEG 2000 files.

These include:
    nemo.jp2:  converted from the original JPEG photo of the aftermath of NEMO,
        the nor'easter that shutdown Boston in February of 2013.
    goodstuff.j2k:  my favorite bevorage.

"""
import importlib.resources as ir
import sys


def nemo():
    """Shortcut for specifying path to nemo.jp2.

    Returns
    -------
    file : str
        Platform-independent path to nemo.jp2.
    """
    return _str_path_to('nemo.jp2')


def goodstuff():
    """Shortcut for specifying path to goodstuff.j2k.

    Returns
    -------
    file : str
        Platform-independent path to goodstuff.j2k.
    """
    return _str_path_to('goodstuff.j2k')


def jpxfile():
    """Shortcut for specifying path to heliov.jpx.

    Returns
    -------
    file : str
        Platform-independent path to heliov.jpx
    """
    return _str_path_to('heliov.jpx')


def _str_path_to(filename):
    """Hide differences between 3.9.0 and below."""
    if sys.version_info[1] >= 9:
        return str(ir.files('glymur.data').joinpath(filename))
    else:
        with ir.path('glymur.data', filename) as path:
            return str(path)
