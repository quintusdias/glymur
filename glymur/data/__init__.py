"""Shipping JPEG 2000 files.

These include:
    nemo.jp2:  converted from the original JPEG photo of the aftermath of NEMO,
        the nor'easter that shutdown Boston in February of 2013.
    goodstuff.j2k:  my favorite bevorage.

"""
import importlib.resources as ir


def nemo():
    """Shortcut for specifying path to nemo.jp2.

    Returns
    -------
    file : str
        Platform-independent path to nemo.jp2.
    """
    with ir.path('glymur.data', 'nemo.jp2') as filename:
        return str(filename)


def goodstuff():
    """Shortcut for specifying path to goodstuff.j2k.

    Returns
    -------
    file : str
        Platform-independent path to goodstuff.j2k.
    """
    with ir.path('glymur.data', 'goodstuff.j2k') as filename:
        return str(filename)


def jpxfile():
    """Shortcut for specifying path to heliov.jpx.

    Returns
    -------
    file : str
        Platform-independent path to 12-v6.4.jpx
    """
    with ir.path('glymur.data', 'heliov.jpx') as filename:
        return str(filename)
