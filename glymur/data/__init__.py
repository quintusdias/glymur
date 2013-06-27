"""Shipping JPEG 2000 files.

These include:
    nemo.jp2:  converted from the original JPEG photo of the aftermath of NEMO,
        the nor'easter that shutdown Boston in February of 2013.
    goodstuff.j2k:  my favorite bevorage.

"""
import pkg_resources


def nemo():
    """Shortcut for specifying path to nemo.jp2.

    Returns
    -------
    file : str
        Platform-independent path to nemo.jp2.
    """
    file = pkg_resources.resource_filename(__name__, "nemo.jp2")
    return file


def goodstuff():
    """Shortcut for specifying path to goodstuff.j2k.

    Returns
    -------
    file : str
        Platform-independent path to goodstuff.j2k.
    """
    file = pkg_resources.resource_filename(__name__, "goodstuff.j2k")
    return file
