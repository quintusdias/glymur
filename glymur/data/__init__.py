"""Shipping JPEG 2000 files.

There is only one JP2 file at the moment, "nemo.jp2", converted from the
original JPEG photo of the aftermath of NEMO, the nor'easter that shutdown
Boston in February of 2013.
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
