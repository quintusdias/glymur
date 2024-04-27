"""Shipping JPEG 2000 files.

These include:
    nemo.jp2:  converted from the original JPEG photo of the aftermath of NEMO,
        the nor'easter that shutdown Boston in February of 2013.
    goodstuff.j2k:  my favorite beverage.

"""
import importlib.resources as ir


def nemo():
    """Shortcut for specifying path to nemo.jp2.

    Returns
    -------
    file : str
        Platform-independent path to nemo.jp2.
    """
    return str(ir.files('glymur.data').joinpath('nemo.jp2'))


def goodstuff():
    """Shortcut for specifying path to goodstuff.j2k.

    Returns
    -------
    file : str
        Platform-independent path to goodstuff.j2k.
    """
    return str(ir.files('glymur.data').joinpath('goodstuff.j2k'))


def jpxfile():
    """Shortcut for specifying path to heliov.jpx.

    Returns
    -------
    file : str
        Platform-independent path to heliov.jpx
    """
    return str(ir.files('glymur.data').joinpath('heliov.jpx'))
