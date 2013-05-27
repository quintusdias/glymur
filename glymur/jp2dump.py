"""
Entry point for jp2dump script.
"""

from .jp2k import Jp2k


def jp2dump(filename, codestream=False):
    """Prints JPEG2000 metadata.

    Parameters
    ----------
    filename : string
        The input JPEG2000 file.
    codestream : optional, logical scalar
        Whether or not to dump codestream contents.
    """
    j = Jp2k(filename)
    if codestream:
        print(j.get_codestream(header_only=False))
    else:
        print(j)
