"""
Entry point for jp2dump script.
"""
import warnings

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
    with warnings.catch_warnings(record=True) as wctx:

        # JP2 metadata can be extensive, so don't print any warnings until we
        # are done with the metadata.
        j = Jp2k(filename)
        if codestream:
            print(j.get_codestream(header_only=False))
        else:
            print(j)

        # Re-emit any warnings that may have been suppressed.
        if len(wctx) > 0:
            print("\n")
        for warning in wctx:
            print("{0}:{1}: {2}: {3}".format(warning.filename,
                                             warning.lineno,
                                             warning.category.__name__,
                                             warning.message))
