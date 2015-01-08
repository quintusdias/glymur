"""
Entry point for console script jp2dump.
"""
import argparse
import os
import warnings

from . import Jp2k, set_printoptions, set_parseoptions, lib


def main():
    """
    Entry point for console script jp2dump.
    """

    kwargs = {'description': 'Print JPEG2000 metadata.',
              'formatter_class': argparse.ArgumentDefaultsHelpFormatter}
    parser = argparse.ArgumentParser(**kwargs)

    parser.add_argument('-x', '--noxml',
                        help='suppress XML',
                        action='store_true')
    parser.add_argument('-s', '--short',
                        help='only print box id, offset, and length',
                        action='store_true')

    chelp = 'Level of codestream information.  0 suppresses all details, '
    chelp += '1 prints the main header, 2 prints the full codestream.'
    parser.add_argument('-c', '--codestream',
                        help=chelp,
                        metavar='LEVEL',
                        nargs=1,
                        type=int,
                        default=[1])

    parser.add_argument('filename')

    args = parser.parse_args()
    if args.noxml:
        set_printoptions(xml=False)
    if args.short:
        set_printoptions(short=True)

    codestream_level = args.codestream[0]
    if codestream_level not in [0, 1, 2]:
        raise ValueError("Invalid level of codestream information specified.")

    if codestream_level == 0:
        set_printoptions(codestream=False)
    elif codestream_level == 2:
        set_parseoptions(full_codestream=True)

    filename = args.filename

    with warnings.catch_warnings(record=True) as wctx:

        # JP2 metadata can be extensive, so don't print any warnings until we
        # are done with the metadata.
        jp2 = Jp2k(filename)
        if jp2._codec_format == lib.openjp2.CODEC_J2K:
            if codestream_level == 0:
                print('File:  {0}'.format(os.path.basename(filename)))
            elif codestream_level == 1:
                print(jp2)
            elif codestream_level == 2:
                print('File:  {0}'.format(os.path.basename(filename)))
                print(jp2.get_codestream(header_only=False))
        else:
            print(jp2)

        # Re-emit any warnings that may have been suppressed.
        if len(wctx) > 0:
            print("\n")
        for warning in wctx:
            print("{0}:{1}: {2}: {3}".format(warning.filename,
                                             warning.lineno,
                                             warning.category.__name__,
                                             warning.message))
