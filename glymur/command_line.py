"""
Entry point for console script jp2dump.
"""
# Standard library imports ...
import argparse
import logging
import pathlib
import warnings

# Local imports ...
from . import Jp2k, set_option, lib
from .tiff import Tiff2Jp2k


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
        set_option('print.xml', False)
    if args.short:
        set_option('print.short', True)

    codestream_level = args.codestream[0]
    if codestream_level not in [0, 1, 2]:
        raise ValueError("Invalid level of codestream information specified.")

    if codestream_level == 0:
        set_option('print.codestream', False)
    elif codestream_level == 2:
        set_option('parse.full_codestream', True)

    path = pathlib.Path(args.filename)

    # JP2 metadata can be extensive, so don't print any warnings until we
    # are done with the metadata.
    with warnings.catch_warnings(record=True) as wctx:

        jp2 = Jp2k(path)
        if jp2._codec_format == lib.openjp2.CODEC_J2K:
            if codestream_level == 0:
                print(f'File:  {path.name}')
            elif codestream_level == 1:
                print(jp2)
            elif codestream_level == 2:
                print(f'File:  {path.name}')
                print(jp2.get_codestream(header_only=False))
        else:
            print(jp2)

        # Now re-emit any suppressed warnings.
        if len(wctx) > 0:
            print("\n")
        for warning in wctx:
            print(
                f"{warning.filename}:{warning.lineno}: "
                f"{warning.category.__name__}: {warning.message}"
            )


def tiff2jp2():
    """
    Entry point for console script tiff2jp2.
    """

    kwargs = {
        'description': 'Convert TIFF to JPEG 2000.',
        'formatter_class': argparse.ArgumentDefaultsHelpFormatter
    }
    parser = argparse.ArgumentParser(**kwargs)

    help = 'Dimensions of JP2K tile.'
    parser.add_argument(
        '--tilesize', nargs=2, type=int, help=help, metavar=('h', 'w')
    )

    help = (
        'Logging level, one of "critical", "error", "warning", "info", '
        'or "debug".'
    )
    parser.add_argument(
        '--verbosity', help=help, default='warning',
        choices=['critical', 'error', 'warning', 'info', 'debug']
    )

    help = (
        'Compression ratio for successive layers.  You may specify more '
        'than once to get multiple layers.'
    )
    parser.add_argument(
        '--cratio', action='append', type=int, help=help,
    )

    help = 'Codeblock size.'
    parser.add_argument(
        '--codeblocksize', nargs=2, type=int, help=help,
        metavar=('cblkh', 'cblkw')
    )

    help = 'Number of decomposition levels.'
    parser.add_argument('--numres', type=int, help=help, default=6)

    help = 'Generate PLT markers.'
    parser.add_argument('--plt', type=bool, help=help, default=False)

    parser.add_argument('tifffile')
    parser.add_argument('jp2kfile')

    args = parser.parse_args()

    logging_level = getattr(logging, args.verbosity.upper())

    tiffpath = pathlib.Path(args.tifffile)
    jp2kpath = pathlib.Path(args.jp2kfile)

    with Tiff2Jp2k(
        tiffpath, jp2kpath, tilesize=args.tilesize, verbosity=logging_level,
        cbsize=args.codeblocksize, cratios=args.cratio, numres=args.numres,
        plt=args.plt
    ) as j:
        j.run()
