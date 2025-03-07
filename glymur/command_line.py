"""Entry point for console script jp2dump."""
# Standard library imports ...
import argparse
import logging
import pathlib
import warnings

# Local imports ...
from . import Jp2k, set_option, lib
from . import tiff, jpeg


def main():
    """Entry point for console script jp2dump."""

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


def jpeg2jp2():
    """Entry point for console script jpeg2jp2."""

    kwargs = {
        'description': 'Convert JPEG to JPEG 2000.',
        'formatter_class': argparse.ArgumentDefaultsHelpFormatter,
        'add_help': False
    }
    parser = argparse.ArgumentParser(**kwargs)

    group1 = parser.add_argument_group(
        'JP2K', 'Pass-through arguments to Jp2k.'
    )

    help = 'Capture resolution parameters'
    group1.add_argument(
        '--capture-resolution', nargs=2, type=float, help=help,
        metavar=('VRESC', 'HRESC')
    )

    help = 'Display resolution parameters'
    group1.add_argument(
        '--display-resolution', nargs=2, type=float, help=help,
        metavar=('VRESD', 'HRESD')
    )

    help = 'Compression ratios for successive layers.'
    group1.add_argument('--cratio', nargs='+', type=int, help=help)

    help = 'PSNR for successive layers.'
    group1.add_argument('--psnr', nargs='+', type=int, help=help)

    help = 'Codeblock size.'
    group1.add_argument(
        '--codeblocksize', nargs=2, type=int, help=help,
        metavar=('cblkh', 'cblkw')
    )

    help = 'Number of decomposition levels.'
    group1.add_argument('--numres', type=int, help=help, default=6)

    help = 'Progression order.'
    choices = ['lrcp', 'rlcp', 'rpcl', 'prcl', 'cprl']
    group1.add_argument('--prog', choices=choices, help=help, default='lrcp')

    help = 'Use irreversible 9x7 transform.'
    group1.add_argument('--irreversible', help=help, action='store_true')

    help = 'Generate EPH markers.'
    group1.add_argument('--eph', help=help, action='store_true')

    help = 'Generate PLT markers.'
    group1.add_argument('--plt', help=help, action='store_true')

    help = 'Generate SOP markers.'
    group1.add_argument('--sop', help=help, action='store_true')

    help = 'Use this many threads/cores.'
    group1.add_argument(
        '--num-threads', type=int, default=1, help=help,
    )

    help = (
        'Dimensions of JP2K tile.  If not provided, the JPEG2000 image will '
        'be written as a single tile.'
    )
    group1.add_argument(
        '--tilesize', nargs=2, type=int, help=help, metavar=('NROWS', 'NCOLS')
    )

    group2 = parser.add_argument_group(
        'JPEG', 'Arguments specific to conversion of JPEG imagery.'
    )

    help = (
        'If specified, subsume any ICC profile found in an APP2 segment(s) '
        'into the colour specification box.  This will involve a file copy '
        'and is therefore a potentially costly operation.'
    )
    group2.add_argument(
        '--include-icc-profile', help=help, action='store_true'
    )

    group2.add_argument('jpeg', help='Input JPEG file.')
    group2.add_argument('jp2k', help='Output JPEG 2000 file.')

    # These arguments are not specific to either group.
    help = 'Show this help message and exit'
    parser.add_argument('--help', '-h', action='help', help=help)

    help = (
        'Logging level, one of "critical", "error", "warning", "info", '
        'or "debug".'
    )
    parser.add_argument(
        '--verbosity', help=help, default='warning',
        choices=['critical', 'error', 'warning', 'info', 'debug']
    )

    args = parser.parse_args()

    logging_level = getattr(logging, args.verbosity.upper())

    jpegp = pathlib.Path(args.jpeg)
    jp2kp = pathlib.Path(args.jp2k)

    kwargs = {
        'cbsize': args.codeblocksize,
        'cratios': args.cratio,
        'capture_resolution': args.capture_resolution,
        'display_resolution': args.display_resolution,
        'eph': args.eph,
        'include_icc_profile': args.include_icc_profile,
        'irreversible': args.irreversible,
        'numres': args.numres,
        'num_threads': args.num_threads,
        'plt': args.plt,
        'prog': args.prog,
        'psnr': args.psnr,
        'sop': args.sop,
        'tilesize': args.tilesize,
        'verbosity': logging_level,
    }

    with jpeg.JPEG2JP2(jpegp, jp2kp, **kwargs) as j:
        j.run()


def tiff2jp2():
    """Entry point for console script tiff2jp2."""

    epilog = (
        "Normally you should at least provide the tilesize argument.  "
        "Even if the TIFF is tiled, tiff2jp2 will NOT automatically use the "
        "TIFF tile dimensions."
    )
    kwargs = {
        'description': 'Convert TIFF to JPEG 2000.',
        'formatter_class': argparse.ArgumentDefaultsHelpFormatter,
        'epilog': epilog,
        'add_help': False
    }
    parser = argparse.ArgumentParser(**kwargs)

    group1 = parser.add_argument_group(
        'JP2K', 'Pass-through arguments to Jp2k.'
    )

    help = 'Capture resolution parameters'
    group1.add_argument(
        '--capture-resolution', nargs=2, type=float, help=help,
        metavar=('VRESC', 'HRESC')
    )

    help = 'Display resolution parameters'
    group1.add_argument(
        '--display-resolution', nargs=2, type=float, help=help,
        metavar=('VRESD', 'HRESD')
    )

    help = 'Compression ratios for successive layers.'
    group1.add_argument('--cratio', nargs='+', type=int, help=help)

    help = 'PSNR for successive layers.'
    group1.add_argument('--psnr', nargs='+', type=int, help=help)

    help = 'Codeblock size.'
    group1.add_argument(
        '--codeblocksize', nargs=2, type=int, help=help,
        metavar=('cblkh', 'cblkw')
    )

    help = 'Number of decomposition levels.'
    group1.add_argument('--numres', type=int, help=help, default=6)

    help = 'Progression order.'
    choices = ['lrcp', 'rlcp', 'rpcl', 'prcl', 'cprl']
    group1.add_argument('--prog', choices=choices, help=help, default='lrcp')

    help = 'Use irreversible 9x7 transform.'
    group1.add_argument('--irreversible', help=help, action='store_true')

    help = 'Generate EPH markers.'
    group1.add_argument('--eph', help=help, action='store_true')

    help = 'Generate PLT markers.'
    group1.add_argument('--plt', help=help, action='store_true')

    help = 'Generate SOP markers.'
    group1.add_argument('--sop', help=help, action='store_true')

    group2 = parser.add_argument_group(
        'TIFF', 'Arguments specific to conversion of TIFF imagery.'
    )

    help = 'Create Exif UUID box from TIFF metadata.'
    group2.add_argument(
        '--create-exif-uuid', help=help, action='store_true', default=True
    )

    help = (
        'Extract XMLPacket tag value from TIFF IFD and store in XMP UUID box. '
        'This will exclude the XMLPacket tag from the Exif UUID box.'
    )
    group2.add_argument(
        '--create-xmp-uuid', help=help, action='store_true', default=True
    )

    help = (
        'If specified, subsume any ICC profile (tag 34675) from the '
        'TIFF IFD into the colour specification box.'
    )
    group2.add_argument(
        '--include-icc-profile', help=help, action='store_true'
    )

    help = (
        'Exclude TIFF tag(s) from EXIF UUID.  This option may be specified as '
        'tag numbers or names.'
    )
    group2.add_argument('--exclude-tags', help=help, nargs='*')

    help = (
        'Dimensions of JP2K tile.  If not provided, the JPEG2000 image will '
        'be written as a single tile.'
    )
    group2.add_argument(
        '--tilesize', nargs=2, type=int, help=help, metavar=('NROWS', 'NCOLS')
    )

    help = 'Use this many threads/cores.'
    group1.add_argument(
        '--num-threads', type=int, default=1, help=help,
    )

    group2.add_argument('tifffile', help='Input TIFF file.')
    group2.add_argument('jp2kfile', help='Output JPEG 2000 file.')

    # These arguments are not specific to either group.
    help = 'Show this help message and exit'
    parser.add_argument('--help', '-h', action='help', help=help)

    help = (
        'Logging level, one of "critical", "error", "warning", "info", '
        'or "debug".'
    )
    parser.add_argument(
        '--verbosity', help=help, default='warning',
        choices=['critical', 'error', 'warning', 'info', 'debug']
    )

    args = parser.parse_args()

    logging_level = getattr(logging, args.verbosity.upper())

    tiffpath = pathlib.Path(args.tifffile)
    jp2kpath = pathlib.Path(args.jp2kfile)

    kwargs = {
        'cbsize': args.codeblocksize,
        'cratios': args.cratio,
        'include_icc_profile': args.include_icc_profile,
        'capture_resolution': args.capture_resolution,
        'create_exif_uuid': args.create_exif_uuid,
        'create_xmp_uuid': args.create_xmp_uuid,
        'display_resolution': args.display_resolution,
        'eph': args.eph,
        'exclude_tags': args.exclude_tags,
        'irreversible': args.irreversible,
        'numres': args.numres,
        'num_threads': args.num_threads,
        'plt': args.plt,
        'prog': args.prog,
        'psnr': args.psnr,
        'sop': args.sop,
        'tilesize': args.tilesize,
        'verbosity': logging_level,
    }

    with tiff.Tiff2Jp2k(tiffpath, jp2kpath, **kwargs) as j:
        j.run()
