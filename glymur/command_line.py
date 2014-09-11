#!/usr/bin/env python

import argparse
import sys
from . import jp2dump, set_printoptions

def main():

    description='Print JPEG2000 metadata.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-x', '--noxml',
            help='Suppress XML.',
            action='store_true')
    parser.add_argument('-s', '--short',
            help='Only print box id, offset, and length.',
            action='store_true')

    chelp = 'Level of codestream information.  0 suppressed all details, '
    chelp += '1 prints headers, 2 prints the full codestream'
    parser.add_argument('-c', '--codestream',
            help=chelp,
            nargs=1,
            type=int,
            default=[0])

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
        print_full_codestream = False
    elif codestream_level == 1:
        print_full_codestream = False
    else:
        print_full_codestream = True
    
    filename = args.filename
    jp2dump(args.filename, codestream=print_full_codestream)
    
