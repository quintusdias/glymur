import re
import sys
import warnings

import numpy as np

import glymur

# Need to know the openjpeg version.  If openjpeg is not installed, we use
# '0.0.0'
OPENJPEG_VERSION = '0.0.0'
if glymur.lib.openjpeg.OPENJPEG is not None:
    OPENJPEG_VERSION = glymur.lib.openjpeg.version()

# Need to know of the libopenjp2 version is the official 2.0.0 release and NOT
# the 2.0+ development version.
OPENJP2_IS_V2_OFFICIAL = False
if glymur.lib.openjp2.OPENJP2 is not None:
    if not hasattr(glymur.lib.openjp2.OPENJP2,
                   'opj_stream_create_default_file_stream_v3'):
        OPENJP2_IS_V2_OFFICIAL = True


msg = "Matplotlib with the PIL backend must be available in order to run the "
msg += "tests in this suite."
no_read_backend_msg = msg
try:
    from PIL import Image
    from matplotlib.pyplot import imread
    no_read_backend = False
except:
    no_read_backend = True

def read_image(infile):
    # PIL issues warnings which we do not care about, so suppress them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = imread(infile)
    return data


def mse(amat, bmat):
    """Mean Square Error"""
    diff = amat.astype(np.double) - bmat.astype(np.double)
    err = np.mean(diff**2)
    return err


def peak_tolerance(amat, bmat):
    """Peak Tolerance"""
    diff = np.abs(amat.astype(np.double) - bmat.astype(np.double))
    ptol = diff.max()
    return ptol


def read_pgx(pgx_file):
    """Helper function for reading the PGX comparison files.

    Open the file in ascii mode and read the header line.
    Will look something like

    PG ML + 8 128 128
    PG%[ \t]%c%c%[ \t+-]%d%[ \t]%d%[ \t]%d"
    """
    header = ''
    with open(pgx_file, 'rb') as fptr:
        while True:
            char = fptr.read(1)
            if char[0] == 10 or char == '\n':
                pos = fptr.tell()
                break
            else:
                if sys.hexversion < 0x03000000:
                    header += char
                else:
                    header += chr(char[0])

    header = header.rstrip()
    tokens = re.split(r'\s', header)

    if (tokens[1][0] == 'M') and (sys.byteorder == 'little'):
        swapbytes = True
    elif (tokens[1][0] == 'L') and (sys.byteorder == 'big'):
        swapbytes = True
    else:
        swapbytes = False

    if (len(tokens) == 6):
        bitdepth = int(tokens[3])
        signed = bitdepth < 0
        if signed:
            bitdepth = -1 * bitdepth
        nrows = int(tokens[5])
        ncols = int(tokens[4])
    else:
        bitdepth = int(tokens[2])
        signed = bitdepth < 0
        if signed:
            bitdepth = -1 * bitdepth
        nrows = int(tokens[4])
        ncols = int(tokens[3])

    if signed:
        if bitdepth <= 8:
            dtype = np.int8
        elif bitdepth <= 16:
            dtype = np.int16
        else:
            raise RuntimeError("unhandled bitdepth")
    else:
        if bitdepth <= 8:
            dtype = np.uint8
        elif bitdepth <= 16:
            dtype = np.uint16
        else:
            raise RuntimeError("unhandled bitdepth")

    shape = [nrows, ncols]

    # Reopen the file in binary mode and seek to the start of the binary
    # data
    with open(pgx_file, 'rb') as fptr:
        fptr.seek(pos)
        data = np.fromfile(file=fptr, dtype=dtype).reshape(shape)

    return(data.byteswap(swapbytes))
