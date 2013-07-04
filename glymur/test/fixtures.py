import re
import sys

import numpy as np

def mse(A, B):
    """Mean Square Error"""
    diff = A.astype(np.double) - B.astype(np.double)
    #e = np.sqrt(np.mean(diff**2))
    e = np.mean(diff**2)
    return e


def peak_tolerance(A, B):
    """Peak Tolerance"""
    diff = np.abs(A.astype(np.double) - B.astype(np.double))
    p = diff.max()
    return p


def read_pgx(pgx_file):
    """Helper function for reading the PGX comparison files.

    Open the file in ascii mode and read the header line.
    Will look something like

    PG ML + 8 128 128
    PG%[ \t]%c%c%[ \t+-]%d%[ \t]%d%[ \t]%d"
    """
    header = ''
    with open(pgx_file, 'rb') as fp:
        while True:
            x = fp.read(1)
            if x[0] == 10 or x == '\n':
                pos = fp.tell()
                break
            else:
                if sys.hexversion < 0x03000000:
                    header += x
                else:
                    header += chr(x[0])

    header = header.rstrip()
    n = re.split('\s', header)

    if (n[1][0] == 'M') and (sys.byteorder == 'little'):
        swapbytes = True
    elif (n[1][0] == 'L') and (sys.byteorder == 'big'):
        swapbytes = True
    else:
        swapbytes = False

    if (len(n) == 6):
        bitdepth = int(n[3])
        signed = bitdepth < 0
        if signed:
            bitdepth = -1 * bitdepth
        nrows = int(n[5])
        ncols = int(n[4])
    else:
        bitdepth = int(n[2])
        signed = bitdepth < 0
        if signed:
            bitdepth = -1 * bitdepth
        nrows = int(n[4])
        ncols = int(n[3])

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
    with open(pgx_file, 'rb') as fp:
        fp.seek(pos)
        data = np.fromfile(file=fp, dtype=dtype).reshape(shape)

    return(data.byteswap(swapbytes))


