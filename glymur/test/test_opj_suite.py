"""
The tests defined here roughly correspond to what is in the OpenJPEG test
suite.
"""
import os
import re
import sys
import unittest
import warnings

import numpy as np

import glymur
from glymur import Jp2k

from .fixtures import mse, OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG

try:
    OPJ_DATA_ROOT = os.environ['OPJ_DATA_ROOT']
except KeyError:
    OPJ_DATA_ROOT = None


def opj_data_file(relative_file_name):
    """Compact way of forming a full filename from OpenJPEG's test suite."""
    jfile = os.path.join(OPJ_DATA_ROOT, relative_file_name)
    return jfile


def peak_tolerance(amat, bmat):
    """Peak Tolerance"""
    diff = np.abs(amat.astype(np.double) - bmat.astype(np.double))
    ptol = diff.max()
    return ptol


def read_pgx(pgx_file):
    """Helper function for reading the PGX comparison files.
    """
    header, pos = read_pgx_header(pgx_file)

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

    dtype = determine_pgx_datatype(signed, bitdepth)

    shape = [nrows, ncols]

    # Reopen the file in binary mode and seek to the start of the binary
    # data
    with open(pgx_file, 'rb') as fptr:
        fptr.seek(pos)
        data = np.fromfile(file=fptr, dtype=dtype).reshape(shape)

    return(data.byteswap(swapbytes))


def determine_pgx_datatype(signed, bitdepth):
    """Determine the datatype of the PGX file.

    Parameters
    ----------
    signed : bool
        True if the datatype is signed, false otherwise
    bitdepth : int
        How many bits are used to make up an image plane.  Should be 8 or 16.
    """
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

    return dtype


def read_pgx_header(pgx_file):
    """Open the file in ascii mode (not really) and read the header line.
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
    return header, pos


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
@unittest.skipIf(glymur.version.openjpeg_version_tuple[0] != 2,
                 "Feature not supported in glymur until openjpeg 2.0")
class TestSuite(unittest.TestCase):
    """
    Test the read_bands method.
    """

    def test_ETS_C1P1_p1_03_j2k(self):
        jfile = opj_data_file('input/conformance/p1_03.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read_bands()

        pgxfile = opj_data_file('baseline/conformance/c1p1_03_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[0], pgxdata) < 2)
        self.assertTrue(mse(jpdata[0], pgxdata) < 0.3)

        pgxfile = opj_data_file('baseline/conformance/c1p1_03_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[1], pgxdata) < 2)
        self.assertTrue(mse(jpdata[1], pgxdata) < 0.21)

        pgxfile = opj_data_file('baseline/conformance/c1p1_03_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[2], pgxdata) <= 1)
        self.assertTrue(mse(jpdata[2], pgxdata) < 0.2)

        pgxfile = opj_data_file('baseline/conformance/c1p1_03_3.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[3], pgxdata)

    def test_ETS_C1P0_p0_05_j2k(self):
        jfile = opj_data_file('input/conformance/p0_05.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read_bands()

        pgxfile = opj_data_file('baseline/conformance/c1p0_05_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[0], pgxdata) < 2)
        self.assertTrue(mse(jpdata[0], pgxdata) < 0.302)

        pgxfile = opj_data_file('baseline/conformance/c1p0_05_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[1], pgxdata) < 2)
        self.assertTrue(mse(jpdata[1], pgxdata) < 0.307)

        pgxfile = opj_data_file('baseline/conformance/c1p0_05_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[2], pgxdata) < 2)
        self.assertTrue(mse(jpdata[2], pgxdata) < 0.269)

        pgxfile = opj_data_file('baseline/conformance/c1p0_05_3.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[3], pgxdata) == 0)
        self.assertTrue(mse(jpdata[3], pgxdata) == 0)

    def test_ETS_C1P0_p0_06_j2k(self):
        jfile = opj_data_file('input/conformance/p0_06.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read_bands()

        pgxfile = opj_data_file('baseline/conformance/c1p0_06_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[0], pgxdata) < 635)
        self.assertTrue(mse(jpdata[0], pgxdata) < 11287)

        pgxfile = opj_data_file('baseline/conformance/c1p0_06_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[1], pgxdata) < 403)
        self.assertTrue(mse(jpdata[1], pgxdata) < 6124)

        pgxfile = opj_data_file('baseline/conformance/c1p0_06_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[2], pgxdata) < 378)
        self.assertTrue(mse(jpdata[2], pgxdata) < 3968)

        pgxfile = opj_data_file('baseline/conformance/c1p0_06_3.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[3], pgxdata) == 0)
        self.assertTrue(mse(jpdata[3], pgxdata) == 0)

    def test_NR_DEC_merged_jp2_19_decode(self):
        jfile = opj_data_file('input/nonregression/merged.jp2')
        Jp2k(jfile).read_bands()
        self.assertTrue(True)

    def test_NR_DEC_broken2_jp2_5_decode(self):
        """
        Invalid marker ID on codestream
        """
        jfile = opj_data_file('input/nonregression/broken2.jp2')
        exp_error = glymur.lib.openjp2.OpenJPEGLibraryError
        with self.assertRaises(exp_error):
            with warnings.catch_warnings():
                # Suppress an UnrecognizedMarkerWarning
                warnings.simplefilter("ignore")
                Jp2k(jfile)[:]

    @unittest.skipIf(re.match(r'''0|1|2.0.0''',
                              glymur.version.openjpeg_version) is not None,
                     "Only supported in 2.0.1 or higher")
    def test_NR_DEC_p1_04_j2k_58_decode_0p7_backwards_compatibility(self):
        """
        Test ability to read specified tiles.  Requires 2.0.1 or higher.

        0.7.x read usage deprecated, should use slicing
        """
        jfile = opj_data_file('input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tdata = jp2k.read(tile=63, rlevel=2)  # last tile
        odata = jp2k[::4, ::4]
        np.testing.assert_array_equal(tdata, odata[224:256, 224:256])
