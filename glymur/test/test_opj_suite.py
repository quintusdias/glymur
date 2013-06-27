"""
The tests defined here roughly correspond to what is in the OpenJPEG test
suite.
"""

from contextlib import contextmanager
import os
import platform
import re
import sys
from xml.etree import cElementTree as ET
import unittest
import warnings

if sys.hexversion <= 0x03030000:
    from mock import patch
    from StringIO import StringIO
else:
    from unittest.mock import patch
    from io import StringIO

import numpy as np

from glymur import Jp2k
import glymur

try:
    data_root = os.environ['OPJ_DATA_ROOT']
except KeyError:
    data_root = None
except:
    raise


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


@unittest.skipIf(glymur.lib.openjp2._OPENJP2 is None,
                 "Missing openjp2 library.")
@unittest.skipIf(data_root is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ETS_C0P0_p0_01_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_01.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c0p0_01.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C0P0_p0_02_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_02.j2k')
        with warnings.catch_warnings():
            # There's a 0xff30 marker segment.  Not illegal, but we don't
            # really know what to do with it.  Just ignore.
            warnings.simplefilter("ignore")
            jp2k = Jp2k(jfile)
            jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c0p0_02.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_03_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_03.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c0p0_03r0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C0P0_p0_03_j2k_r1(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_03.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=1)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_03r1.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_04_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=3)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_04.pgx')
        pgxdata = read_pgx(pgxfile)

        self.assertTrue(peak_tolerance(jpdata[:, :, 2], pgxdata) < 33)
        self.assertTrue(mse(jpdata[:, :, 2], pgxdata) < 55.8)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_05_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_05.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read_bands(reduce=3)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_05.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[0], pgxdata) < 54)
        self.assertTrue(mse(jpdata[0], pgxdata) < 68)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_06_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_06.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=3)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_06.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 109)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 743)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_07_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_07.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_07.pgx')
        pgxdata = read_pgx(pgxfile)

        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 10)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 0.34)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_08_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_08.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=5)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_08.pgx')
        pgxdata = read_pgx(pgxfile)

        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 7)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 6.72)

    def test_ETS_C0P0_p0_09_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_09.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=2)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_09.pgx')
        pgxdata = read_pgx(pgxfile)

        self.assertTrue(peak_tolerance(jpdata, pgxdata) < 4)
        self.assertTrue(mse(jpdata, pgxdata) < 1.47)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_10_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_10.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_10.pgx')
        pgxdata = read_pgx(pgxfile)

        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 10)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 2.84)

    def test_ETS_C0P0_p0_11_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_11.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_11.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_ETS_C0P0_p0_12_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_12.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_12.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_13_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_13.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_13.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_14_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_14.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=2)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_14.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata[:, :, 0], pgxdata)

    @unittest.skip("Known failure in OPENJPEG test suite.")
    def test_ETS_C0P0_p0_15_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_15.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_15r0.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C0P0_p0_15_j2k_r1(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_15.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=1)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_15r1.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C0P0_p0_16_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_16.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_16.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C0P1_p1_01_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_01.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p1_01.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    @unittest.skip("Known failure in OPENJPEG test suite operation.")
    def test_ETS_C0P1_p1_02_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_02.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=3)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p1_02.pgx')
        pgxdata = read_pgx(pgxfile)

        print(peak_tolerance(jpdata[:, :, 0], pgxdata))
        print(peak_tolerance(jpdata[:, :, 1], pgxdata))
        print(peak_tolerance(jpdata[:, :, 2], pgxdata))
        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 35)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 74)

    @unittest.skip("Known failure in OPENJPEG test suite operation.")
    def test_ETS_C0P1_p1_03_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_03.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=3)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p1_03.pgx')
        pgxdata = read_pgx(pgxfile)

        print(peak_tolerance(jpdata[:, :, 0], pgxdata))
        print(peak_tolerance(jpdata[:, :, 1], pgxdata))
        print(peak_tolerance(jpdata[:, :, 2], pgxdata))
        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 28)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 18.8)

    @unittest.skip("Known failure in OPENJPEG test suite operation.")
    def test_ETS_C0P1_p1_04_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p1_04r0.pgx')
        pgxdata = read_pgx(pgxfile)

        print(peak_tolerance(jpdata, pgxdata))
        self.assertTrue(peak_tolerance(jpdata, pgxdata) < 2)
        self.assertTrue(mse(jpdata, pgxdata) < 0.55)

    @unittest.skip("Known failure in OPENJPEG test suite, precision issue.")
    def test_ETS_C0P1_p1_04_j2k_r3(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=3)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p1_04r3.pgx')
        pgxdata = read_pgx(pgxfile)

        print(peak_tolerance(jpdata, pgxdata))
        self.assertTrue(peak_tolerance(jpdata, pgxdata) < 2)
        self.assertTrue(mse(jpdata, pgxdata) < 0.55)

    @unittest.skip("Known failure in OPENJPEG test suite operation.")
    def test_ETS_C0P1_p1_05_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_05.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=4)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p1_05.pgx')
        pgxdata = read_pgx(pgxfile)

        print(peak_tolerance(jpdata[:, :, 0], pgxdata))
        print(peak_tolerance(jpdata[:, :, 1], pgxdata))
        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 128)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 16384)

    @unittest.skip("Known failure in OPENJPEG test suite operation.")
    def test_ETS_C0P1_p1_06_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=1)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p1_06.pgx')
        pgxdata = read_pgx(pgxfile)

        print(peak_tolerance(jpdata[:, :, 0], pgxdata))
        print(peak_tolerance(jpdata[:, :, 1], pgxdata))
        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 128)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 16384)

    @unittest.skip("Known failure in OPENJPEG test suite operation.")
    def test_ETS_C0P1_p1_07_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_07.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p1_07.pgx')
        pgxdata = read_pgx(pgxfile)

        # This one works.
        np.testing.assert_array_equal(jpdata[:, :, 0], pgxdata)
        # This one does not.
        np.testing.assert_array_equal(jpdata[:, :, 1], pgxdata)

    def test_ETS_C1P0_p0_01_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_01.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_01_0.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C1P0_p0_02_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_02.j2k')
        with warnings.catch_warnings():
            # There's a 0xff30 marker segment.  Not illegal, but we don't
            # really know what to do with it.  Just ignore.
            warnings.simplefilter("ignore")
            jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_02_0.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C1P0_p0_03_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_03.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_03_0.pgx')
        pgxdata = read_pgx(pgxfile)

        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C1P0_p0_04_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_04_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 5)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 0.776)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_04_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 1], pgxdata) < 4)
        self.assertTrue(mse(jpdata[:, :, 1], pgxdata) < 0.626)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_04_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 2], pgxdata) < 6)
        self.assertTrue(mse(jpdata[:, :, 2], pgxdata) < 1.07)

    def test_ETS_C1P0_p0_05_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_05.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read_bands()

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_05_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[0], pgxdata) < 2)
        self.assertTrue(mse(jpdata[0], pgxdata) < 0.302)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_05_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[1], pgxdata) < 2)
        self.assertTrue(mse(jpdata[1], pgxdata) < 0.307)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_05_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[2], pgxdata) < 2)
        self.assertTrue(mse(jpdata[2], pgxdata) < 0.269)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_05_3.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[3], pgxdata) == 0)
        self.assertTrue(mse(jpdata[3], pgxdata) == 0)

    def test_ETS_C1P0_p0_06_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_06.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read_bands()

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_06_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[0], pgxdata) < 635)
        self.assertTrue(mse(jpdata[0], pgxdata) < 11287)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_06_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[1], pgxdata) < 403)
        self.assertTrue(mse(jpdata[1], pgxdata) < 6124)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_06_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[2], pgxdata) < 378)
        self.assertTrue(mse(jpdata[2], pgxdata) < 3968)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_06_3.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[3], pgxdata) == 0)
        self.assertTrue(mse(jpdata[3], pgxdata) == 0)

    @unittest.skip("Known failure in OPENJPEG test suite operation.")
    def test_ETS_C1P0_p0_07_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_07.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_07_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 0], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_07_1.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, : 1], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_07_2.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, : 2], pgxdata)

    def test_ETS_C1P0_p0_08_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_08.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=1)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_08_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 0], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_08_1.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 1], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_08_2.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 2], pgxdata)

    def test_ETS_C1P0_p0_09_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_09.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_09_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C1P0_p0_10_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_10.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_10_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 0], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_10_1.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 1], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_10_2.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 2], pgxdata)

    def test_ETS_C1P0_p0_11_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_11.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_11_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_ETS_C1P0_p0_12_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_12.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_12_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_ETS_C1P0_p0_13_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_13.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_13_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 0], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_13_1.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 1], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_13_2.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 2], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_13_3.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 3], pgxdata)

    def test_ETS_C1P0_p0_14_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_14.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_14_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 0], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_14_1.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 1], pgxdata)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_14_2.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[:, :, 2], pgxdata)

    def test_ETS_C1P0_p0_15_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_15.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_15_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C1P0_p0_16_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_16.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p0_16_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C1P1_p1_01_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_01.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_01_0.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata, pgxdata)

    def test_ETS_C1P1_p1_02_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_02.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_02_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 5)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 0.765)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_02_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 1], pgxdata) < 4)
        self.assertTrue(mse(jpdata[:, :, 1], pgxdata) < 0.616)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_02_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 2], pgxdata) < 6)
        self.assertTrue(mse(jpdata[:, :, 2], pgxdata) < 1.051)

    def test_ETS_C1P1_p1_03_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_03.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read_bands()

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_03_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[0], pgxdata) < 2)
        self.assertTrue(mse(jpdata[0], pgxdata) < 0.3)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_03_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[1], pgxdata) < 2)
        self.assertTrue(mse(jpdata[1], pgxdata) < 0.21)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_03_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[2], pgxdata) <= 1)
        self.assertTrue(mse(jpdata[2], pgxdata) < 0.2)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_03_3.pgx')
        pgxdata = read_pgx(pgxfile)
        np.testing.assert_array_equal(jpdata[3], pgxdata)

    def test_ETS_C1P1_p1_04_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_04_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata, pgxdata) < 624)
        self.assertTrue(mse(jpdata, pgxdata) < 3080)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_ETS_C1P1_p1_05_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_05.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_05_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 40)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 8.458)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_05_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 1], pgxdata) < 40)
        self.assertTrue(mse(jpdata[:, :, 1], pgxdata) < 9.816)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_05_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 2], pgxdata) < 40)
        self.assertTrue(mse(jpdata[:, :, 2], pgxdata) < 10.154)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_ETS_C1P1_p1_06_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_06_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 0], pgxdata) < 2)
        self.assertTrue(mse(jpdata[:, :, 0], pgxdata) < 0.6)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_06_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 1], pgxdata) < 2)
        self.assertTrue(mse(jpdata[:, :, 1], pgxdata) < 0.6)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_06_2.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[:, :, 2], pgxdata) < 2)
        self.assertTrue(mse(jpdata[:, :, 2], pgxdata) < 0.6)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_ETS_C1P1_p1_07_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_07.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read_bands()

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_07_0.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[0], pgxdata) <= 0)
        self.assertTrue(mse(jpdata[0], pgxdata) <= 0)

        pgxfile = os.path.join(data_root, 'baseline/conformance/c1p1_07_1.pgx')
        pgxdata = read_pgx(pgxfile)
        self.assertTrue(peak_tolerance(jpdata[1], pgxdata) <= 0)
        self.assertTrue(mse(jpdata[1], pgxdata) <= 0)

    def test_ETS_JP2_file1(self):
        jfile = os.path.join(data_root, 'input/conformance/file1.jp2')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()
        self.assertEqual(jpdata.shape, (512, 768, 3))

    def test_ETS_JP2_file2(self):
        jfile = os.path.join(data_root, 'input/conformance/file2.jp2')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()
        self.assertEqual(jpdata.shape, (640, 480, 3))

    def test_ETS_JP2_file3(self):
        jfile = os.path.join(data_root, 'input/conformance/file3.jp2')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read_bands()
        self.assertEqual(jpdata[0].shape, (640, 480))
        self.assertEqual(jpdata[1].shape, (320, 240))
        self.assertEqual(jpdata[2].shape, (320, 240))

    def test_ETS_JP2_file4(self):
        jfile = os.path.join(data_root, 'input/conformance/file4.jp2')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()
        self.assertEqual(jpdata.shape, (512, 768))

    def test_ETS_JP2_file5(self):
        jfile = os.path.join(data_root, 'input/conformance/file5.jp2')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()
        self.assertEqual(jpdata.shape, (512, 768, 3))

    def test_ETS_JP2_file6(self):
        jfile = os.path.join(data_root, 'input/conformance/file6.jp2')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()
        self.assertEqual(jpdata.shape, (512, 768))

    def test_ETS_JP2_file7(self):
        jfile = os.path.join(data_root, 'input/conformance/file7.jp2')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()
        self.assertEqual(jpdata.shape, (640, 480, 3))

    def test_ETS_JP2_file8(self):
        jfile = os.path.join(data_root, 'input/conformance/file8.jp2')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()
        self.assertEqual(jpdata.shape, (400, 700))

    def test_ETS_JP2_file9(self):
        jfile = os.path.join(data_root, 'input/conformance/file9.jp2')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read()
        self.assertEqual(jpdata.shape, (512, 768, 3))

    def test_NR_DEC_Bretagne2_j2k_1_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/Bretagne2.j2k')
        jp2 = Jp2k(jfile)
        data = jp2.read()
        self.assertTrue(True)

    def test_NR_DEC__00042_j2k_2_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/_00042.j2k')
        jp2 = Jp2k(jfile)
        data = jp2.read()
        self.assertTrue(True)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_123_j2c_3_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/123.j2c')
        jp2 = Jp2k(jfile)
        data = jp2.read()
        self.assertTrue(True)

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_DEC_broken_jp2_4_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/broken.jp2')
        with self.assertWarns(UserWarning) as cw:
            # colr box has bad length.
            jp2 = Jp2k(jfile)
        with self.assertRaises(IOError):
            data = jp2.read()
        self.assertTrue(True)

    def test_NR_DEC_broken2_jp2_5_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/broken2.jp2')
        with self.assertRaises(IOError):
            data = Jp2k(jfile).read()
        self.assertTrue(True)

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_DEC_broken3_jp2_6_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/broken3.jp2')
        with self.assertWarns(UserWarning) as cw:
            # colr box has bad length.
            j = Jp2k(jfile)

        with self.assertRaises(IOError) as ce:
            d = j.read()

    def test_NR_DEC_broken4_jp2_7_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/broken4.jp2')
        with self.assertRaises(IOError):
            data = Jp2k(jfile).read()
        self.assertTrue(True)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_bug_j2c_8_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/bug.j2c')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_buxI_j2k_9_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/buxI.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_buxR_j2k_10_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/buxR.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_Cannotreaddatawithnosizeknown_j2k_11_decode(self):
        relpath = 'input/nonregression/Cannotreaddatawithnosizeknown.j2k'
        jfile = os.path.join(data_root, relpath)
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_cthead1_j2k_12_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/cthead1.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_CT_Phillips_JPEG2K_Decompr_Problem_j2k_13_decode(self):
        relpath = 'input/nonregression/CT_Phillips_JPEG2K_Decompr_Problem.j2k'
        jfile = os.path.join(data_root, relpath)
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_illegalcolortransform_j2k_14_decode(self):
        # Stream too short, expected SOT.
        jfile = os.path.join(data_root,
                             'input/nonregression/illegalcolortransform.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_j2k32_j2k_15_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/j2k32.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_kakadu_v4_4_openjpegv2_broken_j2k_16_decode(self):
        relpath = 'input/nonregression/kakadu_v4-4_openjpegv2_broken.j2k'
        jfile = os.path.join(data_root, relpath)
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_MarkerIsNotCompliant_j2k_17_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/MarkerIsNotCompliant.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_Marrin_jp2_18_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/Marrin.jp2')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_merged_jp2_19_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/merged.jp2')
        data = Jp2k(jfile).read_bands()
        self.assertTrue(True)

    def test_NR_DEC_movie_00000_j2k_20_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/movie_00000.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_movie_00001_j2k_21_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/movie_00001.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_movie_00002_j2k_22_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/movie_00002.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_orb_blue_lin_j2k_j2k_23_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/orb-blue10-lin-j2k.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_orb_blue_win_j2k_j2k_24_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/orb-blue10-win-j2k.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_orb_blue_lin_jp2_25_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/orb-blue10-lin-jp2.jp2')
        with warnings.catch_warnings():
            # This file has an invalid ICC profile
            warnings.simplefilter("ignore")
            data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_orb_blue_win_jp2_26_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/orb-blue10-win-jp2.jp2')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_relax_jp2_27_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/relax.jp2')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_test_lossless_j2k_28_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/test_lossless.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_text_GBR_jp2_29_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/text_GBR.jp2')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_pacs_ge_j2k_30_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/pacs.ge.j2k')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_kodak_2layers_lrcp_j2c_31_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/kodak_2layers_lrcp.j2c')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_kodak_2layers_lrcp_j2c_32_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/kodak_2layers_lrcp.j2c')
        data = Jp2k(jfile).read(layer=2)
        self.assertTrue(True)

    def test_NR_DEC_issue104_jpxstream_jp2_33_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/issue104_jpxstream.jp2')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_mem_b2ace68c_1381_jp2_34_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/mem-b2ace68c-1381.jp2')
        with warnings.catch_warnings():
            # This file has a bad pclr box, we test for this elsewhere.
            warnings.simplefilter("ignore")
            j = Jp2k(jfile)
        data = j.read()
        self.assertTrue(True)

    def test_NR_DEC_mem_b2b86b74_2753_jp2_35_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/mem-b2b86b74-2753.jp2')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_gdal_fuzzer_unchecked_num_resolutions_jp2_36_decode(self):
        f = 'input/nonregression/gdal_fuzzer_unchecked_numresolutions.jp2'
        jfile = os.path.join(data_root, f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            j = Jp2k(jfile)
            with self.assertRaises(IOError):
                data = j.read()

    def test_NR_DEC_jp2_36_decode(self):
        lst = ('input',
               'nonregression',
               'gdal_fuzzer_assert_in_opj_j2k_read_SQcd_SQcc.patch.jp2')
        jfile = os.path.join(data_root, '/'.join(lst))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            j = Jp2k(jfile)
            with self.assertRaises(IOError):
                data = j.read()

    def test_NR_DEC_gdal_fuzzer_check_number_of_tiles_jp2_38_decode(self):
        relpath = 'input/nonregression/gdal_fuzzer_check_number_of_tiles.jp2'
        jfile = os.path.join(data_root, relpath)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            j = Jp2k(jfile)
            with self.assertRaises(IOError):
                data = j.read()

    def test_NR_DEC_gdal_fuzzer_check_comp_dx_dy_jp2_39_decode(self):
        relpath = 'input/nonregression/gdal_fuzzer_check_comp_dx_dy.jp2'
        jfile = os.path.join(data_root, relpath)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(IOError):
                j = Jp2k(jfile).read()

    def test_NR_DEC_file_409752_jp2_40_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/file409752.jp2')
        with self.assertRaises(IOError):
            data = Jp2k(jfile).read()

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_DEC_issue188_beach_64bitsbox_jp2_41_decode(self):
        # Has an 'XML ' box instead of 'xml '.  Yes that is pedantic, but it
        # really does deserve a warning.
        relpath = 'input/nonregression/issue188_beach_64bitsbox.jp2'
        jfile = os.path.join(data_root, relpath)
        with self.assertWarns(UserWarning) as cw:
            data = Jp2k(jfile).read()

    def test_NR_DEC_issue206_image_000_jp2_42_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/issue206_image-000.jp2')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    def test_NR_DEC_p1_04_j2k_43_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(0, 0, 1024, 1024))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata)

    def test_NR_DEC_p1_04_j2k_44_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(640, 512, 768, 640))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[640:768, 512:640])

    def test_NR_DEC_p1_04_j2k_45_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(896, 896, 1024, 1024))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[896:1024, 896:1024])

    def test_NR_DEC_p1_04_j2k_46_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(500, 100, 800, 300))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[500:800, 100:300])

    def test_NR_DEC_p1_04_j2k_47_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(520, 260, 600, 360))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[520:600, 260:360])

    def test_NR_DEC_p1_04_j2k_48_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(520, 260, 660, 360))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[520:660, 260:360])

    def test_NR_DEC_p1_04_j2k_49_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(520, 360, 600, 400))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[520:600, 360:400])

    def test_NR_DEC_p1_04_j2k_50_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(0, 0, 1024, 1024), reduce=2)
        odata = jp2k.read(reduce=2)

        np.testing.assert_array_equal(ssdata, odata[0:256, 0:256])

    def test_NR_DEC_p1_04_j2k_51_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(640, 512, 768, 640), reduce=2)
        odata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(ssdata, odata[160:192, 128:160])

    def test_NR_DEC_p1_04_j2k_52_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(896, 896, 1024, 1024), reduce=2)
        odata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(ssdata, odata[224:352, 224:352])

    def test_NR_DEC_p1_04_j2k_53_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(500, 100, 800, 300), reduce=2)
        odata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(ssdata, odata[125:200, 25:75])

    def test_NR_DEC_p1_04_j2k_54_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(520, 260, 600, 360), reduce=2)
        odata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(ssdata, odata[130:150, 65:90])

    def test_NR_DEC_p1_04_j2k_55_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(520, 260, 660, 360), reduce=2)
        odata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(ssdata, odata[130:165, 65:90])

    def test_NR_DEC_p1_04_j2k_56_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(520, 360, 600, 400), reduce=2)
        odata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(ssdata, odata[130:150, 90:100])

    def test_NR_DEC_p1_04_j2k_57_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tdata = jp2k.read(tile=63)  # last tile
        odata = jp2k.read()
        np.testing.assert_array_equal(tdata, odata[896:1024, 896:1024])

    def test_NR_DEC_p1_04_j2k_58_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tdata = jp2k.read(tile=63, reduce=2)  # last tile
        odata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(tdata, odata[224:256, 224:256])

    def test_NR_DEC_p1_04_j2k_59_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tdata = jp2k.read(tile=12)  # 2nd row, 5th column
        odata = jp2k.read()
        np.testing.assert_array_equal(tdata, odata[128:256, 512:640])

    def test_NR_DEC_p1_04_j2k_60_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tdata = jp2k.read(tile=12, reduce=1)  # 2nd row, 5th column
        odata = jp2k.read(reduce=1)
        np.testing.assert_array_equal(tdata, odata[64:128, 256:320])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_61_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(0, 0, 12, 12))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[0:12, 0:12])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_62_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(1, 8, 8, 11))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[1:8, 8:11])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_63_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(9, 9, 12, 12))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[9:12, 9:12])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_64_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(10, 4, 12, 10))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[10:12, 4:10])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_65_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(3, 3, 9, 9))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[3:9, 3:9])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_66_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(4, 4, 7, 7))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[4:7, 4:7])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_67_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(4, 4, 5, 5))
        odata = jp2k.read()
        np.testing.assert_array_equal(ssdata, odata[4:5, 4: 5])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_68_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(0, 0, 12, 12), reduce=1)
        odata = jp2k.read(reduce=1)
        np.testing.assert_array_equal(ssdata, odata[0:6, 0:6])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_69_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(1, 8, 8, 11), reduce=1)
        self.assertEqual(ssdata.shape, (3, 2, 3))

    def test_NR_DEC_p1_06_j2k_70_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(9, 9, 12, 12), reduce=1)
        self.assertEqual(ssdata.shape, (1, 1, 3))

    def test_NR_DEC_p1_06_j2k_71_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(10, 4, 12, 10), reduce=1)
        self.assertEqual(ssdata.shape, (1, 3, 3))

    def test_NR_DEC_p1_06_j2k_72_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(3, 3, 9, 9), reduce=1)
        self.assertEqual(ssdata.shape, (3, 3, 3))

    def test_NR_DEC_p1_06_j2k_73_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(4, 4, 7, 7), reduce=1)
        self.assertEqual(ssdata.shape, (2, 2, 3))

    def test_NR_DEC_p1_06_j2k_74_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(4, 4, 5, 5), reduce=1)
        self.assertEqual(ssdata.shape, (1, 1, 3))

    def test_NR_DEC_p1_06_j2k_75_decode(self):
        # Image size would be 0 x 0.
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        with self.assertRaises((IOError, OSError)) as ce:
            ssdata = jp2k.read(area=(9, 9, 12, 12), reduce=2)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_76_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        fulldata = jp2k.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiledata = jp2k.read(tile=0)
        np.testing.assert_array_equal(tiledata, fulldata[0:3, 0:3])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_77_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        fulldata = jp2k.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiledata = jp2k.read(tile=5)
        np.testing.assert_array_equal(tiledata, fulldata[3:6, 3:6])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_78_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        fulldata = jp2k.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiledata = jp2k.read(tile=9)
        np.testing.assert_array_equal(tiledata, fulldata[6:9, 3:6])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_79_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        fulldata = jp2k.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiledata = jp2k.read(tile=15)
        np.testing.assert_array_equal(tiledata, fulldata[9:12, 9:12])

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_80_decode(self):
        # Just read the data, don't bother verifying.
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiledata = jp2k.read(tile=0, reduce=2)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_81_decode(self):
        # Just read the data, don't bother verifying.
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiledata = jp2k.read(tile=5, reduce=2)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_82_decode(self):
        # Just read the data, don't bother verifying.
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiledata = jp2k.read(tile=9, reduce=2)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_83_decode(self):
        # tile size is 3x3.  Reducing two levels results in no data.
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises((IOError, OSError)) as ce:
                tiledata = jp2k.read(tile=15, reduce=2)

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_DEC_p1_06_j2k_84_decode(self):
        # Just read the data, don't bother verifying.
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        jp2k = Jp2k(jfile)
        data = jp2k.read(reduce=4)

    def test_NR_DEC_p0_04_j2k_85_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(0, 0, 256, 256))
        fulldata = jp2k.read()
        np.testing.assert_array_equal(fulldata[0:256, 0:256], ssdata)

    def test_NR_DEC_p0_04_j2k_86_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(0, 128, 128, 256))
        fulldata = jp2k.read()
        np.testing.assert_array_equal(fulldata[0:128, 128:256], ssdata)

    def test_NR_DEC_p0_04_j2k_87_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(10, 50, 200, 120))
        fulldata = jp2k.read()
        np.testing.assert_array_equal(fulldata[10:200, 50:120], ssdata)

    def test_NR_DEC_p0_04_j2k_88_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(150, 10, 210, 190))
        fulldata = jp2k.read()
        np.testing.assert_array_equal(fulldata[150:210, 10:190], ssdata)

    def test_NR_DEC_p0_04_j2k_89_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(80, 100, 150, 200))
        fulldata = jp2k.read()
        np.testing.assert_array_equal(fulldata[80:150, 100:200], ssdata)

    def test_NR_DEC_p0_04_j2k_90_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(20, 150, 50, 200))
        fulldata = jp2k.read()
        np.testing.assert_array_equal(fulldata[20:50, 150:200], ssdata)

    def test_NR_DEC_p0_04_j2k_91_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(0, 0, 256, 256), reduce=2)
        fulldata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(fulldata[0:64, 0:64], ssdata)

    def test_NR_DEC_p0_04_j2k_92_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(0, 128, 128, 256), reduce=2)
        fulldata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(fulldata[0:32, 32:64], ssdata)

    def test_NR_DEC_p0_04_j2k_93_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(10, 50, 200, 120), reduce=2)
        fulldata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(fulldata[3:50, 13:30], ssdata)

    def test_NR_DEC_p0_04_j2k_94_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(150, 10, 210, 190), reduce=2)
        fulldata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(fulldata[38:53, 3:48], ssdata)

    def test_NR_DEC_p0_04_j2k_95_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(80, 100, 150, 200), reduce=2)
        fulldata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(fulldata[20:38, 25:50], ssdata)

    def test_NR_DEC_p0_04_j2k_96_decode(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        jp2k = Jp2k(jfile)
        ssdata = jp2k.read(area=(20, 150, 50, 200), reduce=2)
        fulldata = jp2k.read(reduce=2)
        np.testing.assert_array_equal(fulldata[5:13, 38:50], ssdata)


@unittest.skipIf(data_root is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuiteDump(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_NR_p0_01_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_01.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # Segment IDs.
        actual = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'QCD', 'COD', 'SOT', 'SOD', 'EOC']
        self.assertEqual(actual, expected)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 128)
        self.assertEqual(c.segment[1].Ysiz, 128)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # QCD: Quantization default
        self.assertEqual(c.segment[2].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[2]._guardBits, 2)
        self.assertEqual(c.segment[2]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])
        self.assertEqual(c.segment[2]._mantissa,
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # COD: Coding style default
        self.assertFalse(c.segment[3].Scod & 2)  # no sop
        self.assertFalse(c.segment[3].Scod & 4)  # no eph
        self.assertEqual(c.segment[3].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[3]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[3].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[3].SPcod[4], 3)  # layers
        self.assertEqual(tuple(c.segment[3]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[3].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # SOT: start of tile part
        self.assertEqual(c.segment[4].Isot, 0)
        self.assertEqual(c.segment[4].Psot, 7314)
        self.assertEqual(c.segment[4].TPsot, 0)
        self.assertEqual(c.segment[4].TNsot, 1)

    def test_NR_p0_02_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_02.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 127)
        self.assertEqual(c.segment[1].Ysiz, 126)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (127, 126))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(2, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)  # sop
        self.assertTrue(c.segment[2].Scod & 4)  # eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 6)  # layers = 6
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[3].Ccoc, 0)
        self.assertEqual(c.segment[3].SPcoc[0], 3)  # levels
        self.assertEqual(tuple(c.segment[3]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[3].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[3].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[3].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4]._guardBits, 3)
        self.assertEqual(c.segment[4]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])
        self.assertEqual(c.segment[4]._mantissa,
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[5].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # One unknown marker
        self.assertEqual(c.segment[6].id, '0xff30')

        # SOT: start of tile part
        self.assertEqual(c.segment[7].Isot, 0)
        self.assertEqual(c.segment[7].Psot, 6047)
        self.assertEqual(c.segment[7].TPsot, 0)
        self.assertEqual(c.segment[7].TNsot, 1)

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[8].id, 'SOD')

        # SOP, EPH
        sop = [x.id for x in c.segment if x.id == 'SOP']
        eph = [x.id for x in c.segment if x.id == 'EPH']
        self.assertEqual(len(sop), 24)
        self.assertEqual(len(eph), 24)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].id, 'EOC')

    def test_NR_p0_03_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_03.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 256)
        self.assertEqual(c.segment[1].Ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (4,))
        # signed
        self.assertEqual(c.segment[1]._signed, (True,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2]._layers, 8)  # 8
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 1)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 1)  # scalar implicit
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._exponent, [0])
        self.assertEqual(c.segment[3]._mantissa, [0])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].Cqcc, 0)
        self.assertEqual(c.segment[4]._guardBits, 2)
        # quantization type
        self.assertEqual(c.segment[4].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[4]._exponent, [4, 5, 5, 6])
        self.assertEqual(c.segment[4]._mantissa, [0, 0, 0, 0])

        # POD: progression order change
        self.assertEqual(c.segment[5].RSpod, (0,))
        self.assertEqual(c.segment[5].CSpod, (0,))
        self.assertEqual(c.segment[5].LYEpod, (8,))
        self.assertEqual(c.segment[5].REpod, (33,))
        self.assertEqual(c.segment[5].CEpod, (255,))
        self.assertEqual(c.segment[5].Ppod, (glymur.core.LRCP,))

        # CRG:  component registration
        self.assertEqual(c.segment[6].Xcrg, (65424,))
        self.assertEqual(c.segment[6].Ycrg, (32558,))

        # COM: comment
        # Registration
        self.assertEqual(c.segment[7].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[7].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,"
                         + "2001 Algo Vision Technology")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[9].Rcme, glymur.core.RCME_BINARY)
        # Comment value
        self.assertEqual(len(c.segment[9].Ccme), 62)

        # TLM (tile-part length)
        self.assertEqual(c.segment[10].Ztlm, 0)
        self.assertEqual(c.segment[10].Ttlm, (0, 1, 2, 3))
        self.assertEqual(c.segment[10].Ptlm, (4267, 2117, 4080, 2081))

        # SOT: start of tile part
        self.assertEqual(c.segment[11].Isot, 0)
        self.assertEqual(c.segment[11].Psot, 4267)
        self.assertEqual(c.segment[11].TPsot, 0)
        self.assertEqual(c.segment[11].TNsot, 1)

        # RGN: region of interest
        self.assertEqual(c.segment[12].Crgn, 0)
        self.assertEqual(c.segment[12].Srgn, 0)
        self.assertEqual(c.segment[12].SPrgn, 7)

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[13].id, 'SOD')

    def test_NR_p0_04_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_04.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 640)
        self.assertEqual(c.segment[1].Ysiz, 480)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (640, 480))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1), (1, 1), (1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2]._layers, 20)  # 20
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 6)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(c.segment[2]._precinct_size,
                         [(128, 128), (128, 128), (128, 128), (128, 128),
                          (128, 128), (128, 128), (128, 128)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)  # scalar expounded
        self.assertEqual(c.segment[3]._guardBits, 3)
        self.assertEqual(c.segment[3]._exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13,
                          11, 11, 11, 11, 11, 11])
        self.assertEqual(c.segment[3]._mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                          1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                          1888])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].Cqcc, 1)
        # quantization type
        self.assertEqual(c.segment[4].Sqcc & 0x1f, 2)  # none
        self.assertEqual(c.segment[4]._guardBits, 3)
        self.assertEqual(c.segment[4]._exponent,
                         [14, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11,
                          9, 9, 9, 9, 9, 9])
        self.assertEqual(c.segment[4]._mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845,
                          1845, 1868, 1925, 1925, 2007, 32, 32, 131, 2002,
                          2002, 1888])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].Cqcc, 2)
        # quantization type
        self.assertEqual(c.segment[5].Sqcc & 0x1f, 2)  # none
        self.assertEqual(c.segment[5]._guardBits, 3)
        self.assertEqual(c.segment[5]._exponent,
                         [14, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11,
                          9, 9, 9, 9, 9, 9])
        self.assertEqual(c.segment[5]._mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845,
                          1845, 1868, 1925, 1925, 2007, 32, 32, 131, 2002,
                          2002, 1888])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[6].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[6].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[7].Isot, 0)
        self.assertEqual(c.segment[7].Psot, 264383)
        self.assertEqual(c.segment[7].TPsot, 0)
        self.assertEqual(c.segment[7].TNsot, 1)

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[8].id, 'SOD')

    def test_NR_p0_05_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_05.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1024)
        self.assertEqual(c.segment[1].Ysiz, 1024)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1024, 1024))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1), (1, 1), (2, 2), (2, 2)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2]._layers, 7)  # 7
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 6)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].Ccoc, 1)
        self.assertEqual(c.segment[3].SPcoc[0], 3)  # levels
        self.assertEqual(tuple(c.segment[3]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[4].Ccoc, 3)
        self.assertEqual(c.segment[4].SPcoc[0], 6)  # levels
        self.assertEqual(tuple(c.segment[4]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[4].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[4].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[4].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[4].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[4].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[4].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[4].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[5].Sqcd & 0x1f, 2)  # scalar expounded
        self.assertEqual(c.segment[5]._guardBits, 3)
        self.assertEqual(c.segment[5]._exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13,
                          11, 11, 11, 11, 11, 11])
        self.assertEqual(c.segment[5]._mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845,
                          1845, 1868, 1925, 1925, 2007, 32, 32, 131, 2002,
                          2002, 1888])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].Cqcc, 0)
        # quantization type
        self.assertEqual(c.segment[6].Sqcc & 0x1f, 1)  # scalar derived
        self.assertEqual(c.segment[6]._guardBits, 3)
        self.assertEqual(c.segment[6]._exponent, [14])
        self.assertEqual(c.segment[6]._mantissa, [0])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[7].Cqcc, 3)
        # quantization type
        self.assertEqual(c.segment[7].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[7]._guardBits, 3)
        self.assertEqual(c.segment[7]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10,
                          9, 9, 10])
        self.assertEqual(c.segment[7]._mantissa, [0] * 19)

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # TLM (tile-part length)
        self.assertEqual(c.segment[9].Ztlm, 0)
        self.assertEqual(c.segment[9].Ttlm, (0,))
        self.assertEqual(c.segment[9].Ptlm, (1310540,))

        # SOT: start of tile part
        self.assertEqual(c.segment[10].Isot, 0)
        self.assertEqual(c.segment[10].Psot, 1310540)
        self.assertEqual(c.segment[10].TPsot, 0)
        self.assertEqual(c.segment[10].TNsot, 1)

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[11].id, 'SOD')

    def test_NR_p0_06_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_06.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 513)
        self.assertEqual(c.segment[1].Ysiz, 129)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (513, 129))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (12, 12, 12, 12))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1), (2, 1), (1, 2), (2, 2)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RPCL)
        self.assertEqual(c.segment[2]._layers, 4)  # 4
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 6)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)  # scalar expounded
        self.assertEqual(c.segment[3]._guardBits, 3)
        self.assertEqual(c.segment[3]._mantissa,
                         [512, 518, 522, 524, 516, 524, 522, 527, 523, 549,
                          557, 561, 853, 852, 700, 163, 78, 1508, 1831])
        self.assertEqual(c.segment[3]._exponent,
                         [7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 2, 1, 2,
                          1])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].Cqcc, 1)
        # quantization type
        self.assertEqual(c.segment[4].Sqcc & 0x1f, 2)  # scalar derived
        self.assertEqual(c.segment[4]._guardBits, 4)
        self.assertEqual(c.segment[4]._mantissa,
                         [1527, 489, 665, 506, 487, 502, 493, 493, 500, 485,
                          505, 491, 490, 491, 499, 509, 503, 496, 558])
        self.assertEqual(c.segment[4]._exponent,
                         [10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6,
                          5, 5, 5])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].Cqcc, 2)
        # quantization type
        self.assertEqual(c.segment[5].Sqcc & 0x1f, 2)  # scalar derived
        self.assertEqual(c.segment[5]._guardBits, 5)
        self.assertEqual(c.segment[5]._mantissa,
                         [1337, 728, 890, 719, 716, 726, 700, 718, 704, 704,
                          712, 712, 717, 719, 701, 749, 753, 718, 841])
        self.assertEqual(c.segment[5]._exponent,
                         [10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6,
                          5, 5, 5])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].Cqcc, 3)
        # quantization type
        self.assertEqual(c.segment[6].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[6]._guardBits, 6)
        self.assertEqual(c.segment[6]._mantissa, [0] * 19)
        self.assertEqual(c.segment[6]._exponent,
                         [12, 13, 13, 14, 13, 13, 14, 13, 13, 14, 13, 13, 14,
                          13, 13, 14, 13, 13, 14])

        # COC: Coding style component
        self.assertEqual(c.segment[7].Ccoc, 3)
        self.assertEqual(c.segment[7].SPcoc[0], 6)  # levels
        self.assertEqual(tuple(c.segment[7]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[7].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[7].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[7].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[7].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[7].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[7].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[7].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # RGN: region of interest
        self.assertEqual(c.segment[8].Crgn, 0)  # component
        self.assertEqual(c.segment[8].Srgn, 0)  # implicit
        self.assertEqual(c.segment[8].SPrgn, 11)

        # SOT: start of tile part
        self.assertEqual(c.segment[9].Isot, 0)
        self.assertEqual(c.segment[9].Psot, 33582)
        self.assertEqual(c.segment[9].TPsot, 0)
        self.assertEqual(c.segment[9].TNsot, 1)

        # RGN: region of interest
        self.assertEqual(c.segment[10].Crgn, 0)  # component
        self.assertEqual(c.segment[10].Srgn, 0)  # implicit
        self.assertEqual(c.segment[10].SPrgn, 9)

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[11].id, 'SOD')

    def test_NR_p0_07_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_07.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 2048)
        self.assertEqual(c.segment[1].Ysiz, 2048)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (12, 12, 12))
        # signed
        self.assertEqual(c.segment[1]._signed, (True, True, True))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1), (1, 1), (1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)
        self.assertTrue(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2]._layers, 8)  # 8
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa, [0] * 10)
        self.assertEqual(c.segment[3]._exponent,
                         [14, 15, 15, 16, 15, 15, 16, 15, 15, 16])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "Kakadu-3.0.7")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].Isot, 0)
        self.assertEqual(c.segment[5].Psot, 9951)
        self.assertEqual(c.segment[5].TPsot, 0)
        self.assertEqual(c.segment[5].TNsot, 0)  # unknown

        # POD: progression order change
        self.assertEqual(c.segment[6].RSpod, (0,))
        self.assertEqual(c.segment[6].CSpod, (0,))
        self.assertEqual(c.segment[6].LYEpod, (9,))
        self.assertEqual(c.segment[6].REpod, (3,))
        self.assertEqual(c.segment[6].CEpod, (3,))
        self.assertEqual(c.segment[6].Ppod, (glymur.core.LRCP,))

        # PLT: packet length, tile part
        self.assertEqual(c.segment[7].Zplt, 0)
        #self.assertEqual(c.segment[7].iplt), 99)

        # SOD:  start of data
        self.assertEqual(c.segment[8].id, 'SOD')

    def test_NR_p0_08_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_08.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 513)
        self.assertEqual(c.segment[1].Ysiz, 3072)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (513, 3072))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (12, 12, 12))
        # signed
        self.assertEqual(c.segment[1]._signed, (True, True, True))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1), (1, 1), (1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)
        self.assertTrue(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.CPRL)
        self.assertEqual(c.segment[2]._layers, 30)  # 30
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 7)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].Ccoc, 0)
        self.assertEqual(c.segment[3].SPcoc[0], 6)  # levels
        self.assertEqual(tuple(c.segment[3]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[4].Ccoc, 1)
        self.assertEqual(c.segment[4].SPcoc[0], 7)  # levels
        self.assertEqual(tuple(c.segment[4]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[4].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[4].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[4].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[4].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[4].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[4].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[4].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[5].Ccoc, 2)
        self.assertEqual(c.segment[5].SPcoc[0], 8)  # levels
        self.assertEqual(tuple(c.segment[5]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[5].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[5].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[5].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[5].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[5].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[5].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[5].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[6].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[6]._guardBits, 4)
        self.assertEqual(c.segment[6]._mantissa, [0] * 22)
        self.assertEqual(c.segment[6]._exponent,
                         [11, 12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13,
                          12, 12, 13, 12, 12, 13, 12, 12, 13])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[7].Cqcc, 0)
        # quantization type
        self.assertEqual(c.segment[7].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[7]._guardBits, 4)
        self.assertEqual(c.segment[7]._mantissa, [0] * 19)
        self.assertEqual(c.segment[7]._exponent,
                         [11, 12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13,
                             12, 12, 13, 12, 12, 13])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[8].Cqcc, 2)
        # quantization type
        self.assertEqual(c.segment[8].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[8]._guardBits, 4)
        self.assertEqual(c.segment[8]._mantissa, [0] * 25)
        self.assertEqual(c.segment[8]._exponent,
                         [11, 12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13,
                          12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[9].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[9].Ccme.decode('latin-1'),
                         "Kakadu-3.0.7")

        # SOT: start of tile part
        self.assertEqual(c.segment[10].Isot, 0)
        self.assertEqual(c.segment[10].Psot, 3820593)
        self.assertEqual(c.segment[10].TPsot, 0)
        self.assertEqual(c.segment[10].TNsot, 1)  # unknown

    def test_NR_p0_09_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_09.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "0" means profile 2, or full capabilities
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 17)
        self.assertEqual(c.segment[1].Ysiz, 37)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (17, 37))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)  # scalar expounded
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa,
                         [1915, 1884, 1884, 1853, 1884, 1884, 1853, 1962, 1962,
                          1986, 53, 53, 120, 26, 26, 1983])
        self.assertEqual(c.segment[3]._exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 12, 12, 12,
                          11, 11, 12])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "Kakadu-3.0.7")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].Isot, 0)
        self.assertEqual(c.segment[5].Psot, 478)
        self.assertEqual(c.segment[5].TPsot, 0)
        self.assertEqual(c.segment[5].TNsot, 1)  # unknown

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[6].id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[7].id, 'EOC')

    def test_NR_p0_10_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_10.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 256)
        self.assertEqual(c.segment[1].Ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(4, 4), (4, 4), (4, 4)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 2)  # 2
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3]._guardBits, 0)
        self.assertEqual(c.segment[3]._mantissa, [0] * 10)
        self.assertEqual(c.segment[3]._exponent,
                         [11, 12, 12, 13, 12, 12, 13, 12, 12, 13])

        # SOT: start of tile part
        self.assertEqual(c.segment[4].Isot, 0)
        self.assertEqual(c.segment[4].Psot, 2453)
        self.assertEqual(c.segment[4].TPsot, 0)
        self.assertEqual(c.segment[4].TNsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[5].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[6].Isot, 1)
        self.assertEqual(c.segment[6].Psot, 2403)
        self.assertEqual(c.segment[6].TPsot, 0)
        self.assertEqual(c.segment[6].TNsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[7].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[8].Isot, 2)
        self.assertEqual(c.segment[8].Psot, 2420)
        self.assertEqual(c.segment[8].TPsot, 0)
        self.assertEqual(c.segment[8].TNsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[9].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[10].Isot, 3)
        self.assertEqual(c.segment[10].Psot, 2472)
        self.assertEqual(c.segment[10].TPsot, 0)
        self.assertEqual(c.segment[10].TNsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[11].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[12].Isot, 0)
        self.assertEqual(c.segment[12].Psot, 1043)
        self.assertEqual(c.segment[12].TPsot, 1)
        self.assertEqual(c.segment[12].TNsot, 2)

        # SOD:  start of data
        self.assertEqual(c.segment[13].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[14].Isot, 1)
        self.assertEqual(c.segment[14].Psot, 1101)
        self.assertEqual(c.segment[14].TPsot, 1)
        self.assertEqual(c.segment[14].TNsot, 2)

        # SOD:  start of data
        self.assertEqual(c.segment[15].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[16].Isot, 3)
        self.assertEqual(c.segment[16].Psot, 1054)
        self.assertEqual(c.segment[16].TPsot, 1)
        self.assertEqual(c.segment[16].TNsot, 2)

        # SOD:  start of data
        self.assertEqual(c.segment[17].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[18].Isot, 2)
        self.assertEqual(c.segment[18].Psot, 14)
        self.assertEqual(c.segment[18].TPsot, 1)
        self.assertEqual(c.segment[18].TNsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[19].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[20].Isot, 2)
        self.assertEqual(c.segment[20].Psot, 1089)
        self.assertEqual(c.segment[20].TPsot, 2)
        self.assertEqual(c.segment[20].TNsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[21].id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[22].id, 'EOC')

    def test_NR_p0_11_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_11.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 128)
        self.assertEqual(c.segment[1].Ysiz, 1)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)
        self.assertTrue(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 0)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(c.segment[2]._precinct_size, [(128, 2)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3]._guardBits, 3)
        self.assertEqual(c.segment[3]._mantissa, [0])
        self.assertEqual(c.segment[3]._exponent, [8])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].Isot, 0)
        self.assertEqual(c.segment[5].Psot, 118)
        self.assertEqual(c.segment[5].TPsot, 0)
        self.assertEqual(c.segment[5].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[6].id, 'SOD')

        # SOP, EPH
        sop = [x.id for x in c.segment if x.id == 'SOP']
        eph = [x.id for x in c.segment if x.id == 'EPH']
        self.assertEqual(len(sop), 0)
        self.assertEqual(len(eph), 1)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].id, 'EOC')

    def test_NR_p0_12_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_12.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 3)
        self.assertEqual(c.segment[1].Ysiz, 5)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (3, 5))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3]._guardBits, 3)
        self.assertEqual(c.segment[3]._mantissa, [0] * 10)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].Isot, 0)
        self.assertEqual(c.segment[5].Psot, 162)
        self.assertEqual(c.segment[5].TPsot, 0)
        self.assertEqual(c.segment[5].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[6].id, 'SOD')

        # SOP, EPH
        sop = [x.id for x in c.segment if x.id == 'SOP']
        eph = [x.id for x in c.segment if x.id == 'EPH']
        self.assertEqual(len(sop), 4)
        self.assertEqual(len(eph), 0)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].id, 'EOC')

    def test_NR_p0_13_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_13.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1)
        self.assertEqual(c.segment[1].Ysiz, 1)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (1, 1))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, tuple([8] * 257))
        # signed
        self.assertEqual(c.segment[1]._signed, tuple([False] * 257))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 257)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 1)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].Ccoc, 2)
        self.assertEqual(c.segment[3].SPcoc[0], 1)  # levels
        self.assertEqual(tuple(c.segment[3]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4]._guardBits, 2)
        self.assertEqual(c.segment[4]._mantissa, [0] * 4)
        self.assertEqual(c.segment[4]._exponent,
                         [8, 9, 9, 10])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].Cqcc, 1)
        self.assertEqual(c.segment[5]._guardBits, 3)
        # quantization type
        self.assertEqual(c.segment[5].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[5]._exponent, [9, 10, 10, 11])
        self.assertEqual(c.segment[5]._mantissa, [0, 0, 0, 0])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].Cqcc, 2)
        self.assertEqual(c.segment[6]._guardBits, 2)
        # quantization type
        self.assertEqual(c.segment[6].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[6]._exponent, [9, 10, 10, 11])
        self.assertEqual(c.segment[6]._mantissa, [0, 0, 0, 0])

        # RGN: region of interest
        self.assertEqual(c.segment[7].Crgn, 3)
        self.assertEqual(c.segment[7].Srgn, 0)
        self.assertEqual(c.segment[7].SPrgn, 11)

        # POD:  progression order change
        self.assertEqual(c.segment[8].RSpod, (0, 0))
        self.assertEqual(c.segment[8].CSpod, (0, 128))
        self.assertEqual(c.segment[8].LYEpod, (1, 1))
        self.assertEqual(c.segment[8].REpod, (33, 33))
        self.assertEqual(c.segment[8].CEpod, (128, 257))
        self.assertEqual(c.segment[8].Ppod,
                         (glymur.core.RLCP, glymur.core.CPRL))

        # COM: comment
        # Registration
        self.assertEqual(c.segment[9].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[9].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[10].Isot, 0)
        self.assertEqual(c.segment[10].Psot, 1537)
        self.assertEqual(c.segment[10].TPsot, 0)
        self.assertEqual(c.segment[10].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[11].id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[12].id, 'EOC')

    def test_NR_p0_14_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_14.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "0" means profile 2
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 49)
        self.assertEqual(c.segment[1].Ysiz, 49)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (49, 49))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # 1 layer
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [10, 11, 11, 12, 11, 11, 12, 11, 11, 12, 11, 11, 12,
                          11, 11, 12])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "Kakadu-3.0.7")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].Isot, 0)
        self.assertEqual(c.segment[5].Psot, 1528)
        self.assertEqual(c.segment[5].TPsot, 0)
        self.assertEqual(c.segment[5].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[6].id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[7].id, 'EOC')

    def test_NR_p0_15_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_15.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].Rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 256)
        self.assertEqual(c.segment[1].Ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (4,))
        # signed
        self.assertEqual(c.segment[1]._signed, (True,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2]._layers, 8)  # layers = 8
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 1)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 1)  # derived
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0])
        self.assertEqual(c.segment[3]._exponent, [0])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].Cqcc, 0)
        self.assertEqual(c.segment[4]._guardBits, 2)
        # quantization type
        self.assertEqual(c.segment[4].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[4]._mantissa, [0] * 4)
        self.assertEqual(c.segment[4]._exponent, [4, 5, 5, 6])

        # POD: progression order change
        self.assertEqual(c.segment[5].RSpod, (0,))
        self.assertEqual(c.segment[5].CSpod, (0,))
        self.assertEqual(c.segment[5].LYEpod, (8,))
        self.assertEqual(c.segment[5].REpod, (33,))
        self.assertEqual(c.segment[5].CEpod, (255,))
        self.assertEqual(c.segment[5].Ppod, (glymur.core.LRCP,))

        # CRG:  component registration
        self.assertEqual(c.segment[6].Xcrg, (65424,))
        self.assertEqual(c.segment[6].Ycrg, (32558,))

        # COM: comment
        # Registration
        self.assertEqual(c.segment[7].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[7].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,"
                         + "2001 Algo Vision Technology")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[9].Rcme, glymur.core.RCME_BINARY)
        # Comment value
        self.assertEqual(len(c.segment[9].Ccme), 62)

        # TLM: tile-part length
        self.assertEqual(c.segment[10].Ztlm, 0)
        self.assertEqual(c.segment[10].Ttlm, (0, 1, 2, 3))
        self.assertEqual(c.segment[10].Ptlm, (4267, 2117, 4080, 2081))

        # SOT: start of tile part
        self.assertEqual(c.segment[11].Isot, 0)
        self.assertEqual(c.segment[11].Psot, 4267)
        self.assertEqual(c.segment[11].TPsot, 0)
        self.assertEqual(c.segment[11].TNsot, 1)

        # RGN: region of interest
        self.assertEqual(c.segment[12].Crgn, 0)
        self.assertEqual(c.segment[12].Srgn, 0)
        self.assertEqual(c.segment[12].SPrgn, 7)

        # SOD:  start of data
        self.assertEqual(c.segment[13].id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        # SOT: start of tile part
        self.assertEqual(c.segment[31].Isot, 1)
        self.assertEqual(c.segment[31].Psot, 2117)
        self.assertEqual(c.segment[31].TPsot, 0)
        self.assertEqual(c.segment[31].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[32].id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        # SOT: start of tile part
        self.assertEqual(c.segment[49].Isot, 2)
        self.assertEqual(c.segment[49].Psot, 4080)
        self.assertEqual(c.segment[49].TPsot, 0)
        self.assertEqual(c.segment[49].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[50].id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        # SOT: start of tile part
        self.assertEqual(c.segment[67].Isot, 3)
        self.assertEqual(c.segment[67].Psot, 2081)
        self.assertEqual(c.segment[67].TPsot, 0)
        self.assertEqual(c.segment[67].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[68].id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        # EOC:  end of codestream
        self.assertEqual(c.segment[85].id, 'EOC')

    def test_NR_p0_16_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_16.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "0" means profile 2
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 128)
        self.assertEqual(c.segment[1].Ysiz, 128)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)
        self.assertFalse(c.segment[2].Scod & 4)
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2]._layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 10)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])

        # SOT: start of tile part
        self.assertEqual(c.segment[4].Isot, 0)
        self.assertEqual(c.segment[4].Psot, 7331)
        self.assertEqual(c.segment[4].TPsot, 0)
        self.assertEqual(c.segment[4].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[5].id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[6].id, 'EOC')

    def test_NR_p1_01_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_01.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].Rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 127)
        self.assertEqual(c.segment[1].Ysiz, 227)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (5, 128))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (127, 126))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (1, 101))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(2, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)  # SOP
        self.assertTrue(c.segment[2].Scod & 4)  # EPH
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 5)  # layers = 5
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 3)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].Ccoc, 0)
        self.assertEqual(c.segment[3].SPcoc[0], 3)  # level
        self.assertEqual(tuple(c.segment[3]._code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[3].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[3].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[3].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4]._guardBits, 3)
        self.assertEqual(c.segment[4]._mantissa, [0] * 10)
        self.assertEqual(c.segment[4]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[5].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[6].Isot, 0)
        self.assertEqual(c.segment[6].Psot, 4627)
        self.assertEqual(c.segment[6].TPsot, 0)
        self.assertEqual(c.segment[6].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[7].id, 'SOD')

        # SOP, EPH
        sop = [x.id for x in c.segment if x.id == 'SOP']
        eph = [x.id for x in c.segment if x.id == 'EPH']
        self.assertEqual(len(sop), 20)
        self.assertEqual(len(eph), 20)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].id, 'EOC')

    def test_NR_p1_02_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_02.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].Rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 640)
        self.assertEqual(c.segment[1].Ysiz, 480)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (640, 480))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, tuple([8] * 3))
        # signed
        self.assertEqual(c.segment[1]._signed, tuple([False] * 3))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 19)  # layers = 19
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 6)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertTrue(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertTrue(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(c.segment[2]._precinct_size,
                         [(128, 128), (256, 256), (512, 512), (1024, 1024),
                          (2048, 2048), (4096, 4096), (8192, 8192)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[3]._guardBits, 3)
        self.assertEqual(c.segment[3]._mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                          1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                          1888])
        self.assertEqual(c.segment[3]._exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13,
                          11, 11, 11, 11, 11, 11])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].Cqcc, 1)
        self.assertEqual(c.segment[4]._guardBits, 3)
        # quantization type
        self.assertEqual(c.segment[4].Sqcc & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[4]._mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                          1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                          1888])
        self.assertEqual(c.segment[4]._exponent,
                         [14, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11,
                          9, 9, 9, 9, 9, 9])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].Cqcc, 2)
        self.assertEqual(c.segment[5]._guardBits, 3)
        # quantization type
        self.assertEqual(c.segment[5].Sqcc & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[5]._mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                          1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                          1888])
        self.assertEqual(c.segment[5]._exponent,
                         [14, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11,
                          9, 9, 9, 9, 9, 9])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[6].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[6].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[7].Isot, 0)
        self.assertEqual(c.segment[7].Psot, 262838)
        self.assertEqual(c.segment[7].TPsot, 0)
        self.assertEqual(c.segment[7].TNsot, 1)

        # PPT:  packed packet headers, tile-part header
        self.assertEqual(c.segment[8].id, 'PPT')
        self.assertEqual(c.segment[8].Zppt, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[9].id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[10].id, 'EOC')

    def test_NR_p1_03_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_03.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].Rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1024)
        self.assertEqual(c.segment[1].Ysiz, 1024)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1024, 1024))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, tuple([8] * 4))
        # signed
        self.assertEqual(c.segment[1]._signed, tuple([False] * 4))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1), (1, 1), (2, 2), (2, 2)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2]._layers, 10)  # layers = 10
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 6)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertTrue(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].Ccoc, 1)
        self.assertEqual(c.segment[3].SPcoc[0], 3)  # level
        self.assertEqual(tuple(c.segment[3]._code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertTrue(c.segment[3].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[3].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[4].Ccoc, 3)
        self.assertEqual(c.segment[4].SPcoc[0], 6)  # level
        self.assertEqual(tuple(c.segment[4]._code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertTrue(c.segment[4].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[4].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[4].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[4].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[4].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[4].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[4].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[5].Sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[5]._guardBits, 3)
        self.assertEqual(c.segment[5]._mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                             1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                             1888])
        self.assertEqual(c.segment[5]._exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13,
                             11, 11, 11, 11, 11, 11])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].Cqcc, 0)
        self.assertEqual(c.segment[6]._guardBits, 3)
        # quantization type
        self.assertEqual(c.segment[6].Sqcc & 0x1f, 1)  # derived
        self.assertEqual(c.segment[6]._mantissa, [0])
        self.assertEqual(c.segment[6]._exponent, [14])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[7].Cqcc, 3)
        self.assertEqual(c.segment[7]._guardBits, 3)
        # quantization type
        self.assertEqual(c.segment[7].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[7]._mantissa, [0] * 19)
        self.assertEqual(c.segment[7]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10,
                          9, 9, 10])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # PPM:  packed packet headers, main header
        self.assertEqual(c.segment[9].id, 'PPM')
        self.assertEqual(c.segment[9].Zppm, 0)

        # TLM (tile-part length)
        self.assertEqual(c.segment[10].Ztlm, 0)
        self.assertEqual(c.segment[10].Ttlm, (0,))
        self.assertEqual(c.segment[10].Ptlm, (1366780,))

        # SOT: start of tile part
        self.assertEqual(c.segment[11].Isot, 0)
        self.assertEqual(c.segment[11].Psot, 1366780)
        self.assertEqual(c.segment[11].TPsot, 0)
        self.assertEqual(c.segment[11].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[12].id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[13].id, 'EOC')

    def test_NR_p1_04_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_04.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].Rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1024)
        self.assertEqual(c.segment[1].Ysiz, 1024)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (12,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 3)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa,
                         [84, 423, 408, 435, 450, 435, 470, 549, 520, 618])
        self.assertEqual(c.segment[3]._exponent,
                         [8, 10, 10, 10, 9, 9, 9, 8, 8, 8])

        # TLM (tile-part length)
        self.assertEqual(c.segment[4].Ztlm, 0)
        self.assertIsNone(c.segment[4].Ttlm)
        self.assertEqual(c.segment[4].Ptlm,
                         (350, 356, 402, 245, 402, 564, 675, 283, 317, 299,
                          330, 333, 346, 403, 839, 667, 328, 349, 274, 325,
                          501, 561, 756, 710, 779, 620, 628, 675, 600, 66195,
                          721, 719, 565, 565, 546, 586, 574, 641, 713, 634,
                          573, 528, 544, 597, 771, 665, 624, 706, 568, 537,
                          554, 546, 542, 635, 826, 667, 617, 606, 813, 586,
                          641, 654, 669, 623))

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[5].Ccme.decode('latin-1'),
                         "Created by Aware, Inc.")

        # SOT: start of tile part
        self.assertEqual(c.segment[6].Isot, 0)
        self.assertEqual(c.segment[6].Psot, 350)
        self.assertEqual(c.segment[6].TPsot, 0)
        self.assertEqual(c.segment[6].TNsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[7].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[8].Isot, 1)
        self.assertEqual(c.segment[8].Psot, 356)
        self.assertEqual(c.segment[8].TPsot, 0)
        self.assertEqual(c.segment[8].TNsot, 1)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[9].Sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[9]._guardBits, 2)
        self.assertEqual(c.segment[9]._mantissa,
                         [75, 1093, 1098, 1115, 1157, 1134, 1186, 1217, 1245,
                          1248])
        self.assertEqual(c.segment[9]._exponent,
                         [8, 10, 10, 10, 9, 9, 9, 8, 8, 8])

        # SOD:  start of data
        self.assertEqual(c.segment[10].id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[11].Isot, 2)
        self.assertEqual(c.segment[11].Psot, 402)
        self.assertEqual(c.segment[11].TPsot, 0)
        self.assertEqual(c.segment[11].TNsot, 1)

        # and so on

        # There should be 64 SOD, SOT, QCD segments.
        ids = [x.id for x in c.segment if x.id == 'SOT']
        self.assertEqual(len(ids), 64)
        ids = [x.id for x in c.segment if x.id == 'SOD']
        self.assertEqual(len(ids), 64)
        ids = [x.id for x in c.segment if x.id == 'QCD']
        self.assertEqual(len(ids), 64)

        # Tiles should be in order, right?
        tiles = [x.Isot for x in c.segment if x.id == 'SOT']
        self.assertEqual(tiles, list(range(64)))

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].id, 'EOC')

    def test_NR_p1_05_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_05.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].Rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 529)
        self.assertEqual(c.segment[1].Ysiz, 524)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (17, 12))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (37, 37))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (8, 2))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)  # sop
        self.assertTrue(c.segment[2].Scod & 4)  # eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2]._layers, 2)  # levels = 2
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 7)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 8))  # cblk
        # Selective arithmetic coding bypass
        self.assertTrue(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertTrue(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(c.segment[2]._precinct_size, [(16, 16)] * 8)

        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[3]._guardBits, 3)
        self.assertEqual(c.segment[3]._mantissa,
                         [1813, 1814, 1814, 1814, 1815, 1815, 1817, 1821,
                          1821, 1827, 1845, 1845, 1868, 1925, 1925, 2007,
                          32, 32, 131, 2002, 2002, 1888])
        self.assertEqual(c.segment[3]._exponent,
                         [17, 17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14,
                         14, 13, 13, 13, 11, 11, 11, 11, 11, 11])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # 225 consecutive PPM segments.
        zppm = [x.Zppm for x in c.segment[5:230]]
        self.assertEqual(zppm, list(range(225)))

        # SOT: start of tile part
        self.assertEqual(c.segment[230].Isot, 0)
        self.assertEqual(c.segment[230].Psot, 580)
        self.assertEqual(c.segment[230].TPsot, 0)
        self.assertEqual(c.segment[230].TNsot, 1)

        # 225 total SOT segments
        Isot = [x.Isot for x in c.segment if x.id == 'SOT']
        self.assertEqual(Isot, list(range(225)))

        # scads of SOP, EPH segments
        sop = [x.id for x in c.segment if x.id == 'SOP']
        eph = [x.id for x in c.segment if x.id == 'EPH']
        self.assertEqual(len(sop), 26472)
        self.assertEqual(len(eph), 0)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].id, 'EOC')

    def test_NR_p1_06_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].Rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 12)
        self.assertEqual(c.segment[1].Ysiz, 12)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (3, 3))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)  # sop
        self.assertTrue(c.segment[2].Scod & 4)  # eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 4)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (32, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertTrue(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[3]._guardBits, 3)
        self.assertEqual(c.segment[3]._mantissa,
                         [1821, 1845, 1845, 1868, 1925, 1925, 2007, 32,
                          32, 131, 2002, 2002, 1888])
        self.assertEqual(c.segment[3]._exponent,
                         [14, 14, 14, 14, 13, 13, 13, 11, 11, 11,
                          11, 11, 11])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].Isot, 0)
        self.assertEqual(c.segment[5].Psot, 349)
        self.assertEqual(c.segment[5].TPsot, 0)
        self.assertEqual(c.segment[5].TNsot, 1)

        # PPT:  packed packet headers, tile-part header
        self.assertEqual(c.segment[6].id, 'PPT')
        self.assertEqual(c.segment[6].Zppt, 0)

        # scads of SOP, EPH segments

        # 16 SOD segments
        sods = [x for x in c.segment if x.id == 'SOD']
        self.assertEqual(len(sods), 16)

        # 16 PPT segments
        ppts = [x for x in c.segment if x.id == 'PPT']
        self.assertEqual(len(ppts), 16)

        # 16 SOT segments
        Isots = [x.Isot for x in c.segment if x.id == 'SOT']
        self.assertEqual(Isots, list(range(16)))

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].id, 'EOC')

    def test_NR_p1_07_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/p1_07.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].Rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 12)
        self.assertEqual(c.segment[1].Ysiz, 12)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (4, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (12, 12))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (4, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(4, 1), (1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].Scod & 2)  # sop
        self.assertTrue(c.segment[2].Scod & 4)  # eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RPCL)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 1)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(c.segment[2]._precinct_size, [(1, 1), (2, 2)])

        # COC: Coding style component
        self.assertEqual(c.segment[3].Ccoc, 1)
        self.assertEqual(c.segment[3].SPcoc[0], 1)  # level
        self.assertEqual(tuple(c.segment[3]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(c.segment[3]._precinct_size, [(2, 2), (4, 4)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].Sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4]._guardBits, 2)
        self.assertEqual(c.segment[4]._mantissa, [0] * 4)
        self.assertEqual(c.segment[4]._exponent, [8, 9, 9, 10])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[5].Ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[6].Isot, 0)
        self.assertEqual(c.segment[6].Psot, 434)
        self.assertEqual(c.segment[6].TPsot, 0)
        self.assertEqual(c.segment[6].TNsot, 1)

        # scads of SOP, EPH segments

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].id, 'EOC')

    def test_NR_file1_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/file1.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'xml ', 'jp2h', 'xml ',
                               'jp2c'])

        ids = [box.id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # XML box
        tags = [x.tag for x in jp2.box[2].xml]
        self.assertEqual(tags,
                         ['{http://www.jpeg.org/jpx/1.0/xml}'
                          + 'GENERAL_CREATION_INFO'])

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 512)
        self.assertEqual(jp2.box[3].box[0].width, 768)
        self.assertEqual(jp2.box[3].box[0].num_components, 3)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method, 1)
        self.assertEqual(jp2.box[3].box[1].precedence, 0)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact ??
        self.assertEqual(jp2.box[3].box[1].colorspace, glymur.core.SRGB)

        # XML box
        tags = [x.tag for x in jp2.box[4].xml]
        self.assertEqual(tags, ['{http://www.jpeg.org/jpx/1.0/xml}CAPTION',
                                '{http://www.jpeg.org/jpx/1.0/xml}LOCATION',
                                '{http://www.jpeg.org/jpx/1.0/xml}EVENT'])

    def test_NR_file2_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/file2.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'cdef'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 640)
        self.assertEqual(jp2.box[2].box[0].width, 480)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact??
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.YCC)

        # Jp2 Header
        # Channel Definition
        self.assertEqual(jp2.box[2].box[2].index, (0, 1, 2))
        self.assertEqual(jp2.box[2].box[2].channel_type, (0, 0, 0))  # color
        self.assertEqual(jp2.box[2].box[2].association, (3, 2, 1))  # reverse

    def test_NR_file3_dump(self):
        # Three 8-bit components in the sRGB-YCC colourspace, with the Cb and
        # Cr components being subsampled 2x in both the horizontal and
        # vertical directions. The components are stored in the standard
        # order.
        jfile = os.path.join(data_root, 'input/conformance/file3.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 640)
        self.assertEqual(jp2.box[2].box[0].width, 480)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.YCC)

        # sub-sampling
        codestream = jp2.get_codestream()
        self.assertEqual(codestream.segment[1].XRsiz[0], 1)
        self.assertEqual(codestream.segment[1].YRsiz[0], 1)
        self.assertEqual(codestream.segment[1].XRsiz[1], 2)
        self.assertEqual(codestream.segment[1].YRsiz[1], 2)
        self.assertEqual(codestream.segment[1].XRsiz[2], 2)
        self.assertEqual(codestream.segment[1].YRsiz[2], 2)

    def test_NR_file4_dump(self):
        # One 8-bit component in the sRGB-grey colourspace.
        jfile = os.path.join(data_root, 'input/conformance/file4.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 512)
        self.assertEqual(jp2.box[2].box[0].width, 768)
        self.assertEqual(jp2.box[2].box[0].num_components, 1)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact?
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_NR_file5_dump(self):
        # Three 8-bit components in the ROMM-RGB colourspace, encapsulated in a
        # JP2 compatible JPX file. The components have been transformed using
        # the RCT. The colourspace is specified using both a Restricted ICC
        # profile and using the JPX-defined enumerated code for the ROMM-RGB
        # colourspace.
        jfile = os.path.join(data_root, 'input/conformance/file5.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jpx ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[2], 'jpx ')
        self.assertEqual(jp2.box[1].compatibility_list[3], 'jpxb')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 512)
        self.assertEqual(jp2.box[3].box[0].width, 768)
        self.assertEqual(jp2.box[3].box[0].num_components, 3)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method, 2)  # enumerated
        self.assertEqual(jp2.box[3].box[1].precedence, 0)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[3].box[1].icc_profile['Size'], 546)
        self.assertIsNone(jp2.box[3].box[1].colorspace)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[2].method, 1)  # enumerated
        self.assertEqual(jp2.box[3].box[2].precedence, 1)
        self.assertEqual(jp2.box[3].box[2].approximation, 1)  # JPX exact
        self.assertIsNone(jp2.box[3].box[2].icc_profile)
        self.assertEqual(jp2.box[3].box[2].colorspace,
                         glymur.core.ROMM_RGB)

    def test_NR_file6_dump(self):
        jfile = os.path.join(data_root, 'input/conformance/file6.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 512)
        self.assertEqual(jp2.box[2].box[0].width, 768)
        self.assertEqual(jp2.box[2].box[0].num_components, 1)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 12)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact
        self.assertIsNone(jp2.box[2].box[1].icc_profile)
        self.assertEqual(jp2.box[2].box[1].colorspace,
                         glymur.core.GREYSCALE)

    def test_NR_file7_dump(self):
        # Three 16-bit components in the e-sRGB colourspace, encapsulated in a
        # JP2 compatible JPX file. The components have been transformed using
        # the RCT. The colourspace is specified using both a Restricted ICC
        # profile and using the JPX-defined enumerated code for the e-sRGB
        # colourspace.
        jfile = os.path.join(data_root, 'input/conformance/file7.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jpx ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[2], 'jpx ')
        self.assertEqual(jp2.box[1].compatibility_list[3], 'jpxb')
        self.assertEqual(jp2.box[1].minor_version, 0)

        # Reader requirements talk.
        self.assertTrue(glymur.core.RREQ_E_SRGB_ENUMERATED_COLORSPACE
                        in jp2.box[2].standard_flag)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 640)
        self.assertEqual(jp2.box[3].box[0].width, 480)
        self.assertEqual(jp2.box[3].box[0].num_components, 3)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 16)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method, 2)
        self.assertEqual(jp2.box[3].box[1].precedence, 0)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[3].box[1].icc_profile['Size'], 13332)
        self.assertIsNone(jp2.box[3].box[1].colorspace)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[2].method, 1)  # enumerated cspace
        self.assertEqual(jp2.box[3].box[2].precedence, 1)
        self.assertEqual(jp2.box[3].box[2].approximation, 1)  # JPX exact
        self.assertIsNone(jp2.box[3].box[2].icc_profile)
        self.assertEqual(jp2.box[3].box[2].colorspace,
                         glymur.core.E_SRGB)

    def test_NR_file8_dump(self):
        # One 8-bit component in a gamma 1.8 space. The colourspace is
        # specified using a Restricted ICC profile.
        jfile = os.path.join(data_root, 'input/conformance/file8.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'xml ', 'jp2c',
                               'xml '])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 400)
        self.assertEqual(jp2.box[2].box[0].width, 700)
        self.assertEqual(jp2.box[2].box[0].num_components, 1)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 2)  # enumerated
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[2].box[1].icc_profile['Size'], 414)
        self.assertIsNone(jp2.box[2].box[1].colorspace)

        # XML box
        tags = [x.tag for x in jp2.box[3].xml]
        self.assertEqual(tags,
                         ['{http://www.jpeg.org/jpx/1.0/xml}'
                          + 'GENERAL_CREATION_INFO'])

        # XML box
        tags = [x.tag for x in jp2.box[5].xml]
        self.assertEqual(tags,
                         ['{http://www.jpeg.org/jpx/1.0/xml}CAPTION',
                          '{http://www.jpeg.org/jpx/1.0/xml}LOCATION',
                          '{http://www.jpeg.org/jpx/1.0/xml}THING',
                          '{http://www.jpeg.org/jpx/1.0/xml}EVENT'])

    def test_NR_file9_dump(self):
        # Colormap
        jfile = os.path.join(data_root, 'input/conformance/file9.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'pclr', 'cmap', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 512)
        self.assertEqual(jp2.box[2].box[0].width, 768)
        self.assertEqual(jp2.box[2].box[0].num_components, 1)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Palette box.
        self.assertEqual(len(jp2.box[2].box[1].palette), 3)
        self.assertEqual(len(jp2.box[2].box[1].palette[0]), 256)
        self.assertEqual(len(jp2.box[2].box[1].palette[1]), 256)
        self.assertEqual(len(jp2.box[2].box[1].palette[2]), 256)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[0][0], 0)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[1][0], 0)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[2][0], 0)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[0][128], 73)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[1][128], 92)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[2][128], 53)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[0][-1], 245)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[1][-1], 245)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[2][-1], 245)

        # Component mapping box
        self.assertEqual(jp2.box[2].box[2].component_index, (0, 0, 0))
        self.assertEqual(jp2.box[2].box[2].mapping_type, (1, 1, 1))
        self.assertEqual(jp2.box[2].box[2].palette_index, (0, 1, 2))

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[3].method, 1)
        self.assertEqual(jp2.box[2].box[3].precedence, 0)
        self.assertEqual(jp2.box[2].box[3].approximation, 1)  # JPX exact
        self.assertIsNone(jp2.box[2].box[3].icc_profile)
        self.assertEqual(jp2.box[2].box[3].colorspace, glymur.core.SRGB)

    def test_NR_00042_j2k_dump(self):
        # Profile 3.
        jfile = os.path.join(data_root, 'input/nonregression/_00042.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "3" means profile 3
        self.assertEqual(c.segment[1].Rsiz, 3)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1920)
        self.assertEqual(c.segment[1].Ysiz, 1080)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1920, 1080))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (12, 12, 12))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.CPRL)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(c.segment[2]._precinct_size[0], (128, 128))
        self.assertEqual(c.segment[2]._precinct_size[1:], [(256, 256)] * 5)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa,
                         [1824, 1776, 1776, 1728, 1792, 1792, 1760, 1872,
                          1872, 1896, 5, 5, 71, 2003, 2003, 1890])
        self.assertEqual(c.segment[3]._exponent,
                         [18, 18, 18, 18, 17, 17, 17, 16,
                          16, 16, 14, 14, 14, 14, 14, 14])

        # COC: Coding style component
        self.assertEqual(c.segment[4].Ccoc, 1)
        self.assertEqual(c.segment[4].SPcoc[0], 5)  # level
        self.assertEqual(tuple(c.segment[4]._code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[4].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[4].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[4].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[4].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[4].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[4].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[4].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].Cqcc, 1)
        self.assertEqual(c.segment[5]._guardBits, 2)
        # quantization type
        self.assertEqual(c.segment[5].Sqcc & 0x1f, 2)
        self.assertEqual(c.segment[5]._mantissa,
                         [1824, 1776, 1776, 1728, 1792, 1792, 1760, 1872,
                          1872, 1896, 5, 5, 71, 2003, 2003, 1890])
        self.assertEqual(c.segment[5]._exponent,
                         [18, 18, 18, 18, 17, 17, 17, 16, 16, 16, 14, 14, 14,
                          14, 14, 14])

        # COC: Coding style component
        self.assertEqual(c.segment[6].Ccoc, 2)
        self.assertEqual(c.segment[6].SPcoc[0], 5)  # level
        self.assertEqual(tuple(c.segment[6]._code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[6].SPcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[6].SPcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[6].SPcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[6].SPcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[6].SPcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[6].SPcoc[3] & 0x0020)
        self.assertEqual(c.segment[6].SPcoc[4],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[7].Cqcc, 2)
        self.assertEqual(c.segment[7]._guardBits, 2)
        # quantization type
        self.assertEqual(c.segment[7].Sqcc & 0x1f, 2)  # none
        self.assertEqual(c.segment[7]._mantissa,
                         [1824, 1776, 1776, 1728, 1792, 1792, 1760, 1872,
                         1872, 1896, 5, 5, 71, 2003, 2003, 1890])
        self.assertEqual(c.segment[7]._exponent,
                         [18, 18, 18, 18, 17, 17, 17, 16, 16, 16, 14, 14,
                         14, 14, 14, 14])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].Ccme.decode('latin-1'),
                         "Created by OpenJPEG version 1.3.0")

        # TLM (tile-part length)
        self.assertEqual(c.segment[9].Ztlm, 0)
        self.assertEqual(c.segment[9].Ttlm, (0, 0, 0))
        self.assertEqual(c.segment[9].Ptlm, (45274, 20838, 8909))

        # 3 tiles, one for each component
        idx = [x.Isot for x in c.segment if x.id == 'SOT']
        self.assertEqual(idx, [0, 0, 0])
        lens = [x.Psot for x in c.segment if x.id == 'SOT']
        self.assertEqual(lens, [45274, 20838, 8909])
        TPsot = [x.TPsot for x in c.segment if x.id == 'SOT']
        self.assertEqual(TPsot, [0, 1, 2])

        sods = [x for x in c.segment if x.id == 'SOD']
        self.assertEqual(len(sods), 3)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].id, 'EOC')

    def test_Bretagne2_j2k_dump(self):
        # Profile 3.
        jfile = os.path.join(data_root, 'input/nonregression/Bretagne2.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "3" means profile 3
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 2592)
        self.assertEqual(c.segment[1].Ysiz, 1944)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (640, 480))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(c.segment[2]._precinct_size,
                         [(16, 16), (32, 32), (64, 64), (128, 128),
                          (128, 128), (128, 128)])

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        expected += ['SOT', 'COC', 'QCC', 'COC', 'QCC', 'SOD'] * 25
        expected += ['EOC']
        self.assertEqual(ids, expected)

    def test_NR_buxI_j2k_dump(self):
        jfile = os.path.join(data_root, 'input/nonregression/buxI.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 512)
        self.assertEqual(c.segment[1].Ysiz, 512)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (512, 512))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'SOT', 'SOD', 'EOC']
        self.assertEqual(ids, expected)

    def test_NR_buxR_j2k_dump(self):
        jfile = os.path.join(data_root, 'input/nonregression/buxR.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 512)
        self.assertEqual(c.segment[1].Ysiz, 512)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (512, 512))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'SOT', 'SOD', 'EOC']
        self.assertEqual(ids, expected)

    def test_NR_Cannotreaddatawithnosizeknown_j2k(self):
        lst = ['input', 'nonregression',
               'Cannotreaddatawithnosizeknown.j2k']
        path = '/'.join(lst)

        jfile = os.path.join(data_root, path)
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1420)
        self.assertEqual(c.segment[1].Ysiz, 1416)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1420, 1416))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 4)
        self.assertEqual(c.segment[3]._mantissa, [0] * 34)
        self.assertEqual(c.segment[3]._exponent, [16] + [17, 17, 18] * 11)

    def test_NR_CT_Phillips_JPEG2K_Decompr_Problem_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/'
                             + 'CT_Phillips_JPEG2K_Decompr_Problem.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 512)
        self.assertEqual(c.segment[1].Ysiz, 614)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (512, 614))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (12,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa,
                         [442, 422, 422, 403, 422, 422, 403, 472, 472, 487,
                          591, 591, 676, 558, 558, 485])
        self.assertEqual(c.segment[3]._exponent,
                         [22, 22, 22, 22, 21, 21, 21, 20, 20, 20, 19, 19, 19,
                          18, 18, 18])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'), "Kakadu-3.2")

    def test_NR_cthead1_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/cthead1.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 256)
        self.assertEqual(c.segment[1].Ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (256, 256))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [9, 10, 10, 11, 10, 10, 11, 10, 10, 11, 10, 10, 10,
                          9, 9, 10])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'), "Kakadu-v6.3.1")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'), "Kakadu-v6.3.1")

    @unittest.skip("fprintf stderr output in r2343.")
    def test_NR_illegalcolortransform_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/illegalcolortransform.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1420)
        self.assertEqual(c.segment[1].Ysiz, 1416)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1420, 1416))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 4)
        self.assertEqual(c.segment[3]._mantissa, [0] * 34)
        self.assertEqual(c.segment[3]._exponent, [16] + [17, 17, 18] * 11)

    def test_NR_j2k32_dump(self):
        jfile = os.path.join(data_root, 'input/nonregression/j2k32.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 256)
        self.assertEqual(c.segment[1].Ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (256, 256))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (True, True, True))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent, [8, 9, 9, 10, 9, 9, 10, 9, 9,
                         10, 9, 9, 10, 9, 9, 10])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_BINARY)
        # Comment value
        self.assertEqual(len(c.segment[4].Ccme), 36)

    def test_NR_kakadu_v4_4_openjpegv2_broken_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/'
                             + 'kakadu_v4-4_openjpegv2_broken.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 2048)
        self.assertEqual(c.segment[1].Ysiz, 2500)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (2048, 2500))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 12)  # layers = 12
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 8)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa, [0] * 25)
        self.assertEqual(c.segment[3]._exponent,
                         [17, 18, 18, 19, 18, 18, 19, 18, 18, 19, 18, 18, 19,
                          18, 18, 19, 18, 18, 19, 18, 18, 19, 18, 18, 19])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'), "Kakadu-v4.4")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        expected = "Kdu-Layer-Info: log_2{Delta-D(MSE)/[2^16*Delta-L(bytes)]},"
        expected += " L(bytes)\n"
        expected += " -65.4, 6.8e+004\n"
        expected += " -66.3, 1.0e+005\n"
        expected += " -67.3, 2.0e+005\n"
        expected += " -68.5, 4.1e+005\n"
        expected += " -69.0, 5.1e+005\n"
        expected += " -69.5, 5.9e+005\n"
        expected += " -69.7, 6.8e+005\n"
        expected += " -70.3, 8.2e+005\n"
        expected += " -70.8, 1.0e+006\n"
        expected += " -71.9, 1.4e+006\n"
        expected += " -73.8, 2.0e+006\n"
        expected += "-256.0, 3.7e+006\n"
        self.assertEqual(c.segment[5].Ccme.decode('latin-1'), expected)

    def test_NR_MarkerIsNotCompliant_j2k_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/MarkerIsNotCompliant.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1420)
        self.assertEqual(c.segment[1].Ysiz, 1416)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1420, 1416))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 4)
        self.assertEqual(c.segment[3]._mantissa, [0] * 34)
        self.assertEqual(c.segment[3]._exponent,
                         [16, 17, 17, 18, 17, 17, 18, 17, 17, 18, 17, 17, 18,
                          17, 17, 18, 17, 17, 18, 17, 17, 18, 17, 17, 18, 17,
                          17, 18, 17, 17, 18, 17, 17, 18])

    def test_NR_movie_00000(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/movie_00000.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1920)
        self.assertEqual(c.segment[1].Ysiz, 1080)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1920, 1080))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_movie_00001(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/movie_00001.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1920)
        self.assertEqual(c.segment[1].Ysiz, 1080)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1920, 1080))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_movie_00002(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/movie_00002.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1920)
        self.assertEqual(c.segment[1].Ysiz, 1080)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1920, 1080))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_orb_blue10_lin_j2k_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/orb-blue10-lin-j2k.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 117)
        self.assertEqual(c.segment[1].Ysiz, 117)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (117, 117))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 4)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_orb_blue10_win_j2k_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/orb-blue10-win-j2k.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 117)
        self.assertEqual(c.segment[1].Ysiz, 117)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (117, 117))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 4)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_pacs_ge_j2k_dump(self):
        jfile = os.path.join(data_root, 'input/nonregression/pacs.ge.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 512)
        self.assertEqual(c.segment[1].Ysiz, 512)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (512, 512))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1]._signed, (True,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 16)  # layers = 16
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [18, 19, 19, 20, 19, 19, 20, 19, 19, 20, 19, 19, 20,
                         19, 19, 20])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "Kakadu-2.0.2")

    def test_NR_test_lossless_j2k_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/test_lossless.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1024)
        self.assertEqual(c.segment[1].Ysiz, 1024)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1024, 1024))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (12,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [12, 13, 13, 14, 13, 13, 14, 13, 13, 14, 13, 13, 14,
                          13, 13, 14])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "ClearCanvas DICOM OpenJPEG")

    def test_NR_123_j2c_dump(self):
        jfile = os.path.join(data_root, 'input/nonregression/123.j2c')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1800)
        self.assertEqual(c.segment[1].Ysiz, 1800)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1800, 1800))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 4)
        self.assertEqual(c.segment[3]._mantissa, [0] * 34)
        self.assertEqual(c.segment[3]._exponent,
                         [16] + [17, 17, 18] * 11)

    def test_NR_bug_j2c_dump(self):
        jfile = os.path.join(data_root, 'input/nonregression/bug.j2c')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 1800)
        self.assertEqual(c.segment[1].Ysiz, 1800)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (1800, 1800))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 4)
        self.assertEqual(c.segment[3]._mantissa, [0] * 34)
        self.assertEqual(c.segment[3]._exponent,
                         [16] + [17, 17, 18] * 11)

    def test_NR_kodak_2layers_lrcp_j2c_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/kodak_2layers_lrcp.j2c')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 2048)
        self.assertEqual(c.segment[1].Ysiz, 1556)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                         (2048, 1556))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (12, 12, 12))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(c.segment[2]._precinct_size,
                         [(128, 128)] + [(256, 256)] * 5)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [13, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
                          13, 13, 13])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "DCP-Werkstatt")

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_broken_jp2_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/broken.jp2')
        with self.assertWarns(UserWarning) as cw:
            # colr box has bad length.
            jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 152)
        self.assertEqual(jp2.box[2].box[0].width, 203)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # not allowed?
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)

        c = jp2.box[3].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'CME', 'COD', 'QCD', 'QCC', 'QCC']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 203)
        self.assertEqual(c.segment[1].Ysiz, 152)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (203, 152))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COM: comment
        # Registration
        self.assertEqual(c.segment[2].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[2].Ccme.decode('latin-1'),
                         "Creator: JasPer Version 1.701.0")

        # COD: Coding style default
        self.assertFalse(c.segment[3].Scod & 2)  # no sop
        self.assertFalse(c.segment[3].Scod & 4)  # no eph
        self.assertEqual(c.segment[3].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[3]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[3].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[3].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[3]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[3].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[3].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[4].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[4]._guardBits, 2)
        self.assertEqual(c.segment[4]._mantissa, [0] * 16)
        self.assertEqual(c.segment[4]._exponent,
                         [8] + [9, 9, 10] * 5)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].Cqcc, 1)
        self.assertEqual(c.segment[5]._guardBits, 2)
        # quantization type
        self.assertEqual(c.segment[5].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[5]._mantissa, [0] * 16)
        self.assertEqual(c.segment[5]._exponent,
                         [8] + [9, 9, 10] * 5)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].Cqcc, 2)
        self.assertEqual(c.segment[6]._guardBits, 2)
        # quantization type
        self.assertEqual(c.segment[6].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[6]._mantissa, [0] * 16)
        self.assertEqual(c.segment[6]._exponent,
                         [8] + [9, 9, 10] * 5)

    def test_NR_broken2_jp2_dump(self):
        # Invalid marker ID on codestream.
        jfile = os.path.join(data_root,
                             'input/nonregression/broken2.jp2')
        with self.assertRaises(IOError):
            jp2 = Jp2k(jfile)

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_broken3_jp2_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/broken3.jp2')
        with self.assertWarns(UserWarning) as cw:
            # colr box has bad length.
            jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 152)
        self.assertEqual(jp2.box[2].box[0].width, 203)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)

        c = jp2.box[3].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'CME', 'COD', 'QCD', 'QCC', 'QCC']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 203)
        self.assertEqual(c.segment[1].Ysiz, 152)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (203, 152))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COM: comment
        # Registration
        self.assertEqual(c.segment[2].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[2].Ccme.decode('latin-1'),
                         "Creator: JasPer Vers)on 1.701.0")

        # COD: Coding style default
        self.assertFalse(c.segment[3].Scod & 2)  # no sop
        self.assertFalse(c.segment[3].Scod & 4)  # no eph
        self.assertEqual(c.segment[3].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[3]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[3].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[3].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[3]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[3].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[3].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[4].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[4]._guardBits, 2)
        self.assertEqual(c.segment[4]._mantissa, [0] * 16)
        self.assertEqual(c.segment[4]._exponent,
                         [8] + [9, 9, 10] * 5)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].Cqcc, 1)
        self.assertEqual(c.segment[5]._guardBits, 2)
        # quantization type
        self.assertEqual(c.segment[5].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[5]._mantissa, [0] * 16)
        self.assertEqual(c.segment[5]._exponent,
                         [8] + [9, 9, 10] * 5)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].Cqcc, 2)
        self.assertEqual(c.segment[6]._guardBits, 2)
        # quantization type
        self.assertEqual(c.segment[6].Sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[6]._mantissa, [0] * 16)
        self.assertEqual(c.segment[6]._exponent,
                         [8] + [9, 9, 10] * 5)

    def test_NR_broken4_jp2_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/broken4.jp2')
        with self.assertRaises(IOError):
            jp2 = Jp2k(jfile)

    def test_NR_file409752(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/file409752.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 243)
        self.assertEqual(jp2.box[2].box[0].width, 720)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.YCC)

        c = jp2.box[3].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 720)
        self.assertEqual(c.segment[1].Ysiz, 243)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (720, 243))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1), (2, 1), (2, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 128))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa,
                         [1816, 1792, 1792, 1724, 1770, 1770, 1724, 1868,
                          1868, 1892, 3, 3, 69, 2002, 2002, 1889])
        self.assertEqual(c.segment[3]._exponent,
                         [13] * 4 + [12] * 3 + [11] * 3 + [9] * 6)

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_gdal_fuzzer_assert_in_opj_j2k_read_SQcd_SQcc_patch_jp2(self):
        lst = ['input', 'nonregression',
               'gdal_fuzzer_assert_in_opj_j2k_read_SQcd_SQcc.patch.jp2']
        jfile = os.path.join(data_root, '/'.join(lst))
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_gdal_fuzzer_check_comp_dx_dy_jp2_dump(self):
        lst = ['input', 'nonregression', 'gdal_fuzzer_check_comp_dx_dy.jp2']
        jfile = os.path.join(data_root, '/'.join(lst))
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_gdal_fuzzer_check_number_of_tiles(self):
        # Has an impossible tiling setup.
        lst = ['input', 'nonregression',
               'gdal_fuzzer_check_number_of_tiles.jp2']
        jfile = os.path.join(data_root, '/'.join(lst))
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_gdal_fuzzer_unchecked_numresolutions_dump(self):
        # Has an invalid number of resolutions.
        lst = ['input', 'nonregression',
               'gdal_fuzzer_unchecked_numresolutions.jp2']
        jfile = os.path.join(data_root, '/'.join(lst))
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

    def test_NR_issue104_jpxstream_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/issue104_jpxstream.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'pclr', 'cmap'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jpxb')
        self.assertEqual(jp2.box[1].compatibility_list[2], 'jpx ')

        # Reader requirements talk.
        self.assertTrue(glymur.core.RREQ_UNRESTRICTED_JPEG2000_PART_1
                        in jp2.box[2].standard_flag)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 203)
        self.assertEqual(jp2.box[3].box[0].width, 479)
        self.assertEqual(jp2.box[3].box[0].num_components, 1)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method, 1)  # enumerated
        self.assertEqual(jp2.box[3].box[1].precedence, 2)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # exact
        self.assertEqual(jp2.box[3].box[1].colorspace, glymur.core.SRGB)

        # Jp2 Header
        # Palette box.
        self.assertEqual(len(jp2.box[3].box[2].palette), 3)
        self.assertEqual(len(jp2.box[3].box[2].palette[0]), 256)
        self.assertEqual(len(jp2.box[3].box[2].palette[1]), 256)
        self.assertEqual(len(jp2.box[3].box[2].palette[2]), 256)

        # Jp2 Header
        # Component mapping box
        self.assertEqual(jp2.box[3].box[3].component_index, (0, 0, 0))
        self.assertEqual(jp2.box[3].box[3].mapping_type, (1, 1, 1))
        self.assertEqual(jp2.box[3].box[3].palette_index, (0, 1, 2))

        c = jp2.box[4].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 479)
        self.assertEqual(c.segment[1].Ysiz, 203)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (256, 203))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent, [8] + [9, 9, 10] * 5)

    def test_NR_issue188_beach_64bitsbox(self):
        lst = ['input', 'nonregression', 'issue188_beach_64bitsbox.jp2']
        jfile = os.path.join(data_root, '/'.join(lst))
        with warnings.catch_warnings():
            # There's a warning for an unknown box.  We explicitly test for
            # that down below.
            warnings.simplefilter("ignore")
            jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'XML ', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 200)
        self.assertEqual(jp2.box[2].box[0].width, 200)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)

        # Skip the 4th box, it is uknown.

        c = jp2.box[4].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 200)
        self.assertEqual(c.segment[1].Ysiz, 200)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (200, 200))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3]._guardBits, 1)

    def test_NR_issue206_image_000_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/issue206_image-000.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jpxb')
        self.assertEqual(jp2.box[1].compatibility_list[2], 'jpx ')

        # Reader requirements talk.
        self.assertTrue(glymur.core.RREQ_UNRESTRICTED_JPEG2000_PART_1
                        in jp2.box[2].standard_flag)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 326)
        self.assertEqual(jp2.box[3].box[0].width, 431)
        self.assertEqual(jp2.box[3].box[0].num_components, 3)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method, 1)  # ICC
        self.assertEqual(jp2.box[3].box[1].precedence, 2)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[3].box[1].colorspace, glymur.core.SRGB)

        c = jp2.box[4].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 431)
        self.assertEqual(c.segment[1].Ysiz, 326)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (256, 256))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent, [8] + [9, 9, 10] * 5)

    def test_NR_Marrin_jp2_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/Marrin.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'cdef', 'res '])

        ids = [box.id for box in jp2.box[2].box[3].box]
        self.assertEqual(ids, ['resd'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 135)
        self.assertEqual(jp2.box[2].box[0].width, 135)
        self.assertEqual(jp2.box[2].box[0].num_components, 2)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.GREYSCALE)

        # Jp2 Header
        # Channel Definition
        self.assertEqual(jp2.box[2].box[2].index, (0, 1))
        self.assertEqual(jp2.box[2].box[2].channel_type, (0, 1))   # opacity
        self.assertEqual(jp2.box[2].box[2].association, (0, 0))  # both main

        c = jp2.box[3].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 135)
        self.assertEqual(c.segment[1].Ysiz, 135)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (135, 135))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 2)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_9x7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa,
                         [1822, 1770, 1770, 1724, 1792, 1792, 1762, 1868, 1868,
                          1892, 3, 3, 69, 2002, 2002, 1889])
        self.assertEqual(c.segment[3]._exponent,
                         [14] * 4 + [13] * 3 + [12] * 3 + [10] * 6)

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].Rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].Ccme.decode('latin-1'),
                         "Kakadu-v5.2.1")

    def test_NR_mem_b2ace68c_1381_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/mem-b2ace68c-1381.jp2')
        with warnings.catch_warnings():
            # This file has a bad pclr box, we test for this elsewhere.
            warnings.simplefilter("ignore")
            jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'pclr', 'cmap'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jpxb')
        self.assertEqual(jp2.box[1].compatibility_list[2], 'jpx ')

        # Reader requirements talk.
        self.assertTrue(glymur.core.RREQ_CMYK_ENUMERATED_COLORSPACE
                        in jp2.box[2].standard_flag)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 865)
        self.assertEqual(jp2.box[3].box[0].width, 649)
        self.assertEqual(jp2.box[3].box[0].num_components, 1)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 1)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method, 1)  # enumerated
        self.assertEqual(jp2.box[3].box[1].precedence, 2)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[3].box[1].colorspace, glymur.core.CMYK)

        # Jp2 Header
        # Palette box.
        self.assertEqual(len(jp2.box[3].box[2].palette), 4)
        self.assertEqual(len(jp2.box[3].box[2].palette[0]), 1)
        self.assertEqual(len(jp2.box[3].box[2].palette[1]), 1)
        self.assertEqual(len(jp2.box[3].box[2].palette[2]), 1)
        self.assertEqual(len(jp2.box[3].box[2].palette[3]), 1)

        # Jp2 Header
        # Component mapping box
        self.assertEqual(jp2.box[3].box[3].component_index, (0, 1, 2))
        self.assertEqual(jp2.box[3].box[3].mapping_type, (1, 1, 0))
        self.assertEqual(jp2.box[3].box[3].palette_index, (0, 0, 1))

        c = jp2.box[4].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 649)
        self.assertEqual(c.segment[1].Ysiz, 865)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (256, 256))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (1,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 3)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent, [1] + [2, 2, 3] * 5)

    def test_NR_mem_b2b86b74_2753_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/mem-b2b86b74-2753.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'pclr', 'cmap'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jpxb')
        self.assertEqual(jp2.box[1].compatibility_list[2], 'jpx ')

        # Reader requirements talk.
        self.assertTrue(glymur.core.RREQ_UNRESTRICTED_JPEG2000_PART_1
                        in jp2.box[2].standard_flag)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 46)
        self.assertEqual(jp2.box[3].box[0].width, 124)
        self.assertEqual(jp2.box[3].box[0].num_components, 1)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 4)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method, 1)  # enumerated
        self.assertEqual(jp2.box[3].box[1].precedence, 2)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[3].box[1].colorspace, glymur.core.SRGB)

        # Jp2 Header
        # Palette box.
        # 3 columns with 16 entries.
        self.assertEqual(len(jp2.box[3].box[2].palette), 3)
        self.assertEqual(len(jp2.box[3].box[2].palette[0]), 16)
        self.assertEqual(len(jp2.box[3].box[2].palette[1]), 16)
        self.assertEqual(len(jp2.box[3].box[2].palette[2]), 16)

        # Jp2 Header
        # Component mapping box
        self.assertEqual(jp2.box[3].box[3].component_index, (0, 0, 0))
        self.assertEqual(jp2.box[3].box[3].mapping_type, (1, 1, 1))
        self.assertEqual(jp2.box[3].box[3].palette_index, (0, 1, 2))

        c = jp2.box[4].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 124)
        self.assertEqual(c.segment[1].Ysiz, 46)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (124, 46))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (4,))
        # signed
        self.assertEqual(c.segment[1]._signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent, [4] + [5, 5, 6] * 5)

    def test_NR_merged_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/merged.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 576)
        self.assertEqual(jp2.box[2].box[0].width, 766)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 1)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.YCC)

        c = jp2.box[3].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'POD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 766)
        self.assertEqual(c.segment[1].Ysiz, 576)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (766, 576))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1), (2, 1), (2, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 128))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 1)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent, [8] + [9, 9, 10] * 5)

        # POD: progression order change
        self.assertEqual(c.segment[4].RSpod, (0, 0))
        self.assertEqual(c.segment[4].CSpod, (0, 1))
        self.assertEqual(c.segment[4].LYEpod, (1, 1))
        self.assertEqual(c.segment[4].REpod, (6, 6))
        self.assertEqual(c.segment[4].CEpod, (1, 3))

        podvals = (glymur.core.LRCP, glymur.core.LRCP)
        self.assertEqual(c.segment[4].Ppod, podvals)

    def test_NR_orb_blue10_lin_jp2_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/orb-blue10-lin-jp2.jp2')
        with warnings.catch_warnings():
            # This file has an invalid ICC profile
            warnings.simplefilter("ignore")
            jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 117)
        self.assertEqual(jp2.box[2].box[0].width, 117)
        self.assertEqual(jp2.box[2].box[0].num_components, 4)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 2)  # res icc
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertIsNone(jp2.box[2].box[1].icc_profile)
        self.assertIsNone(jp2.box[2].box[1].colorspace)

        c = jp2.box[3].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 117)
        self.assertEqual(c.segment[1].Ysiz, 117)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (117, 117))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 4)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_orb_blue10_win_jp2_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/orb-blue10-win-jp2.jp2')
        with warnings.catch_warnings():
            # This file has an invalid ICC profile
            warnings.simplefilter("ignore")
            jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 117)
        self.assertEqual(jp2.box[2].box[0].width, 117)
        self.assertEqual(jp2.box[2].box[0].num_components, 4)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method, 2)  # restricted icc
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertIsNone(jp2.box[2].box[1].icc_profile)
        self.assertIsNone(jp2.box[2].box[1].colorspace)

        c = jp2.box[3].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 117)
        self.assertEqual(c.segment[1].Ysiz, 117)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (117, 117))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 4)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_text_GBR_dump(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/text_GBR.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.id for box in jp2.box]
        lst = ['jP  ', 'ftyp', 'rreq', 'jp2h',
               'uuid', 'uuid', 'uuid', 'uuid', 'jp2c']
        self.assertEqual(ids, lst)

        ids = [box.id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'res '])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Reader requirements.
        # Compositing layer uses any icc profile
        self.assertTrue(44 in jp2.box[2].standard_flag)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 400)
        self.assertEqual(jp2.box[3].box[0].width, 400)
        self.assertEqual(jp2.box[3].box[0].num_components, 3)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method, 3)  # any icc
        self.assertEqual(jp2.box[3].box[1].precedence, 2)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[3].box[1].icc_profile['Size'], 1328)
        self.assertIsNone(jp2.box[3].box[1].colorspace)

        # UUID boxes.  All mentioned in the RREQ box.
        self.assertEqual(jp2.box[2].vendor_feature[0], jp2.box[4].uuid)
        self.assertEqual(jp2.box[2].vendor_feature[1], jp2.box[5].uuid)
        self.assertEqual(jp2.box[2].vendor_feature[2], jp2.box[6].uuid)
        self.assertEqual(jp2.box[2].vendor_feature[3], jp2.box[7].uuid)

        c = jp2.box[8].main_header

        ids = [x.id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].Rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].Xsiz, 400)
        self.assertEqual(c.segment[1].Ysiz, 400)
        # Reference grid offset
        self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1]._signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].Scod & 2)  # no sop
        self.assertFalse(c.segment[2].Scod & 4)  # no eph
        self.assertEqual(c.segment[2].SPcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2]._layers, 6)  # layers = 6
        self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
        self.assertEqual(c.segment[2].SPcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2]._code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].SPcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].SPcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].SPcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].SPcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
        self.assertEqual(c.segment[2].SPcod[8],
                         glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].SPcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].Sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3]._guardBits, 2)
        self.assertEqual(c.segment[3]._mantissa, [0] * 16)
        self.assertEqual(c.segment[3]._exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

if __name__ == "__main__":
    unittest.main()
