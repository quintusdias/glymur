import os
import sys
import unittest
import warnings

import numpy as np

import glymur
from glymur import Jp2k
from glymur.lib import openjpeg as opj

from .fixtures import *

try:
    data_root = os.environ['OPJ_DATA_ROOT']
except KeyError:
    data_root = None
except:
    raise


@unittest.skipIf(glymur.lib.openjpeg._OPENJPEG is None,
                 "Missing openjpeg library.")
class TestJp2k(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Monkey patch the package so as to use OPENJPEG instead of OPENJP2
        cls.openjp2 = glymur.lib.openjp2._OPENJP2
        glymur.lib.openjp2._OPENJP2 = None

    @classmethod
    def tearDownClass(cls):
        # Restore OPENJP2
        glymur.lib.openjp2._OPENJP2 = cls.openjp2

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    def test_bands(self):
        # Reading individual bands is an advanced maneuver.
        jp2k = Jp2k(self.j2kfile)
        with self.assertRaises(NotImplementedError) as ce:
            jpdata = jp2k.read_bands()

    def test_area(self):
        # Area option not allowed for 1.5.1.
        j2k = Jp2k(self.j2kfile)
        with self.assertRaises(TypeError) as ce:
            d = j2k.read(area=(0, 0, 100, 100))

    def test_tile(self):
        # tile option not allowed for 1.5.1.
        j2k = Jp2k(self.j2kfile)
        with self.assertRaises(TypeError) as ce:
            d = j2k.read(tile=0)

    def test_layer(self):
        # layer option not allowed for 1.5.1.
        j2k = Jp2k(self.j2kfile)
        with self.assertRaises(TypeError) as ce:
            d = j2k.read(layer=1)

    def test_basic_jp2(self):
        # This test is only useful when openjp2 is not available
        # and OPJ_DATA_ROOT is not set.  We need at least one
        # working JP2 test.
        j2k = Jp2k(self.jp2file)
        d = j2k.read(reduce=1)

    def test_basic_j2k(self):
        # This test is only useful when openjp2 is not available
        # and OPJ_DATA_ROOT is not set.  We need at least one
        # working J2K test.
        j2k = Jp2k(self.j2kfile)
        d = j2k.read()


@unittest.skipIf(glymur.lib.openjpeg._OPENJPEG is None,
                 "Missing openjpeg library.")
@unittest.skipIf(data_root is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Monkey patch the package so as to use OPENJPEG instead of OPENJP2
        cls.openjp2 = glymur.lib.openjp2._OPENJP2
        glymur.lib.openjp2._OPENJP2 = None

    @classmethod
    def tearDownClass(cls):
        # Restore OPENJP2
        glymur.lib.openjp2._OPENJP2 = cls.openjp2

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

    def test_ETS_C0P0_p0_09_j2k(self):
        jfile = os.path.join(data_root, 'input/conformance/p0_09.j2k')
        jp2k = Jp2k(jfile)
        jpdata = jp2k.read(reduce=2)

        pgxfile = os.path.join(data_root,
                               'baseline/conformance/c0p0_09.pgx')
        pgxdata = read_pgx(pgxfile)

        self.assertTrue(peak_tolerance(jpdata, pgxdata) < 4)
        self.assertTrue(mse(jpdata, pgxdata) < 1.47)

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

    @unittest.skip("reading separate bands not allowed")
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

    @unittest.skip("reading separate bands not allowed")
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
        with warnings.catch_warnings():
            # This file has an invalid ICC profile
            warnings.simplefilter("ignore")
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

    @unittest.skip("reading separate bands not allowed")
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

    @unittest.skip("reading separate bands not allowed")
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

    @unittest.skip("reading separate bands not allowed")
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

    @unittest.skip("Should have worked, must be investigated.")
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

    @unittest.skip("Should have worked, must be investigated.")
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
        with warnings.catch_warnings():
            # This file has an invalid ICC profile
            warnings.simplefilter("ignore")
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

    @unittest.skip("reading separate bands not allowed")
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

    def test_NR_DEC_issue104_jpxstream_jp2_33_decode(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/issue104_jpxstream.jp2')
        data = Jp2k(jfile).read()
        self.assertTrue(True)

    @unittest.skip("Should have worked, must be investigated.")
    def test_NR_DEC_file_409752_jp2_40_decode(self):
        jfile = os.path.join(data_root, 'input/nonregression/file409752.jp2')
        j = Jp2k(jfile)
        data = j.read()
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
