"""
Tests for the slicing protocol.
"""
# Standard library imports ...
import unittest
import warnings

# Third party library imports ...
import numpy as np

# Local imports
import glymur
from glymur import Jp2k
from glymur.jp2box import InvalidJp2kError
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):
    """
    Test slice protocol
    """
    @classmethod
    def setUpClass(self):

        self.jp2 = Jp2k(glymur.data.nemo())
        self.jp2_data = self.jp2[:]

        self.j2k = Jp2k(glymur.data.goodstuff())
        self.j2k_data = self.j2k[:]
        self.j2k_r1_data = self.j2k[::2, ::2]
        self.j2k_r2_data = self.j2k[::4, ::4]
        self.j2k_r5_data = self.j2k[::32, ::32]

    def test_NR_DEC_p1_04_j2k_43_decode(self):
        actual = self.j2k[:800, :480]
        expected = self.j2k_data
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_45_decode(self):
        """
        Get bottom right
        """
        actual = self.j2k[672:800, 352:480]
        expected = self.j2k_data[672:800, 352:480]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_46_decode(self):
        actual = self.j2k[500:800, 100:300]
        expected = self.j2k_data[500:800, 100:300]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_47_decode(self):
        actual = self.j2k[520:600, 260:360]
        expected = self.j2k_data[520:600, 260:360]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_48_decode(self):
        actual = self.j2k[520:660, 260:360]
        expected = self.j2k_data[520:660, 260:360]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_49_decode(self):
        actual = self.j2k[520:600, 360:400]
        expected = self.j2k_data[520:600, 360:400]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_50_decode(self):
        actual = self.j2k[:800:4, :480:4]
        expected = self.j2k_r2_data
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_51_decode(self):
        """
        NR_DEC_p1_04_j2k_51_decode

        Original extents were

        actual = self.j2k[640:768:4, 512:640:4]
        expected = self.j2k_r2_data[160:192, 128:160]

        Just needed to shift the columns to the left to make it work with
        our own image.
        """
        actual = self.j2k[640:768:4, 256:384:4]
        expected = self.j2k_r2_data[160:192, 64:96]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_53_decode(self):
        actual = self.j2k[500:800:4, 100:300:4]
        expected = self.j2k_r2_data[125:200, 25:75]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_54_decode(self):
        actual = self.j2k[520:600:4, 260:360:4]
        expected = self.j2k_r2_data[130:150, 65:90]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_55_decode(self):
        actual = self.j2k[520:660:4, 260:360:4]
        expected = self.j2k_r2_data[130:165, 65:90]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_56_decode(self):
        actual = self.j2k[520:600:4, 360:400:4]
        expected = self.j2k_r2_data[130:150, 90:100]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_06_j2k_70_decode(self):
        actual = self.j2k[9:12:2, 9:12:2]
        self.assertEqual(actual.shape, (1, 1, 3))

    def test_NR_DEC_p1_06_j2k_71_decode(self):
        actual = self.j2k[10:12:2, 4:10:2]
        self.assertEqual(actual.shape, (1, 3, 3))

    def test_NR_DEC_p1_06_j2k_72_decode(self):
        ssdata = self.j2k[3:9:2, 3:9:2]
        self.assertEqual(ssdata.shape, (3, 3, 3))

    def test_NR_DEC_p1_06_j2k_73_decode(self):
        ssdata = self.j2k[4:7:2, 4:7:2]
        self.assertEqual(ssdata.shape, (2, 2, 3))

    def test_NR_DEC_p1_06_j2k_74_decode(self):
        ssdata = self.j2k[4:5:2, 4:5:2]
        self.assertEqual(ssdata.shape, (1, 1, 3))

    def test_NR_DEC_p1_06_j2k_75_decode(self):
        """
        SCENARIO:  Try to read an image area with an impossible stride.

        EXPECTED RESULT:  An error is raised.
        """
        if glymur.version.openjpeg_version >= '2.4.0':
            # The library catches this on its own.
            expected_error = glymur.lib.openjp2.OpenJPEGLibraryError
        else:
            # Image size would be 0 x 0.  We have to manually detect this.
            expected_error = InvalidJp2kError
        with self.assertRaises(expected_error):
            with warnings.catch_warnings():
                # Only openjpeg 2.4.0 issues warnings
                warnings.simplefilter('ignore')
                self.j2k[9:12:4, 9:12:4]

    def test_NR_DEC_p0_04_j2k_85_decode(self):
        actual = self.j2k[:256, :256]
        expected = self.j2k_data[:256, :256]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_86_decode(self):
        actual = self.j2k[:128, 128:256]
        expected = self.j2k_data[:128, 128:256]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_87_decode(self):
        actual = self.j2k[10:200, 50:120]
        expected = self.j2k_data[10:200, 50:120]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_88_decode(self):
        actual = self.j2k[150:210, 10:190]
        expected = self.j2k_data[150:210, 10:190]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_89_decode(self):
        actual = self.j2k[80:150, 100:200]
        expected = self.j2k_data[80:150, 100:200]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_90_decode(self):
        actual = self.j2k[20:50, 150:200]
        expected = self.j2k_data[20:50, 150:200]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_91_decode(self):
        actual = self.j2k[:256:4, :256:4]
        expected = self.j2k_r2_data[0:64, 0:64]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_92_decode(self):
        actual = self.j2k[:128:4, 128:256:4]
        expected = self.j2k_r2_data[:32, 32:64]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_93_decode(self):
        actual = self.j2k[10:200:4, 50:120:4]
        expected = self.j2k_r2_data[3:50, 13:30]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_94_decode(self):
        actual = self.j2k[150:210:4, 10:190:4]
        expected = self.j2k_r2_data[38:53, 3:48]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_95_decode(self):
        actual = self.j2k[80:150:4, 100:200:4]
        expected = self.j2k_r2_data[20:38, 25:50]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_96_decode(self):
        actual = self.j2k[20:50:4, 150:200:4]
        expected = self.j2k_r2_data[5:13, 38:50]
        np.testing.assert_array_equal(actual, expected)

    def test_resolution_strides_cannot_differ(self):
        with self.assertRaises(ValueError):
            # Strides in x/y directions cannot differ.
            self.j2k[::2, ::3]

    def test_resolution_strides_must_be_powers_of_two(self):
        with self.assertRaises(ValueError):
            self.j2k[::3, ::3]

    def test_integer_index_in_3d(self):

        for j in [0, 1, 2]:
            band = self.j2k[:, :, j]
            np.testing.assert_array_equal(self.j2k_data[:, :, j], band)

    def test_slice_in_third_dimension(self):
        actual = self.j2k[:, :, 1:3]
        expected = self.j2k_data[:, :, 1:3]
        np.testing.assert_array_equal(actual, expected)

    def test_reduce_resolution_and_slice_in_third_dimension(self):
        actual = self.j2k[::2, ::2, 1:3]
        expected = self.j2k_r1_data[:, :, 1:3]
        np.testing.assert_array_equal(actual, expected)

    def test_retrieve_single_row(self):
        actual = self.jp2[0]
        expected = self.jp2_data[0]
        np.testing.assert_array_equal(actual, expected)

    def test_retrieve_single_pixel(self):
        actual = self.jp2[0, 0]
        expected = self.jp2_data[0, 0]
        np.testing.assert_array_equal(actual, expected)

    def test_retrieve_single_component(self):
        actual = self.jp2[20, 20, 2]
        expected = self.jp2_data[20, 20, 2]
        np.testing.assert_array_equal(actual, expected)

    def test_ellipsis_full_read(self):
        actual = self.j2k[...]
        expected = self.j2k_data
        np.testing.assert_array_equal(actual, expected)

    def test_ellipsis_band_select(self):
        actual = self.j2k[..., 0]
        expected = self.j2k_data[..., 0]
        np.testing.assert_array_equal(actual, expected)

    def test_ellipsis_row_select(self):
        actual = self.j2k[0, ...]
        expected = self.j2k_data[0, ...]
        np.testing.assert_array_equal(actual, expected)

    def test_two_ellipsis_band_select(self):
        actual = self.j2k[..., ..., 1]
        expected = self.j2k_data[:, :, 1]
        np.testing.assert_array_equal(actual, expected)

    def test_two_ellipsis_row_select(self):
        actual = self.j2k[1, ..., ...]
        expected = self.j2k_data[1, :, :]
        np.testing.assert_array_equal(actual, expected)

    def test_two_ellipsis_and_full_slice(self):
        actual = self.j2k[..., ..., :]
        expected = self.j2k_data[:]
        np.testing.assert_array_equal(actual, expected)

    def test_single_slice(self):
        rows = slice(3, 8)
        actual = self.j2k[rows]
        expected = self.j2k_data[3:8, :, :]
        np.testing.assert_array_equal(actual, expected)

    def test_region_rlevel5(self):
        """
        maximim rlevel

        There seems to be a difference between version of openjpeg, as
        openjp2 produces an image of size (16, 13, 3) and openjpeg produced
        (17, 12, 3).
        """
        actual = self.j2k[5:533:32, 27:423:32]
        expected = self.j2k_r5_data[1:17, 1:14]
        np.testing.assert_array_equal(actual, expected)

    def test_write_ellipsis(self):
        """
        SCENARIO:  write image by specifying ellipsis in slice

        EXPECTED RESULT:  image is validated
        """
        expected = self.j2k_data

        j = Jp2k(self.temp_j2k_filename, shape=expected.shape)
        j[...] = expected
        actual = j[:]

        np.testing.assert_array_equal(actual, expected)

    def test_basic_write(self):
        """
        SCENARIO:  write image by specifying image data with slice protocol

        EXPECTED RESULT:  image is validated
        """
        expected = self.j2k_data

        j = Jp2k(self.temp_j2k_filename, shape=expected.shape)
        j[:] = expected
        actual = j[:]

        np.testing.assert_array_equal(actual, expected)

    def test_cannot_write_with_non_default_single_slice(self):
        """
        SCENARIO:  Write image using single non-default slices.  Only ':' is
        currently valid as a single slice argument.

        EXPECTED RESULT:  ValueError
        """
        j = Jp2k(self.temp_j2k_filename, shape=self.j2k_data.shape)
        with self.assertRaises(ValueError):
            j[slice(None, 0)] = self.j2k_data
        with self.assertRaises(ValueError):
            j[slice(0, None)] = self.j2k_data
        with self.assertRaises(ValueError):
            j[slice(0, 0, None)] = self.j2k_data
        with self.assertRaises(ValueError):
            j[slice(0, 640)] = self.j2k_data

    def test_cannot_write_a_row(self):
        """
        SCENARIO:  Write image row by specifying slicing.  Only ':' is
        currently valid as a single slice argument.

        EXPECTED RESULT:  ValueError
        """
        j = Jp2k(self.temp_j2k_filename, shape=self.j2k_data.shape)
        with self.assertRaises(ValueError):
            j[5] = self.j2k_data

    def test_cannot_write_a_pixel(self):
        """
        SCENARIO:  Write pixel by specifying slicing.  Only ':' is
        currently valid as a single slice argument.

        EXPECTED RESULT:  ValueError
        """
        j = Jp2k(self.temp_j2k_filename, shape=self.j2k_data.shape)
        with self.assertRaises(ValueError):
            j[25, 35] = self.j2k_data[25, 35]

    def test_cannot_write_a_column(self):
        """
        SCENARIO:  Write column by specifying slicing.  Only ':' is
        currently valid as a single slice argument.

        EXPECTED RESULT:  ValueError
        """
        j = Jp2k(self.temp_j2k_filename, shape=self.j2k_data.shape)
        with self.assertRaises(ValueError):
            j[:, 25, :] = self.j2k_data[:, :25, :]

    def test_cannot_write_a_band(self):
        """
        SCENARIO:  Write band by specifying slicing.  Only ':' is
        currently valid as a single slice argument.

        EXPECTED RESULT:  ValueError
        """
        j = Jp2k(self.temp_j2k_filename, shape=self.j2k_data.shape)
        with self.assertRaises(ValueError):
            j[:, :, 0] = self.j2k_data[:, :, 0]

    def test_cannot_write_a_subarray(self):
        """
        SCENARIO:  Write area by specifying slicing.  Only ':' is
        currently valid as a single slice argument.

        EXPECTED RESULT:  ValueError
        """
        j = Jp2k(self.temp_j2k_filename, shape=self.j2k_data.shape)
        with self.assertRaises(ValueError):
            j[:25, :45, :] = self.j2k_data[:25, :25, :]
