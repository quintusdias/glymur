"""
Tests for general glymur functionality.
"""
# Standard library imports ...
import os
import re
import tempfile
import unittest

# Third party library imports ...
import numpy as np

# Local imports
import glymur
from glymur import Jp2k
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG
from . import fixtures


class SliceProtocolBase(unittest.TestCase):
    """
    Test slice protocol, i.e. when using [ ] to read image data.
    """
    @classmethod
    def setUpClass(self):

        self.jp2 = Jp2k(glymur.data.nemo())
        self.jp2_data = self.jp2[:]
        self.jp2_data_r1 = self.jp2[::2, ::2]

        self.j2k = Jp2k(glymur.data.goodstuff())
        self.j2k_data = self.j2k[:]

        self.j2k_data_r1 = self.j2k[::2, ::2]
        self.j2k_data_r5 = self.j2k[::32, ::32]


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(re.match("1.5|2", glymur.version.openjpeg_version) is None,
                 "Must have openjpeg 1.5 or higher to run")
@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
class TestSliceProtocolBaseWrite(SliceProtocolBase):

    def test_write_ellipsis(self):
        expected = self.j2k_data

        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, shape=expected.shape)
            j[...] = expected
            actual = j[:]

        np.testing.assert_array_equal(actual, expected)

    def test_basic_write(self):
        expected = self.j2k_data

        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.j2k_data)
            actual = j[:]

        np.testing.assert_array_equal(actual, expected)

    def test_cannot_write_with_non_default_single_slice(self):
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, shape=self.j2k_data.shape)
            with self.assertRaises(TypeError):
                j[slice(None, 0)] = self.j2k_data
            with self.assertRaises(TypeError):
                j[slice(0, None)] = self.j2k_data
            with self.assertRaises(TypeError):
                j[slice(0, 0, None)] = self.j2k_data
            with self.assertRaises(TypeError):
                j[slice(0, 640)] = self.j2k_data

    def test_cannot_write_a_row(self):
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, shape=self.j2k_data.shape)
            with self.assertRaises(TypeError):
                j[5] = self.j2k_data

    def test_cannot_write_a_pixel(self):
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, shape=self.j2k_data.shape)
            with self.assertRaises(TypeError):
                j[25, 35] = self.j2k_data[25, 35]

    def test_cannot_write_a_column(self):
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, shape=self.j2k_data.shape)
            with self.assertRaises(TypeError):
                j[:, 25, :] = self.j2k_data[:, :25, :]

    def test_cannot_write_a_band(self):
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, shape=self.j2k_data.shape)
            with self.assertRaises(TypeError):
                j[:, :, 0] = self.j2k_data[:, :, 0]

    def test_cannot_write_a_subarray(self):
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, shape=self.j2k_data.shape)
            with self.assertRaises(TypeError):
                j[:25, :45, :] = self.j2k_data[:25, :25, :]


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSliceProtocolRead(SliceProtocolBase):

    def test_resolution_strides_cannot_differ(self):
        with self.assertRaises(IndexError):
            # Strides in x/y directions cannot differ.
            self.j2k[::2, ::3]

    def test_resolution_strides_must_be_powers_of_two(self):
        with self.assertRaises(IndexError):
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
        expected = self.j2k_data_r1[:, :, 1:3]
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

    @unittest.skipIf(re.match("0|1", glymur.version.openjpeg_version),
                     "Must have openjpeg 2 or higher to run")
    def test_region_rlevel5(self):
        """
        maximim rlevel

        There seems to be a difference between version of openjpeg, as
        openjp2 produces an image of size (16, 13, 3) and openjpeg produced
        (17, 12, 3).
        """
        actual = self.j2k[5:533:32, 27:423:32]
        expected = self.j2k_data_r5[1:17, 1:14]
        np.testing.assert_array_equal(actual, expected)
