"""
Tests for general glymur functionality.
"""
import doctest
import os
import re
import struct
import sys
import tempfile
import unittest
import uuid
import warnings
from xml.etree import cElementTree as ET

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch

import numpy as np
import pkg_resources

import glymur
from glymur import Jp2k
from glymur.version import openjpeg_version

from .fixtures import HAS_PYTHON_XMP_TOOLKIT
from .fixtures import WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG

if HAS_PYTHON_XMP_TOOLKIT:
    import libxmp
    from libxmp import XMPMeta

from .fixtures import OPJ_DATA_ROOT, opj_data_file
from . import fixtures


def docTearDown(doctest_obj):
    glymur.set_parseoptions(full_codestream=False)


# Doc tests should be run as well.
def load_tests(loader, tests, ignore):
    """Should run doc tests as well"""
    if os.name == "nt":
        # Can't do it on windows, temporary file issue.
        return tests
    if glymur.lib.openjp2.OPENJP2 is not None:
        tests.addTests(doctest.DocTestSuite('glymur.jp2k',
                                            tearDown=docTearDown))
    return tests


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


class TestJp2k(unittest.TestCase):
    """These tests should be run by just about all configuration."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()
        self.jpxfile = glymur.data.jpxfile()

    def tearDown(self):
        pass

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    @unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
    def test_warn_if_using_read_method(self):
        """Should warn if deprecated read method is called"""
        with self.assertWarns(DeprecationWarning):
            Jp2k(self.jp2file).read()

    def test_shape_jp2(self):
        """verify shape attribute for JP2 file
        """
        jp2 = Jp2k(self.jp2file)
        self.assertEqual(jp2.shape, (1456, 2592, 3))

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_shape_greyscale_jp2(self):
        """verify shape attribute for greyscale JP2 file
        """
        with warnings.catch_warnings():
            # Suppress a warning due to bad compatibility list entry.
            warnings.simplefilter("ignore")
            jfile = opj_data_file('input/conformance/file4.jp2')
            jp2 = Jp2k(jfile)
        self.assertEqual(jp2.shape, (512, 768))

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_shape_single_channel_j2k(self):
        """verify shape attribute for single channel J2K file
        """
        jfile = opj_data_file('input/conformance/p0_01.j2k')
        jp2 = Jp2k(jfile)
        self.assertEqual(jp2.shape, (128, 128))

    def test_shape_j2k(self):
        """verify shape attribute for J2K file
        """
        j2k = Jp2k(self.j2kfile)
        self.assertEqual(j2k.shape, (800, 480, 3))

    def test_shape_jpx_jp2(self):
        """verify shape attribute for JPX file with JP2 compatibility
        """
        jpx = Jp2k(self.jpxfile)
        self.assertEqual(jpx.shape, (1024, 1024, 3))

    @unittest.skipIf(re.match("0|1.[0-4]", glymur.version.openjpeg_version),
                     "Must have openjpeg 1.5 or higher to run")
    @unittest.skipIf(os.name == "nt", "Unexplained failure on windows")
    def test_irreversible(self):
        """Irreversible"""
        j = Jp2k(self.jp2file)
        expdata = j[:]
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j2 = Jp2k(tfile.name, data=expdata, irreversible=True)

            codestream = j2.get_codestream()
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

            actdata = j2[:]
            self.assertTrue(fixtures.mse(actdata[0], expdata[0]) < 0.38)

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    @unittest.skipIf(re.match('1.[0-4]', openjpeg_version) is not None,
                     "Not supported on OpenJPEG {0}".format(openjpeg_version))
    @unittest.skipIf(re.match('1.5.(1|2)', openjpeg_version) is not None,
                     "Mysteriously fails in 1.5.1 and 1.5.2")
    def test_no_cxform_pclr_jpx(self):
        """Indices for pclr jpxfile if no color transform"""
        j = Jp2k(self.jpxfile)
        rgb = j[:]
        j.ignore_pclr_cmap_cdef = True
        idx = j[:]
        nr, nc = 1024, 1024
        self.assertEqual(rgb.shape, (nr, nc, 3))
        self.assertEqual(idx.shape, (nr, nc))

        # Should be able to manually reconstruct the RGB image from the palette
        # and indices.
        palette = j.box[2].box[2].palette
        rgb_from_idx = np.zeros(rgb.shape, dtype=np.uint8)
        for r in np.arange(nr):
            for c in np.arange(nc):
                rgb_from_idx[r, c] = palette[idx[r, c]]
        np.testing.assert_array_equal(rgb, rgb_from_idx)

    @unittest.skipIf(os.name == "nt", "Unexplained failure on windows")
    def test_repr(self):
        """Verify that results of __repr__ are eval-able."""
        j = Jp2k(self.j2kfile)
        newjp2 = eval(repr(j))

        self.assertEqual(newjp2.filename, self.j2kfile)
        self.assertEqual(len(newjp2.box), 0)

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    @unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
    def test_rlevel_max_backwards_compatibility(self):
        """
        Verify that rlevel=-1 gets us the lowest resolution image

        This is an old option only available via the read method, not via
        array-style slicing.
        """
        j = Jp2k(self.j2kfile)
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings():
                # Suppress a warning due to deprecated syntax
                # Not as easy to verify the warning under python2.
                warnings.simplefilter("ignore")
                thumbnail1 = j.read(rlevel=-1)
        else:
            with self.assertWarns(DeprecationWarning):
                thumbnail1 = j.read(rlevel=-1)
        thumbnail2 = j[::32, ::32]
        np.testing.assert_array_equal(thumbnail1, thumbnail2)
        self.assertEqual(thumbnail1.shape, (25, 15, 3))

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_rlevel_too_high(self):
        """Should error out appropriately if reduce level too high"""
        j = Jp2k(self.jp2file)
        with self.assertRaises(IOError):
            j[::64, ::64]

    def test_not_jpeg2000(self):
        """Should error out appropriately if not given a JPEG 2000 file."""
        filename = pkg_resources.resource_filename(glymur.__name__, "jp2k.py")
        with self.assertRaises(IOError):
            Jp2k(filename)

    def test_file_not_present(self):
        """Should error out if reading from a file that does not exist"""
        # Verify that we error out appropriately if not given an existing file
        # at all.
        with self.assertRaises(OSError):
            filename = 'this file does not actually exist on the file system.'
            Jp2k(filename)

    def test_jp2_boxes(self):
        """Verify the boxes of a JP2 file.  Basic jp2 test."""
        jp2k = Jp2k(self.jp2file)

        # top-level boxes
        self.assertEqual(len(jp2k.box), 5)

        self.assertEqual(jp2k.box[0].box_id, 'jP  ')
        self.assertEqual(jp2k.box[0].offset, 0)
        self.assertEqual(jp2k.box[0].length, 12)
        self.assertEqual(jp2k.box[0].longname, 'JPEG 2000 Signature')

        self.assertEqual(jp2k.box[1].box_id, 'ftyp')
        self.assertEqual(jp2k.box[1].offset, 12)
        self.assertEqual(jp2k.box[1].length, 20)
        self.assertEqual(jp2k.box[1].longname, 'File Type')

        self.assertEqual(jp2k.box[2].box_id, 'jp2h')
        self.assertEqual(jp2k.box[2].offset, 32)
        self.assertEqual(jp2k.box[2].length, 45)
        self.assertEqual(jp2k.box[2].longname, 'JP2 Header')

        self.assertEqual(jp2k.box[3].box_id, 'uuid')
        self.assertEqual(jp2k.box[3].offset, 77)
        self.assertEqual(jp2k.box[3].length, 3146)

        self.assertEqual(jp2k.box[4].box_id, 'jp2c')
        self.assertEqual(jp2k.box[4].offset, 3223)
        self.assertEqual(jp2k.box[4].length, 1132296)

        # jp2h super box
        self.assertEqual(len(jp2k.box[2].box), 2)

        self.assertEqual(jp2k.box[2].box[0].box_id, 'ihdr')
        self.assertEqual(jp2k.box[2].box[0].offset, 40)
        self.assertEqual(jp2k.box[2].box[0].length, 22)
        self.assertEqual(jp2k.box[2].box[0].longname, 'Image Header')
        self.assertEqual(jp2k.box[2].box[0].height, 1456)
        self.assertEqual(jp2k.box[2].box[0].width, 2592)
        self.assertEqual(jp2k.box[2].box[0].num_components, 3)
        self.assertEqual(jp2k.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2k.box[2].box[0].signed, False)
        self.assertEqual(jp2k.box[2].box[0].compression, 7)
        self.assertEqual(jp2k.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2k.box[2].box[0].ip_provided, False)

        self.assertEqual(jp2k.box[2].box[1].box_id, 'colr')
        self.assertEqual(jp2k.box[2].box[1].offset, 62)
        self.assertEqual(jp2k.box[2].box[1].length, 15)
        self.assertEqual(jp2k.box[2].box[1].longname, 'Colour Specification')
        self.assertEqual(jp2k.box[2].box[1].precedence, 0)
        self.assertEqual(jp2k.box[2].box[1].approximation, 0)
        self.assertEqual(jp2k.box[2].box[1].colorspace, glymur.core.SRGB)
        self.assertIsNone(jp2k.box[2].box[1].icc_profile)

    def test_j2k_box(self):
        """A J2K/J2C file must not have any boxes."""
        # Verify that a J2K file has no boxes.
        jp2k = Jp2k(self.j2kfile)
        self.assertEqual(len(jp2k.box), 0)

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_64bit_xl_field(self):
        """XL field should be supported"""
        # Verify that boxes with the XL field are properly read.
        # Don't have such a file on hand, so we create one.  Copy our example
        # file, but making the codestream have a 64-bit XL field.
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                write_buffer = ifile.read(3223)
                tfile.write(write_buffer)

                # The L field must be 1 in order to signal the presence of the
                # XL field.  The actual length of the jp2c box increased by 8
                # (8 bytes for the XL field).
                length = 1
                typ = b'jp2c'
                xlen = 1133427 + 8
                write_buffer = struct.pack('>I4sQ', int(length), typ, xlen)
                tfile.write(write_buffer)

                # Get the rest of the input file (minus the 8 bytes for L and
                # T.
                ifile.seek(8, 1)
                write_buffer = ifile.read()
                tfile.write(write_buffer)
                tfile.flush()

            jp2k = Jp2k(tfile.name)

            self.assertEqual(jp2k.box[4].box_id, 'jp2c')
            self.assertEqual(jp2k.box[4].offset, 3223)
            self.assertEqual(jp2k.box[4].length, 1133427 + 8)

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_length_field_is_zero(self):
        """L=0 (length field in box header) is allowed"""
        # Verify that boxes with the L field as zero are correctly read.
        # This should only happen in the last box of a JPEG 2000 file.
        # Our example image has its last box at byte 588458.
        baseline_jp2 = Jp2k(self.jp2file)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                write_buffer = ifile.read(588458)
                tfile.write(write_buffer)

                length = 0
                typ = b'uuid'
                write_buffer = struct.pack('>I4s', int(length), typ)
                tfile.write(write_buffer)

                # Get the rest of the input file (minus the 8 bytes for L and
                # T.
                ifile.seek(8, 1)
                write_buffer = ifile.read()
                tfile.write(write_buffer)
                tfile.flush()

            new_jp2 = Jp2k(tfile.name)

            # The top level boxes in each file should match.
            for j in range(len(baseline_jp2.box)):
                self.assertEqual(new_jp2.box[j].box_id,
                                 baseline_jp2.box[j].box_id)
                self.assertEqual(new_jp2.box[j].offset,
                                 baseline_jp2.box[j].offset)
                self.assertEqual(new_jp2.box[j].length,
                                 baseline_jp2.box[j].length)

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_basic_jp2(self):
        """Just a very basic test that reading a JP2 file does not error out.
        """
        j2k = Jp2k(self.jp2file)
        j2k[::2, ::2]

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_basic_j2k(self):
        """This test is only useful when openjp2 is not available
        and OPJ_DATA_ROOT is not set.  We need at least one
        working J2K test.
        """
        j2k = Jp2k(self.j2kfile)
        j2k[:]

    def test_empty_box_with_j2k(self):
        """Verify that the list of boxes in a J2C/J2K file is present, but
        empty.
        """
        j = Jp2k(self.j2kfile)
        self.assertEqual(j.box, [])

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_uinf_ulst_url_boxes(self):
        """Verify that we can read UINF, ULST, and URL boxes"""
        # Verify that we can read UINF, ULST, and URL boxes.  I don't have
        # easy access to such a file, and there's no such file in the
        # openjpeg repository, so I'll fake one.
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                write_buffer = ifile.read(77)
                tfile.write(write_buffer)

                # Write the UINF superbox
                # Length = 50, id is uinf.
                write_buffer = struct.pack('>I4s', int(50), b'uinf')
                tfile.write(write_buffer)

                # Write the ULST box.
                # Length is 26, 1 UUID, hard code that UUID as zeros.
                write_buffer = struct.pack('>I4sHIIII', int(26), b'ulst',
                                           int(1), int(0), int(0), int(0),
                                           int(0))
                tfile.write(write_buffer)

                # Write the URL box.
                # Length is 16, version is one byte, flag is 3 bytes, url
                # is the rest.
                write_buffer = struct.pack('>I4sBBBB',
                                           int(16), b'url ',
                                           int(0), int(0), int(0), int(0))
                tfile.write(write_buffer)
                write_buffer = struct.pack('>ssss', b'a', b'b', b'c', b'd')
                tfile.write(write_buffer)

                # Get the rest of the input file.
                write_buffer = ifile.read()
                tfile.write(write_buffer)
                tfile.flush()

            jp2k = Jp2k(tfile.name)

            self.assertEqual(jp2k.box[3].box_id, 'uinf')
            self.assertEqual(jp2k.box[3].offset, 77)
            self.assertEqual(jp2k.box[3].length, 50)

            self.assertEqual(jp2k.box[3].box[0].box_id, 'ulst')
            self.assertEqual(jp2k.box[3].box[0].offset, 85)
            self.assertEqual(jp2k.box[3].box[0].length, 26)
            ulst = []
            ulst.append(uuid.UUID('00000000-0000-0000-0000-000000000000'))
            self.assertEqual(jp2k.box[3].box[0].ulst, ulst)

            self.assertEqual(jp2k.box[3].box[1].box_id, 'url ')
            self.assertEqual(jp2k.box[3].box[1].offset, 111)
            self.assertEqual(jp2k.box[3].box[1].length, 16)
            self.assertEqual(jp2k.box[3].box[1].version, 0)
            self.assertEqual(jp2k.box[3].box[1].flag, (0, 0, 0))
            self.assertEqual(jp2k.box[3].box[1].url, 'abcd')

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_xml_with_trailing_nulls(self):
        """ElementTree doesn't like trailing null chars after valid XML text"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                write_buffer = ifile.read(77)
                tfile.write(write_buffer)

                # Write the xml box
                # Length = 36, id is 'xml '.
                write_buffer = struct.pack('>I4s', int(36), b'xml ')
                tfile.write(write_buffer)

                write_buffer = '<test>this is a test</test>' + chr(0)
                write_buffer = write_buffer.encode()
                tfile.write(write_buffer)

                # Get the rest of the input file.
                write_buffer = ifile.read()
                tfile.write(write_buffer)
                tfile.flush()

            jp2k = Jp2k(tfile.name)

            self.assertEqual(jp2k.box[3].box_id, 'xml ')
            self.assertEqual(jp2k.box[3].offset, 77)
            self.assertEqual(jp2k.box[3].length, 36)
            self.assertEqual(ET.tostring(jp2k.box[3].xml.getroot()),
                             b'<test>this is a test</test>')

    @unittest.skipIf(not HAS_PYTHON_XMP_TOOLKIT,
                     "Requires Python XMP Toolkit >= 2.0")
    def test_xmp_attribute(self):
        """Verify the XMP packet in the shipping example file can be read."""
        j = Jp2k(self.jp2file)

        xmp = j.box[3].data
        ns0 = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
        ns1 = '{http://ns.adobe.com/xap/1.0/}'
        name = '{0}RDF/{0}Description/{1}CreatorTool'.format(ns0, ns1)
        elt = xmp.find(name)
        self.assertEqual(elt.text, 'Google')

        xmp = XMPMeta()
        xmp.parse_from_str(j.box[3].raw_data.decode('utf-8'),
                           xmpmeta_wrap=False)
        creator_tool = xmp.get_property(libxmp.consts.XMP_NS_XMP,
                                        'CreatorTool')
        self.assertEqual(creator_tool, 'Google')

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    @unittest.skipIf(re.match(r'''(1|2.0.0)''',
                              glymur.version.openjpeg_version) is not None,
                     "Not supported until 2.0.1")
    def test_jpx_mult_codestreams_jp2_brand(self):
        """Read JPX codestream when jp2-compatible."""
        # The file in question has multiple codestreams.
        jpx = Jp2k(self.jpxfile)
        data = jpx[:]
        self.assertEqual(data.shape, (1024, 1024, 3))

    def test_read_without_openjpeg(self):
        """Don't have openjpeg or openjp2 library?  Must error out."""
        with patch('glymur.version.openjpeg_version_tuple', new=(0, 0, 0)):
            with patch('glymur.version.openjpeg_version', new='0.0.0'):
                with self.assertRaises(RuntimeError):
                    with warnings.catch_warnings():
                        # Suppress a deprecation warning for raw read method.
                        warnings.simplefilter("ignore")
                        glymur.Jp2k(self.jp2file).read()
                with self.assertRaises(RuntimeError):
                    glymur.Jp2k(self.jp2file)[:]

    def test_read_bands_without_openjp2(self):
        """Don't have openjp2 library?  Must error out."""
        with patch('glymur.version.openjpeg_version_tuple', new=(1, 5, 0)):
            with patch('glymur.version.openjpeg_version', new='1.5.0'):
                with self.assertRaises(RuntimeError):
                    glymur.Jp2k(self.jp2file).read_bands()


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(re.match('1.[0-4]', openjpeg_version) is not None,
                 "Not supported with OpenJPEG {0}".format(openjpeg_version))
@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
class TestJp2k_write(unittest.TestCase):
    """Write tests, can be run by versions 1.5+"""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    def test_precinct_size_too_small(self):
        """first precinct size must be >= 2x that of the code block size"""
        data = np.zeros((640, 480), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cbsize=(16, 16), psizes=[(16, 16)])

    def test_precinct_size_not_power_of_two(self):
        """must be power of two"""
        data = np.zeros((640, 480), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cbsize=(16, 16), psizes=[(48, 48)])

    def test_unsupported_int32(self):
        """Should raise a runtime error if trying to write int32"""
        data = np.zeros((128, 128), dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data)

    def test_unsupported_uint32(self):
        """Should raise a runtime error if trying to write uint32"""
        data = np.zeros((128, 128), dtype=np.uint32)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data)

    def test_write_with_version_too_early(self):
        """Should raise a runtime error if trying to write with version 1.3"""
        data = np.zeros((128, 128), dtype=np.uint8)
        versions = ["1.0.0", "1.1.0", "1.2.0", "1.3.0"]
        for version in versions:
            with patch('glymur.version.openjpeg_version', new=version):
                with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
                    with self.assertRaises(RuntimeError):
                        Jp2k(tfile.name, data=data)

    def test_cblkh_different_than_width(self):
        """Verify that we can set a code block size where height does not equal
        width.
        """
        data = np.zeros((128, 128), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            # The code block dimensions are given as rows x columns.
            j = Jp2k(tfile.name, data=data, cbsize=(16, 32))
            codestream = j.get_codestream()

            # Code block size is reported as XY in the codestream.
            self.assertEqual(tuple(codestream.segment[2].spcod[5:7]), (3, 2))

    def test_too_many_dimensions(self):
        """OpenJP2 only allows 2D or 3D images."""
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name,
                     data=np.zeros((128, 128, 2, 2), dtype=np.uint8))

    def test_2d_rgb(self):
        """RGB must have at least 3 components."""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name,
                     data=np.zeros((128, 128, 2), dtype=np.uint8),
                     colorspace='rgb')

    def test_colorspace_with_j2k(self):
        """Specifying a colorspace with J2K does not make sense"""
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name,
                     data=np.zeros((128, 128, 3), dtype=np.uint8),
                     colorspace='rgb')

    def test_specify_rgb(self):
        """specify RGB explicitly"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            j = Jp2k(tfile.name,
                     data=np.zeros((128, 128, 3), dtype=np.uint8),
                     colorspace='rgb')
            self.assertEqual(j.box[2].box[1].colorspace, glymur.core.SRGB)

    def test_specify_gray(self):
        """test gray explicitly specified (that's GRAY, not GREY)"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data, colorspace='gray')
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    def test_specify_grey(self):
        """test grey explicitly specified"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data, colorspace='grey')
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    def test_grey_with_two_extra_comps(self):
        """should be able to write gray + two extra components"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128, 3), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data, colorspace='gray')
            self.assertEqual(j.box[2].box[0].height, 128)
            self.assertEqual(j.box[2].box[0].width, 128)
            self.assertEqual(j.box[2].box[0].num_components, 3)
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    def test_specify_ycc(self):
        """Should reject YCC"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                data = np.zeros((128, 128, 3), dtype=np.uint8)
                Jp2k(tfile.name, data=data, colorspace='ycc')

    def test_write_with_jp2_in_caps(self):
        """should be able to write with JP2 suffix."""
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        with tempfile.NamedTemporaryFile(suffix='.JP2') as tfile:
            ofile = Jp2k(tfile.name, data=expdata)
            actdata = ofile[:]
            np.testing.assert_array_equal(actdata, expdata)

    def test_write_srgb_without_mct(self):
        """should be able to write RGB without specifying mct"""
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, data=expdata, mct=False)
            actdata = ofile[:]
            np.testing.assert_array_equal(actdata, expdata)

            codestream = ofile.get_codestream()
            self.assertEqual(codestream.segment[2].spcod[3], 0)  # no mct

    def test_write_grayscale_with_mct(self):
        """MCT usage makes no sense for grayscale images."""
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=expdata[:, :, 0], mct=True)

    def test_write_cprl(self):
        """Must be able to write a CPRL progression order file"""
        # Issue 17
        j = Jp2k(self.jp2file)
        expdata = j[::2, ::2]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, data=expdata, prog='CPRL')
            actdata = ofile[:]
            np.testing.assert_array_equal(actdata, expdata)

            codestream = ofile.get_codestream()
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.CPRL)


class TestJp2k_1_x(unittest.TestCase):
    """Test suite for openjpeg 1.x, not appropriate for 2.x"""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    def test_tile(self):
        """tile option not allowed for 1.x.
        """
        with patch('glymur.version.openjpeg_version_tuple', new=(1, 5, 0)):
            with patch('glymur.version.openjpeg_version', new="1.5.0"):
                j2k = Jp2k(self.j2kfile)
                with warnings.catch_warnings():
                    # The tile keyword is deprecated, so suppress the warning.
                    warnings.simplefilter('ignore')
                    with self.assertRaises(TypeError):
                        j2k.read(tile=0)

    def test_layer(self):
        """layer option not allowed for 1.x.
        """
        with patch('glymur.version.openjpeg_version_tuple', new=(1, 5, 0)):
            j2k = Jp2k(self.j2kfile)
            with self.assertRaises(RuntimeError):
                j2k.layer = 1


@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class Test_2p0_official(unittest.TestCase):
    """Tests specific to v2.0.0"""

    def test_extra_components_on_v2(self):
        """Can only write 4 components on 2.0+, should error out otherwise."""
        with patch('glymur.version.openjpeg_version', new="2.0.0"):
            with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
                data = np.zeros((128, 128, 4), dtype=np.uint8)
                with self.assertRaises(IOError):
                    Jp2k(tfile.name, data=data)


@unittest.skipIf(glymur.version.openjpeg_version_tuple[0] < 2,
                 "Requires as least v2.0")
class TestJp2k_2_0(unittest.TestCase):
    """Test suite requiring at least version 2.0"""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    def test_bad_area_parameter(self):
        """Should error out appropriately if given a bad area parameter."""
        j = Jp2k(self.jp2file)
        with self.assertRaises(IOError):
            # Start corner must be >= 0
            j[-1:1, -1:1]
        with self.assertRaises(IOError):
            # End corner must be > 0
            j[10:0, 10:0]
        with self.assertRaises(IOError):
            # End corner must be >= start corner
            j[10:8, 10:8]

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_unrecognized_jp2_clrspace(self):
        """We only allow RGB and GRAYSCALE.  Should error out with others"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128, 3), dtype=np.uint8)
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data, colorspace='cmyk')

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_asoc_label_box(self):
        """Test asoc and label box"""
        # Construct a fake file with an asoc and a label box, as
        # OpenJPEG doesn't have such a file.
        data = Jp2k(self.jp2file)[::2, ::2]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            Jp2k(tfile.name, data=data)

            with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile2:

                # Offset of the codestream is where we start.
                read_buffer = tfile.read(77)
                tfile2.write(read_buffer)

                # read the rest of the file, it's the codestream.
                codestream = tfile.read()

                # Write the asoc superbox.
                # Length = 36, id is 'asoc'.
                write_buffer = struct.pack('>I4s', int(56), b'asoc')
                tfile2.write(write_buffer)

                # Write the contained label box
                write_buffer = struct.pack('>I4s', int(13), b'lbl ')
                tfile2.write(write_buffer)
                tfile2.write('label'.encode())

                # Write the xml box
                # Length = 36, id is 'xml '.
                write_buffer = struct.pack('>I4s', int(35), b'xml ')
                tfile2.write(write_buffer)

                write_buffer = '<test>this is a test</test>'
                write_buffer = write_buffer.encode()
                tfile2.write(write_buffer)

                # Now append the codestream.
                tfile2.write(codestream)
                tfile2.flush()

                jasoc = Jp2k(tfile2.name)
                self.assertEqual(jasoc.box[3].box_id, 'asoc')
                self.assertEqual(jasoc.box[3].box[0].box_id, 'lbl ')
                self.assertEqual(jasoc.box[3].box[0].label, 'label')
                self.assertEqual(jasoc.box[3].box[1].box_id, 'xml ')


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(re.match(r'''(1|2.0.0)''',
                          glymur.version.openjpeg_version) is not None,
                 "Not to be run until unless 2.0.1 or higher is present")
class TestJp2k_2_1(unittest.TestCase):
    """Only to be run in 2.0+."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_grey_with_extra_component(self):
        """version 2.0 cannot write gray + extra"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128, 2), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data)
            self.assertEqual(j.box[2].box[0].height, 128)
            self.assertEqual(j.box[2].box[0].width, 128)
            self.assertEqual(j.box[2].box[0].num_components, 2)
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_rgb_with_extra_component(self):
        """v2.0+ should be able to write extra components"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128, 4), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data)
            self.assertEqual(j.box[2].box[0].height, 128)
            self.assertEqual(j.box[2].box[0].width, 128)
            self.assertEqual(j.box[2].box[0].num_components, 4)
            self.assertEqual(j.box[2].box[1].colorspace, glymur.core.SRGB)

    @unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_openjpeg_library_message(self):
        """Verify the error message produced by the openjpeg library"""
        # This will confirm that the error callback mechanism is working.
        with open(self.jp2file, 'rb') as fptr:
            data = fptr.read()
            with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
                # Codestream starts at byte 3323. SIZ marker at 3233.
                # COD marker at 3282.  Subsampling at 3276.
                offset = 3223
                tfile.write(data[0:offset+52])

                # Make the DY bytes of the SIZ segment zero.  That means that
                # a subsampling factor is zero, which is illegal.
                tfile.write(b'\x00')
                tfile.write(data[offset+53:offset+55])
                tfile.write(b'\x00')
                tfile.write(data[offset+57:offset+59])
                tfile.write(b'\x00')

                tfile.write(data[offset+59:])
                tfile.flush()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    j = Jp2k(tfile.name)
                    regexp = re.compile(r'''OpenJPEG\slibrary\serror:\s+
                                            Invalid\svalues\sfor\scomp\s=\s0\s+
                                            :\sdx=1\sdy=0''', re.VERBOSE)
                    if sys.hexversion < 0x03020000:
                        with self.assertRaisesRegexp((IOError, OSError),
                                                     regexp):
                            j[::2, ::2]
                    else:
                        with self.assertRaisesRegex((IOError, OSError),
                                                    regexp):
                            j[::2, ::2]


@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestParsing(unittest.TestCase):
    """Tests for verifying how parsing may be altered."""
    def setUp(self):
        self.jp2file = glymur.data.nemo()
        # Reset parseoptions for every test.
        glymur.set_parseoptions(full_codestream=False)

    def tearDown(self):
        glymur.set_parseoptions(full_codestream=False)

    @unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
    def test_bad_rsiz(self):
        """Should not warn if RSIZ when parsing is turned off."""
        filename = opj_data_file('input/nonregression/edf_c2_1002767.jp2')
        glymur.set_parseoptions(full_codestream=False)
        Jp2k(filename)

        glymur.set_parseoptions(full_codestream=True)
        with self.assertWarnsRegex(UserWarning, 'Invalid profile'):
            Jp2k(filename)

    def test_main_header(self):
        """verify that the main header isn't loaded during normal parsing"""
        # The hidden _main_header attribute should show up after accessing it.
        jp2 = Jp2k(self.jp2file)
        jp2c = jp2.box[4]
        self.assertIsNone(jp2c._codestream)
        jp2c.codestream
        self.assertIsNotNone(jp2c._codestream)


@unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestJp2kOpjDataRootWarnings(unittest.TestCase):
    """These tests should be run by just about all configuration."""

    def test_undecodeable_box_id(self):
        """Should warn in case of undecodeable box ID but not error out."""
        filename = opj_data_file('input/nonregression/edf_c2_1013627.jp2')
        with self.assertWarnsRegex(UserWarning, 'Unrecognized box'):
            jp2 = Jp2k(filename)

        # Now make sure we got all of the boxes.  Ignore the last, which was
        # bad.
        box_ids = [box.box_id for box in jp2.box[:-1]]
        self.assertEqual(box_ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

    def test_bad_ftyp_brand(self):
        """Should warn in case of bad ftyp brand."""
        filename = opj_data_file('input/nonregression/edf_c2_1000290.jp2')
        with self.assertWarns(UserWarning):
            Jp2k(filename)

    def test_invalid_approximation(self):
        """Should warn in case of invalid approximation."""
        filename = opj_data_file('input/nonregression/edf_c2_1015644.jp2')
        with self.assertWarnsRegex(UserWarning, 'Invalid approximation'):
            Jp2k(filename)

    def test_invalid_colorspace(self):
        """
        Should warn in case of invalid colorspace.

        There are multiple warnings, so there's no good way to regex them all.
        """
        filename = opj_data_file('input/nonregression/edf_c2_1103421.jp2')
        with self.assertWarns(UserWarning):
            Jp2k(filename)

    def test_stupid_windows_eol_at_end(self):
        """Garbage characters at the end of the file."""
        filename = opj_data_file('input/nonregression/issue211.jp2')
        with self.assertWarns(UserWarning):
            Jp2k(filename)


@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestJp2kOpjDataRoot(unittest.TestCase):
    """These tests should be run by just about all configurations."""

    def test_zero_length_reserved_segment(self):
        """
        Zero length reserved segment.  Unsure if this is invalid or not.

        Much of the rest of the file is invalid, so just be sure we can parse
        all of it without erroring out.
        """
        filename = opj_data_file('input/nonregression/issue476.jp2')
        with warnings.catch_warnings():
            # Suppress warnings for this corrupt file
            warnings.simplefilter("ignore")
            cstr = Jp2k(filename).get_codestream()
            self.assertEqual(cstr.segment[5].marker_id, '0xff00')
            self.assertEqual(cstr.segment[5].length, 0)

    @unittest.skipIf(re.match("0|1.[0-4]", glymur.version.openjpeg_version),
                     "Must have openjpeg 1.5 or higher to run")
    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_irreversible(self):
        """Irreversible"""
        filename = opj_data_file('input/nonregression/issue141.rawl')
        expdata = np.fromfile(filename, dtype=np.uint16)
        expdata.resize((32, 2048))
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=expdata, irreversible=True, numres=5)

            codestream = j.get_codestream()
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

            actdata = j[:]
            self.assertTrue(fixtures.mse(actdata, expdata) < 250)

    @unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
    def test_no_cxform_pclr_jp2(self):
        """Indices for pclr jpxfile if no color transform"""
        filename = opj_data_file('input/conformance/file9.jp2')
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(filename)
        rgb = jp2[:]
        jp2.ignore_pclr_cmap_cdef = True
        idx = jp2[:]
        self.assertEqual(rgb.shape, (512, 768, 3))
        self.assertEqual(idx.shape, (512, 768))

        # Should be able to manually reconstruct the RGB image from the palette
        # and indices.
        palette = jp2.box[2].box[1].palette
        rgb_from_idx = np.zeros(rgb.shape, dtype=np.uint8)
        for r in np.arange(rgb.shape[0]):
            for c in np.arange(rgb.shape[1]):
                rgb_from_idx[r, c] = palette[idx[r, c]]
        np.testing.assert_array_equal(rgb, rgb_from_idx)

    def test_read_differing_subsamples(self):
        """should error out with read used on differently subsampled images"""
        # Verify that we error out appropriately if we use the read method
        # on an image with differing subsamples
        #
        # Issue 86.
        filename = opj_data_file('input/conformance/p0_05.j2k')
        j = Jp2k(filename)
        with self.assertRaises(RuntimeError):
            j[:]

    @unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
    def test_no_cxform_cmap(self):
        """Bands as physically ordered, not as physically intended"""
        # This file has the components physically reversed.  The cmap box
        # tells the decoder how to order them, but this flag prevents that.
        filename = opj_data_file('input/conformance/file2.jp2')
        with self.assertWarns(UserWarning):
            # The file has a bad compatibility list entry.  Not important here.
            j = Jp2k(filename)
        ycbcr = j[:]
        j.ignore_pclr_cmap_cdef = True
        crcby = j[:]

        expected = np.zeros(ycbcr.shape, ycbcr.dtype)
        for k in range(crcby.shape[2]):
            expected[:, :, crcby.shape[2] - k - 1] = crcby[:, :, k]

        np.testing.assert_array_equal(ycbcr, expected)


@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestCodestreamOpjData(unittest.TestCase):
    """Test suite for unusual codestream cases.  Uses OPJ_DATA_ROOT"""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_reserved_marker_segment(self):
        """Reserved marker segments are ok."""

        # Some marker segments were reserved in FCD15444-1.  Since that
        # standard is old, some of them may have come into use.
        #
        # Let's inject a reserved marker segment into a file that
        # we know something about to make sure we can still parse it.
        filename = os.path.join(OPJ_DATA_ROOT, 'input/conformance/p0_01.j2k')
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with open(filename, 'rb') as ifile:
                # Everything up until the first QCD marker.
                read_buffer = ifile.read(45)
                tfile.write(read_buffer)

                # Write the new marker segment, 0xff6f = 65391
                read_buffer = struct.pack('>HHB', int(65391), int(3), int(0))
                tfile.write(read_buffer)

                # Get the rest of the input file.
                read_buffer = ifile.read()
                tfile.write(read_buffer)
                tfile.flush()

            codestream = Jp2k(tfile.name).get_codestream()

            self.assertEqual(codestream.segment[2].marker_id, '0xff6f')
            self.assertEqual(codestream.segment[2].length, 3)
            self.assertEqual(codestream.segment[2].data, b'\x00')

    def test_psot_is_zero(self):
        """Psot=0 in SOT is perfectly legal.  Issue #78."""
        filename = os.path.join(OPJ_DATA_ROOT,
                                'input/nonregression/123.j2c')
        j = Jp2k(filename)
        codestream = j.get_codestream(header_only=False)

        # The codestream is valid, so we should be able to get the entire
        # codestream, so the last one is EOC.
        self.assertEqual(codestream.segment[-1].marker_id, 'EOC')

    def test_siz_segment_ssiz_signed(self):
        """ssiz attribute to be removed in future release"""
        filename = os.path.join(OPJ_DATA_ROOT, 'input/conformance/p0_03.j2k')
        j = Jp2k(filename)
        codestream = j.get_codestream()

        # The ssiz attribute was simply a tuple of raw bytes.
        # The first 7 bits are interpreted as the bitdepth, the MSB determines
        # whether or not it is signed.
        self.assertEqual(codestream.segment[1].ssiz, (131,))


class TestCodestreamRepr(unittest.TestCase):

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_soc(self):
        """Test SOC segment repr"""
        segment = glymur.codestream.SOCsegment()
        newseg = eval(repr(segment))
        self.assertEqual(newseg.marker_id, 'SOC')

    def test_siz(self):
        """Test SIZ segment repr"""
        kwargs = {'rsiz': 0,
                  'xysiz': (2592, 1456),
                  'xyosiz': (0, 0),
                  'xytsiz': (2592, 1456),
                  'xytosiz': (0, 0),
                  'Csiz': 3,
                  'bitdepth': (8, 8, 8),
                  'signed':  (False, False, False),
                  'xyrsiz': ((1, 1, 1), (1, 1, 1))}
        segment = glymur.codestream.SIZsegment(**kwargs)
        newseg = eval(repr(segment))
        self.assertEqual(newseg.marker_id, 'SIZ')
        self.assertEqual(newseg.xsiz, 2592)
        self.assertEqual(newseg.ysiz, 1456)
        self.assertEqual(newseg.xosiz, 0)
        self.assertEqual(newseg.yosiz, 0)
        self.assertEqual(newseg.xtsiz, 2592)
        self.assertEqual(newseg.ytsiz, 1456)
        self.assertEqual(newseg.xtosiz, 0)
        self.assertEqual(newseg.ytosiz, 0)

        self.assertEqual(newseg.xrsiz, (1, 1, 1))
        self.assertEqual(newseg.yrsiz, (1, 1, 1))
        self.assertEqual(newseg.bitdepth, (8, 8, 8))
        self.assertEqual(newseg.signed, (False, False, False))

    def test_siz_segment_ssiz_unsigned(self):
        """ssiz attribute to be removed in future release"""
        j = Jp2k(self.jp2file)
        codestream = j.get_codestream()

        # The ssiz attribute was simply a tuple of raw bytes.
        # The first 7 bits are interpreted as the bitdepth, the MSB determines
        # whether or not it is signed.
        self.assertEqual(codestream.segment[1].ssiz, (7, 7, 7))


class TestCodestream(unittest.TestCase):
    """Test suite for unusual codestream cases."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_siz_segment_ssiz_unsigned(self):
        """ssiz attribute to be removed in future release"""
        j = Jp2k(self.jp2file)
        codestream = j.get_codestream()

        # The ssiz attribute was simply a tuple of raw bytes.
        # The first 7 bits are interpreted as the bitdepth, the MSB determines
        # whether or not it is signed.
        self.assertEqual(codestream.segment[1].ssiz, (7, 7, 7))
