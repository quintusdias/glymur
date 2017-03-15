"""
Tests for general glymur functionality.
"""
# Standard library imports ...
import datetime
import doctest
from io import BytesIO
import os
import re
import struct
import sys
import tempfile
import unittest
import uuid
import warnings
if sys.hexversion >= 0x03030000:
    from unittest.mock import patch
else:
    from mock import patch
if sys.hexversion >= 0x03040000:
    import pathlib
else:
    import pathlib2 as pathlib
from xml.etree import cElementTree as ET

# Third party library imports ...
import numpy as np
import pkg_resources as pkg

# Local imports
import glymur
from glymur import Jp2k
from glymur.core import COLOR, RED, GREEN, BLUE, RESTRICTED_ICC_PROFILE
from glymur.codestream import SIZsegment
from glymur.version import openjpeg_version

from .fixtures import WINDOWS_TMP_FILE_MSG
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG

from . import fixtures


def docTearDown(doctest_obj):
    glymur.set_option('parse.full_codestream', False)


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


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestJp2k(unittest.TestCase):
    """These tests should be run by just about all configuration."""

    @classmethod
    def setUpClass(cls):
        cls.jp2file = glymur.data.nemo()
        cls.j2kfile = glymur.data.goodstuff()
        cls.jpxfile = glymur.data.jpxfile()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_pathlib(self):
        p = pathlib.Path(self.jp2file)
        jp2 = Jp2k(p)
        self.assertEqual(jp2.shape, (1456, 2592, 3))

    @unittest.skipIf(re.match('1.5.(1|2)', openjpeg_version) is not None,
                     "Mysteriously fails in 1.5.1 and 1.5.2")
    def test_no_cxform_pclr_jpx(self):
        """
        Indices for pclr jpxfile still usable if no color transform specified
        """
        with warnings.catch_warnings():
            # Suppress a Compatibility list item warning.  We already test
            # for this elsewhere.
            warnings.simplefilter("ignore")
            jp2 = Jp2k(self.jpxfile)
        rgb = jp2[:]
        jp2.ignore_pclr_cmap_cdef = True
        idx = jp2[:]
        self.assertEqual(rgb.shape, (1024, 1024, 3))
        self.assertEqual(idx.shape, (1024, 1024))

        # Should be able to manually reconstruct the RGB image from the palette
        # and indices.
        palette = jp2.box[2].box[2].palette
        rgb_from_idx = np.zeros(rgb.shape, dtype=np.uint8)
        for r in np.arange(rgb.shape[0]):
            for c in np.arange(rgb.shape[1]):
                rgb_from_idx[r, c] = palette[idx[r, c]]
        np.testing.assert_array_equal(rgb, rgb_from_idx)

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_no_cxform_cmap(self):
        """
        Reorder the components.
        """
        j2k = Jp2k(self.j2kfile)
        rgb = j2k[:]
        height, width, ncomps = rgb.shape

        # Rewrap the J2K file to reorder the components
        boxes = [
            glymur.jp2box.JPEG2000SignatureBox(),
            glymur.jp2box.FileTypeBox()
        ]
        jp2h = glymur.jp2box.JP2HeaderBox()
        jp2h.box = [
            glymur.jp2box.ImageHeaderBox(height, width, num_components=ncomps),
            glymur.jp2box.ColourSpecificationBox(colorspace=glymur.core.SRGB)
        ]

        channel_type = [COLOR, COLOR, COLOR]
        association = [BLUE, GREEN, RED]
        cdef = glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                                  association=association)
        jp2h.box.append(cdef)

        boxes.append(jp2h)
        boxes.append(glymur.jp2box.ContiguousCodestreamBox())

        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            jp2 = j2k.wrap(tfile.name, boxes=boxes)

            jp2.ignore_pclr_cmap_cdef = False
            bgr = jp2[:]

        np.testing.assert_array_equal(rgb, bgr[:, :, [2, 1, 0]])

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_bad_tile_part_pointer(self):
        """
        Should error out if we don't read a valid marker.

        Rewrite the Psot value such that the SOT marker segment points far
        beyond the end of the EOC marker (and the end of the file).
        """
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Copy up until Psot field.
                ofile.write(ifile.read(3350))

                # Write a bad Psot value.
                ofile.write(struct.pack('>I', 2000000))

                # copy the rest of the file as-is.
                ifile.seek(3354)
                ofile.write(ifile.read())
                ofile.flush()

            j = Jp2k(ofile.name)
            with self.assertRaises(IOError):
                j.get_codestream(header_only=False)

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_read_differing_subsamples(self):
        """
        should error out with read used on differently subsampled images

        Verify that we error out appropriately if we use the read method
        on an image with differing subsamples

        Issue 86.

        Copy nemo.jp2 but change the SIZ segment to have differing subsamples.
        """
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Copy up until codestream box.
                ofile.write(ifile.read(3223))

                # Write the jp2c header and SOC marker.
                ofile.write(ifile.read(10))

                # Read the SIZ segment, modify the last y subsampling value,
                # and write it back out
                buffer = bytearray(ifile.read(49))
                buffer[-1] = 2
                ofile.write(buffer)

                # Write the rest of the file.
                ofile.write(ifile.read())
                ofile.flush()

            j = Jp2k(ofile.name)
            with self.assertRaises(IOError):
                j[:]

    def test_shape_jp2(self):
        """verify shape attribute for JP2 file
        """
        jp2 = Jp2k(self.jp2file)
        self.assertEqual(jp2.shape, (1456, 2592, 3))

    def test_shape_3_channel_j2k(self):
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
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

            actdata = j2[:]
            self.assertTrue(fixtures.mse(actdata[0], expdata[0]) < 0.38)

    @unittest.skipIf(os.name == "nt", "Unexplained failure on windows")
    def test_repr(self):
        """Verify that results of __repr__ are eval-able."""
        j = Jp2k(self.j2kfile)
        newjp2 = eval(repr(j))

        self.assertEqual(newjp2.filename, self.j2kfile)
        self.assertEqual(len(newjp2.box), 0)

    def test_rlevel_max_backwards_compatibility(self):
        """
        Verify that rlevel=-1 gets us the lowest resolution image

        This is an old option only available via the read method, not via
        array-style slicing.
        """
        j = Jp2k(self.j2kfile)
        with warnings.catch_warnings():
            # Suppress the DeprecationWarning
            warnings.simplefilter("ignore")
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
        filename = pkg.resource_filename(glymur.__name__, "jp2k.py")
        with self.assertRaises(IOError):
            Jp2k(filename)

    def test_file_not_present(self):
        """Should error out if reading from a file that does not exist"""
        # Verify that we error out appropriately if not given an existing file
        # at all.
        filename = 'this file does not actually exist on the file system.'
        with self.assertRaises(OSError):
            Jp2k(filename)

    def test_codestream(self):
        """
        Verify the markers and segments of a JP2 file codestream.
        """
        jp2 = Jp2k(self.jp2file)
        c = jp2.get_codestream(header_only=False)

        # SOC
        self.assertEqual(c.segment[0].marker_id, 'SOC')

        # SIZ
        self.assertEqual(c.segment[1].marker_id, 'SIZ')
        self.assertEqual(c.segment[1].rsiz, 0)
        self.assertEqual(c.segment[1].xsiz, 2592)
        self.assertEqual(c.segment[1].ysiz, 1456)
        self.assertEqual(c.segment[1].xosiz, 0)
        self.assertEqual(c.segment[1].yosiz, 0)
        self.assertEqual(c.segment[1].xtsiz, 2592)
        self.assertEqual(c.segment[1].ytsiz, 1456)
        self.assertEqual(c.segment[1].xtosiz, 0)
        self.assertEqual(c.segment[1].ytosiz, 0)
        self.assertEqual(c.segment[1].Csiz, 3)
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        self.assertEqual(c.segment[1].signed, (False, False, False))
        self.assertEqual(c.segment[1].xrsiz, (1, 1, 1))
        self.assertEqual(c.segment[1].yrsiz, (1, 1, 1))

        self.assertEqual(c.segment[2].marker_id, 'COD')
        self.assertEqual(c.segment[2].offset, 3282)
        self.assertEqual(c.segment[2].length, 12)
        self.assertEqual(c.segment[2].scod, 0)
        self.assertEqual(c.segment[2].layers, 2)
        self.assertEqual(c.segment[2].code_block_size, (64.0, 64.0))
        self.assertEqual(c.segment[2].prog_order, 0)
        self.assertEqual(c.segment[2].xform, 1)
        self.assertEqual(c.segment[2].precinct_size, ((32768, 32768)))

        self.assertEqual(c.segment[3].marker_id, 'QCD')
        self.assertEqual(c.segment[3].offset, 3296)
        self.assertEqual(c.segment[3].length, 7)
        self.assertEqual(c.segment[3].sqcd, 64)
        self.assertEqual(c.segment[3].mantissa, [0, 0, 0, 0])
        self.assertEqual(c.segment[3].exponent, [8, 9, 9, 10])
        self.assertEqual(c.segment[3].guard_bits, 2)

        self.assertEqual(c.segment[4].marker_id, 'CME')
        self.assertEqual(c.segment[4].rcme, 1)
        self.assertEqual(c.segment[4].ccme,
                         b'Created by OpenJPEG version 2.0.0')

        self.assertEqual(c.segment[5].marker_id, 'SOT')
        self.assertEqual(c.segment[5].offset, 3344)
        self.assertEqual(c.segment[5].length, 10)
        self.assertEqual(c.segment[5].isot, 0)
        self.assertEqual(c.segment[5].psot, 1132173)
        self.assertEqual(c.segment[5].tpsot, 0)
        self.assertEqual(c.segment[5].tnsot, 1)

        self.assertEqual(c.segment[6].marker_id, 'COC')
        self.assertEqual(c.segment[6].offset, 3356)
        self.assertEqual(c.segment[6].length, 9)
        self.assertEqual(c.segment[6].ccoc, 1)
        np.testing.assert_array_equal(c.segment[6].scoc,
                                      np.array([0]))
        np.testing.assert_array_equal(c.segment[6].spcoc,
                                      np.array([1, 4, 4, 0, 1]))
        self.assertEqual(c.segment[6].precinct_size,
                         ((32768, 32768)))

        self.assertEqual(c.segment[7].marker_id, 'QCC')
        self.assertEqual(c.segment[7].offset, 3367)
        self.assertEqual(c.segment[7].length, 8)
        self.assertEqual(c.segment[7].cqcc, 1)
        self.assertEqual(c.segment[7].sqcc, 64)
        self.assertEqual(c.segment[7].mantissa, [0, 0, 0, 0])
        self.assertEqual(c.segment[7].exponent, [8, 9, 9, 10])
        self.assertEqual(c.segment[7].guard_bits, 2)

        self.assertEqual(c.segment[8].marker_id, 'COC')
        self.assertEqual(c.segment[8].offset, 3377)
        self.assertEqual(c.segment[8].length, 9)
        self.assertEqual(c.segment[8].ccoc, 2)
        np.testing.assert_array_equal(c.segment[8].scoc,
                                      np.array([0]))
        np.testing.assert_array_equal(c.segment[8].spcoc,
                                      np.array([1, 4, 4, 0, 1]))
        self.assertEqual(c.segment[8].precinct_size,
                         ((32768, 32768)))

        self.assertEqual(c.segment[9].marker_id, 'QCC')
        self.assertEqual(c.segment[9].offset, 3388)
        self.assertEqual(c.segment[9].length, 8)
        self.assertEqual(c.segment[9].cqcc, 2)
        self.assertEqual(c.segment[9].sqcc, 64)
        self.assertEqual(c.segment[9].mantissa, [0, 0, 0, 0])
        self.assertEqual(c.segment[9].exponent, [8, 9, 9, 10])
        self.assertEqual(c.segment[9].guard_bits, 2)

        self.assertEqual(c.segment[10].marker_id, 'SOD')

        self.assertEqual(c.segment[11].marker_id, 'EOC')

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
        """
        Just a very basic test that reading a JP2 file does not error out.
        """
        j2k = Jp2k(self.jp2file)
        j2k[::2, ::2]

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_basic_j2k(self):
        """
        Just a very basic test that reading a J2K file does not error out.
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

    def test_xmp_attribute(self):
        """Verify the XMP packet in the shipping example file can be read."""
        j = Jp2k(self.jp2file)

        xmp = j.box[3].data
        ns0 = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
        ns1 = '{http://ns.adobe.com/xap/1.0/}'
        name = '{0}RDF/{0}Description/{1}CreatorTool'.format(ns0, ns1)
        elt = xmp.find(name)
        self.assertEqual(elt.text, 'Google')

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
        """
        Don't have openjpeg or openjp2 library?  Must error out.
        """
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
        """
        Don't have openjp2 library?  Must error out.
        """
        exp_error = IOError
        with patch('glymur.version.openjpeg_version_tuple', new=(1, 5, 0)):
            with patch('glymur.version.openjpeg_version', new='1.5.0'):
                with self.assertRaises(exp_error):
                    glymur.Jp2k(self.jp2file).read_bands()

        
    @unittest.skipIf(sys.platform == 'win32', WINDOWS_TMP_FILE_MSG)
    def test_zero_length_reserved_segment(self):
        """
        Zero length reserved segment.  Unsure if this is invalid or not.

        Just make sure we can parse all of it without erroring out.
        """
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Copy up until codestream box.
                ofile.write(ifile.read(3223))

                # Write the new codestream length (+4) and the box ID.
                buffer = struct.pack('>I4s', 1132296 + 4, b'jp2c')
                ofile.write(buffer)

                # Copy up until the EOC marker.
                ifile.seek(3231)
                ofile.write(ifile.read(1132286))

                # Write the zero-length reserved segment.
                buffer = struct.pack('>BBH', 255, 0, 0)
                ofile.write(buffer)

                # Write the EOC marker and be done with it.
                ofile.write(ifile.read())
                ofile.flush()

            cstr = Jp2k(ofile.name).get_codestream(header_only=False)
            self.assertEqual(cstr.segment[11].marker_id, '0xff00')
            self.assertEqual(cstr.segment[11].length, 0)

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_psot_is_zero(self):
        """
        Psot=0 in SOT is perfectly legal.  Issue #78.
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as ofile:
            with open(self.j2kfile, 'rb') as ifile:
                # Write up until the SOD segment.
                ofile.write(ifile.read(164))

                # Write a SOT box with Psot = 0
                buffer = struct.pack('>HHHIBB', 0xff90, 10, 0, 0, 0, 1)
                ofile.write(buffer)

                # Write the rest of it.
                ofile.write(ifile.read())
                ofile.flush()

            j = Jp2k(ofile.name)
            codestream = j.get_codestream(header_only=False)

            # The codestream is valid, so we should be able to get the entire
            # codestream, so the last one is EOC.
            self.assertEqual(codestream.segment[-3].marker_id, 'SOT')
            self.assertEqual(codestream.segment[-2].marker_id, 'SOD')
            self.assertEqual(codestream.segment[-1].marker_id, 'EOC')

    def test_basic_icc_profile(self):
        """
        basic ICC profile

        Original file tested was input/conformance/file5.jp2
        """
        fp = BytesIO()

        # Write the colr box header.
        buffer = struct.pack('>I4s', 557, b'colr')
        buffer += struct.pack('>BBB', RESTRICTED_ICC_PROFILE, 2, 1)

        buffer += struct.pack('>IIBB', 546, 0, 2, 32)
        buffer += b'\x00' * 2 + b'scnr' + b'RGB ' + b'XYZ '
        # Need a date in bytes 24:36
        buffer += struct.pack('>HHHHHH', 2001, 8, 30, 13, 32, 37)
        buffer += 'acsp'.encode('utf-8')
        buffer += b'\x00\x00\x00\x00'
        buffer += b'\x00\x00\x00\x01'  # platform
        buffer += 'KODA'.encode('utf-8')  # 48 - 52
        buffer += 'ROMM'.encode('utf-8')  # Device Model
        buffer += b'\x00' * 12
        buffer += struct.pack('>III', 63190, 65536, 54061)  # 68 - 80
        buffer += 'JPEG'.encode('utf-8')  # 80 - 84
        buffer += b'\x00' * 44
        fp.write(buffer)
        fp.seek(8)

        box = glymur.jp2box.ColourSpecificationBox.parse(fp, 0, 557)
        profile = box.icc_profile

        self.assertEqual(profile['Size'], 546)
        self.assertEqual(profile['Preferred CMM Type'], 0)
        self.assertEqual(profile['Version'], '2.2.0')
        self.assertEqual(profile['Device Class'], 'input device profile')
        self.assertEqual(profile['Color Space'], 'RGB')
        self.assertEqual(profile['Datetime'],
                         datetime.datetime(2001, 8, 30, 13, 32, 37))
        self.assertEqual(profile['File Signature'], 'acsp')
        self.assertEqual(profile['Platform'], 'unrecognized')
        self.assertEqual(profile['Flags'],
                         'embedded, can be used independently')

        self.assertEqual(profile['Device Manufacturer'], 'KODA')
        self.assertEqual(profile['Device Model'], 'ROMM')

        self.assertEqual(profile['Device Attributes'],
                         ('reflective, glossy, positive media polarity, '
                          'color media'))
        self.assertEqual(profile['Rendering Intent'], 'perceptual')

        np.testing.assert_almost_equal(profile['Illuminant'],
                                       (0.964203, 1.000000, 0.824905),
                                       decimal=6)

        self.assertEqual(profile['Creator'], 'JPEG')

    @unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None, "Needs openjp2")
    def test_different_layers(self):
        """
        Verify that setting the layer property results in different images.
        """
        file = os.path.join('data', 'p0_03.j2k')
        file = pkg.resource_filename(__name__, file)
        j = Jp2k(file)
        d0 = j[:]
        
        j.layer = 1
        d1 = j[:]

        np.alltrue(d0 != d1)

    def test_default_verbosity(self):
        """
        By default, verbosity should be false.
        """
        file = os.path.join('data', 'p0_03.j2k')
        file = pkg.resource_filename(__name__, file)
        j = Jp2k(file)
        self.assertFalse(j.verbose)

    def test_default_layer(self):
        """
        By default, the layer should be 0
        """
        file = os.path.join('data', 'p0_03.j2k')
        file = pkg.resource_filename(__name__, file)
        j = Jp2k(file)
        self.assertEqual(j.layer, 0)


class CinemaBase(fixtures.MetadataBase):

    def verify_cinema_cod(self, cod_segment):

        self.assertFalse(cod_segment.scod & 2)  # no sop
        self.assertFalse(cod_segment.scod & 4)  # no eph
        self.assertEqual(cod_segment.prog_order, glymur.core.CPRL)
        self.assertEqual(cod_segment.layers, 1)
        self.assertEqual(cod_segment.mct, 1)
        self.assertEqual(cod_segment.num_res, 5)  # levels
        self.assertEqual(tuple(cod_segment.code_block_size), (32, 32))

    def check_cinema4k_codestream(self, codestream, image_size):

        kwargs = {'rsiz': 4, 'xysiz': image_size, 'xyosiz': (0, 0),
                  'xytsiz': image_size, 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12), 'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(codestream.segment[1], SIZsegment(**kwargs))

        self.verify_cinema_cod(codestream.segment[2])

    def check_cinema2k_codestream(self, codestream, image_size):

        kwargs = {'rsiz': 3, 'xysiz': image_size, 'xyosiz': (0, 0),
                  'xytsiz': image_size, 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12), 'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(codestream.segment[1], SIZsegment(**kwargs))

        self.verify_cinema_cod(codestream.segment[2])


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
@unittest.skipIf(re.match(r'''(1|2.0.0)''',
                          glymur.version.openjpeg_version) is not None,
                 "Uses features not supported until 2.0.1")
class WriteCinema(CinemaBase):

    @classmethod
    def setUpClass(cls):
        cls.jp2file = glymur.data.nemo()
        cls.jp2_data = glymur.Jp2k(cls.jp2file)[:]

    def test_NR_ENC_X_6_2K_24_FULL_CBR_CIRCLE_000_tif_17_encode(self):
        """
        Original test file was

            input/nonregression/X_6_2K_24_FULL_CBR_CIRCLE_000.tif

        """
        # Need to provide the proper size image
        data = np.concatenate((self.jp2_data, self.jp2_data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:1080, :2048, :]

        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with warnings.catch_warnings():
                # Ignore a warning issued by the library.
                warnings.simplefilter('ignore')
                j = Jp2k(tfile.name, data=data, cinema2k=24)

            codestream = j.get_codestream()
            self.check_cinema2k_codestream(codestream, (2048, 1080))

    def test_NR_ENC_X_6_2K_24_FULL_CBR_CIRCLE_000_tif_20_encode(self):
        """
        Original test file was

            input/nonregression/X_6_2K_24_FULL_CBR_CIRCLE_000.tif

        """
        # Need to provide the proper size image
        data = np.concatenate((self.jp2_data, self.jp2_data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:1080, :2048, :]

        with warnings.catch_warnings():
            # Ignore a warning issued by the library.
            warnings.simplefilter('ignore')
            with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
                j = Jp2k(tfile.name, data=data, cinema2k=48)
                codestream = j.get_codestream()
                self.check_cinema2k_codestream(codestream, (2048, 1080))

    def test_NR_ENC_ElephantDream_4K_tif_21_encode(self):
        """
        Verify basic cinema4k write

        Original test file is input/nonregression/ElephantDream_4K.tif
        """
        # Need to provide the proper size image
        data = np.concatenate((self.jp2_data, self.jp2_data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:2160, :4096, :]

        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with warnings.catch_warnings():
                # Ignore a warning issued by the library.
                warnings.simplefilter('ignore')
                j = Jp2k(tfile.name, data=data, cinema4k=True)

            codestream = j.get_codestream()
            self.check_cinema4k_codestream(codestream, (4096, 2160))


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
class TestJp2k_write(fixtures.MetadataBase):
    """Write tests, can be run by versions 1.5+"""

    @classmethod
    def setUpClass(cls):
        cls.jp2file = glymur.data.nemo()
        cls.j2kfile = glymur.data.goodstuff()

        cls.j2k_data = glymur.Jp2k(cls.j2kfile)[:]
        cls.jp2_data = glymur.Jp2k(cls.jp2file)[:]

        # Make single channel jp2 and j2k files.
        obj = tempfile.NamedTemporaryFile(delete=False, suffix=".j2k")
        glymur.Jp2k(obj.name, data=cls.j2k_data[:, :, 0])
        cls.single_channel_j2k = obj

        obj = tempfile.NamedTemporaryFile(delete=False, suffix=".jp2")
        glymur.Jp2k(obj.name, data=cls.j2k_data[:, :, 0])
        cls.single_channel_jp2 = obj

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.single_channel_j2k.name)
        os.unlink(cls.single_channel_jp2.name)

    def test_no_jp2c_box_in_outermost_jp2_list(self):
        """
        There must be a JP2C box in the outermost list of boxes.
        """
        j = glymur.Jp2k(self.jp2file)

        # Remove the last box, which is a codestream.
        boxes = j.box[:-1]

        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            with self.assertRaises(IOError):
                j2 = j.wrap(tfile.name, boxes=boxes)

    @unittest.skipIf(glymur.version.openjpeg_version_tuple[0] < 2,
                     "Requires as least v2.0")
    def test_null_data(self):
        """
        Verify that we prevent trying to write images with one dimension zero.
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=np.zeros((0, 256), dtype=np.uint8))

    def test_NR_ENC_Bretagne1_ppm_2_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne1.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data,
                     psnr=[30, 35, 40], numres=2)

            codestream = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 1)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self.verify_codeblock_style(codestream.segment[2].cstyle,
                                    [False, False, False, False, False, False])
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(codestream.segment[2].precinct_size,
                         ((32768, 32768)))

    def test_NR_ENC_Bretagne1_ppm_1_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne1.ppm

        """
        data = self.jp2_data
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            # Should be written with 3 layers.
            j = Jp2k(tfile.name, data=data, cratios=[200, 100, 50])
            c = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].mct, 1)  # mct
        self.assertEqual(c.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self.verify_codeblock_style(c.segment[2].cstyle,
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, ((32768, 32768)))

    @unittest.skipIf(glymur.config.load_openjpeg_library('openjpeg') is None,
                     "Needs openjpeg before this test make sense.")
    def test_NR_ENC_Bretagne1_ppm_1_encode_v15(self):
        """
        Test JPEG writing with version 1.5

        Original file tested was

            input/nonregression/Bretagne1.ppm

        """
        data = self.jp2_data
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with patch('glymur.jp2k.version.openjpeg_version_tuple', new=(1, 5, 0)):
                with patch('glymur.jp2k.opj2.OPENJP2', new=None):
                    j = Jp2k(tfile.name, shape=data.shape)
                    j[:] = data
                    c = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 3
        self.assertEqual(c.segment[2].mct, 1)  # mct
        self.assertEqual(c.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self.verify_codeblock_style(c.segment[2].cstyle,
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, ((32768, 32768)))

    @unittest.skipIf(fixtures.low_memory_linux_machine(), "Low memory machine")
    def test_NR_ENC_Bretagne1_ppm_3_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne1.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name,
                     data=self.jp2_data,
                     psnr=[30, 35, 40], cbsize=(16, 16), psizes=[(64, 64)])

            codestream = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (16, 16))  # cblksz
        self.verify_codeblock_style(codestream.segment[2].cstyle,
                                    [False, False,
                                     False, False, False, False])
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(codestream.segment[2].precinct_size,
                         ((2, 2), (4, 4), (8, 8), (16, 16), (32, 32),
                          (64, 64)))

    def test_NR_ENC_Bretagne2_ppm_4_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name,
                     data=self.jp2_data,
                     psizes=[(128, 128)] * 3,
                     cratios=[100, 20, 2],
                     tilesize=(480, 640),
                     cbsize=(32, 32))

            # Should be three layers.
            codestream = j.get_codestream()

            # RSIZ
            self.assertEqual(codestream.segment[1].xtsiz, 640)
            self.assertEqual(codestream.segment[1].ytsiz, 480)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (32, 32))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False,
                                         False, False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((16, 16), (32, 32), (64, 64), (128, 128),
                              (128, 128), (128, 128)))

    def test_NR_ENC_Bretagne2_ppm_5_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data,
                     tilesize=(127, 127), prog="PCRL")

            codestream = j.get_codestream()

            # RSIZ
            self.assertEqual(codestream.segment[1].xtsiz, 127)
            self.assertEqual(codestream.segment[1].ytsiz, 127)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.PCRL)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False,
                                         False, False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_Bretagne2_ppm_6_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, subsam=(2, 2), sop=True)

            codestream = j.get_codestream(header_only=False)

            # RSIZ
            self.assertEqual(codestream.segment[1].xrsiz, (2, 2, 2))
            self.assertEqual(codestream.segment[1].yrsiz, (2, 2, 2))

            # COD: Coding style default
            self.assertTrue(codestream.segment[2].scod & 2)  # sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

            # 18 SOP segments.
            nsops = [x.nsop for x in codestream.segment
                     if x.marker_id == 'SOP']
            self.assertEqual(nsops, list(range(18)))

    def test_NR_ENC_Bretagne2_ppm_7_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, modesw=38, eph=True)

            codestream = j.get_codestream(header_only=False)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertTrue(codestream.segment[2].scod & 4)  # eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, True, True,
                                         False, False, True])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

            # 18 EPH segments.
            ephs = [x for x in codestream.segment if x.marker_id == 'EPH']
            self.assertEqual(len(ephs), 18)

    def test_NR_ENC_Bretagne2_ppm_8_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name,
                     data=self.jp2_data, grid_offset=[300, 150], cratios=[800])

            codestream = j.get_codestream(header_only=False)

            # RSIZ
            self.assertEqual(codestream.segment[1].xosiz, 150)
            self.assertEqual(codestream.segment[1].yosiz, 300)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_Cevennes1_bmp_9_encode(self):
        """
        Original file tested was

            input/nonregression/Cevennes1.bmp

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, cratios=[800])

            codestream = j.get_codestream(header_only=False)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_Cevennes2_ppm_10_encode(self):
        """
        Original file tested was

            input/nonregression/Cevennes2.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, cratios=[50])

            codestream = j.get_codestream(header_only=False)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_Rome_bmp_11_encode(self):
        """
        Original file tested was

            input/nonregression/Rome.bmp

        """
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            jp2 = Jp2k(tfile.name,
                       data=self.jp2_data,
                       psnr=[30, 35, 50], prog='LRCP', numres=3)

            ids = [box.box_id for box in jp2.box]
            self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

            ids = [box.box_id for box in jp2.box[2].box]
            self.assertEqual(ids, ['ihdr', 'colr'])

            # Signature box.  Check for corruption.
            self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

            # File type box.
            self.assertEqual(jp2.box[1].brand, 'jp2 ')
            self.assertEqual(jp2.box[1].minor_version, 0)
            self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

            # Jp2 Header
            # Image header
            self.assertEqual(jp2.box[2].box[0].height, 1456)
            self.assertEqual(jp2.box[2].box[0].width, 2592)
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
            self.assertEqual(jp2.box[2].box[1].approximation, 0)
            self.assertIsNone(jp2.box[2].box[1].icc_profile)
            self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)

            codestream = jp2.box[3].codestream

            kwargs = {'rsiz': 0, 'xysiz': (2592, 1456), 'xyosiz': (0, 0),
                      'xytsiz': (2592, 1456), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 2)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_random_issue_0005_tif_12_encode(self):
        """
        Original file tested was

            input/nonregression/random-issue-0005.tif
        """
        data = self.jp2_data[:1024, :1024, 0].astype(np.uint16)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=data)

            codestream = j.get_codestream(header_only=False)

            kwargs = {'rsiz': 0, 'xysiz': (1024, 1024), 'xyosiz': (0, 0),
                      'xytsiz': (1024, 1024), 'xytosiz': (0, 0),
                      'bitdepth': (16,), 'signed': (False,),
                      'xyrsiz': [(1,), (1,)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 0)
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_issue141_rawl_23_encode(self):
        """
        Test irreversible option

        Original file tested was

            input/nonregression/issue141.rawl

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, irreversible=True)

            codestream = j.get_codestream()
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

    def test_cinema_mode_with_too_old_version_of_openjpeg(self):
        """
        Cinema mode not allowed for anything less than 2.0.1

        Origin file tested was

            input/nonregression/X_4_2K_24_185_CBR_WB_000.tif

        """
        data = np.zeros((857, 2048, 3), dtype=np.uint8)
        versions = ["1.5.0", "2.0.0"]
        for version in versions:
            with patch('glymur.version.openjpeg_version', new=version):
                with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
                    with self.assertRaises(IOError):
                        Jp2k(tfile.name, data=data, cinema2k=48)

    def test_cinema2K_with_others(self):
        """
        Can't specify cinema2k with any other options.

        Original test file was
        input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif
        """
        data = np.zeros((857, 2048, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cinema2k=48, cratios=[200, 100, 50])

    def test_cinema4K_with_others(self):
        """
        Can't specify cinema4k with any other options.

        Original test file was input/nonregression/ElephantDream_4K.tif
        """
        data = np.zeros((4096, 2160, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cinema4k=True, cratios=[200, 100, 50])

    def test_cblk_size_precinct_size(self):
        """
        code block sizes should never exceed half that of precinct size.
        """
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data,
                     cbsize=(64, 64), psizes=[(64, 64)])

    def test_cblk_size_not_power_of_two(self):
        """
        code block sizes should be powers of two.
        """
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data, cbsize=(13, 12))

    def test_precinct_size_not_p2(self):
        """
        precinct sizes should be powers of two.
        """
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data, psizes=[(173, 173)])

    def test_code_block_dimensions(self):
        """
        don't allow extreme codeblock sizes
        """
        # opj_compress doesn't allow the dimensions of a codeblock
        # to be too small or too big, so neither will we.
        data = self.j2k_data
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            # opj_compress doesn't allow code block area to exceed 4096.
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data, cbsize=(256, 256))

            # opj_compress doesn't allow either dimension to be less than 4.
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data, cbsize=(2048, 2))
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data, cbsize=(2, 2048))

    def test_psnr_with_cratios(self):
        """
        Using psnr with cratios options is not allowed.
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data, psnr=[30, 35, 40],
                     cratios=[2, 3, 4])

    def test_cinema2K_bad_frame_rate(self):
        """
        Cinema2k frame rate must be either 24 or 48.

        Original test input file was
        input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data, cinema2k=36)

    def test_irreversible(self):
        """
        Verify that the Irreversible option works
        """
        expdata = self.j2k_data
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=expdata, irreversible=True, numres=5)

            codestream = j.get_codestream()
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

            actdata = j[:]
            self.assertTrue(fixtures.mse(actdata, expdata) < 0.28)

    def test_shape_greyscale_jp2(self):
        """verify shape attribute for greyscale JP2 file
        """
        jp2 = Jp2k(self.single_channel_jp2.name)
        self.assertEqual(jp2.shape, (800, 480))
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_shape_single_channel_j2k(self):
        """verify shape attribute for single channel J2K file
        """
        j2k = Jp2k(self.single_channel_j2k.name)
        self.assertEqual(j2k.shape, (800, 480))

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
            self.assertEqual(codestream.segment[2].code_block_size, (16, 32))

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
        data = np.zeros((128, 128, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
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
            self.assertEqual(codestream.segment[2].mct, 0)  # no mct

    def test_write_grayscale_with_mct(self):
        """
        MCT usage makes no sense for grayscale images.
        """
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
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.CPRL)


class TestJp2k_1_x(unittest.TestCase):
    """Test suite for openjpeg 1.x, not appropriate for 2.x"""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

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
        with patch('glymur.version.openjpeg_version', new="1.5.0"):
            with patch('glymur.version.openjpeg_version_tuple', new=(1, 5, 0)):
                j2k = Jp2k(self.j2kfile)
                with self.assertRaises(IOError):
                    j2k.layer = 1

    @unittest.skipIf(((glymur.lib.openjpeg.OPENJPEG is None) or
                      (glymur.lib.openjpeg.version() < '1.5.0')),
                     "OpenJPEG version one must be present")
    def test_read_version_15(self):
        """
        Test read using version 1.5
        """
        j = Jp2k(self.j2kfile)
        expected = j[:]
        with patch('glymur.jp2k.opj2.OPENJP2', new=None):
            actual = j._read_openjpeg()
            np.testing.assert_array_equal(actual, expected)

            actual = j._read_openjpeg(area=(0, 0, 250, 250))
            np.testing.assert_array_equal(actual, expected[:250, :250])

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
        data = np.zeros((128, 128, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
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


@unittest.skipIf(glymur.version.openjpeg_version < '2.0.0',
                 "Not to be run until unless 2.0.1 or higher is present")
class TestJp2k_2_1(unittest.TestCase):
    """Only to be run in 2.0+."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()
        self.jpxfile = glymur.data.jpxfile()

    def tearDown(self):
        pass

    def test_ignore_pclr_cmap_cdef_on_old_read(self):
        """
        The old "read" interface allowed for passing ignore_pclr_cmap_cdef
        to read a palette dataset "uninterpolated".
        """
        jpx = Jp2k(self.jpxfile)
        jpx.ignore_pclr_cmap_cdef = True
        expected = jpx[:]

        jpx2 = Jp2k(self.jpxfile)
        with warnings.catch_warnings():
            # Ignore a deprecation warning.
            warnings.simplefilter('ignore')
            actual = jpx2.read(ignore_pclr_cmap_cdef=True)

        np.testing.assert_array_equal(actual, expected)

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

    @unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
    def test_openjpeg_library_message(self):
        """
        Verify the error message produced by the openjpeg library
        """
        # This will confirm that the error callback mechanism is working.
        with open(self.jp2file, 'rb') as fptr:
            data = fptr.read()
            with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
                # Codestream starts at byte 3323. SIZ marker at 3233.
                # COD marker at 3282.  Subsampling at 3276.
                offset = 3223
                tfile.write(data[0:offset + 52])

                # Make the DY bytes of the SIZ segment zero.  That means that
                # a subsampling factor is zero, which is illegal.
                tfile.write(b'\x00')
                tfile.write(data[offset + 53:offset + 55])
                tfile.write(b'\x00')
                tfile.write(data[offset + 57:offset + 59])
                tfile.write(b'\x00')

                tfile.write(data[offset + 59:])
                tfile.flush()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    j = Jp2k(tfile.name)
                    with self.assertRaises((IOError, OSError)):
                        j[::2, ::2]


class TestParsing(unittest.TestCase):
    """
    Tests for verifying how parsing may be altered.
    """
    def setUp(self):
        self.jp2file = glymur.data.nemo()
        # Reset parseoptions for every test.
        glymur.set_option('parse.full_codestream', False)

    def tearDown(self):
        glymur.set_option('parse.full_codestream', False)

    def test_main_header(self):
        """verify that the main header isn't loaded during normal parsing"""
        # The hidden _main_header attribute should show up after accessing it.
        jp2 = Jp2k(self.jp2file)
        jp2c = jp2.box[4]
        self.assertIsNone(jp2c._codestream)
        jp2c.codestream
        self.assertIsNotNone(jp2c._codestream)


@unittest.skipIf(re.match(r'''0|1|2.0.0''',
                          glymur.version.openjpeg_version) is not None,
                 "Only supported in 2.0.1 or higher")
class TestReadArea(unittest.TestCase):
    """
    Runs tests introduced in version 2.0+ or that pass only in 2.0+

    These tests are only slightly different than their counterparts in the
    OpenJPEG test suite.  The difference is in the file that is tested and
    their extents.  The purpose is the same, though.
    """
    @classmethod
    def setUpClass(self):

        self.j2k = Jp2k(glymur.data.goodstuff())
        self.j2k_data = self.j2k[:]
        self.j2k_half_data = self.j2k[::2, ::2]
        self.j2k_quarter_data = self.j2k[::4, ::4]

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
        expected = self.j2k_quarter_data
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_51_decode(self):
        """
        NR_DEC_p1_04_j2k_51_decode

        Original extents were

        actual = self.j2k[640:768:4, 512:640:4]
        expected = self.j2k_quarter_data[160:192, 128:160]

        Just needed to shift the columns to the left to make it work with
        our own image.
        """
        actual = self.j2k[640:768:4, 256:384:4]
        expected = self.j2k_quarter_data[160:192, 64:96]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_53_decode(self):
        actual = self.j2k[500:800:4, 100:300:4]
        expected = self.j2k_quarter_data[125:200, 25:75]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_54_decode(self):
        actual = self.j2k[520:600:4, 260:360:4]
        expected = self.j2k_quarter_data[130:150, 65:90]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_55_decode(self):
        actual = self.j2k[520:660:4, 260:360:4]
        expected = self.j2k_quarter_data[130:165, 65:90]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p1_04_j2k_56_decode(self):
        actual = self.j2k[520:600:4, 360:400:4]
        expected = self.j2k_quarter_data[130:150, 90:100]
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
        # Image size would be 0 x 0.
        with self.assertRaises((IOError, OSError)):
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
        expected = self.j2k_quarter_data[0:64, 0:64]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_92_decode(self):
        actual = self.j2k[:128:4, 128:256:4]
        expected = self.j2k_quarter_data[:32, 32:64]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_93_decode(self):
        actual = self.j2k[10:200:4, 50:120:4]
        expected = self.j2k_quarter_data[3:50, 13:30]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_94_decode(self):
        actual = self.j2k[150:210:4, 10:190:4]
        expected = self.j2k_quarter_data[38:53, 3:48]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_95_decode(self):
        actual = self.j2k[80:150:4, 100:200:4]
        expected = self.j2k_quarter_data[20:38, 25:50]
        np.testing.assert_array_equal(actual, expected)

    def test_NR_DEC_p0_04_j2k_96_decode(self):
        actual = self.j2k[20:50:4, 150:200:4]
        expected = self.j2k_quarter_data[5:13, 38:50]
        np.testing.assert_array_equal(actual, expected)
