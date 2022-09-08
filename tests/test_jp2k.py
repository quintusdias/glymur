"""
Tests for general glymur functionality.
"""
# Standard library imports ...
import collections
import datetime
import doctest
import importlib.resources as ir
from io import BytesIO
import os
import pathlib
import shutil
import struct
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid
import warnings

# Third party library imports ...
from lxml import etree as ET
import numpy as np

# Local imports
import glymur
from glymur import Jp2k
from glymur.jp2box import InvalidJp2kError
from glymur.core import COLOR, RED, GREEN, BLUE, RESTRICTED_ICC_PROFILE

from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG

from . import fixtures, data


def docTearDown(doctest_obj):  # pragma: no cover
    glymur.set_option('parse.full_codestream', False)


# Doc tests should be run as well.
def load_tests(loader, tests, ignore):  # pragma: no cover
    """Should run doc tests as well"""
    if os.name == "nt":
        # Can't do it on windows, temporary file issue.
        return tests
    if glymur.lib.openjp2.OPENJP2 is not None:
        tests.addTests(
            doctest.DocTestSuite('glymur.jp2k', tearDown=docTearDown)
        )
    return tests


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(glymur.version.openjpeg_version < '2.3.0',
                 "Requires as least v2.3.0")
class TestJp2k(fixtures.TestCommon):
    """These tests should be run by just about all configuration."""

    def setUp(self):
        super(TestJp2k, self).setUp()
        glymur.reset_option('all')

    def test_dtype_jp2(self):
        """
        Scenario:  An RGB image is read from a JP2 file.

        Expected response:  the dtype property is np.uint8
        """
        j = Jp2k(self.jp2file)
        self.assertEqual(j.dtype, np.uint8)

    def test_dtype_j2k_uint16(self):
        """
        Scenario:  A uint16 monochrome image is read from a J2K file.

        Expected response:  the dtype property is np.uint16
        """
        with ir.path('tests.data', 'uint16.j2k') as path:
            j = Jp2k(path)
        self.assertEqual(j.dtype, np.uint16)

    def test_cod_segment_not_3rd(self):
        """
        Scenario:  Normally the COD segment is the 3rd segment.
        Here it is 4th.  Read the image.

        Expected response:  No errors.
        """
        j = Jp2k(self.j2kfile)
        j.codestream.segment.insert(2, j.codestream.segment[1])
        j[::2, ::2]

    def test_dtype_prec4_signd1(self):
        """
        Scenario:  A 4-bit signed image is read from a J2k file.

        Expected response:  the dtype property is np.int8
        """
        with ir.path('tests.data', 'p0_03.j2k') as path:
            j = Jp2k(path)
        self.assertEqual(j.dtype, np.int8)

    def test_dtype_inconsistent_bitdetph(self):
        """
        Scenario:  The image has different bitdepths in different components.

        Expected response:  TypeError when accessing the dtype property.
        """
        with ir.path('tests.data', 'issue982.j2k') as path:
            j = Jp2k(path)
        with self.assertRaises(TypeError):
            j.dtype

    def test_ndims_jp2(self):
        """
        Scenario:  An RGB image is read from a JP2 file.

        Expected response:  the ndim attribute/property is 3
        """
        j = Jp2k(self.jp2file)
        self.assertEqual(j.ndim, 3)

    def test_ndims_j2k(self):
        """
        Scenario:  An RGB image is read from a raw codestream.

        Expected response:  the ndim attribute/property is 3
        """
        j = Jp2k(self.j2kfile)
        self.assertEqual(j.ndim, 3)

    def test_ndims_monochrome_j2k(self):
        """
        Scenario:  An monochrome image is read from a raw codestream.

        Expected response:  the ndim attribute/property is 2
        """
        with ir.path('tests.data', 'p0_02.j2k') as path:
            j = Jp2k(path)
        self.assertEqual(j.ndim, 2)

    def test_read_bands_unequal_subsampling(self):
        """
        SCENARIO:  The read_bands method is used on an image with unequal
        subsampling.

        EXPECTED RESPONSE: The image is a list of arrays of unequal size.
        """
        with ir.path(data, 'p0_06.j2k') as path:
            d = Jp2k(path).read_bands()

        actual = [band.shape for band in d]
        expected = [(129, 513), (129, 257), (65, 513), (65, 257)]
        self.assertEqual(actual, expected)

    def test_read_bands(self):
        """
        SCENARIO:  The read_bands method really should only be used on images
        with different subsampling values.  But for backwards compatibility
        it also reads images with the same subsampling value.  Read data via
        both the slicing protocol and the read_bands method.

        EXPECTED RESULT: The shape of the data read via the slicing
        protocol should be the same as the shape read by the
        read_bands method.
        """
        j = Jp2k(self.jp2file)
        d1 = j[:]
        d2 = j.read_bands()
        self.assertEqual(d1.shape, d2.shape)

    def test_pathlib(self):
        """
        SCENARIO: Provide a pathlib.Path instead of a string for the filename.
        """
        p = pathlib.Path(self.jp2file)
        jp2 = Jp2k(p)
        self.assertEqual(jp2.shape, (1456, 2592, 3))

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

    def test_no_cxform_cmap(self):
        """
        SCENARIO:  Write an RGB image as a JP2 file, but with a colr box
        specifying the components in reverse order.  The codestream is the
        same, it's just the colr box that is different.

        EXPECTED RESULT:  The output image has bands reversed from the input
        image.
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
        cdef = glymur.jp2box.ChannelDefinitionBox(
            channel_type=channel_type, association=association
        )
        jp2h.box.append(cdef)

        boxes.append(jp2h)
        boxes.append(glymur.jp2box.ContiguousCodestreamBox())

        # Write the image back out with reversed definition.  The codestream
        # is the same, it's just the JP2 wrapper that is different.
        jp2 = j2k.wrap(self.temp_jp2_filename, boxes=boxes)

        jp2.ignore_pclr_cmap_cdef = False
        bgr = jp2[:]

        np.testing.assert_array_equal(rgb, bgr[:, :, [2, 1, 0]])

    def test_bad_tile_part_pointer(self):
        """
        SCENARIO:  A bad SOT marker segment is encountered (Psot value pointing
        far beyond the end of the EOC marker) when requesting a fully parsed
        codestream.

        EXPECTED RESULT:  struct.error
        """
        with open(self.temp_jp2_filename, 'wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Copy up until Psot field.
                ofile.write(ifile.read(3350))

                # Write a bad Psot value.
                ofile.write(struct.pack('>I', 2000000))

                # copy the rest of the file as-is.
                ifile.seek(3354)
                ofile.write(ifile.read())
                ofile.flush()

        j = Jp2k(self.temp_jp2_filename)
        with self.assertRaises(struct.error):
            j.get_codestream(header_only=False)

    def test_read_differing_subsamples(self):
        """
        SCENARIO:  Attempt to read a file where the components have differing
        subsampling.  This causes the decoded components to have different
        sizes.

        EXPECTED RESULT:  RuntimeError
        """
        # copy nemo.jp2 but change the SIZ segment to have differing subsamples
        with open(self.temp_jp2_filename, mode='wb') as ofile:
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
            with self.assertRaises(RuntimeError):
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

    def test_irreversible(self):
        """
        SCENARIO:  Write a J2K file with the irreversible transform.

        EXPECTED RESULT:  the 9-7 wavelet transform is detected in the
        codestream.
        """
        j = Jp2k(self.jp2file)
        expdata = j[:]
        j2 = Jp2k(self.temp_j2k_filename, data=expdata, irreversible=True)

        codestream = j2.get_codestream()
        self.assertEqual(
            codestream.segment[2].xform,
            glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE
        )

        actdata = j2[:]
        self.assertTrue(fixtures.mse(actdata[0], expdata[0]) < 0.38)

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
        with self.assertRaises(ValueError):
            j[::64, ::64]

    def test_not_jpeg2000(self):
        """
        SCENARIO:  The Jp2k constructor is passed a file that is not JPEG 2000.

        EXPECTED RESULT:  RuntimeError
        """
        with ir.path(glymur, 'jp2k.py') as path:
            with self.assertRaises(InvalidJp2kError):
                Jp2k(path)

    def test_file_does_not_exist(self):
        """
        Scenario:  The Jp2k construtor is passed a file that does not exist
        and the intent is reading.

        Expected Result:  FileNotFoundError
        """
        # Verify that we error out appropriately if not given an existing file
        # at all.
        filename = 'this file does not actually exist on the file system.'
        j = Jp2k(filename)
        with self.assertRaises(FileNotFoundError):
            j[:]

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_write_to_a_file_using_context_manager(self):
        """
        SCENARIO:  Write to a file using a context manager, read the data back
        using a context manager.

        EXPECTED RESULT: the data matches
        """
        expected = fixtures.skimage.data.astronaut()

        j1 = Jp2k(self.temp_jp2_filename)
        j1[:] = expected

        j2 = Jp2k(self.temp_jp2_filename)
        actual = j2[:]

        np.testing.assert_array_equal(actual, expected)

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

    def test_64bit_xl_field(self):
        """
        SCENARIO:  A JP2 file is encountered with a jp2c file with the XL field
        properly set.

        EXPECTED RESULT:  The file should parse and be read without errors.  In
        particular, the jp2c box should be 8 bytes longer than in the original.
        """
        # Don't have such a file on hand, so we create one.  Copy our example
        # file, but making the codestream have a 64-bit XL field.
        with open(self.temp_jp2_filename, mode='wb') as tfile:
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

    def test_length_field_is_zero(self):
        """
        SCENARIO:  A JP2 file has in its last box and L field with value 0.

        EXPECTED RESULT:  The file is parsed without error.  In particular, the
        length of that last box is correctly computed.
        """
        # Verify that boxes with the L field as zero are correctly read.
        # This should only happen in the last box of a JPEG 2000 file.
        # Our example image has its last box at byte 588458.
        baseline_jp2 = Jp2k(self.jp2file)
        with open(self.temp_jp2_filename, mode='wb') as tfile:
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

    def test_uinf_ulst_url_boxes(self):
        """
        SCENARIO:  A JP2 file with UINF, ULST, and URL boxes is encountered.

        EXPECTED RESULT:  The file is parsed without error.
        """
        # Must create the file.
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                write_buffer = ifile.read(77)
                tfile.write(write_buffer)

                # Write the UINF superbox
                # Length = 50, id is uinf.
                uinf_len = 50
                write_buffer = struct.pack('>I4s', int(uinf_len), b'uinf')
                tfile.write(write_buffer)

                # Write the ULST box.
                # Length is 26, 1 UUID, hard code that UUID as zeros.
                ulst_len = 26
                write_buffer = struct.pack(
                    '>I4sHIIII',
                    ulst_len, b'ulst', int(1), int(0), int(0), int(0), int(0)
                )
                tfile.write(write_buffer)

                # Write the URL box.
                # Length is 16, version is one byte, flag is 3 bytes, url
                # is the rest.
                url_box_len = 16
                write_buffer = struct.pack(
                    '>I4sBBBB',
                    url_box_len, b'url ', int(0), int(0), int(0), int(0)
                )
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
            self.assertEqual(jp2k.box[3].length, uinf_len)

            self.assertEqual(jp2k.box[3].box[0].box_id, 'ulst')
            self.assertEqual(jp2k.box[3].box[0].offset, 85)
            self.assertEqual(jp2k.box[3].box[0].length, ulst_len)
            ulst = []
            ulst.append(uuid.UUID('00000000-0000-0000-0000-000000000000'))
            self.assertEqual(jp2k.box[3].box[0].ulst, ulst)

            self.assertEqual(jp2k.box[3].box[1].box_id, 'url ')
            self.assertEqual(jp2k.box[3].box[1].offset, 111)
            self.assertEqual(jp2k.box[3].box[1].length, url_box_len)
            self.assertEqual(jp2k.box[3].box[1].version, 0)
            self.assertEqual(jp2k.box[3].box[1].flag, (0, 0, 0))
            self.assertEqual(jp2k.box[3].box[1].url, 'abcd')

    def test_xml_with_trailing_nulls(self):
        """
        SCENARIO:  An xml box is encountered with null chars trailing the valid
        XML.  This causes problems for ElementTree.

        EXPECTED RESULT:  The xml box is parsed without issue and the original
        XML is recovered.
        """
        with open(self.temp_jp2_filename, mode='wb') as tfile:
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
        exp_error = RuntimeError
        with patch('glymur.version.openjpeg_version_tuple', new=(1, 5, 0)):
            with patch('glymur.version.openjpeg_version', new='1.5.0'):
                with self.assertRaises(exp_error):
                    glymur.Jp2k(self.jp2file).read_bands()

    def test_zero_length_reserved_segment(self):
        """
        SCENARIO:  There is a zero-length reserved marker segment just before
        the EOC marker segment.  It is unclear to me if this is valid or not.
        It looks valid.

        EXPECTED RESULT:  The file is parsed without error and the zero-length
        segment is detected in the codestream.  No warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
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
                buffer = struct.pack('>BBH', 0xff, 0x00, 0)
                ofile.write(buffer)

                # Write the EOC marker and be done with it.
                ofile.write(ifile.read())
                ofile.flush()

        cstr = Jp2k(ofile.name).get_codestream(header_only=False)
        self.assertEqual(cstr.segment[11].marker_id, '0xff00')
        self.assertEqual(cstr.segment[11].length, 0)

    def test_psot_is_zero(self):
        """
        SCENARIO:  An SOT marker segment is encountered with a Psot value of 0.
        GH #98.

        EXPECTED RESULT:  The file should parse without error.  The SOT marker
        should be detected.
        """
        with open(self.temp_j2k_filename, mode='wb') as ofile:
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
        profile = box.icc_profile_header

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

        np.testing.assert_almost_equal(
            profile['Illuminant'],
            (0.964203, 1.000000, 0.824905),
            decimal=6
        )

        self.assertEqual(profile['Creator'], 'JPEG')

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_different_layers(self):
        """
        SCENARIO:  Set the layer property to specify the 2nd layer.

        EXPECTED RESULT:  The 2nd image read in is not the same as the first.
        """
        with ir.path(data, 'p0_03.j2k') as path:
            j = Jp2k(path)
        d0 = j[:]

        j.layer = 1
        d1 = j[:]

        np.alltrue(d0 != d1)

    def test_invalid_layers(self):
        """
        SCENARIO:  an improper layer value is set

        EXPECTED RESULT:  RuntimeError when an invalid layer number is supplied
        """
        # There are 8 layers, so only values [0-7] are valid.
        with ir.path(data, 'p0_03.j2k') as path:
            j = Jp2k(path)

        with self.assertRaises(ValueError):
            j.layer = -1

        for layer in range(8):
            # 0-7 are all valid.
            j.layer

        with self.assertRaises(ValueError):
            j.layer = 8

    def test_default_verbosity(self):
        """
        SCENARIO:  Check the default verbosity property.

        EXPECTED RESULT:  The default verbosity setting is False.
        """
        with ir.path(data, 'p0_03.j2k') as path:
            j = Jp2k(path)

        self.assertFalse(j.verbose)

    def test_default_layer(self):
        """
        SCENARIO:  Check the default layer property.

        EXPECTED RESULT:  The default layer property value is 0.
        """
        with ir.path(data, 'p0_03.j2k') as path:
            j = Jp2k(path)

        self.assertEqual(j.layer, 0)

    @unittest.skipIf(os.cpu_count() < 4, "makes no sense if 4 cores not there")
    def test_thread_support(self):
        """
        SCENARIO:  Set a non-default thread support value.

        EXPECTED RESULTS:  Using more threads speeds up a full read.
        """
        jp2 = Jp2k(self.jp2file)
        t0 = time.time()
        jp2[:]
        t1 = time.time()
        delta0 = t1 - t0

        glymur.set_option('lib.num_threads', 4)
        t0 = time.time()
        jp2[:]
        t1 = time.time()
        delta1 = t1 - t0

        self.assertTrue(delta1 < delta0)

    @unittest.skipIf(os.cpu_count() < 4, "makes no sense if 4 cores not there")
    def test_thread_support_on_openjpeg_lt_220(self):
        """
        SCENARIO:  Set number of threads on openjpeg < 2.2.0

        EXPECTED RESULTS:  RuntimeError
        """
        with patch('glymur.jp2k.version.openjpeg_version', new='2.1.0'):
            with self.assertRaises(RuntimeError):
                glymur.set_option('lib.num_threads', 4)

    @unittest.skipIf(os.cpu_count() < 4, "makes no sense if 4 cores not there")
    @patch('glymur.lib.openjp2.has_thread_support')
    def test_thread_support_not_compiled_into_library(self, mock_ts):
        """
        SCENARIO:  Set number of threads on openjpeg >= 2.2.0, but openjpeg
        has not been compiled with thread support.

        EXPECTED RESULTS:  RuntimeError
        """
        mock_ts.return_value = False
        with patch('glymur.jp2k.version.openjpeg_version', new='2.2.0'):
            with self.assertRaises(RuntimeError):
                glymur.set_option('lib.num_threads', 4)


class TestComponent(unittest.TestCase):
    """
    Test how a component's precision translates into a datatype.
    """
    @classmethod
    def setUpClass(cls):
        cls.jp2file = glymur.data.nemo()

    def test_nbits_lt_9(self):
        """
        SCENARIO:  A layer has less than 9 bits per sample

        EXPECTED RESULT:  np.int8
        """
        j = Jp2k(self.jp2file)

        # Fake a data structure that resembles the openjpeg component.
        Component = collections.namedtuple('Component', ['prec', 'sgnd'])
        c = Component(prec=7, sgnd=True)
        dtype = j._component2dtype(c)
        self.assertEqual(dtype, np.int8)

    def test_nbits_lt_16_gt_8(self):
        """
        SCENARIO:  A layer has between 9 and 16 bits per sample.

        EXPECTED RESULT:  np.int16
        """
        j = Jp2k(self.jp2file)

        # Fake a data structure that resembles the openjpeg component.
        Component = collections.namedtuple('Component', ['prec', 'sgnd'])
        c = Component(prec=15, sgnd=True)
        dtype = j._component2dtype(c)
        self.assertEqual(dtype, np.int16)

    def test_nbits_gt_16(self):
        """
        SCENARIO:  One of the layers has more than 16 bits per sample.

        EXPECTED RESULT:  RuntimeError
        """
        j = Jp2k(self.jp2file)

        # Fake a data structure that resembles the openjpeg component.
        Component = collections.namedtuple('Component', ['prec', 'sgnd'])
        c = Component(prec=17, sgnd=True)
        with self.assertRaises(ValueError):
            j._component2dtype(c)


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestJp2k_write(fixtures.MetadataBase):
    """Test writing Jpeg2000 files"""

    @classmethod
    def setUpClass(cls):
        cls.jp2file = glymur.data.nemo()
        cls.j2kfile = glymur.data.goodstuff()

        cls.j2k_data = glymur.Jp2k(cls.j2kfile)[:]
        cls.jp2_data = glymur.Jp2k(cls.jp2file)[:]

        # Make single channel jp2 and j2k files.
        test_dir = tempfile.mkdtemp()
        test_dir_path = pathlib.Path(test_dir)

        cls.single_channel_j2k = test_dir_path / 'single_channel.j2k'
        glymur.Jp2k(cls.single_channel_j2k, data=cls.j2k_data[:, :, 0])

        cls.single_channel_jp2 = test_dir_path / 'single_channel.jp2'
        glymur.Jp2k(cls.single_channel_jp2, data=cls.j2k_data[:, :, 0])

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.single_channel_j2k)
        os.unlink(cls.single_channel_jp2)

    @unittest.skipIf(os.cpu_count() < 2, "makes no sense if 2 cores not there")
    def test_threads(self):
        """
        SCENARIO:  Attempt to encode with threading support.  This feature is
        new as of openjpeg library version 2.4.0.

        EXPECTED RESULT:  In library versions prior to 2.4.0, a warning is
        issued.
        """
        glymur.set_option('lib.num_threads', 2)
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with warnings.catch_warnings(record=True) as w:
                Jp2k(tfile.name, data=self.jp2_data)
                if glymur.version.openjpeg_version >= '2.4.0':
                    self.assertEqual(len(w), 0)
                else:
                    self.assertEqual(len(w), 1)

    def test_capture_resolution(self):
        """
        SCENARIO:  The capture_resolution keyword is specified.

        EXPECTED RESULT:  The cres box is created.
        """
        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4
        j = glymur.Jp2k(
            self.temp_jp2_filename, data=self.jp2_data,
            capture_resolution=[vresc, hresc],
            display_resolution=[vresd, hresd],
        )

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resc')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresc)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresc)

        self.assertEqual(j.box[2].box[2].box[1].box_id, 'resd')
        self.assertEqual(j.box[2].box[2].box[1].vertical_resolution, vresd)
        self.assertEqual(j.box[2].box[2].box[1].horizontal_resolution, hresd)

    def test_capture_resolution_when_j2k_specified(self):
        """
        Scenario:  Capture/Display resolution boxes are specified when the file
        name indicates J2K.

        Expected Result:  InvalidJp2kError
        """

        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4
        with self.assertRaises(InvalidJp2kError):
            glymur.Jp2k(
                self.temp_j2k_filename, data=self.jp2_data,
                capture_resolution=[vresc, hresc],
                display_resolution=[vresd, hresd],
            )

    def test_capture_resolution_when_not_writing(self):
        """
        Scenario:  Jp2k is invoked in a read-only situation but capture/display
        resolution arguments are supplied.

        Expected result:  RuntimeError
        """
        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4

        shutil.copyfile(self.jp2file, self.temp_jp2_filename)

        with self.assertRaises(RuntimeError):
            glymur.Jp2k(
                self.temp_jp2_filename,
                capture_resolution=[vresc, hresc],
                display_resolution=[vresd, hresd],
            )

    def test_capture_resolution_supplied_but_not_display(self):
        """
        Scenario:  Writing a JP2 is intended, but only a capture resolution
        box is specified, and not a display resolution box.

        Expected Result:  No errors, the boxes are validated.
        """
        vresc, hresc = 0.1, 0.2

        j = glymur.Jp2k(
            self.temp_jp2_filename, data=self.jp2_data,
            capture_resolution=[vresc, hresc],
        )

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resc')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresc)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresc)

        # there's just one child box
        self.assertEqual(len(j.box[2].box[2].box), 1)

    def test_display_resolution_supplied_but_not_capture(self):
        """
        Scenario:  Writing a JP2 is intended, but only a capture resolution
        box is specified, and not a display resolution box.

        Expected Result:  No errors, the boxes are validated.
        """
        vresd, hresd = 0.3, 0.4

        j = glymur.Jp2k(
            self.temp_jp2_filename, data=self.jp2_data,
            display_resolution=[vresd, hresd],
        )

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resd')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresd)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresd)

        # there's just one child box
        self.assertEqual(len(j.box[2].box[2].box), 1)

    def test_no_jp2c_box_in_outermost_jp2_list(self):
        """
        SCENARIO:  A JP2 file is encountered without a JP2C box in the outer-
        most list of boxes.

        EXPECTED RESULT:  RuntimeError
        """
        j = glymur.Jp2k(self.jp2file)

        # Remove the last box, which is a codestream.
        boxes = j.box[:-1]

        with open(self.temp_jp2_filename, mode="wb") as tfile:
            with self.assertRaises(RuntimeError):
                j.wrap(tfile.name, boxes=boxes)

    def test_null_data(self):
        """
        SCENARIO:  An image with a dimension with length 0 is provided.

        EXPECTED RESULT:  RuntimeError
        """
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(InvalidJp2kError):
                Jp2k(tfile.name, data=np.zeros((0, 256), dtype=np.uint8))

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_psnr_zero_value_not_last(self):
        """
        SCENARIO:  The PSNR keyword argument has a zero value, but it is not
        the last value.

        EXPECTED RESULT:  RuntimeError
        """
        kwargs = {
            'data': fixtures.skimage.data.camera(),
            'psnr': [0, 35, 40, 30],
        }
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_jp2_filename, **kwargs)

    @unittest.skipIf(glymur.version.openjpeg_version < '2.5.0',
                     "Requires as least v2.5.0")
    def test_tlm_yes(self):
        """
        SCENARIO:  Use the tlm keyword.

        EXPECTED RESULT:  A TLM segment is detected.
        """
        kwargs = {
            'data': self.jp2_data,
            'tlm': True
        }
        j = Jp2k(self.temp_jp2_filename, **kwargs)

        codestream = j.get_codestream(header_only=False)

        at_least_one_tlm_segment = any(
            isinstance(seg, glymur.codestream.TLMsegment)
            for seg in codestream.segment
        )
        self.assertTrue(at_least_one_tlm_segment)

    def test_tlm_no(self):
        """
        SCENARIO:  Use the tlm keyword set to False

        EXPECTED RESULT:  A TLM segment not detected.
        """
        kwargs = {
            'data': self.jp2_data,
            'tlm': False
        }
        j = Jp2k(self.temp_jp2_filename, **kwargs)

        codestream = j.get_codestream(header_only=False)

        at_least_one_tlm_segment = any(
            isinstance(seg, glymur.codestream.TLMsegment)
            for seg in codestream.segment
        )
        self.assertFalse(at_least_one_tlm_segment)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_plt_yes(self):
        """
        SCENARIO:  Use the plt keyword.

        EXPECTED RESULT:  Plt segment is detected.
        """
        kwargs = {
            'data': fixtures.skimage.data.camera(),
            'plt': True
        }
        j = Jp2k(self.temp_jp2_filename, **kwargs)

        codestream = j.get_codestream(header_only=False)

        at_least_one_plt = any(
            isinstance(seg, glymur.codestream.PLTsegment)
            for seg in codestream.segment
        )
        self.assertTrue(at_least_one_plt)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_plt_no(self):
        """
        SCENARIO:  Use the plt keyword set to false.

        EXPECTED RESULT:  Plt segment is not detected.
        """
        kwargs = {
            'data': fixtures.skimage.data.camera(),
            'plt': False
        }
        j = Jp2k(self.temp_jp2_filename, **kwargs)

        codestream = j.get_codestream(header_only=False)

        at_least_one_plt = any(
            isinstance(seg, glymur.codestream.PLTsegment)
            for seg in codestream.segment
        )
        self.assertFalse(at_least_one_plt)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_psnr_non_zero_non_monotonically_decreasing(self):
        """
        SCENARIO:  The PSNR keyword argument is non-monotonically increasing
        and does not contain zero.

        EXPECTED RESULT:  RuntimeError
        """
        kwargs = {
            'data': fixtures.skimage.data.camera(),
            'psnr': [30, 35, 40, 30],
        }
        with self.assertRaises(RuntimeError):
            Jp2k(self.temp_jp2_filename, **kwargs)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_psnr(self):
        """
        SCENARIO:  Four peak signal-to-noise ratio values are supplied, the
        last is zero.

        EXPECTED RESULT:  Four quality layers, the first should be lossless.
        """
        kwargs = {
            'data': fixtures.skimage.data.camera(),
            'psnr': [30, 35, 40, 0],
        }
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            j = Jp2k(tfile.name, **kwargs)

            d = {}
            for layer in range(4):
                j.layer = layer
                d[layer] = j[:]

        with warnings.catch_warnings():
            # MSE is zero for that first image, resulting in a divide-by-zero
            # warning
            warnings.simplefilter('ignore')
            psnr = [
                fixtures.skimage.metrics.peak_signal_noise_ratio(
                    fixtures.skimage.data.camera(), d[j]
                )
                for j in range(4)
            ]

        # That first image should be lossless.
        self.assertTrue(np.isinf(psnr[0]))

        # None of the subsequent images should have inf PSNR.
        self.assertTrue(not np.any(np.isinf(psnr[1:])))

        # PSNR should increase for the remaining images.
        self.assertTrue(np.all(np.diff(psnr[1:])) > 0)

    def test_NR_ENC_Bretagne1_ppm_2_encode(self):
        """
        SCENARIO:  Three peak signal-to-noise ratio values, two resolutions are
        supplied.

        EXPECTED RESULT:  Three quality layers, two resolutions.
        """
        kwargs = {
            'data': self.jp2_data,
            'psnr': [30, 35, 40],
            'numres': 2,
        }
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            j = Jp2k(tfile.name, **kwargs)

            codestream = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res + 1, 2)  # levels
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
        SCENARIO:  Create a JP2 image with three compression ratios.

        EXPECTED RESULT:  There are three layers.
        """
        data = self.jp2_data
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            # Should be written with 3 layers.
            j = Jp2k(tfile.name, data=data, cratios=[200, 100, 50])
            c = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].mct, 1)  # mct
        self.assertEqual(c.segment[2].num_res + 1, 6)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self.verify_codeblock_style(c.segment[2].cstyle,
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, ((32768, 32768)))

    def test_NR_ENC_Bretagne1_ppm_3_encode(self):
        """
        SCENARIO:  Three peak signal to noise rations are provided, along with
        specific code block sizes and precinct sizes.

        EXPECTED RESULT:  Three quality layers and the specified code block
        size are present.  The precinct sizes validate.
        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            j = Jp2k(
                tfile.name,
                data=self.jp2_data,
                psnr=[30, 35, 40],
                cbsize=(16, 16), psizes=[(64, 64)]
            )

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
        self.verify_codeblock_style(
            codestream.segment[2].cstyle,
            [False, False, False, False, False, False]
        )
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(
            codestream.segment[2].precinct_size,
            ((2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64))
        )

    def test_NR_ENC_Bretagne2_ppm_4_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            j = Jp2k(
                tfile.name,
                data=self.jp2_data,
                psizes=[(128, 128)] * 3,
                cratios=[100, 20, 2],
                tilesize=(480, 640),
                cbsize=(32, 32)
            )

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
            self.assertEqual(
                tuple(codestream.segment[2].code_block_size),
                (32, 32)
            )  # cblksz
            self.verify_codeblock_style(
                codestream.segment[2].cstyle,
                [False, False, False, False, False, False]
            )
            self.assertEqual(
                codestream.segment[2].xform,
                glymur.core.WAVELET_XFORM_5X3_REVERSIBLE
            )
            self.assertEqual(
                codestream.segment[2].precinct_size,
                (
                    (16, 16), (32, 32), (64, 64), (128, 128), (128, 128),
                    (128, 128)
                )
            )

    def test_NR_ENC_Bretagne2_ppm_5_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
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
            self.assertEqual(codestream.segment[2].layers, 1)
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(
                codestream.segment[2].cstyle,
                [False, False, False, False, False, False]
            )
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_Bretagne2_ppm_6_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm
        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
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
        with open(self.temp_j2k_filename, mode='wb') as tfile:
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
        with open(self.temp_j2k_filename, mode='wb') as tfile:
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
        with open(self.temp_j2k_filename, mode='wb') as tfile:
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
        with open(self.temp_j2k_filename, mode='wb') as tfile:

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
        with open(self.temp_jp2_filename, mode='wb') as tfile:

            jp2 = Jp2k(tfile.name, data=self.jp2_data, psnr=[30, 35, 50],
                       prog='LRCP', numres=3)

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

            kwargs = {
                'rsiz': 0,
                'xysiz': (2592, 1456),
                'xyosiz': (0, 0),
                'xytsiz': (2592, 1456),
                'xytosiz': (0, 0),
                'bitdepth': (8, 8, 8),
                'signed': (False, False, False),
                'xyrsiz': [(1, 1, 1), (1, 1, 1)]
            }
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
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            j = Jp2k(tfile.name, data=data)

            codestream = j.get_codestream(header_only=False)

            kwargs = {
                'rsiz': 0,
                'xysiz': (1024, 1024),
                'xyosiz': (0, 0),
                'xytsiz': (1024, 1024),
                'xytosiz': (0, 0),
                'bitdepth': (16,),
                'signed': (False,),
                'xyrsiz': [(1,), (1,)]
            }
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
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, irreversible=True)

            codestream = j.get_codestream()
            self.assertEqual(
                codestream.segment[2].xform,
                glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE
            )

    def test_cinema2K_with_others(self):
        """
        Can't specify cinema2k with any other options.

        Original test file was
        input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif
        """
        data = np.zeros((857, 2048, 3), dtype=np.uint8)
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data,
                     cinema2k=48, cratios=[200, 100, 50])

    def test_cinema4K_with_others(self):
        """
        Can't specify cinema4k with any other options.

        Original test file was input/nonregression/ElephantDream_4K.tif
        """
        data = np.zeros((4096, 2160, 3), dtype=np.uint8)
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data,
                     cinema4k=True, cratios=[200, 100, 50])

    def test_cblk_size_precinct_size(self):
        """
        code block sizes should never exceed half that of precinct size.
        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=self.j2k_data,
                     cbsize=(64, 64), psizes=[(64, 64)])

    def test_cblk_size_not_power_of_two(self):
        """
        code block sizes should be powers of two.
        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=self.j2k_data, cbsize=(13, 12))

    def test_precinct_size_not_p2(self):
        """
        precinct sizes should be powers of two.
        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=self.j2k_data, psizes=[(173, 173)])

    def test_code_block_dimensions(self):
        """
        don't allow extreme codeblock sizes
        """
        # opj_compress doesn't allow the dimensions of a codeblock
        # to be too small or too big, so neither will we.
        data = self.j2k_data
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            # opj_compress doesn't allow code block area to exceed 4096.
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data, cbsize=(256, 256))

            # opj_compress doesn't allow either dimension to be less than 4.
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data, cbsize=(2048, 2))
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data, cbsize=(2, 2048))

    def test_psnr_with_cratios(self):
        """
        Using psnr with cratios options is not allowed.
        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=self.j2k_data, psnr=[30, 35, 40],
                     cratios=[2, 3, 4])

    def test_irreversible(self):
        """
        Verify that the Irreversible option works
        """
        expdata = self.j2k_data
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            j = Jp2k(tfile.name, data=expdata, irreversible=True, numres=5)

            codestream = j.get_codestream()
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

            actdata = j[:]
            self.assertTrue(fixtures.mse(actdata, expdata) < 0.28)

    def test_shape_greyscale_jp2(self):
        """verify shape attribute for greyscale JP2 file
        """
        jp2 = Jp2k(self.single_channel_jp2)
        self.assertEqual(jp2.shape, (800, 480))
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_shape_single_channel_j2k(self):
        """verify shape attribute for single channel J2K file
        """
        j2k = Jp2k(self.single_channel_j2k)
        self.assertEqual(j2k.shape, (800, 480))

    def test_precinct_size_too_small(self):
        """
        SCENARIO:  The first precinct size is less than 2x that of the code
        block size.

        EXPECTED RESULT:  InvalidJp2kError
        """
        data = np.zeros((640, 480), dtype=np.uint8)
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(InvalidJp2kError):
                Jp2k(tfile.name, data=data, cbsize=(16, 16), psizes=[(16, 16)])

    def test_precinct_size_not_power_of_two(self):
        """
        SCENARIO:  A precinct size is specified that is not a power of 2.

        EXPECTED RESULT:  InvalidJp2kError
        """
        data = np.zeros((640, 480), dtype=np.uint8)
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(InvalidJp2kError):
                Jp2k(tfile.name, data=data,
                     cbsize=(16, 16), psizes=[(48, 48)])

    def test_unsupported_int32(self):
        """Should raise a runtime error if trying to write int32"""
        data = np.zeros((128, 128), dtype=np.int32)
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data)

    def test_unsupported_uint32(self):
        """Should raise a runtime error if trying to write uint32"""
        data = np.zeros((128, 128), dtype=np.uint32)
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data)

    def test_write_with_version_too_early(self):
        """Should raise a runtime error if trying to write with version 1.3"""
        data = np.zeros((128, 128), dtype=np.uint8)
        versions = [
            "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "2.0.0",
            "2.1.0", "2.2.0"
        ]
        for version in versions:
            with patch('glymur.version.openjpeg_version', new=version):
                with open(self.temp_j2k_filename, mode='wb') as tfile:
                    with self.assertRaises(RuntimeError):
                        Jp2k(tfile.name, data=data)

    def test_cblkh_different_than_width(self):
        """Verify that we can set a code block size where height does not equal
        width.
        """
        data = np.zeros((128, 128), dtype=np.uint8)
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            # The code block dimensions are given as rows x columns.
            j = Jp2k(tfile.name, data=data, cbsize=(16, 32))
            codestream = j.get_codestream()

            # Code block size is reported as XY in the codestream.
            self.assertEqual(codestream.segment[2].code_block_size, (16, 32))

    def test_too_many_dimensions(self):
        """OpenJP2 only allows 2D or 3D images."""
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name,
                     data=np.zeros((128, 128, 2, 2), dtype=np.uint8))

    def test_2d_rgb(self):
        """RGB must have at least 3 components."""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name,
                     data=np.zeros((128, 128, 2), dtype=np.uint8),
                     colorspace='rgb')

    def test_colorspace_with_j2k(self):
        """Specifying a colorspace with J2K does not make sense"""
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name,
                     data=np.zeros((128, 128, 3), dtype=np.uint8),
                     colorspace='rgb')

    def test_specify_rgb(self):
        """specify RGB explicitly"""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            j = Jp2k(tfile.name,
                     data=np.zeros((128, 128, 3), dtype=np.uint8),
                     colorspace='rgb')
            self.assertEqual(j.box[2].box[1].colorspace, glymur.core.SRGB)

    def test_specify_gray(self):
        """test gray explicitly specified (that's GRAY, not GREY)"""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            data = np.zeros((128, 128), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data, colorspace='gray')
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    def test_specify_grey(self):
        """test grey explicitly specified"""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            data = np.zeros((128, 128), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data, colorspace='grey')
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    def test_grey_with_two_extra_comps(self):
        """should be able to write gray + two extra components"""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
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
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data, colorspace='ycc')

    def test_write_with_jp2_in_caps(self):
        """should be able to write with JP2 suffix."""
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]

        filename = str(self.temp_jp2_filename).replace('.jp2', '.JP2')

        with open(filename, mode='wb') as tfile:
            ofile = Jp2k(tfile.name, data=expdata)
            actdata = ofile[:]
            np.testing.assert_array_equal(actdata, expdata)

    def test_write_srgb_without_mct(self):
        """should be able to write RGB without specifying mct"""
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
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
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=expdata[:, :, 0], mct=True)

    def test_write_cprl(self):
        """Must be able to write a CPRL progression order file"""
        # Issue 17
        j = Jp2k(self.jp2file)
        expdata = j[::2, ::2]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            ofile = Jp2k(tfile.name, data=expdata, prog='CPRL')
            actdata = ofile[:]
            np.testing.assert_array_equal(actdata, expdata)

            codestream = ofile.get_codestream()
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.CPRL)

    def test_bad_area_parameter(self):
        """Should error out appropriately if given a bad area parameter."""
        j = Jp2k(self.jp2file)
        error = glymur.lib.openjp2.OpenJPEGLibraryError
        with self.assertRaises(ValueError):
            # Start corner must be >= 0
            j[-1:1, -1:1]
        with self.assertRaises(ValueError):
            # End corner must be > 0
            j[10:0, 10:0]
        with self.assertRaises(error):
            # End corner must be >= start corner
            j[10:8, 10:8]

    def test_unrecognized_jp2_clrspace(self):
        """We only allow RGB and GRAYSCALE.  Should error out with others"""
        data = np.zeros((128, 128, 3), dtype=np.uint8)
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data, colorspace='cmyk')

    def test_asoc_label_box(self):
        """Test asoc and label box"""
        # Construct a fake file with an asoc and a label box, as
        # OpenJPEG doesn't have such a file.
        data = Jp2k(self.jp2file)[::2, ::2]
        file1 = self.test_dir_path / 'file1.jp2'
        Jp2k(file1, data=data)

        with open(file1, mode='rb') as tfile:

            file2 = self.test_dir_path / 'file2.jp2'
            with open(file2, mode='wb') as tfile2:

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

    def test_grey_with_extra_component(self):
        """version 2.0 cannot write gray + extra"""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            data = np.zeros((128, 128, 2), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data)
            self.assertEqual(j.box[2].box[0].height, 128)
            self.assertEqual(j.box[2].box[0].width, 128)
            self.assertEqual(j.box[2].box[0].num_components, 2)
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    def test_rgb_with_extra_component(self):
        """v2.0+ should be able to write extra components"""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            data = np.zeros((128, 128, 4), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data)
            self.assertEqual(j.box[2].box[0].height, 128)
            self.assertEqual(j.box[2].box[0].width, 128)
            self.assertEqual(j.box[2].box[0].num_components, 4)
            self.assertEqual(j.box[2].box[1].colorspace, glymur.core.SRGB)

    def test_openjpeg_library_error(self):
        """
        SCENARIO:  A zero subsampling factor should produce as error by the
        library.

        EXPECTED RESULT:  OpenJPEGLibraryError
        """
        # This will confirm that the error callback mechanism is working.
        with open(self.jp2file, 'rb') as fptr:
            data = fptr.read()
            with open(self.temp_jp2_filename, mode='wb') as tfile:
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
                    error = glymur.lib.openjp2.OpenJPEGLibraryError
                    with self.assertRaises(error):
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
