"""
Tests for reading JPEG 2000 files with the reader class.
"""
# Standard library imports ...
import collections
import datetime
import importlib.resources as ir
from io import BytesIO
import pathlib
import struct
import unittest
from unittest.mock import patch
import uuid
import warnings

# Third party library imports ...
from lxml import etree as ET
import numpy as np

# Local imports
import glymur
from glymur import Jp2kr
from glymur.jp2box import InvalidJp2kError
from glymur.core import RESTRICTED_ICC_PROFILE

from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG

from . import fixtures


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(glymur.version.openjpeg_version < '2.4.0',
                 "Requires as least v2.4.0")
class TestJp2kr(fixtures.TestCommon):
    """These tests should be run by just about all configuration."""

    def setUp(self):
        super().setUp()
        glymur.reset_option('all')

    def test_repr(self):
        """
        Scenario:  repr is run on a Jp2kr object

        Expected response:  Should clearly indicate Jp2kr, not Jp2k
        """
        j = Jp2kr(self.j2kfile)
        self.assertRegex(repr(j), 'glymur.Jp2kr(.*?)')

    def test_last_decomposition(self):
        """
        Scenario:  The last decomposition image is requested using [::-1]
        notation.

        Expected response:  the image size is verified
        """
        j = Jp2kr(self.j2kfile)
        d = j[::-1, ::-1]
        self.assertEqual(d.shape, (25, 15, 3))

    def test_dtype_jp2(self):
        """
        Scenario:  An RGB image is read from a JP2 file.

        Expected response:  the dtype property is np.uint8
        """
        j = Jp2kr(self.jp2file)
        self.assertEqual(j.dtype, np.uint8)

    def test_dtype_j2k_uint16(self):
        """
        Scenario:  A uint16 monochrome image is read from a J2K file.

        Expected response:  the dtype property is np.uint16
        """
        path = ir.files('tests.data').joinpath('uint16.j2k')
        j = Jp2kr(path)
        self.assertEqual(j.dtype, np.uint16)

    def test_cod_segment_not_3rd(self):
        """
        Scenario:  Normally the COD segment is the 3rd segment.
        Here it is 4th.  Read the image.

        Expected response:  No errors.
        """
        j = Jp2kr(self.j2kfile)
        j.codestream.segment.insert(2, j.codestream.segment[1])
        j[::2, ::2]

    def test_dtype_prec4_signd1(self):
        """
        Scenario:  A 4-bit signed image is read from a J2k file.

        Expected response:  the dtype property is np.int8
        """
        path = ir.files('tests.data').joinpath('p0_03.j2k')
        j = Jp2kr(path)
        self.assertEqual(j.dtype, np.int8)

    def test_dtype_inconsistent_bitdetph(self):
        """
        Scenario:  The image has different bitdepths in different components.

        Expected response:  TypeError when accessing the dtype property.
        """
        path = ir.files('tests.data').joinpath('issue392.jp2')

        with warnings.catch_warnings():
            # There's a warning due to an unrecognized colorspace.  Don't care
            # about that here.
            warnings.simplefilter("ignore")
            j = Jp2kr(path)

        with self.assertRaises(TypeError):
            j.dtype

    def test_ndims_jp2(self):
        """
        Scenario:  An RGB image is read from a JP2 file.

        Expected response:  the ndim attribute/property is 3
        """
        j = Jp2kr(self.jp2file)
        self.assertEqual(j.ndim, 3)

    def test_ndims_j2k(self):
        """
        Scenario:  An RGB image is read from a raw codestream.

        Expected response:  the ndim attribute/property is 3
        """
        j = Jp2kr(self.j2kfile)
        self.assertEqual(j.ndim, 3)

    def test_ndims_monochrome_j2k(self):
        """
        Scenario:  An monochrome image is read from a raw codestream.

        Expected response:  the ndim attribute/property is 2
        """
        path = ir.files('tests.data').joinpath('p0_02.j2k')
        j = Jp2kr(path)
        self.assertEqual(j.ndim, 2)

    def test_read_bands_unequal_subsampling(self):
        """
        SCENARIO:  The read_bands method is used on an image with unequal
        subsampling.

        EXPECTED RESPONSE: The image is a list of arrays of unequal size.
        """
        path = ir.files('tests.data').joinpath('p0_06.j2k')
        d = Jp2kr(path).read_bands()

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
        j = Jp2kr(self.jp2file)
        d1 = j[:]
        d2 = j.read_bands()
        self.assertEqual(d1.shape, d2.shape)

    def test_pathlib(self):
        """
        SCENARIO: Provide a pathlib.Path instead of a string for the filename.
        """
        p = pathlib.Path(self.jp2file)
        jp2 = Jp2kr(p)
        self.assertEqual(jp2.shape, (1456, 2592, 3))

    def test_no_cxform_pclr_jpx(self):
        """
        Indices for pclr jpxfile still usable if no color transform specified
        """
        with warnings.catch_warnings():
            # Suppress a Compatibility list item warning.  We already test
            # for this elsewhere.
            warnings.simplefilter("ignore")
            jp2 = Jp2kr(self.jpxfile)
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
                ofile.write(ifile.read(77))

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

            j = Jp2kr(ofile.name)
            with self.assertRaises(RuntimeError):
                j[:]

    def test_shape_jp2(self):
        """verify shape attribute for JP2 file
        """
        jp2 = Jp2kr(self.jp2file)
        self.assertEqual(jp2.shape, (1456, 2592, 3))

    def test_shape_3_channel_j2k(self):
        """verify shape attribute for J2K file
        """
        j2k = Jp2kr(self.j2kfile)
        self.assertEqual(j2k.shape, (800, 480, 3))

    def test_shape_jpx_jp2(self):
        """verify shape attribute for JPX file with JP2 compatibility
        """
        jpx = Jp2kr(self.jpxfile)
        self.assertEqual(jpx.shape, (1024, 1024, 3))

    def test_rlevel_max_backwards_compatibility(self):
        """
        Verify that rlevel=-1 gets us the lowest resolution image

        This is an old option only available via the read method, not via
        array-style slicing.
        """
        j = Jp2kr(self.j2kfile)
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
        j = Jp2kr(self.jp2file)
        with self.assertRaises(ValueError):
            j[::64, ::64]

    def test_not_jpeg2000(self):
        """
        SCENARIO:  The Jp2kr constructor is passed a file that is not
        JPEG 2000.

        EXPECTED RESULT:  RuntimeError
        """
        path = ir.files('tests.data').joinpath('nemo.txt')
        with self.assertRaises(InvalidJp2kError):
            Jp2kr(path)

    def test_file_does_not_exist(self):
        """
        Scenario:  The Jp2kr construtor is passed a file that does not exist
        and the intent is reading.

        Expected Result:  FileNotFoundError
        """
        # Verify that we error out appropriately if not given an existing file
        # at all.
        filename = 'this file does not actually exist on the file system.'
        with self.assertRaises(FileNotFoundError):
            Jp2kr(filename)

    def test_codestream(self):
        """
        Verify the markers and segments of a JP2 file codestream.
        """
        jp2 = Jp2kr(self.jp2file)
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
        self.assertEqual(c.segment[2].offset, 136)
        self.assertEqual(c.segment[2].length, 12)
        self.assertEqual(c.segment[2].scod, 0)
        self.assertEqual(c.segment[2].layers, 2)
        self.assertEqual(c.segment[2].code_block_size, (64.0, 64.0))
        self.assertEqual(c.segment[2].prog_order, 0)
        self.assertEqual(c.segment[2].xform, 1)
        np.testing.assert_array_equal(
            c.segment[2].precinct_size, np.array(((32768, 32768)))
        )

        self.assertEqual(c.segment[3].marker_id, 'QCD')
        self.assertEqual(c.segment[3].offset, 150)
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
        self.assertEqual(c.segment[5].offset, 198)
        self.assertEqual(c.segment[5].length, 10)
        self.assertEqual(c.segment[5].isot, 0)
        self.assertEqual(c.segment[5].psot, 1132173)
        self.assertEqual(c.segment[5].tpsot, 0)
        self.assertEqual(c.segment[5].tnsot, 1)

        self.assertEqual(c.segment[6].marker_id, 'COC')
        self.assertEqual(c.segment[6].offset, 210)
        self.assertEqual(c.segment[6].length, 9)
        self.assertEqual(c.segment[6].ccoc, 1)
        np.testing.assert_array_equal(c.segment[6].scoc,
                                      np.array([0]))
        np.testing.assert_array_equal(c.segment[6].spcoc,
                                      np.array([1, 4, 4, 0, 1]))
        np.testing.assert_array_equal(
            c.segment[6].precinct_size, np.array(((32768, 32768)))
        )

        self.assertEqual(c.segment[7].marker_id, 'QCC')
        self.assertEqual(c.segment[7].offset, 221)
        self.assertEqual(c.segment[7].length, 8)
        self.assertEqual(c.segment[7].cqcc, 1)
        self.assertEqual(c.segment[7].sqcc, 64)
        self.assertEqual(c.segment[7].mantissa, [0, 0, 0, 0])
        self.assertEqual(c.segment[7].exponent, [8, 9, 9, 10])
        self.assertEqual(c.segment[7].guard_bits, 2)

        self.assertEqual(c.segment[8].marker_id, 'COC')
        self.assertEqual(c.segment[8].offset, 231)
        self.assertEqual(c.segment[8].length, 9)
        self.assertEqual(c.segment[8].ccoc, 2)
        np.testing.assert_array_equal(c.segment[8].scoc,
                                      np.array([0]))
        np.testing.assert_array_equal(c.segment[8].spcoc,
                                      np.array([1, 4, 4, 0, 1]))
        np.testing.assert_array_equal(
            c.segment[8].precinct_size, np.array(((32768, 32768)))
        )

        self.assertEqual(c.segment[9].marker_id, 'QCC')
        self.assertEqual(c.segment[9].offset, 242)
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
        jp2k = Jp2kr(self.jp2file)

        # top-level boxes
        self.assertEqual(len(jp2k.box), 4)

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

        self.assertEqual(jp2k.box[3].box_id, 'jp2c')
        self.assertEqual(jp2k.box[3].offset, 77)
        self.assertEqual(jp2k.box[3].length, 1132296)

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
        jp2k = Jp2kr(self.j2kfile)
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
                write_buffer = ifile.read(77)
                tfile.write(write_buffer)

                # The L field must be 1 in order to signal the presence of the
                # XL field.  The actual length of the jp2c box increased by 8
                # (8 bytes for the XL field).
                length = 1
                typ = b'jp2c'
                xlen = 1132296 + 8
                write_buffer = struct.pack('>I4sQ', int(length), typ, xlen)
                tfile.write(write_buffer)

                # Get the rest of the input file (minus the 8 bytes for L and
                # T.
                ifile.seek(8, 1)
                write_buffer = ifile.read()
                tfile.write(write_buffer)
                tfile.flush()

            jp2k = Jp2kr(tfile.name)

            self.assertEqual(jp2k.box[-1].box_id, 'jp2c')
            self.assertEqual(jp2k.box[-1].offset, 77)
            self.assertEqual(jp2k.box[-1].length, 1132296 + 8)

    def test_length_field_is_zero(self):
        """
        SCENARIO:  A JP2 file has in its last box and L field with value 0.

        EXPECTED RESULT:  The file is parsed without error.  In particular, the
        length of that last box is correctly computed.
        """
        # Verify that boxes with the L field as zero are correctly read.
        # This should only happen in the last box of a JPEG 2000 file.
        # Our example image has its last box at byte 588458.
        baseline_jp2 = Jp2kr(self.jp2file)
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

            new_jp2 = Jp2kr(tfile.name)

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
        j2k = Jp2kr(self.jp2file)
        j2k[::2, ::2]

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_basic_j2k(self):
        """
        Just a very basic test that reading a J2K file does not error out.
        """
        j2k = Jp2kr(self.j2kfile)
        j2k[:]

    def test_empty_box_with_j2k(self):
        """Verify that the list of boxes in a J2C/J2K file is present, but
        empty.
        """
        j = Jp2kr(self.j2kfile)
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

            jp2k = Jp2kr(tfile.name)

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

            jp2k = Jp2kr(tfile.name)

            self.assertEqual(jp2k.box[3].box_id, 'xml ')
            self.assertEqual(jp2k.box[3].offset, 77)
            self.assertEqual(jp2k.box[3].length, 36)
            self.assertEqual(ET.tostring(jp2k.box[3].xml.getroot()),
                             b'<test>this is a test</test>')

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_jpx_mult_codestreams_jp2_brand(self):
        """Read JPX codestream when jp2-compatible."""
        # The file in question has multiple codestreams.
        jpx = Jp2kr(self.jpxfile)
        data = jpx[:]
        self.assertEqual(data.shape, (1024, 1024, 3))

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
                ofile.write(ifile.read(77))

                # Write the new codestream length (+4) and the box ID.
                buffer = struct.pack('>I4s', 1132296 + 4, b'jp2c')
                ofile.write(buffer)

                # Copy up until the EOC marker.
                ifile.seek(85)
                ofile.write(ifile.read(1132286))

                # Write the zero-length reserved segment.
                buffer = struct.pack('>BBH', 0xff, 0x00, 0)
                ofile.write(buffer)

                # Write the EOC marker and be done with it.
                ofile.write(ifile.read())
                ofile.flush()

        cstr = Jp2kr(ofile.name).get_codestream(header_only=False)
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

            j = Jp2kr(ofile.name)
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
        path = ir.files('tests.data').joinpath('p0_03.j2k')
        j = Jp2kr(path)
        d0 = j[:]

        j.layer = 1
        d1 = j[:]

        np.all(d0 != d1)

    def test_invalid_layers(self):
        """
        SCENARIO:  an improper layer value is set

        EXPECTED RESULT:  RuntimeError when an invalid layer number is supplied
        """
        # There are 8 layers, so only values [0-7] are valid.
        path = ir.files('tests.data').joinpath('p0_03.j2k')
        j = Jp2kr(path)

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
        path = ir.files('tests.data').joinpath('p0_03.j2k')
        j = Jp2kr(path)

        self.assertFalse(j.verbose)

    def test_default_layer(self):
        """
        SCENARIO:  Check the default layer property.

        EXPECTED RESULT:  The default layer property value is 0.
        """
        path = ir.files('tests.data').joinpath('p0_03.j2k')
        j = Jp2kr(path)

        self.assertEqual(j.layer, 0)


class TestVersion(fixtures.TestCommon):
    """
    Tests for the version of openjpeg.  These can be run regardless of the
    version of openjpeg installed, or even if openjpeg is not installed,
    because we fully mock the openjpeg version.
    """
    def test_read_minimum_version(self):
        """
        Scenario:  we have openjpeg, but not the minimum supported version.

        Expected Result:  RuntimeError
        """
        with patch('glymur.version.openjpeg_version_tuple', new=(2, 2, 9)):
            with patch('glymur.version.openjpeg_version', new='2.2.9'):
                with self.assertRaises(RuntimeError):
                    glymur.Jp2kr(self.jp2file)[:]

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
                        glymur.Jp2kr(self.jp2file).read()
                with self.assertRaises(RuntimeError):
                    glymur.Jp2kr(self.jp2file)[:]

    def test_read_bands_without_openjp2(self):
        """
        Don't have openjp2 library?  Must error out.
        """
        exp_error = RuntimeError
        with patch('glymur.version.openjpeg_version_tuple', new=(1, 5, 0)):
            with patch('glymur.version.openjpeg_version', new='1.5.0'):
                with self.assertRaises(exp_error):
                    glymur.Jp2kr(self.jp2file).read_bands()


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
        j = Jp2kr(self.jp2file)

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
        j = Jp2kr(self.jp2file)

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
        j = Jp2kr(self.jp2file)

        # Fake a data structure that resembles the openjpeg component.
        Component = collections.namedtuple('Component', ['prec', 'sgnd'])
        c = Component(prec=17, sgnd=True)
        with self.assertRaises(ValueError):
            j._component2dtype(c)


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
        jp2 = Jp2kr(self.jp2file)
        jp2c = jp2.box[-1]
        self.assertIsNone(jp2c._codestream)
        jp2c.codestream
        self.assertIsNotNone(jp2c._codestream)
