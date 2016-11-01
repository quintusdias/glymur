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
from xml.etree import cElementTree as ET

# Third party library imports ...
import numpy as np
import pkg_resources as pkg

# Local imports
import glymur
from glymur import Jp2k
from glymur.core import COLOR, RED, GREEN, BLUE, RESTRICTED_ICC_PROFILE
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


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(unittest.TestCase):
    """These tests should be run by just about all configuration."""

    @classmethod
    def setUpClass(cls):
        cls.jp2file = glymur.data.nemo()
        cls.j2kfile = glymur.data.goodstuff()
        cls.jpxfile = glymur.data.jpxfile()

    @classmethod
    def tearDownClass(cls):
        pass

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
        """
        file = os.path.join('data', 'p0_06.j2k')
        file = pkg.resource_filename(__name__, file)
        j = Jp2k(file)
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

    def test_openjpeg_library_error(self):
        """
        required COD marker not found in main header
        """
        file = os.path.join('data', 'edf_c2_1178956.jp2')
        file = pkg.resource_filename(__name__, file)
        exp_error = glymur.lib.openjp2.OpenJPEGLibraryError
        with self.assertRaises(exp_error):
            with warnings.catch_warnings():
                # Suppress a UserWarning for bad file type compatibility
                warnings.simplefilter("ignore")
                Jp2k(file)[:]

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

    @unittest.skipIf(glymur.version.openjpeg_version < '2.0.0',
                     'Requires 2.0.0 or better.')
    def test_read_bands(self):
        """
        Have to use read_bands if the subsampling is not uniform
        """
        file = os.path.join('data', 'p0_06.j2k')
        file = pkg.resource_filename(__name__, file)
        bands = glymur.Jp2k(file).read_bands()
        self.assertEqual(bands[0].shape, (129, 513))
        self.assertEqual(bands[1].shape, (129, 257))
        self.assertEqual(bands[2].shape, (65, 513))
        self.assertEqual(bands[3].shape, (65, 257))

    @unittest.skipIf(re.match(r'''0|1|2.0.0''',
                              glymur.version.openjpeg_version) is not None,
                     "Only supported in 2.0.1 or higher")
    def test_read_tile_backwards_compatibility(self):
        """
        Test ability to read specified tiles.  Requires 2.0.1 or higher.

        0.7.x read usage deprecated, should use slicing
        """
        file = os.path.join('data', 'p0_03.j2k')
        file = pkg.resource_filename(__name__, file)
        jp2k = Jp2k(file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tdata = jp2k.read(tile=3, rlevel=1)  # last tile
        odata = jp2k[::2, ::2]
        np.testing.assert_array_equal(tdata, odata[64:128, 64:128])

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
