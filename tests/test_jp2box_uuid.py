# -*- coding:  utf-8 -*-
"""Test suite for printing.
"""
# Standard library imports
from io import BytesIO
import os
import pkg_resources as pkg
import shutil
import struct
import sys
import tempfile
import unittest
import uuid
import warnings

# Third party library imports ...
try:
    import lxml.etree
except ImportError:
    import xml.etree.ElementTree as ET

# Local imports
import glymur
from glymur import Jp2k
from . import fixtures
from .fixtures import SimpleRDF

TIFF_ASCII = 2
TIFF_SHORT = 3
TIFF_LONG = 4
TIFF_RATIONAL = 5
TIFF_DOUBLE = 12

SUBFILETYPE = 254
FILETYPE_REDUCEDIMAGE = 0x1
OSUBFILETYPE = 255
IMAGEWIDTH = 256
IMAGELENGTH = 257
BITSPERSAMPLE = 258
COMPRESSION = 259
COMPRESSION_NONE = 1
PHOTOMETRIC = 262
STRIPOFFSETS = 273
ORIENTATION = 274
PHOTOMETRIC_MINISBLACK = 1
SAMPLESPERPIXEL = 277
ROWSPERSTRIP = 278
STRIPBYTECOUNTS = 279
MINSAMPLEVALUE = 280
MAXSAMPLEVALUE = 281
XRESOLUTION = 282
YRESOLUTION = 283
PLANARCONFIG = 284


@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
class TestSuite(unittest.TestCase):
    """Tests for XMP, Exif UUIDs."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def _create_degenerate_geotiff(self, e):
        """
        Create an in-memory degenerate geotiff.

        Parameters
        ----------
        e : str
           Either '<' for little endian or '>' for big endian.

        Returns
        -------
        bytes
            sequence of bytes making up a degenerate geotiff.  Should have
            something like the following structure:

            Magic: 0x4949 <little-endian> Version: 0x2a <ClassicTIFF>
            Directory 0: offset 8 (0x8) next 0 (0)
            SubFileType (254) LONG (4) 1<1>
            ImageWidth (256) SHORT (3) 1<1>
            ImageLength (257) SHORT (3) 1<1>
            BitsPerSample (258) SHORT (3) 1<8>
            Compression (259) SHORT (3) 1<1>
            Photometric (262) SHORT (3) 1<1>
            StripOffsets (273) LONG (4) 1<1>
            SamplesPerPixel (277) SHORT (3) 1<1>
            RowsPerStrip (278) LONG (4) 1<1>
            StripByteCounts (279) LONG (4) 1<1>
            XResolution (282) RATIONAL (5) 1<75>
            YResolution (283) RATIONAL (5) 1<75>
            33550 (0x830e) DOUBLE (12) 3<10 10 0>
            33922 (0x8482) DOUBLE (12) 6<0 0 0 444650 4.64051e+06 0>
            34735 (0x87af) SHORT (3) 24<1 1 0 5 1024 0 1 1 1025 0 1 1 ...>
            34737 (0x87b1) ASCII (2) 45<UTM Zone 16N NAD27"|Clar ...>
        """
        tag_payloads = []

        b = BytesIO()

        # Create the header.
        # Signature, version, offset to IFD
        if e == '<':
            buffer = struct.pack('<2sHI', b'II', 42, 8)
        else:
            buffer = struct.pack('>2sHI', b'MM', 42, 8)
        b.write(buffer)

        offset = b.tell()

        num_tags = 16

        # The CDATA offset is past IFD tag count
        offset += 2

        # The CDATA offset is past the IFD
        offset += num_tags * 12

        # The CDATA offset is past the null offset to next IFD
        offset += 4

        # The CDATA offset is past the image data
        offset += 1

        # Write the tag count
        buffer = struct.pack(e + 'H', num_tags)
        b.write(buffer)

        # Sub file type
        buffer = struct.pack(e + 'HHII', SUBFILETYPE, TIFF_LONG, 1, 1)
        b.write(buffer)

        buffer = struct.pack(e + 'HHII', IMAGEWIDTH, TIFF_SHORT, 1, 1)
        b.write(buffer)
        buffer = struct.pack(e + 'HHII', IMAGELENGTH, TIFF_SHORT, 1, 1)
        b.write(buffer)

        buffer = struct.pack(e + 'HHII', BITSPERSAMPLE, TIFF_SHORT, 1, 8)
        b.write(buffer)

        buffer = struct.pack(e + 'HHII',
                             COMPRESSION, TIFF_SHORT, 1, COMPRESSION_NONE)
        b.write(buffer)

        buffer = struct.pack(e + 'HHII', PHOTOMETRIC, TIFF_SHORT, 1, 1)
        b.write(buffer)

        buffer = struct.pack(e + 'HHII', STRIPOFFSETS, TIFF_LONG, 1, 1)
        b.write(buffer)

        buffer = struct.pack(e + 'HHII', SAMPLESPERPIXEL, TIFF_SHORT, 1, 1)
        b.write(buffer)

        buffer = struct.pack(e + 'HHII', ROWSPERSTRIP, TIFF_LONG, 1, 1)
        b.write(buffer)

        buffer = struct.pack(e + 'HHII', STRIPBYTECOUNTS, TIFF_LONG, 1, 1)
        b.write(buffer)

        buffer = struct.pack(e + 'HHII', XRESOLUTION, TIFF_RATIONAL, 1, offset)
        b.write(buffer)
        tag_payloads.append((e + 'I', 75))
        tag_payloads.append((e + 'I', 1))

        buffer = struct.pack(e + 'HHII',
                             YRESOLUTION, TIFF_RATIONAL, 1, offset + 8)
        b.write(buffer)
        tag_payloads.append((e + 'I', 75))
        tag_payloads.append((e + 'I', 1))

        # Model pixel scale tag
        buffer = struct.pack(e + 'HHII', 33550, TIFF_DOUBLE, 3, offset + 16)
        b.write(buffer)
        tag_payloads.append((e + 'd', 10))
        tag_payloads.append((e + 'd', 10))
        tag_payloads.append((e + 'd', 0))

        buffer = struct.pack(e + 'HHII', 33922, TIFF_DOUBLE, 6, offset + 40)
        b.write(buffer)
        datums = [0.0, 0.0, 0.0, 44650.0, 4640510.0, 0.0]
        for data in datums:
            tag_payloads.append((e + 'd', data))

        buffer = struct.pack(e + 'HHII', 34735, TIFF_SHORT, 24, offset + 88)
        b.write(buffer)
        datums = [
            1, 1, 0, 5,
            1024, 0, 1, 1,
            1025, 0, 1, 1,
            1026, 34737, 20, 0,
            2049, 34737, 24, 20,
            3072, 0, 1, 26716,
        ]
        for data in datums:
            tag_payloads.append((e + 'H', data))

        buffer = struct.pack(e + 'HHII', 34737, TIFF_ASCII, 45, offset + 136)
        b.write(buffer)
        items = (e + '45s', b'UTM Zone 16N NAD27"|Clarke, 1866 by Default| ')
        tag_payloads.append(items)

        # NULL pointer to next IFD
        buffer = struct.pack(e + 'I', 0)
        b.write(buffer)

        # Image data.  Just a single byte will do.
        buffer = struct.pack(e + 'B', 0)
        b.write(buffer)

        # Tag payloads
        for format, datum in tag_payloads:
            buffer = struct.pack(format, datum)
            b.write(buffer)

        b.seek(0)
        return b.read()

    def test_append_xmp_uuid(self):
        """Should be able to append an XMP UUID box."""
        the_uuid = uuid.UUID('be7acfcb-97a9-42e8-9c71-999491e3afac')
        raw_data = SimpleRDF.encode('utf-8')
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            shutil.copyfile(self.jp2file, tfile.name)
            jp2 = Jp2k(tfile.name)
            ubox = glymur.jp2box.UUIDBox(the_uuid=the_uuid, raw_data=raw_data)
            jp2.append(ubox)

            # Should be two UUID boxes now.
            expected_ids = ['jP  ', 'ftyp', 'jp2h', 'uuid', 'jp2c', 'uuid']
            actual_ids = [b.box_id for b in jp2.box]
            self.assertEqual(actual_ids, expected_ids)

            # The data should be an XMP packet, which gets interpreted as
            # an ElementTree.
            if 'lxml' in sys.modules.keys():
                self.assertTrue(isinstance(jp2.box[-1].data,
                                           lxml.etree._ElementTree))
            else:
                self.assertTrue(isinstance(jp2.box[-1].data,
                                           ET.ElementTree))

    @unittest.skipIf(sys.hexversion < 0x03000000, "assertWarns is PY3K")
    def test_bad_exif_tag(self):
        """
        Corrupt the Exif IFD with an invalid tag should produce a warning.
        """
        b = self._create_exif_uuid('<')

        b.seek(0)
        buffer = b.read()

        # The first tag should begin at byte 32.  Replace the entire IDF
        # entry with zeros.
        # import pdb; pdb.set_trace()
        tag = struct.pack('<HHII', 0, 3, 0, 0)
        buffer = buffer[:40] + tag + buffer[52:]

        b = BytesIO()
        b.write(buffer)
        b.seek(8)

        with self.assertWarns(UserWarning):
            box = glymur.jp2box.UUIDBox.parse(b, 0, 418)

        self.assertEqual(box.box_id, 'uuid')

        # Should still get the IFD.  16 tags.
        self.assertEqual(len(box.data.keys()), 16)

    def test_exif(self):
        """
        Verify read of both big and little endian Exif IFDs.
        """
        # Check both little and big endian.
        for endian in ['<', '>']:
            self._test_endian_exif(endian)

    def _create_exif_uuid(self, endian):
        """
        Create a buffer that can be parsed as an Exif UUID.

        Parameters
        ----------
        endian : str
            Either '<' for little endian or '>' for big endian
        """
        b = BytesIO()
        # Write L, T, UUID identifier.
        # 388 = length of degenerate tiff
        # 6 = Exif\x0\x0
        # 16 = length of UUID identifier
        # 8 = length of L, T
        # 388 + 6 + 16 + 8 = 418
        b.write(struct.pack('>I4s', 418, b'uuid'))
        b.write(b'JpgTiffExif->JP2')

        b.write(b'Exif\x00\x00')

        buffer = self._create_degenerate_geotiff(endian)
        b.write(buffer)

        b.seek(8)

        return b

    def _test_endian_exif(self, endian):
        """
        Test Exif IFDs.

        Parameters
        ----------
        endian : str
            Either '<' for little endian or '>' for big endian
        """
        bptr = self._create_exif_uuid(endian)

        box = glymur.jp2box.UUIDBox.parse(bptr, 0, 418)
        self.assertEqual(box.data['XResolution'], 75)

        expected = 'UTM Zone 16N NAD27"|Clarke, 1866 by Default| '
        self.assertEqual(box.data['GeoAsciiParams'], expected)

    def test_print_bad_geotiff(self):
        """
        Geotiff UUID is corrupt.  Print that instead of erroring out.
        """
        relpath = os.path.join('data', 'issue398.dat')
        path = pkg.resource_filename(__name__, relpath)
        with open(path, 'rb') as f:
            f.seek(8)
            with warnings.catch_warnings():
                # Ignore the warnings about invalid TIFF tags, we already know
                # that.
                warnings.simplefilter('ignore')
                box = glymur.jp2box.UUIDBox.parse(f, 0, 380)

        actual = str(box)
        if fixtures.HAVE_GDAL:
            expected = ("UUID Box (uuid) @ (0, 380)\n"
                        "    UUID:  "
                        "b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03 (GeoTIFF)\n"
                        "    UUID Data:  corrupt")
        else:
            expected = ("UUID Box (uuid) @ (0, 380)\n"
                        "    UUID:  "
                        "b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03 (GeoTIFF)\n"
                        "    UUID Data:  None")
        self.assertEqual(actual, expected)

    def test_bad_tag_datatype(self):
        """Only certain datatypes are allowable"""
        f = BytesIO()

        # Write L, T, UUID identifier.
        f.write(struct.pack('>I4s', 52, b'uuid'))
        f.write(b'JpgTiffExif->JP2')

        f.write(b'Exif\x00\x00')
        xbuffer = struct.pack('<BBHI', 74, 73, 42, 8)
        f.write(xbuffer)

        # We will write just a single tag.
        f.write(struct.pack('<H', 1))

        # 2000 is not an allowable TIFF datatype.
        f.write(struct.pack('<HHI4s', 271, 2000, 3, b'HTC\x00'))

        f.seek(8)

        # We should still get a UUID box out of it.  But we get no data.
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                box = glymur.jp2box.UUIDBox.parse(f, -1, 56)
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                box = glymur.jp2box.UUIDBox.parse(f, -1, 56)

        self.assertEqual(box.box_id, 'uuid')
        self.assertIsNone(box.data)

    @unittest.skipIf(sys.hexversion < 0x03000000, "assertWarns not until PY3K")
    def test_bad_tiff_header_byte_order_indication(self):
        """Only b'II' and b'MM' are allowed."""
        f = BytesIO()

        # Write L, T, UUID identifier.
        f.write(struct.pack('>I4s', 52, b'uuid'))
        f.write(b'JpgTiffExif->JP2')

        f.write(b'Exif\x00\x00')
        xbuffer = struct.pack('<BBHI', 74, 73, 42, 8)
        f.write(xbuffer)

        # We will write just a single tag.
        f.write(struct.pack('<H', 1))

        # 271 is the Make.
        f.write(struct.pack('<HHI4s', 271, 2, 3, b'HTC\x00'))

        # Write the null offset to the next IFD.
        f.write(struct.pack('<I', 0))

        f.seek(8)

        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings():
                # Just ignore the warning on Python2.
                warnings.simplefilter('ignore')
                box = glymur.jp2box.UUIDBox.parse(f, -1, 56)
        else:
            with self.assertWarns(UserWarning):
                box = glymur.jp2box.UUIDBox.parse(f, -1, 56)

        # We should still get a UUID box out of it.  But we get no data.
        self.assertEqual(box.box_id, 'uuid')
        self.assertIsNone(box.data)
