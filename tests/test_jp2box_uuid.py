# -*- coding:  utf-8 -*-
"""Test suite for printing.
"""
# Standard library imports
import importlib.resources as ir
import io
import platform
import shutil
import struct
import unittest
import uuid
import warnings

# Third party library imports ...
import lxml.etree

# Local imports
import glymur
from glymur import Jp2k
from glymur.jp2box import UUIDBox
from . import fixtures, data

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
MODELPIXELSCALE = 33550
MODELTIEPOINT = 33922
GEOKEYDIRECTORY = 34735
GEOASCIIPARAMS = 34737


class TestSuite(fixtures.TestCommon):
    """Tests for XMP, Exif UUIDs."""

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
        b = io.BytesIO()

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

        # Write out all the IFD tags.  Any data that exceeds 4 bytes has to
        # be appended later.
        lst = [
            struct.pack(e + 'HHII', SUBFILETYPE, TIFF_LONG, 1, 1),
            struct.pack(e + 'HHII', IMAGEWIDTH, TIFF_SHORT, 1, 1),
            struct.pack(e + 'HHII', IMAGELENGTH, TIFF_SHORT, 1, 1),
            struct.pack(e + 'HHII', BITSPERSAMPLE, TIFF_SHORT, 1, 8),
            struct.pack(e + 'HHII', COMPRESSION, TIFF_SHORT, 1,
                        COMPRESSION_NONE),
            struct.pack(e + 'HHII', PHOTOMETRIC, TIFF_SHORT, 1, 1),
            struct.pack(e + 'HHII', STRIPOFFSETS, TIFF_LONG, 1, 1),
            struct.pack(e + 'HHII', SAMPLESPERPIXEL, TIFF_SHORT, 1, 1),
            struct.pack(e + 'HHII', ROWSPERSTRIP, TIFF_LONG, 1, 1),
            struct.pack(e + 'HHII', STRIPBYTECOUNTS, TIFF_LONG, 1, 1),
            struct.pack(e + 'HHII', XRESOLUTION, TIFF_RATIONAL, 1, offset),
            struct.pack(e + 'HHII', YRESOLUTION, TIFF_RATIONAL, 1, offset + 8),
            struct.pack(e + 'HHII', MODELPIXELSCALE, TIFF_DOUBLE, 3,
                        offset + 16),
            struct.pack(e + 'HHII', MODELTIEPOINT, TIFF_DOUBLE, 6,
                        offset + 40),
            struct.pack(e + 'HHII', GEOKEYDIRECTORY, TIFF_SHORT, 24,
                        offset + 88),
            struct.pack(e + 'HHII', GEOASCIIPARAMS, TIFF_ASCII, 45,
                        offset + 136),
        ]
        for buffer in lst:
            b.write(buffer)

        # NULL pointer to next IFD
        buffer = struct.pack(e + 'I', 0)
        b.write(buffer)

        # Image data.  Just a single byte will do.
        buffer = struct.pack(e + 'B', 0)
        b.write(buffer)

        # Now append the tag payloads that did not fit into the IFD.

        # XResolution
        tag_payloads = [
            (e + 'I', 75),  # XResolution
            (e + 'I', 1),
            (e + 'I', 75),  # YResolution
            (e + 'I', 1),
            (e + 'd', 10),  # Model pixel scale tag
            (e + 'd', 10),
            (e + 'd', 0),
        ]

        # MODELTIEPOINT
        datums = [0.0, 0.0, 0.0, 44650.0, 4640510.0, 0.0]
        for datum in datums:
            tag_payloads.append((e + 'd', datum))

        # GeoKeyDirectory
        datums = [
            1, 1, 0, 5,
            1024, 0, 1, 1,
            1025, 0, 1, 1,
            1026, 34737, 20, 0,
            2049, 34737, 24, 20,
            3072, 0, 1, 26716,
        ]
        for datum in datums:
            tag_payloads.append((e + 'H', datum))

        # GEOASCIIPARAMS
        items = (e + '45s',
                 b'UTM Zone 16N NAD27"|Clarke, 1866 by Default| ')
        tag_payloads.append(items)

        # Tag payloads
        for format, datum in tag_payloads:
            buffer = struct.pack(format, datum)
            b.write(buffer)

        b.seek(0)
        return b.read()

    def test__read_exif_uuid_missing_exif00_lead_in(self):
        """
        SCENARIO:  Parse a JpgTiffExif->Jp2 UUID that is missing the 'EXIF\0\0'
        lead-in.

        EXPECTED RESULT:  Should not error out.  Verify the UUID type.  Verify
        the existance of one of the "Exif.Photo" tags.
        """
        box_data = ir.read_binary('tests.data', 'issue549.dat')
        bf = io.BytesIO(box_data)
        box = UUIDBox.parse(bf, 0, len(box_data))

        actual = box.uuid
        expected = uuid.UUID(bytes=b'JpgTiffExif->JP2')
        self.assertEqual(actual, expected)

        self.assertEqual(box.data['ExifTag']['ExifVersion'], (48, 50, 51, 50))

    def test__read_malformed_exif_uuid(self):
        """
        SCENARIO:  Parse a JpgTiffExif->Jp2 UUID that is not only missing the
        'EXIF\0\0' lead-in, but even the TIFF header is malformed.

        EXPECTED RESULT:  RuntimeError
        """
        box_data = ir.read_binary('tests.data', 'issue549.dat')
        bf = io.BytesIO(box_data[:16] + box_data[20:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            box = UUIDBox.parse(bf, 0, 37700)

        actual = box.uuid
        expected = uuid.UUID(bytes=b'JpgTiffExif->JP2')
        self.assertEqual(actual, expected)

    @unittest.skipIf(
            platform.system().startswith('Windows'),
            "Skipping on windows, see issue 560"
    )
    def test__printing__geotiff_uuid__xml_sidecar(self):
        """
        SCENARIO:  Print a geotiff UUID with XML sidecar file.

        EXPECTED RESULT:  Should not error out.  There is a warning about GDAL
        not being able to print the UUID data as expected.
        """
        box_data = ir.read_binary('tests.data', '0220000800_uuid.dat')
        bf = io.BytesIO(box_data)
        bf.seek(8)
        box = UUIDBox.parse(bf, 0, 703)
        with warnings.catch_warnings(record=True) as w:
            str(box)

        if fixtures._HAVE_GDAL:
            self.assertEqual(len(w), 1)
        else:
            # If no gdal, there's no warning.  It's just an Exif UUID in
            # that case.
            self.assertEqual(len(w), 0)

    def test_append_xmp_uuid(self):
        """
        SCENARIO:  Append an XMP UUID box to an existing JP2 file.

        EXPECTED RESULT:  The new last box in the JP2 file is UUID.
        """
        the_uuid = uuid.UUID('be7acfcb-97a9-42e8-9c71-999491e3afac')
        raw_data = fixtures.SIMPLE_RDF.encode('utf-8')

        shutil.copyfile(self.jp2file, self.temp_jp2_filename)

        jp2 = Jp2k(self.temp_jp2_filename)
        ubox = glymur.jp2box.UUIDBox(the_uuid=the_uuid, raw_data=raw_data)
        jp2.append(ubox)

        # Should be two UUID boxes now.
        expected_ids = ['jP  ', 'ftyp', 'jp2h', 'uuid', 'jp2c', 'uuid']
        actual_ids = [b.box_id for b in jp2.box]
        self.assertEqual(actual_ids, expected_ids)

        # The data should be an XMP packet, which gets interpreted as
        # an ElementTree.
        self.assertTrue(isinstance(jp2.box[-1].data, lxml.etree._ElementTree))

    def test_bad_exif_tag(self):
        """
        Corrupt the Exif IFD with an invalid tag should produce a warning.
        """
        b = self._create_exif_uuid('<')

        b.seek(0)
        buffer = b.read()

        # The first tag should begin at byte 32.  Replace the entire IDF
        # entry with zeros.
        tag = struct.pack('<HHII', 0, 3, 0, 0)
        buffer = buffer[:40] + tag + buffer[52:]

        b = io.BytesIO()
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
        b = io.BytesIO()
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

    @unittest.skipIf(
            platform.system().startswith('Windows'),
            "Skipping on windows, see issue 560"
    )
    def test_print_bad_geotiff(self):
        """
        SCENARIO:  A GeoTIFF UUID is corrupt.

        EXPECTED RESULT:  No errors.  There is a warning issued when we try
        to print the box.
        """
        with ir.path(data, 'issue398.dat') as path:
            with path.open('rb') as f:
                f.seek(8)
                with warnings.catch_warnings(record=True) as w:
                    box = glymur.jp2box.UUIDBox.parse(f, 0, 380)
                    str(box)

        if fixtures._HAVE_GDAL:
            self.assertEqual(len(w), 1)
        else:
            # No warning issued if GDAL is not present.
            self.assertEqual(len(w), 0)


class TestSuiteHiRISE(fixtures.TestCommon):
    """Tests for HiRISE RDRs."""

    def setUp(self):
        super(TestSuiteHiRISE, self).setUp()

        # Hand-create the boxes needed for HiRISE.
        the_uuid = uuid.UUID('2b0d7e97-aa2e-317d-9a33-e53161a2f7d0')
        ulst = glymur.jp2box.UUIDListBox([the_uuid])

        version = 0
        flag = [0, 0, 0]
        url = 'ESP_032436_1755_COLOR.LBL'
        debox = glymur.jp2box.DataEntryURLBox(version, flag, url)

        uuidinfo = glymur.jp2box.UUIDInfoBox([ulst, debox])

        uuid_data = ir.read_binary(data, 'degenerate_geotiff.tif')
        the_uuid = uuid.UUID('b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03')
        geotiff_uuid = glymur.jp2box.UUIDBox(the_uuid, uuid_data)

        # Fabricate a new JP2 file out of the signature, file type, header,
        # and codestream out of nemo.jp2, but add in the UUIDInfo and UUID
        # box from HiRISE.
        jp2 = Jp2k(self.jp2file)
        boxes = [jp2.box[0], jp2.box[1], jp2.box[2], uuidinfo, geotiff_uuid,
                 jp2.box[-1]]

        self.hirise_jp2file_name = self.test_dir_path / 'hirise.jp2'
        jp2.wrap(self.hirise_jp2file_name, boxes=boxes)

    def test_tags(self):
        jp2 = Jp2k(self.hirise_jp2file_name)
        self.assertEqual(jp2.box[4].data['GeoDoubleParams'],
                         (0.0, 180.0, 0.0, 0.0, 3396190.0, 3396190.0))
        self.assertEqual(jp2.box[4].data['GeoAsciiParams'],
                         'Equirectangular MARS|GCS_MARS|')
        self.assertEqual(jp2.box[4].data['GeoKeyDirectory'], (
            1,        1,  0,    18,  # noqa
            1024,     0,  1,     1,  # noqa
            1025,     0,  1,     1,  # noqa
            1026, 34737, 21,     0,  # noqa
            2048,     0,  1, 32767,  # noqa
            2049, 34737,  9,    21,  # noqa
            2050,     0,  1, 32767,  # noqa
            2054,     0,  1,  9102,  # noqa
            2056,     0,  1, 32767,  # noqa
            2057, 34736,  1,     4,  # noqa
            2058, 34736,  1,     5,  # noqa
            3072,     0,  1, 32767,  # noqa
            3074,     0,  1, 32767,  # noqa
            3075,     0,  1,    17,  # noqa
            3076,     0,  1,  9001,  # noqa
            3082, 34736,  1,     2,  # noqa
            3083, 34736,  1,     3,  # noqa
            3088, 34736,  1,     1,  # noqa
            3089, 34736,  1,     0,  # noqa
        ))
        self.assertEqual(jp2.box[4].data['ModelPixelScale'], (0.25, 0.25, 0.0))
        self.assertEqual(jp2.box[4].data['ModelTiePoint'], (
            0.0, 0.0, 0.0, -2523306.125, -268608.875, 0.0
        ))

    @unittest.skipIf(not fixtures._HAVE_GDAL, 'Could not load GDAL')
    def test_printing_geotiff_uuid(self):
        """
        SCENARIO:  Print a geotiff UUID.

        EXPECTED RESULT:  Should match a known geotiff UUID.  The string
        representation validates.
        """
        jp2 = Jp2k(self.hirise_jp2file_name)
        self.maxDiff = None
        actual = str(jp2.box[4])

        expected = fixtures.GEOTIFF_UUID
        self.assertEqual(actual, expected)
