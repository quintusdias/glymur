# -*- coding:  utf-8 -*-
"""Test suite for printing.
"""
import os
import pkg_resources as pkg
import shutil
import struct
import sys
import tempfile
import unittest
import uuid
import warnings

if sys.hexversion <= 0x03000000:
    from mock import patch
    from StringIO import StringIO
else:
    from unittest.mock import patch
    from io import StringIO

import lxml.etree

from . import fixtures

import glymur
from glymur import Jp2k
from .fixtures import SimpleRDF


@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
class TestSuite(unittest.TestCase):
    """Tests for XMP, Exif UUIDs."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

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
            self.assertTrue(isinstance(jp2.box[-1].data,
                                       lxml.etree._ElementTree))

    def test_big_endian_exif(self):
        """Verify read of Exif big-endian IFD."""
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as tfile:

            with open(self.jp2file, 'rb') as ifptr:
                tfile.write(ifptr.read())

            # Write L, T, UUID identifier.
            tfile.write(struct.pack('>I4s', 52, b'uuid'))
            tfile.write(b'JpgTiffExif->JP2')

            tfile.write(b'Exif\x00\x00')
            xbuffer = struct.pack('>BBHI', 77, 77, 42, 8)
            tfile.write(xbuffer)

            # We will write just a single tag.
            tfile.write(struct.pack('>H', 1))

            # The "Make" tag is tag no. 271.
            tfile.write(struct.pack('>HHI4s', 271, 2, 3, b'HTC\x00'))
            tfile.flush()

            jp2 = glymur.Jp2k(tfile.name)
            self.assertEqual(jp2.box[-1].data['Make'], "HTC")


@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
class TestSuiteHiRISE(unittest.TestCase):
    """Tests for HiRISE RDRs."""

    def setUp(self):
        # Hand-create the boxes needed for HiRISE.
        the_uuid = uuid.UUID('2b0d7e97-aa2e-317d-9a33-e53161a2f7d0')
        ulst = glymur.jp2box.UUIDListBox([the_uuid])

        version = 0
        flag = [0, 0, 0]
        url = 'ESP_032436_1755_COLOR.LBL'
        debox = glymur.jp2box.DataEntryURLBox(version, flag, url)

        uuidinfo = glymur.jp2box.UUIDInfoBox([ulst, debox])

        relpath = os.path.join('data', 'degenerate_geotiff.tif')
        path = pkg.resource_filename(__name__, relpath)
        with open(path, 'rb') as fptr:
            uuid_data = fptr.read()
        the_uuid = uuid.UUID('b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03')
        geotiff_uuid = glymur.jp2box.UUIDBox(the_uuid, uuid_data)

        # Fabricate a new JP2 file out of the signature, file type, header,
        # and codestream out of nemo.jp2, but add in the UUIDInfo and UUID
        # box from HiRISE.
        jp2 = Jp2k(glymur.data.nemo())
        boxes = [jp2.box[0], jp2.box[1], jp2.box[2], uuidinfo, geotiff_uuid,
                 jp2.box[-1]]

        with tempfile.NamedTemporaryFile(suffix=".jp2", delete=False) as tfile:
            jp2.wrap(tfile.name, boxes=boxes)
        self.hirise_jp2file_name = tfile.name

    def tearDown(self):
        os.unlink(self.hirise_jp2file_name)

    def test_tags(self):
        jp2 = Jp2k(self.hirise_jp2file_name)
        self.assertEqual(jp2.box[4].data['GeoDoubleParams'],
                         (0.0, 180.0, 0.0, 0.0, 3396190.0, 3396190.0))
        self.assertEqual(jp2.box[4].data['GeoAsciiParams'],
                         'Equirectangular MARS|GCS_MARS|')
        self.assertEqual(jp2.box[4].data['GeoKeyDirectory'], (
            1,        1,  0,    18,
            1024,     0,  1,     1,
            1025,     0,  1,     1,
            1026, 34737, 21,     0,
            2048,     0,  1, 32767,
            2049, 34737,  9,    21,
            2050,     0,  1, 32767,
            2054,     0,  1,  9102,
            2056,     0,  1, 32767,
            2057, 34736,  1,     4,
            2058, 34736,  1,     5,
            3072,     0,  1, 32767,
            3074,     0,  1, 32767,
            3075,     0,  1,    17,
            3076,     0,  1,  9001,
            3082, 34736,  1,     2,
            3083, 34736,  1,     3,
            3088, 34736,  1,     1,
            3089, 34736,  1,     0
        ))
        self.assertEqual(jp2.box[4].data['ModelPixelScale'], (0.25, 0.25, 0.0))
        self.assertEqual(jp2.box[4].data['ModelTiePoint'], (
            0.0, 0.0, 0.0, -2523306.125, -268608.875, 0.0
        ))

    @unittest.skipIf('Anaconda' in sys.version, 'Problem with corner coords')
    def test_printing(self):
        jp2 = Jp2k(self.hirise_jp2file_name)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(jp2.box[4])
            actual = fake_out.getvalue().strip()
        if fixtures.HAVE_GDAL:
            expected = fixtures.geotiff_uuid
        else:
            expected = fixtures.geotiff_uuid_without_gdal
        self.assertEqual(actual, expected)


@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
class TestSuiteWarns(unittest.TestCase):
    """Tests for XMP, Exif UUIDs, issues warnings."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_bad_tag_datatype(self):
        """Only certain datatypes are allowable"""
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as tfile:

            with open(self.jp2file, 'rb') as ifptr:
                tfile.write(ifptr.read())

            # Write L, T, UUID identifier.
            tfile.write(struct.pack('>I4s', 52, b'uuid'))
            tfile.write(b'JpgTiffExif->JP2')

            tfile.write(b'Exif\x00\x00')
            xbuffer = struct.pack('<BBHI', 73, 73, 42, 8)
            tfile.write(xbuffer)

            # We will write just a single tag.
            tfile.write(struct.pack('<H', 1))

            # 2000 is not an allowable TIFF datatype.
            tfile.write(struct.pack('<HHI4s', 271, 2000, 3, b'HTC\x00'))
            tfile.flush()

            with warnings.catch_warnings():
                # Ignore the invalid datatype warnings.
                warnings.simplefilter('ignore')
                j = glymur.Jp2k(tfile.name)

            self.assertEqual(j.box[-1].box_id, 'uuid')

            # Invalid tag, so no data
            self.assertIsNone(j.box[-1].data)

    def test_bad_tiff_header_byte_order_indication(self):
        """Only b'II' and b'MM' are allowed."""
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as tfile:

            with open(self.jp2file, 'rb') as ifptr:
                tfile.write(ifptr.read())

            # Write L, T, UUID identifier.
            tfile.write(struct.pack('>I4s', 52, b'uuid'))
            tfile.write(b'JpgTiffExif->JP2')

            tfile.write(b'Exif\x00\x00')
            xbuffer = struct.pack('<BBHI', 74, 73, 42, 8)
            tfile.write(xbuffer)

            # We will write just a single tag.
            tfile.write(struct.pack('<H', 1))

            # 271 is the Make.
            tfile.write(struct.pack('<HHI4s', 271, 2, 3, b'HTC\x00'))
            tfile.flush()

            with warnings.catch_warnings():
                # Ignore the warning about the endian order, we test for that
                # elsewhere.
                warnings.simplefilter('ignore')
                jp2 = glymur.Jp2k(tfile.name)

            # We should still get a UUID box out of it.  But we get no data.
            self.assertEqual(jp2.box[-1].box_id, 'uuid')
            self.assertIsNone(jp2.box[-1].data)
