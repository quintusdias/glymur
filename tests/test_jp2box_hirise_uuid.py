# -*- coding:  utf-8 -*-
"""Test suite for printing.
"""
# Standard library imports
import os
import pkg_resources as pkg
import sys
import tempfile
import unittest
import uuid

# Local imports
import glymur
from glymur import Jp2k
from . import fixtures


class TestSuite(unittest.TestCase):
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
        with open(path, 'rb') as f:
            uuid_data = f.read()
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
            1, 1, 0, 18,
            1024, 0, 1, 1,
            1025, 0, 1, 1,
            1026, 34737, 21, 0,
            2048, 0, 1, 32767,
            2049, 34737, 9, 21,
            2050, 0, 1, 32767,
            2054, 0, 1, 9102,
            2056, 0, 1, 32767,
            2057, 34736, 1, 4,
            2058, 34736, 1, 5,
            3072, 0, 1, 32767,
            3074, 0, 1, 32767,
            3075, 0, 1, 17,
            3076, 0, 1, 9001,
            3082, 34736, 1, 2,
            3083, 34736, 1, 3,
            3088, 34736, 1, 1,
            3089, 34736, 1, 0
        ))
        self.assertEqual(jp2.box[4].data['ModelPixelScale'], (0.25, 0.25, 0.0))
        self.assertEqual(jp2.box[4].data['ModelTiePoint'], (
            0.0, 0.0, 0.0, -2523306.125, -268608.875, 0.0
        ))

    @unittest.skipIf(sys.hexversion < 0x03000000, "Don't bother testing u''")
    def test_printing(self):
        jp2 = Jp2k(self.hirise_jp2file_name)
        actual = str(jp2.box[4])
        if fixtures.HAVE_GDAL:
            expected = fixtures.geotiff_uuid
        else:
            expected = fixtures.geotiff_uuid_without_gdal
        self.assertEqual(actual, expected)
