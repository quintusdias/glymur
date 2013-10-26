# -*- coding:  utf-8 -*-
"""Test suite for printing.
"""
# C0302:  don't care too much about having too many lines in a test module
# pylint: disable=C0302

# E061:  unittest.mock introduced in 3.3 (python-2.7/pylint issue)
# pylint: disable=E0611,F0401

# R0904:  Not too many methods in unittest.
# pylint: disable=R0904

import os
import re
import struct
import sys
import tempfile
import warnings
from xml.etree import cElementTree as ET

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

if sys.hexversion < 0x03000000:
    from StringIO import StringIO
else:
    from io import StringIO

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch

import glymur
from glymur import Jp2k
from .fixtures import OPJ_DATA_ROOT, opj_data_file, nemo_xmp_box


class TestUUIDExif(unittest.TestCase):
    """Tests for UUIDs of Exif type."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    @unittest.skipIf(sys.hexversion < 0x03000000, "Requires assertWarns, 3.2+")
    def test_unrecognized_exif_tag(self):
        """Verify warning in case of unrecognized tag."""
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

            # The "Make" tag is tag no. 271.  Corrupt it to 171.
            tfile.write(struct.pack('<HHI4s', 171, 2, 3, b'HTC\x00'))
            tfile.flush()

            with self.assertWarns(UserWarning):
                j = glymur.Jp2k(tfile.name)

    @unittest.skipIf(sys.hexversion < 0x03000000, "Requires assertWarns, 3.2+")
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

            with self.assertWarns(UserWarning):
                j = glymur.Jp2k(tfile.name)

            self.assertEqual(j.box[-1].box_id, 'uuid')

    @unittest.skipIf(sys.hexversion < 0x03000000, "Requires assertWarns, 3.2+")
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

            with self.assertWarns(UserWarning):
                j = glymur.Jp2k(tfile.name)

            self.assertEqual(j.box[-1].box_id, 'uuid')

if __name__ == "__main__":
    unittest.main()
