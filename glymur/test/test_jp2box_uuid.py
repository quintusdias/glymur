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
import shutil
import struct
import sys
import tempfile
import uuid
import warnings

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

import lxml.etree

from .fixtures import HAS_PYTHON_XMP_TOOLKIT, OPJ_DATA_ROOT
if HAS_PYTHON_XMP_TOOLKIT:
    from libxmp import XMPMeta

import glymur
from glymur import Jp2k
from .fixtures import OPJ_DATA_ROOT, opj_data_file, SimpleRDF


@unittest.skipIf(os.name == "nt", "Unexplained failure on windows")
class TestUUIDXMP(unittest.TestCase):
    """Tests for UUIDs of XMP type."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_append(self):
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

@unittest.skipIf(os.name == "nt", "Unexplained failure on windows")
class TestUUIDExif(unittest.TestCase):
    """Tests for UUIDs of Exif type."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

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

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                j = glymur.Jp2k(tfile.name)
                self.assertTrue(issubclass(w[0].category, UserWarning))
                msg = 'Unrecognized Exif tag'
                self.assertTrue(msg in str(w[0].message))

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

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                j = glymur.Jp2k(tfile.name)
                self.assertTrue(issubclass(w[0].category, UserWarning))
                msg = 'Invalid TIFF tag'
                self.assertTrue(msg in str(w[0].message))

            self.assertEqual(j.box[-1].box_id, 'uuid')

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

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                j = glymur.Jp2k(tfile.name)
                self.assertTrue(issubclass(w[0].category, UserWarning))
                msg = 'The byte order indication in the TIFF header '
                if sys.hexversion < 0x03000000:
                    msg += "(JI) is invalid.  "
                    msg += "It should be either [73, 73] or [77, 77]."
                else:
                    msg += "(b'JI') is invalid.  "
                    msg += "It should be either b'II' or b'MM'."
                self.assertTrue(msg in str(w[0].message))

            self.assertEqual(j.box[-1].box_id, 'uuid')

    def test_big_endian(self):
        """Verify read of big-endian IFD."""
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

if __name__ == "__main__":
    unittest.main()
