# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting JPX box layout.
"""

import os
import struct
import sys
import tempfile
import warnings
import xml.etree.cElementTree as ET

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

import glymur
from glymur import Jp2k
from glymur.jp2box import ReaderRequirementsBox


@unittest.skipIf(sys.hexversion < 0x03000000, "Warning assert on 2.x.")
@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
class TestReaderRequirements(unittest.TestCase):
    """Test suite for XML boxes."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        pass

    def tearDown(self):
        pass

    def test_mask_length_is_3(self):
        """The standard says that the mask length should be 1, 2, 4, or 8."""
        # Rewrite nemo to include this kind of rreq box.
        with tempfile.NamedTemporaryFile(suffix=".jpx") as tfile:
            with open(self.jp2file, 'rb') as nemof:
                # Read the jP and ftyp boxes as-is.
                write_buffer = nemof.read(32)
                tfile.write(write_buffer)

                # Fake a rreq box with ML = 3.
                write_buffer = struct.pack('>I4sB', 74, b'rreq', 3)
                tfile.write(write_buffer)

                # pad the rest with zeros 
                write_buffer = struct.pack('>65s', b'\x00' * 65)
                tfile.write(write_buffer)

                # Write the rest of nemo.
                tfile.write(nemof.read())
                tfile.flush()

            with self.assertWarns(UserWarning):
                j = Jp2k(tfile.name)
            self.assertEqual(j.box[2].box_id, 'rreq')
            self.assertEqual(type(j.box[2]),
                             glymur.jp2box.ReaderRequirementsBox)

