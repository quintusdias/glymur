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
class TestJPXOther(unittest.TestCase):
    """Test suite for other JPX boxes."""

    def setUp(self):
        self.jpxfile = glymur.data.jpxfile()
        pass

    def tearDown(self):
        pass

    def test_rreq_box_strange_mask_length(self):
        """The standard says that the mask length should be 1, 2, 4, or 8."""
        with warnings.catch_warnings():
            # This file has a rreq mask length that we do not recognize.
            warnings.simplefilter("ignore")
            j = Jp2k(self.jpxfile)
        self.assertEqual(j.box[2].box_id, 'rreq')
        self.assertEqual(type(j.box[2]),
                         glymur.jp2box.ReaderRequirementsBox)

    def test_free_box(self):
        """Verify that we can handle a free box."""
        with warnings.catch_warnings():
            # This file has a rreq mask length that we do not recognize.
            warnings.simplefilter("ignore")
            j = Jp2k(self.jpxfile)
        self.assertEqual(j.box[16].box[0].box_id, 'free')
        self.assertEqual(type(j.box[16].box[0]), glymur.jp2box.FreeBox)

    def test_nlst(self):
        """Verify that we can handle a free box."""
        with warnings.catch_warnings():
            # This file has a rreq mask length that we do not recognize.
            warnings.simplefilter("ignore")
            j = Jp2k(self.jpxfile)
        self.assertEqual(j.box[16].box[1].box[0].box_id, 'nlst')
        self.assertEqual(type(j.box[16].box[1].box[0]),
                         glymur.jp2box.NumberListBox)
        
        # Two associations.
        self.assertEqual(len(j.box[16].box[1].box[0].associations), 2)

        # Codestream 0
        self.assertEqual(j.box[16].box[1].box[0].associations[0], 1 << 24)

        # Compositing Layer 0
        self.assertEqual(j.box[16].box[1].box[0].associations[1], 2 << 24)


