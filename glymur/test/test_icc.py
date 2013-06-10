import datetime
import os
import struct
import sys
import tempfile
import unittest
import warnings
from xml.etree import cElementTree as ET

import numpy as np
import pkg_resources

from glymur import Jp2k
import glymur

try:
    data_root = os.environ['OPJ_DATA_ROOT']
except KeyError:
    data_root = None
except:
    raise


class TestICC(unittest.TestCase):

    def setUp(self):
        self.jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                       "data/nemo.jp2")

    def tearDown(self):
        pass

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_file5(self):
        filename = os.path.join(data_root, 'input/conformance/file5.jp2')
        j = Jp2k(filename)
        profile = j.box[3].box[1].icc_profile
        self.assertEqual(profile.size, 546)
        self.assertEqual(profile.preferred_cmm_type, 0)
        self.assertEqual(profile.version, '2.2.0')
        self.assertEqual(profile.device_class, 'input device profile')
        self.assertEqual(profile.colour_space, 'RGB')
        self.assertEqual(profile.datetime,
                         datetime.datetime(2001,8,30,13,32,37))
        self.assertEqual(profile.file_signature, 'acsp')
        self.assertEqual(profile.platform, 'unrecognized')
        self.assertTrue(profile.flags & 0x01)  # embedded
        self.assertFalse(profile.flags & 0x02)  # use anywhere

        self.assertEqual(profile.device_manufacturer, 'KODA')
        self.assertEqual(profile.device_model, 'ROMM')

        self.assertFalse(profile.device_attributes & 0x01)  # reflective
        self.assertFalse(profile.device_attributes & 0x02)  # glossy
        self.assertFalse(profile.device_attributes & 0x04)  # positive
        self.assertFalse(profile.device_attributes & 0x08)  # colour
        self.assertEqual(profile.rendering_intent & 0x00ff, 0)  # perceptual

        np.testing.assert_almost_equal(profile.illuminant,
                                       (0.964203, 1.000000, 0.824905),
                                       decimal=6)

        self.assertEqual(profile.creator, 'JPEG')

if __name__ == "__main__":
    unittest.main()

