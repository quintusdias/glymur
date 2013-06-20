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


@unittest.skipIf(data_root is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestICC(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_file5(self):
        filename = os.path.join(data_root, 'input/conformance/file5.jp2')
        j = Jp2k(filename)
        profile = j.box[3].box[1].icc_profile
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
                         'reflective, glossy, positive media polarity, '
                         + 'color media')
        self.assertEqual(profile['Rendering Intent'], 'perceptual')

        np.testing.assert_almost_equal(profile['Illuminant'],
                                       (0.964203, 1.000000, 0.824905),
                                       decimal=6)

        self.assertEqual(profile['Creator'], 'JPEG')

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_invalid_profile_header(self):
        jfile = os.path.join(data_root,
                             'input/nonregression/orb-blue10-lin-jp2.jp2')
        with self.assertWarns(UserWarning) as cw:
            j = Jp2k(jfile)

if __name__ == "__main__":
    unittest.main()
