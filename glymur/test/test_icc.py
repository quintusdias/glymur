"""
ICC profile tests.
"""
import datetime
import unittest

import numpy as np

from glymur import Jp2k
from .fixtures import OPJ_DATA_ROOT, opj_data_file
from .fixtures import WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG


@unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestICC(unittest.TestCase):
    """ICC profile tests"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_file5(self):
        """basic ICC profile"""
        filename = opj_data_file('input/conformance/file5.jp2')
        with self.assertWarns(UserWarning):
            # The file has a bad compatibility list entry.  Not important here.
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

    def test_invalid_profile_header(self):
        """invalid ICC header data should cause UserWarning"""
        jfile = opj_data_file('input/nonregression/orb-blue10-lin-jp2.jp2')

        regex = 'ICC profile header is corrupt'
        with self.assertWarnsRegex(UserWarning, regex):
            Jp2k(jfile)
