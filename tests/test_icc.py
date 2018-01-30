# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting ICC profiles
"""

# Standard library imports ...
from datetime import datetime
import os
import struct
import unittest

# Third party library imports
import numpy as np
import pkg_resources as pkg

# Local imports
from glymur import Jp2k
from glymur._iccprofile import _ICCProfile


class TestSuite(unittest.TestCase):
    """Test suite for ICC Profile code."""

    def setUp(self):
        relpath = os.path.join('data', 'sgray.icc')
        iccfile = pkg.resource_filename(__name__, relpath)
        with open(iccfile, mode='rb') as f:
            self.buffer = f.read()

    def test_gray(self):
        """
        The ICC profile colorspace is gray.  The datetime information is null.
        """
        icc_profile = _ICCProfile(self.buffer)
        self.assertEqual(icc_profile.header['Version'], '2.1.0')
        self.assertEqual(icc_profile.header['Color Space'], 'gray')
        self.assertIsNone(icc_profile.header['Datetime'])

        # Only True for version4
        self.assertFalse('Profile Id' in icc_profile.header.keys())

    def test_bad_rendering_intent(self):
        """
        The rendering intent is not in the range 0-4.

        It should be classified as 'unknown'
        """
        intent = struct.pack('>I', 10)
        self.buffer = self.buffer[:64] + intent + self.buffer[68:]

        icc_profile = _ICCProfile(self.buffer)
        self.assertEqual(icc_profile.header['Rendering Intent'], 'unknown')

    def test_version4(self):
        """
        ICC profile is version 4
        """
        leadoff = struct.pack('>IIBB', 416, 0, 4, 0)
        self.buffer = leadoff + self.buffer[10:]

        icc_profile = _ICCProfile(self.buffer)
        self.assertEqual(icc_profile.header['Version'], '4.0.0')
        self.assertTrue('Profile Id' in icc_profile.header.keys())

    def test_icc_profile(self):
        """
        Verify full ICC profile
        """
        relpath = os.path.join('data', 'text_GBR.jp2')
        jfile = pkg.resource_filename(__name__, relpath)
        with self.assertWarns(UserWarning):
            # The brand is wrong, this is JPX, not JP2.
            j = Jp2k(jfile)
        box = j.box[3].box[1]

        self.assertEqual(box.icc_profile['Size'], 1328)
        self.assertEqual(box.icc_profile['Color Space'], 'RGB')
        self.assertEqual(box.icc_profile['Connection Space'], 'XYZ')
        self.assertEqual(box.icc_profile['Datetime'],
                         datetime(2009, 2, 25, 11, 26, 11))
        self.assertEqual(box.icc_profile['File Signature'], 'acsp')
        self.assertEqual(box.icc_profile['Platform'], 'APPL')
        self.assertEqual(box.icc_profile['Flags'],
                         'not embedded, can be used independently')
        self.assertEqual(box.icc_profile['Device Manufacturer'], 'appl')
        self.assertEqual(box.icc_profile['Device Model'], '')
        self.assertEqual(box.icc_profile['Device Attributes'],
                         ('reflective, glossy, positive media polarity, '
                          'color media'))
        self.assertEqual(box.icc_profile['Rendering Intent'], 'perceptual')
        np.testing.assert_almost_equal(box.icc_profile['Illuminant'],
                                       np.array([0.9642023, 1.0, 0.824905]),
                                       decimal=6)
        self.assertEqual(box.icc_profile['Creator'], 'appl')
