# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting ICC profiles
"""

# Standard library imports ...
import os
import struct
import unittest

# Third party library imports
import pkg_resources as pkg

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
