# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting ICC profiles
"""

# Standard library imports ...
from datetime import datetime
try:
    import importlib.resources as ir
except ImportError:
    import importlib_resources as ir
from io import BytesIO
import struct
import unittest

# Third party library imports
import numpy as np

# Local imports
from glymur._iccprofile import _ICCProfile
from glymur.core import ANY_ICC_PROFILE
from glymur.jp2box import ColourSpecificationBox
from . import data


class TestSuite(unittest.TestCase):
    """Test suite for ICC Profile code."""

    def setUp(self):
        self.buffer = ir.read_binary(data, 'sgray.icc')

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
        fp = BytesIO()
        fp.write(b'\x00' * 179)

        # Write the colr box header.
        buffer = struct.pack('>I4s', 1339, b'colr')
        buffer += struct.pack('>BBB', ANY_ICC_PROFILE, 2, 1)

        size = 1328
        preferred_cmm_type = 1634758764
        color_space = 'RGB'
        connection_space = 'XYZ'

        buffer += struct.pack('>IIBB', size, preferred_cmm_type, 2, 32)
        buffer += b'\x00' * 2 + b'mntr'
        buffer += color_space.encode('utf-8') + b' '
        buffer += connection_space.encode('utf-8') + b' '

        # Need a date in bytes 24:36
        dt = datetime(2009, 2, 25, 11, 26, 11)
        pargs = dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
        buffer += struct.pack('>HHHHHH', *pargs)

        file_signature = 'ascp'
        buffer += file_signature.encode('utf-8')

        platform = 'APPL'
        buffer += platform.encode('utf-8')

        buffer += b'\x00' * 4
        buffer += 'appl'.encode('utf-8')  # 48 - 52

        buffer += b'\x00' * 16
        buffer += struct.pack('>III', 63190, 65536, 54061)  # 68 - 80

        device_manufacturer = 'appl'
        buffer += device_manufacturer.encode('utf-8')  # 80 - 84

        buffer += b'\x00' * 44
        fp.write(buffer)
        fp.seek(179 + 8)

        # Should be able to read the colr box now
        box = ColourSpecificationBox.parse(fp, 179, 1339)

        self.assertEqual(box.icc_profile['Size'], size)
        self.assertEqual(box.icc_profile['Preferred CMM Type'],
                         preferred_cmm_type)
        self.assertEqual(box.icc_profile['Color Space'], color_space)
        self.assertEqual(box.icc_profile['Connection Space'], connection_space)
        self.assertEqual(box.icc_profile['Datetime'], dt)
        self.assertEqual(box.icc_profile['File Signature'], file_signature)
        self.assertEqual(box.icc_profile['Platform'], platform)
        self.assertEqual(box.icc_profile['Flags'],
                         'not embedded, can be used independently')
        self.assertEqual(box.icc_profile['Device Manufacturer'],
                         device_manufacturer)
        self.assertEqual(box.icc_profile['Device Model'], '')
        self.assertEqual(box.icc_profile['Device Attributes'],
                         ('reflective, glossy, positive media polarity, '
                          'color media'))
        self.assertEqual(box.icc_profile['Rendering Intent'], 'perceptual')
        np.testing.assert_almost_equal(box.icc_profile['Illuminant'],
                                       np.array([0.9642023, 1.0, 0.824905]),
                                       decimal=6)
