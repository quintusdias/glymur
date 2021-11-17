# -*- coding:  utf-8 -*-
"""
Test suite specifically targeting ICC profiles
"""

# Standard library imports ...
from datetime import datetime
import importlib.resources as ir
import struct
import tempfile
import unittest
import warnings

# Third party library imports
import numpy as np

# Local imports
import glymur
from glymur import Jp2k
from glymur._iccprofile import _ICCProfile
from glymur.jp2box import (
    ColourSpecificationBox, ContiguousCodestreamBox, FileTypeBox,
    ImageHeaderBox, JP2HeaderBox, JPEG2000SignatureBox, InvalidJp2kError
)
from glymur.core import SRGB
from . import fixtures, data


class TestColourSpecificationBox(fixtures.TestCommon):
    """Test suite for colr box instantiation."""

    def setUp(self):
        super(TestColourSpecificationBox, self).setUp()

        j2k = Jp2k(self.j2kfile)
        codestream = j2k.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)

        self.jp2b = JPEG2000SignatureBox()
        self.ftyp = FileTypeBox()
        self.jp2h = JP2HeaderBox()
        self.jp2c = ContiguousCodestreamBox()
        self.ihdr = ImageHeaderBox(
            height=height,
            width=width,
            num_components=num_components
        )

        self.icc_profile = ir.read_binary(data, 'sgray.icc')

    def test_bad_method_printing(self):
        """
        SCENARIO:  An ICC profile is both too short and has an invalid method
        value.

        EXPECTED RESULT:  Warnings are issued.  Printing the string
        representation should not error out.
        """
        with ir.path(data, 'issue405.dat') as path:
            with path.open('rb') as f:
                f.seek(8)
                with warnings.catch_warnings():
                    # Lots of things wrong with this file.
                    warnings.simplefilter('ignore')
                    box = ColourSpecificationBox.parse(f, length=80, offset=0)
                    str(box)

    def test_colr_with_out_enum_cspace(self):
        """must supply an enumerated colorspace when writing"""
        j2k = Jp2k(self.j2kfile)

        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        boxes[2].box = [self.ihdr, ColourSpecificationBox(colorspace=None)]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(InvalidJp2kError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_missing_colr_box(self):
        """jp2h must have a colr box"""
        j2k = Jp2k(self.j2kfile)
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        boxes[2].box = [self.ihdr]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(InvalidJp2kError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_bad_approx_jp2_field(self):
        """JP2 has requirements for approx field"""
        j2k = Jp2k(self.j2kfile)
        boxes = [self.jp2b, self.ftyp, self.jp2h, self.jp2c]
        colr = ColourSpecificationBox(colorspace=SRGB, approximation=1)
        boxes[2].box = [self.ihdr, colr]
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with self.assertRaises(InvalidJp2kError):
                j2k.wrap(tfile.name, boxes=boxes)

    def test_default_colr(self):
        """basic colr instantiation"""
        colr = ColourSpecificationBox(colorspace=SRGB)
        self.assertEqual(colr.method, glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(colr.precedence, 0)
        self.assertEqual(colr.approximation, 0)
        self.assertEqual(colr.colorspace, SRGB)
        self.assertIsNone(colr.icc_profile)

    def test_icc_profile(self):
        """basic colr box with ICC profile"""
        colr = ColourSpecificationBox(icc_profile=self.icc_profile)
        self.assertEqual(colr.method, glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(colr.precedence, 0)
        self.assertEqual(colr.approximation, 0)

        icc_profile = _ICCProfile(colr.icc_profile)
        self.assertEqual(icc_profile.header['Version'], '2.1.0')
        self.assertEqual(icc_profile.header['Color Space'], 'gray')
        self.assertIsNone(icc_profile.header['Datetime'])

        # Only True for version4
        self.assertFalse('Profile Id' in icc_profile.header.keys())

    def test_colr_with_bad_color(self):
        """
        SCENARIO:  A colr box has an invalid colorspace.

        EXPECTED RESULT:  An InvalidJp2kError is raised when attempting to
        write the box.
        """
        with self.assertWarns(UserWarning):
            # A warning is issued due to the bad colorspace.
            colr = ColourSpecificationBox(colorspace=-1, approximation=0)

        with tempfile.TemporaryFile() as tfile:
            with self.assertRaises(InvalidJp2kError):
                colr.write(tfile)

    def test_write_colr_with_bad_method(self):
        """
        SCENARIO:  A colr box has an invalid method value.

        EXPECTED RESULT:  InvalidJp2kError
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            colr = ColourSpecificationBox(colorspace=SRGB, method=5)
        with tempfile.TemporaryFile() as tfile:
            with self.assertRaises(InvalidJp2kError):
                colr.write(tfile)


class TestSuite(unittest.TestCase):
    """Test suite for ICC Profile code."""

    def setUp(self):
        self.buffer = ir.read_binary(data, 'sgray.icc')

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
        SCENARIO:  The ColourDefinitionBox has an ICC profile.

        EXPECTED RESULT:  Verify the ICC profile metadata.
        """
        with ir.path(data, 'text_GBR.jp2') as path:
            with self.assertWarns(UserWarning):
                # The brand is wrong, this is JPX, not JP2.
                j = Jp2k(path)
        box = j.box[3].box[1]

        self.assertEqual(box.icc_profile_header['Size'], 1328)
        self.assertEqual(box.icc_profile_header['Color Space'], 'RGB')
        self.assertEqual(box.icc_profile_header['Connection Space'], 'XYZ')
        self.assertEqual(
            box.icc_profile_header['Datetime'],
            datetime(2009, 2, 25, 11, 26, 11)
        )
        self.assertEqual(box.icc_profile_header['File Signature'], 'acsp')
        self.assertEqual(box.icc_profile_header['Platform'], 'APPL')
        self.assertEqual(
            box.icc_profile_header['Flags'],
            'not embedded, can be used independently'
        )
        self.assertEqual(box.icc_profile_header['Device Manufacturer'], 'appl')
        self.assertEqual(box.icc_profile_header['Device Model'], '')
        self.assertEqual(
            box.icc_profile_header['Device Attributes'],
            'reflective, glossy, positive media polarity, color media'
        )
        self.assertEqual(
            box.icc_profile_header['Rendering Intent'], 'perceptual'
        )
        np.testing.assert_almost_equal(
            box.icc_profile_header['Illuminant'],
            np.array([0.9642023, 1.0, 0.824905]),
            decimal=6
        )
        self.assertEqual(box.icc_profile_header['Creator'], 'appl')
