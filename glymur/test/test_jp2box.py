import doctest
import unittest

import numpy as np
import pkg_resources

import glymur

# Doc tests should be run as well.
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite('glymur.jp2box'))
    return tests

@unittest.skipIf(glymur.lib.openjp2._OPENJP2 is None,
                 "Missing openjp2 library.")
class TestJp2Boxes(unittest.TestCase):

    def setUp(self):
        self.jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                       "data/nemo.jp2")

    def tearDown(self):
        pass

    def test_default_JPEG2000SignatureBox(self):
        # Should be able to instantiate a JPEG2000SignatureBox 
        b = glymur.jp2box.JPEG2000SignatureBox()
        self.assertEqual(b.signature, (13, 10, 135, 10))

    def test_default_FileTypeBox(self):
        # Should be able to instantiate a FileTypeBox 
        b = glymur.jp2box.FileTypeBox()
        self.assertEqual(b.brand, 'jp2 ')
        self.assertEqual(b.minor_version, 0)
        self.assertEqual(b.compatibility_box, ['jp2 '])

    def test_default_ImageHeaderBox(self):
        # Should be able to instantiate an image header box.
        b = glymur.jp2box.ImageHeaderBox(height=512, width=256,
                                         num_components=3)
        self.assertEqual(b.height,  512)
        self.assertEqual(b.width,  256)
        self.assertEqual(b.num_components,  3)
        self.assertEqual(b.bits_per_component, 8)
        self.assertFalse(b.signed)
        self.assertFalse(b.colorspace_unknown)

    def test_default_ColourSpecificationBox(self):
        b = glymur.jp2box.ColourSpecificationBox(colorspace=glymur.core.SRGB)
        self.assertEqual(b.method,  glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(b.precedence, 0)
        self.assertEqual(b.approximation, 1)
        self.assertEqual(b.colorspace, glymur.core.SRGB)
        self.assertIsNone(b.icc_profile)

    def test_ColourSpecificationBox_with_colorspace_and_icc(self):
        # Colour specification boxes can't have both.
        with self.assertRaises(IOError):
            colorspace = glymur.core.SRGB
            icc_profile = b'\x01\x02\x03\x04'
            b = glymur.jp2box.ColourSpecificationBox(colorspace, icc_profile)

    def test_ColourSpecificationBox_with_bad_method(self):
        colorspace = glymur.core.SRGB
        method = -1
        with self.assertRaises(IOError):
            b = glymur.jp2box.ColourSpecificationBox(colorspace, method)

    def test_ColourSpecificationBox_with_bad_approximation(self):
        colorspace = glymur.core.SRGB
        approximation = -1
        with self.assertRaises(IOError):
            b = glymur.jp2box.ColourSpecificationBox(colorspace, approximation)


if __name__ == "__main__":
    unittest.main()

