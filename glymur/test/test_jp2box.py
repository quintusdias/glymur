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
        b = glymur.jp2box.ImageHeaderBox([512, 256, 3])
        self.assertEqual(b.height,  512)
        self.assertEqual(b.width,  256)
        self.assertEqual(b.num_components,  3)
        self.assertEqual(b.bits_per_component, 8)
        self.assertFalse(b.signed)
        self.assertFalse(b.cspace_unknown)


if __name__ == "__main__":
    unittest.main()

