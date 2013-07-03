import os
import unittest

import numpy as np

import glymur
from glymur import Jp2k
from glymur.lib import openjpeg as opj

try:
    data_root = os.environ['OPJ_DATA_ROOT']
except KeyError:
    data_root = None
except:
    raise


@unittest.skipIf(glymur.lib.openjpeg._OPENJPEG is None,
                 "Missing openjpeg library.")
class TestJp2k15(unittest.TestCase):

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    def test_layers(self):
        # Layers not allowed for 1.5.1.
        self.assertTrue(False)

    def test_area(self):
        # Area option not allowed for 1.5.1.
        self.assertTrue(False)

    def test_tile(self):
        # Tile option not allowed for 1.5.1.
        self.assertTrue(False)

    def test_verbose(self):
        # Verbose option not allowed for 1.5.1.
        self.assertTrue(False)

    def test_differing_subsampling(self):
        # Only images with same subsampling is allowed.
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
