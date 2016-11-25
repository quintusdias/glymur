"""
Tests for general glymur functionality.
"""
# Standard library imports ...
import sys
import unittest
import warnings
if sys.hexversion >= 0x03030000:
    from unittest.mock import patch
else:
    from mock import patch

# Third party library imports ...
import numpy as np

# Local imports
import glymur
from glymur import Jp2k


class TestJp2k_1_x(unittest.TestCase):
    """Test suite for openjpeg 1.x, not appropriate for 2.x"""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def test_tile(self):
        """tile option not allowed for 1.x.
        """
        with patch('glymur.version.openjpeg_version_tuple', new=(1, 5, 0)):
            with patch('glymur.version.openjpeg_version', new="1.5.0"):
                j2k = Jp2k(self.j2kfile)
                with warnings.catch_warnings():
                    # The tile keyword is deprecated, so suppress the warning.
                    warnings.simplefilter('ignore')
                    with self.assertRaises(TypeError):
                        j2k.read(tile=0)

    def test_layer(self):
        """layer option not allowed for 1.x.
        """
        with patch('glymur.version.openjpeg_version', new="1.5.0"):
            with patch('glymur.version.openjpeg_version_tuple', new=(1, 5, 0)):
                j2k = Jp2k(self.j2kfile)
                with self.assertRaises(IOError):
                    j2k.layer = 1

    @unittest.skipIf(((glymur.lib.openjpeg.OPENJPEG is None) or
                      (glymur.lib.openjpeg.version() < '1.5.0')),
                     "OpenJPEG version one must be present")
    def test_read_version_15(self):
        """
        Test read using version 1.5
        """
        j = Jp2k(self.j2kfile)
        expected = j[:]
        with patch('glymur.jp2k.opj2.OPENJP2', new=None):
            actual = j._read_openjpeg()
            np.testing.assert_array_equal(actual, expected)

            actual = j._read_openjpeg(area=(0, 0, 250, 250))
            np.testing.assert_array_equal(actual, expected[:250, :250])
