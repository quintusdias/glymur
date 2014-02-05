"""
Test suite for openjpeg's callback functions.
"""
# R0904:  Seems like pylint is fooled in this situation
# pylint: disable=R0904

# 'mock' most certainly is in unittest (Python 3.3)
# pylint: disable=E0611,F0401

import os
import re
import sys
import tempfile
import warnings

import unittest

if sys.hexversion <= 0x03030000:
    from mock import patch
    from StringIO import StringIO
else:
    from unittest.mock import patch
    from io import StringIO

import glymur


@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Missing openjp2 library.")
class TestCallbacks(unittest.TestCase):
    """Test suite for callbacks."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_info_callback_on_write(self):
        """Verify messages printed when writing an image in verbose mode."""
        j = glymur.Jp2k(self.jp2file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiledata = j.read(tile=0)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            j = glymur.Jp2k(tfile.name, 'wb')
            with patch('sys.stdout', new=StringIO()) as fake_out:
                j.write(tiledata, verbose=True)
                actual = fake_out.getvalue().strip()
        expected = '[INFO] tile number 1 / 1'
        self.assertEqual(actual, expected)

    def test_info_callbacks_on_read(self):
        """stdio output when info callback handler is enabled"""

        # Verify that we get the expected stdio output when our internal info
        # callback handler is enabled.
        j = glymur.Jp2k(self.j2kfile)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            j.read(rlevel=1, verbose=True, area=(0, 0, 200, 150))
            actual = fake_out.getvalue().strip()

        lines = ['[INFO] Start to read j2k main header (0).',
                 '[INFO] Main header has been correctly decoded.',
                 '[INFO] Setting decoding area to 0,0,150,200',
                 '[INFO] Header of tile 0 / 0 has been read.',
                 '[INFO] Tile 1/1 has been decoded.',
                 '[INFO] Image data has been updated with tile 1.']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)


@unittest.skipIf(glymur.lib.openjpeg.OPENJPEG is None,
                 "Missing openjpeg library.")
class TestCallbacks15(unittest.TestCase):
    """This test suite is for OpenJPEG 1.5.1 properties.
    """

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    def test_info_callbacks_on_read(self):
        """Verify stdout when reading.

        Verify that we get the expected stdio output when our internal info
        callback handler is enabled.
        """
        with patch('glymur.lib.openjp2.OPENJP2', new=None):
            # Force to use OPENJPEG instead of OPENJP2.
            j = glymur.Jp2k(self.j2kfile)
            with patch('sys.stdout', new=StringIO()) as fake_out:
                j.read(rlevel=1, verbose=True)
                actual = fake_out.getvalue().strip()

            regex = re.compile(r"""\[INFO\]\stile\s1\sof\s1\s+
                                   \[INFO\]\s-\stiers-1\stook\s
                                           [0-9]+\.[0-9]+\ss\s+
                                   \[INFO\]\s-\sdwt\stook\s
                                           (-){0,1}[0-9]+\.[0-9]+\ss\s+
                                   \[INFO\]\s-\stile\sdecoded\sin\s
                                           [0-9]+\.[0-9]+\ss""",
                               re.VERBOSE)

            # assertRegex in Python 3.3 (python2.7/pylint issue)
            # pylint: disable=E1101
            if sys.hexversion <= 0x03020000:
                self.assertRegexpMatches(actual, regex)
            else:
                self.assertRegex(actual, regex)


if __name__ == "__main__":
    unittest.main()
