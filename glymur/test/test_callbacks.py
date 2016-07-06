"""
Test suite for openjpeg's callback functions.
"""
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


class TestCallbacks(unittest.TestCase):
    """Test suite for callbacks."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    @unittest.skipIf(glymur.version.openjpeg_version[0] != '2',
                     "Missing openjp2 library.")
    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_info_callback_on_write_backwards_compatibility(self):
        """Verify messages printed when writing an image in verbose mode."""
        j = glymur.Jp2k(self.jp2file)
        with warnings.catch_warnings():
            # Ignore a library warning.
            warnings.simplefilter('ignore')
            tiledata = j.read(tile=0)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with patch('sys.stdout', new=StringIO()) as fake_out:
                glymur.Jp2k(tfile.name, data=tiledata, verbose=True)
            actual = fake_out.getvalue().strip()
        expected = '[INFO] tile number 1 / 1'
        self.assertEqual(actual, expected)

    @unittest.skipIf(glymur.version.openjpeg_version[0] != '2',
                     "Missing openjp2 library.")
    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_info_callback_on_write(self):
        """Verify messages printed when writing an image in verbose mode."""
        j = glymur.Jp2k(self.jp2file)
        tiledata = j[:]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with patch('sys.stdout', new=StringIO()) as fake_out:
                glymur.Jp2k(tfile.name, data=tiledata, verbose=True)
                actual = fake_out.getvalue().strip()
        expected = '[INFO] tile number 1 / 1'
        self.assertEqual(actual, expected)

    @unittest.skipIf(glymur.version.openjpeg_version[0] == '0',
                     "Missing openjpeg/openjp2 library.")
    def test_info_callbacks_on_read(self):
        """stdio output when info callback handler is enabled"""

        # Verify that we get the expected stdio output when our internal info
        # callback handler is enabled.
        jp2 = glymur.Jp2k(self.j2kfile)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            jp2.verbose = True
            jp2[::2, ::2]
            actual = fake_out.getvalue().strip()

        if glymur.version.openjpeg_version == '2.1.1':
            lines = ['[INFO] Start to read j2k main header (0).',
                     '[INFO] Main header has been correctly decoded.',
                     '[INFO] Setting decoding area to 0,0,480,800',
                     '[INFO] Header of tile 1 / 1 has been read.',
                     '[INFO] Tile 1/1 has been decoded.',
                     '[INFO] Image data has been updated with tile 1.']

            expected = '\n'.join(lines)
            self.assertEqual(actual, expected)
        elif glymur.version.openjpeg_version[0] == '2':
            lines = ['[INFO] Start to read j2k main header (0).',
                     '[INFO] Main header has been correctly decoded.',
                     '[INFO] Setting decoding area to 0,0,480,800',
                     '[INFO] Header of tile 0 / 0 has been read.',
                     '[INFO] Tile 1/1 has been decoded.',
                     '[INFO] Image data has been updated with tile 1.']

            expected = '\n'.join(lines)
            self.assertEqual(actual, expected)
        else:
            regex = re.compile(r"""\[INFO\]\stile\s1\sof\s1\s+
                                   \[INFO\]\s-\stiers-1\stook\s
                                           [0-9]+\.[0-9]+\ss\s+
                                   \[INFO\]\s-\sdwt\stook\s
                                           (-){0,1}[0-9]+\.[0-9]+\ss\s+
                                   \[INFO\]\s-\stile\sdecoded\sin\s
                                           [0-9]+\.[0-9]+\ss""",
                               re.VERBOSE)

            if sys.hexversion <= 0x03020000:
                self.assertRegexpMatches(actual, regex)
            else:
                self.assertRegex(actual, regex)
