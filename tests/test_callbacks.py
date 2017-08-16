"""
Test suite for openjpeg's callback functions.
"""
# Standard library imports ...
from io import StringIO
import os
import tempfile
import warnings
import unittest
from unittest.mock import patch

# Local imports ...
import glymur


@unittest.skipIf(glymur.version.openjpeg_version[0] != '2',
                 "Missing openjp2 library.")
class TestCallbacks(unittest.TestCase):
    """Test suite for callbacks."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

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

    def test_info_callbacks_on_read(self):
        """stdio output when info callback handler is enabled"""

        # Verify that we get the expected stdio output when our internal info
        # callback handler is enabled.
        jp2 = glymur.Jp2k(self.j2kfile)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            jp2.verbose = True
            jp2[::2, ::2]
            actual = fake_out.getvalue().strip()

        expected = ('[INFO] Start to read j2k main header (0).\n'
                    '[INFO] Main header has been correctly decoded.\n'
                    '[INFO] Setting decoding area to 0,0,480,800\n'
                    '[INFO] Header of tile 1 / 1 has been read.\n'
                    '[INFO] Tile 1/1 has been decoded.\n'
                    '[INFO] Image data has been updated with tile 1.')
        self.assertEqual(actual, expected)
