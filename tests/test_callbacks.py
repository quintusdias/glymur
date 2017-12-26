"""
Test suite for openjpeg's callback functions.
"""
# Standard library imports ...
import os
import sys
import tempfile
import warnings
import unittest
if sys.hexversion >= 0x03030000:
    from unittest.mock import patch
    from io import StringIO
else:
    from StringIO import StringIO

    # Third party imports ...
    from mock import patch

# Local imports ...
import glymur


class TestCallbacks(unittest.TestCase):
    """Test suite for callbacks."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    @unittest.skipIf(glymur.version.openjpeg_version < '2.0.0',
                     "Openjpeg/openjp2 library too old.")
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

    @unittest.skipIf(glymur.version.openjpeg_version < '2.0.0',
                     "Openjpeg/openjp2 library too old.")
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

    @unittest.skipIf(glymur.version.openjpeg_version < '1.5.0',
                     "Openjpeg/openjp2 library too old.")
    def test_info_callbacks_on_read(self):
        """
        Set the verbose flag, do a read operation.

        Verify that sys.stdout produces information.  Don't bother matching
        the exact string because this may change the next time OpenJPEG is
        update.
        """
        jp2 = glymur.Jp2k(self.j2kfile)
        with patch('sys.stdout', new=StringIO()) as mock_stdout:
            jp2.verbose = True
            jp2[::2, ::2]

        self.assertIn('[INFO]', mock_stdout.getvalue())
