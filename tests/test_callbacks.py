"""
Test suite for openjpeg's callback functions.
"""
# Standard library imports ...
from io import StringIO
import sys
import tempfile
import warnings
import unittest
from unittest.mock import patch

# Local imports ...
import glymur
from . import fixtures


@unittest.skipIf(fixtures.OPENJPEG_NOT_AVAILABLE,
                 fixtures.OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(unittest.TestCase):
    """Test suite for callbacks."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    @unittest.skipIf(sys.platform == 'win32', fixtures.WINDOWS_TMP_FILE_MSG)
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

    @unittest.skipIf(sys.platform == 'win32', fixtures.WINDOWS_TMP_FILE_MSG)
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
        """
        SCENARIO:  the verbose attribute is set to True

        EXPECTED RESULT:  The info callback handler should be enabled.  There
        should be [INFO] output present in sys.stdout.
        """
        jp2 = glymur.Jp2k(self.j2kfile)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            jp2.verbose = True
            jp2[::2, ::2]
            actual = fake_out.getvalue().strip()

        self.assertIn('[INFO]', actual)
