"""
Test suite for openjpeg's callback functions.
"""
# Standard library imports ...
from io import StringIO
import warnings
import unittest
from unittest.mock import patch

# Local imports ...
import glymur
from . import fixtures


@unittest.skipIf(
    fixtures.OPENJPEG_NOT_AVAILABLE, fixtures.OPENJPEG_NOT_AVAILABLE_MSG
)
class TestSuite(fixtures.TestCommon):
    """Test suite for callbacks."""

    def test_info_callback_on_write_backwards_compatibility(self):
        """
        SCENARIO:  write to a J2K file while in verbose mode

        EXPECTED RESULT:  verify messages from the library
        """
        j = glymur.Jp2k(self.jp2file)
        with warnings.catch_warnings():
            # Ignore a library warning.
            warnings.simplefilter('ignore')
            tiledata = j.read(tile=0)

        with patch('sys.stdout', new=StringIO()) as fake_out:
            glymur.Jp2k(self.temp_j2k_filename, data=tiledata, verbose=True)
            actual = fake_out.getvalue().strip()

        expected = '[INFO] tile number 1 / 1'
        self.assertEqual(actual, expected)

    def test_info_callback_on_write(self):
        """
        SCENARIO:  write to a JP2 file while in verbose mode

        EXPECTED RESULT:  verify messages from the library
        """
        j = glymur.Jp2k(self.jp2file)
        tiledata = j[:]

        with patch('sys.stdout', new=StringIO()) as fake_out:
            glymur.Jp2k(self.temp_jp2_filename, data=tiledata, verbose=True)
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

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_info_callbacks_on_writing_tiles(self):
        """
        SCENARIO:  the verbose attribute is set to True

        EXPECTED RESULT:  The info callback handler should be enabled.  There
        should be [INFO] output present in sys.stdout.
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 3, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = glymur.Jp2k(
            self.temp_jp2_filename, shape=shape, tilesize=tilesize,
            verbose=True
        )
        with patch('sys.stdout', new=StringIO()) as fake_out:
            for tw in j.get_tilewriters():
                tw[:] = jp2_data
            actual = fake_out.getvalue().strip()

        self.assertIn('[INFO] tile number', actual)
