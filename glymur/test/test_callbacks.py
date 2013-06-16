import os
import pkg_resources
import sys
import tempfile
import unittest
import warnings

if sys.hexversion < 0x03000000:
    from StringIO import StringIO
else:
    from io import StringIO

import glymur


@unittest.skipIf(glymur.lib.openjp2._OPENJP2 is None,
                 "Missing openjp2 library.")
class TestCallbacks(unittest.TestCase):

    def setUp(self):
        # Save sys.stdout.
        self.stdout = sys.stdout
        sys.stdout = StringIO()
        self.jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                       "data/nemo.jp2")

    def tearDown(self):
        # Restore stdout.
        sys.stdout = self.stdout

    def test_info_callback_on_write(self):
        # Verify the messages printed when writing an image in verbose mode.
        j = glymur.Jp2k(self.jp2file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiledata = j.read(tile=0)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            j = glymur.Jp2k(tfile.name, 'wb')
            j.write(tiledata, verbose=True)
        actual = sys.stdout.getvalue().strip()
        expected = '[INFO] tile number 1 / 1'
        self.assertEqual(actual, expected)

    def test_info_warning_callbacks_on_read(self):
        # Verify that we get the expected stdio output when our internal info
        # callback handler is enabled.
        j = glymur.Jp2k(self.jp2file)
        d = j.read(reduce=3, verbose=True, area=(0, 0, 512, 1024))
        actual = sys.stdout.getvalue().strip()

        lines = ['[INFO] Start to read j2k main header (3135).',
                 '[INFO] Main header has been correctly decoded.',
                 '[INFO] Setting decoding area to 0,0,1024,512',
                 '[INFO] Header of tile 0 / 17 has been read.',
                 '[INFO] Tile 1/18 has been decoded.',
                 '[INFO] Image data has been updated with tile 1.',
                 '[INFO] Header of tile 1 / 17 has been read.',
                 '[INFO] Tile 2/18 has been decoded.',
                 '[INFO] Image data has been updated with tile 2.',
                 '[INFO] Stream reached its end !']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
