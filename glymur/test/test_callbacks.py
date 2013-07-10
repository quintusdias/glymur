#pylint:  disable-all
import os
import pkg_resources
import re
import sys
import tempfile
import unittest
import warnings

if sys.hexversion < 0x03000000:
    from StringIO import StringIO
else:
    from io import StringIO

import glymur


@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Missing openjp2 library.")
class TestCallbacks(unittest.TestCase):

    def setUp(self):
        # Save sys.stdout.
        self.stdout = sys.stdout
        sys.stdout = StringIO()
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        # Restore stdout.
        sys.stdout = self.stdout

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
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
        j = glymur.Jp2k(self.j2kfile)
        d = j.read(rlevel=1, verbose=True, area=(0, 0, 200, 150))
        actual = sys.stdout.getvalue().strip()

        lines = ['[INFO] Start to read j2k main header (0).',
                 '[INFO] Main header has been correctly decoded.',
                 '[INFO] Setting decoding area to 0,0,150,200',
                 '[INFO] Header of tile 0 / 0 has been read.',
                 '[INFO] Tile 1/1 has been decoded.',
                 '[INFO] Image data has been updated with tile 1.']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)


@unittest.skipIf(glymur.lib.openjp2.OPENJPEG is None,
                 "Missing openjpeg library.")
class TestCallbacks15(unittest.TestCase):
    """This test suite is for OpenJPEG 1.5.1 properties.
    """

    @classmethod
    def setUpClass(cls):
        # Monkey patch the package so as to use OPENJPEG instead of OPENJP2
        cls.openjp2 = glymur.lib.openjp2.OPENJP2
        glymur.lib.openjp2.OPENJP2 = None

    @classmethod
    def tearDownClass(cls):
        # Restore OPENJP2
        glymur.lib.openjp2.OPENJP2 = cls.openjp2

    def setUp(self):
        # Save sys.stdout.
        self.stdout = sys.stdout
        sys.stdout = StringIO()
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        # Restore stdout.
        sys.stdout = self.stdout

    def test_info_callbacks_on_read(self):
        # Verify that we get the expected stdio output when our internal info
        # callback handler is enabled.
        j = glymur.Jp2k(self.j2kfile)
        d = j.read(rlevel=1, verbose=True)
        actual = sys.stdout.getvalue().strip()

        regex = re.compile(r"""\[INFO\]\stile\s1\sof\s1\s+
                               \[INFO\]\s-\stiers-1\stook\s[0-9]+\.[0-9]+\ss\s+
                               \[INFO\]\s-\sdwt\stook\s[0-9]+\.[0-9]+\ss\s+
                               \[INFO\]\s-\stile\sdecoded\sin\s[0-9]+\.[0-9]+\ss""",
                           re.VERBOSE)
        if sys.hexversion <= 0x03020000:
            self.assertRegexpMatches(actual, regex)
        else:
            self.assertRegex(actual, regex)



if __name__ == "__main__":
    unittest.main()
