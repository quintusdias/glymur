"""These tests are for edge cases where OPENJPEG does not exist, but
OPENJP2 may be present in some form or other.
"""
# unittest doesn't work well with R0904.
# pylint: disable=R0904

# tempfile.TemporaryDirectory, unittest.assertWarns introduced in 3.2
# pylint: disable=E1101

# unittest.mock only in Python 3.3 (python2.7/pylint import issue)
# pylint:  disable=E0611,F0401


import imp
import os
import sys
import tempfile
import unittest
import warnings

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch

import glymur
from glymur import Jp2k


@unittest.skipIf(sys.hexversion < 0x03020000,
                 "TemporaryDirectory introduced in 3.2.")
@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Needs openjp2 library first before these tests make sense.")
class TestSuite(unittest.TestCase):
    """Test suite for configuration file operation."""

    @classmethod
    def setUpClass(cls):
        imp.reload(glymur)
        imp.reload(glymur.lib.openjp2)

    @classmethod
    def tearDownClass(cls):
        imp.reload(glymur)
        imp.reload(glymur.lib.openjp2)

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_config_file_via_environ(self):
        """Verify that we can read a configuration file set via environ var."""
        with tempfile.TemporaryDirectory() as tdir:
            configdir = os.path.join(tdir, 'glymur')
            os.mkdir(configdir)
            filename = os.path.join(configdir, 'glymurrc')
            with open(filename, 'wt') as tfile:
                tfile.write('[library]\n')

                # Need to reliably recover the location of the openjp2 library,
                # so using '_name' appears to be the only way to do it.
                # pylint:  disable=W0212
                libloc = glymur.lib.openjp2.OPENJP2._name
                line = 'openjp2: {0}\n'.format(libloc)
                tfile.write(line)
                tfile.flush()
                with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                    imp.reload(glymur.lib.openjp2)
                    Jp2k(self.jp2file)

    def test_xdg_env_config_file_is_bad(self):
        """A non-existant library location should be rejected."""
        with tempfile.TemporaryDirectory() as tdir:
            configdir = os.path.join(tdir, 'glymur')
            os.mkdir(configdir)
            fname = os.path.join(configdir, 'glymurrc')
            with open(fname, 'w') as fptr:
                with tempfile.NamedTemporaryFile(suffix='.dylib') as tfile:
                    fptr.write('[library]\n')
                    fptr.write('openjp2: {0}.not.there\n'.format(tfile.name))
                    fptr.flush()
                    with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                        # Misconfigured new configuration file should
                        # be rejected.
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter('always')
                            imp.reload(glymur.lib.openjp2)
                            self.assertTrue(issubclass(w[0].category,UserWarning))
                            self.assertTrue('could not be loaded' in str(w[0].message))


@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None and
                 glymur.lib.openjpeg.OPENJPEG is None,
                 "Missing openjp2 library.")
class TestConfig(unittest.TestCase):
    """Test suite for reading without proper library in place."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    def test_read_without_library(self):
        """Don't have either openjp2 or openjpeg libraries?  Must error out.
        """
        with patch('glymur.lib.openjp2.OPENJP2', new=None):
            with patch('glymur.lib.openjpeg.OPENJPEG', new=None):
                with self.assertRaises(glymur.jp2k.LibraryNotFoundError):
                    glymur.Jp2k(self.jp2file).read()

    def test_read_bands_without_library(self):
        """Don't have openjp2 library?  Must error out.
        """
        with patch('glymur.lib.openjp2.OPENJP2', new=None):
            with patch('glymur.lib.openjpeg.OPENJPEG', new=None):
                with patch('glymur.version.openjpeg_version_tuple',
                           new=(0, 0, 0)):
                    with self.assertRaises(glymur.jp2k.LibraryNotFoundError):
                        glymur.Jp2k(self.jp2file).read_bands()

    @unittest.skipIf(os.name == "nt", "NamedTemporaryFile issue on windows")
    def test_write_without_library(self):
        """Don't have openjpeg libraries?  Must error out.
        """
        data = glymur.Jp2k(self.j2kfile).read()
        with patch('glymur.lib.openjp2.OPENJP2', new=None):
            with patch('glymur.lib.openjpeg.OPENJPEG', new=None):
                with self.assertRaises(glymur.jp2k.LibraryNotFoundError):
                    with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
                        ofile = Jp2k(tfile.name, 'wb')
                        ofile.write(data)


if __name__ == "__main__":
    unittest.main()
