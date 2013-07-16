"""These tests are for edge cases where OPENJPEG does not exist, but
OPENJP2 may be present in some form or other.
"""
#pylint:  disable-all

import imp
import os
import sys
import tempfile

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch
import warnings

import pkg_resources

import glymur
from glymur import Jp2k
from glymur.lib import openjp2 as opj2

class TestConfig(unittest.TestCase):

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_read_without_library_backing_us_up(self):
        """Don't have either openjp2 or openjpeg libraries?  Must error out.
        """
        with patch('glymur.lib.openjp2.OPENJP2', new=None):
            with  patch('glymur.lib.openjpeg.OPENJPEG', new=None):
                with self.assertRaises((IOError, OSError)):
                    d = glymur.Jp2k(self.jp2file).read()


@unittest.skipIf(sys.hexversion < 0x03020000,
                 "TemporaryDirectory introduced in 3.2.")
@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Needs openjp2 library first before these tests make sense.")
class TestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Monkey patch the package so as to ignore OPENJPEG if it exists.
        cls.openjpeg = glymur.lib.openjpeg.OPENJPEG
        glymur.lib.openjp2.OPENJPEG = None

    @classmethod
    def tearDownClass(cls):
        # Restore OPENJPEG
        glymur.lib.openjpeg.OPENJPEG = cls.openjpeg

    def setUp(self):
        imp.reload(glymur)
        imp.reload(glymur.lib.openjp2)
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        imp.reload(glymur)
        imp.reload(glymur.lib.openjp2)

    def test_config_file_via_environ(self):
        """Verify that we can read a configuration file set via environ var."""
        with tempfile.TemporaryDirectory() as tdir:
            configdir = os.path.join(tdir, 'glymur')
            os.mkdir(configdir)
            filename = os.path.join(configdir, 'glymurrc')
            with open(filename, 'wt') as tfile:
                tfile.write('[library]\n')
                libloc = glymur.lib.openjp2.OPENJP2._name
                line = 'openjp2: {0}\n'.format(libloc)
                tfile.write(line)
                tfile.flush()
                with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                    imp.reload(glymur.lib.openjp2)
                    j = Jp2k(self.jp2file)

    def test_config_file_via_environ_is_wrong(self):
        # A non-existant library location should be rejected.
        with tempfile.TemporaryDirectory() as tdir:
            configdir = os.path.join(tdir, 'glymur')
            os.mkdir(configdir)
            fname = os.path.join(configdir, 'glymurrc')
            with open(fname, 'w') as fp:
                with tempfile.NamedTemporaryFile(suffix='.dylib') as tfile:
                    fp.write('[library]\n')
                    fp.write('openjp2: {0}.not.there\n'.format(tfile.name))
                    fp.flush()
                    with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                        # Misconfigured new configuration file should
                        # be rejected.
                        with self.assertWarns(UserWarning) as cw:
                            imp.reload(glymur.lib.openjp2)

    def test_missing_config_file_via_environ(self):
        # Verify that we error out properly if the configuration file
        # specified via environment variable is not found.
        with tempfile.TemporaryDirectory() as tdir:
            with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                # Misconfigured new configuration file should
                # be rejected.
                with self.assertWarns(UserWarning) as cw:
                    imp.reload(glymur.lib.openjp2)

    def test_home_dir_missing_config_dir(self):
        # Verify no exception is raised if $HOME is missing .config directory.
        with tempfile.TemporaryDirectory() as tdir:
            with patch.dict('os.environ', {'HOME': tdir}):
                # Misconfigured new configuration file should
                # be rejected.
                with self.assertWarns(UserWarning) as cw:
                    imp.reload(glymur.lib.openjp2)

    def test_home_dir_missing_glymur_rc_dir(self):
        # Should warn but not error if $HOME/.config but no glymurrc dir.
        with tempfile.TemporaryDirectory() as tdir:
            # We need the subdirectory to be specifically named as ".config"
            # in order for this test to work.  A specifically-named temporary
            # directory does not seem to be possible, so try to symlink it.
            # Supposedly the symlink gets cleaned up with tdir gets cleaned up.
            with tempfile.TemporaryDirectory(suffix=".config", dir=tdir) \
                    as tdir_config:
                os.symlink(tdir_config, os.path.join(tdir, '.config'))
                with patch.dict('os.environ', {'HOME': tdir}):
                    # Misconfigured new configuration file should
                    # be rejected.
                    with self.assertWarns(UserWarning) as cw:
                        imp.reload(glymur.lib.openjp2)

if __name__ == "__main__":
    unittest.main()
