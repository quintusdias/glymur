"""These tests are for edge cases where OPENJPEG does not exist, but
OPENJP2 may be present in some form or other.
"""
# Standard library imports ...
import contextlib
import imp
import os
import pathlib
import platform
import tempfile
import unittest
from unittest.mock import patch
import warnings

# Local imports ...
import glymur
from glymur import Jp2k
from . import fixtures


@contextlib.contextmanager
def chdir(dirname=None):
    """
    This context manager restores the value of the current working directory
    (cwd) after the enclosed code block completes or raises an exception.  If a
    directory name is supplied to the context manager then the cwd is changed
    prior to running the code block.

    Shamelessly lifted from
    http://www.astropython.org/snippet/2009/10/chdir-context-manager
    """
    curdir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)


class TestSuite(unittest.TestCase):

    @patch('glymur.config.platform.system')
    @patch('glymur.config.sys.version', 'Anaconda')
    @patch('glymur.config.sys.executable', '/opt/anaconda/bin/python')
    def test_anaconda_on_mac(self, mock_platform_system):
        """
        SCENARIO:  the platform is Anaconda on mac.

        EXPECTED RESULT:  the path of the openjp2 library is under the anaconda
        root.
        """
        mock_platform_system.return_value = 'Darwin'

        actual = glymur.config._determine_full_path('openjp2')
        expected = pathlib.Path('/opt/anaconda/lib/libopenjp2.dylib')

        self.assertEqual(actual, expected)

    @unittest.skipIf(platform.system() == 'Windows', 'nonsensical on windows')
    @patch('glymur.config.platform.system')
    @patch('glymur.config.sys.version', 'Anaconda')
    @patch('glymur.config.sys.executable', '/usr/bin/python')
    def test_windows_path(self, mock_platform_system):
        """
        SCENARIO:  the platform is Anaconda on windows, even though we are not
        actually running on windows.

        EXPECTED RESULT:  the path of the openjp2 library is an Anaconda DLL
        """
        mock_platform_system.return_value = 'Windows'

        actual = glymur.config._determine_full_path('openjp2')
        expected = pathlib.Path('/usr/bin/Library/bin/openjp2.dll')

        self.assertEqual(actual, expected)

    @patch('pathlib.Path.exists')
    @patch('glymur.config.sys.version', 'not anaconda')
    @patch('glymur.config.platform.system')
    def test_macports(self, mock_platform_system, mock_path_exists):
        """
        SCENARIO:  the platform is MacPorts.

        EXPECTED RESULT:  the path of the openjp2 library is in /opt/local
        """
        mock_platform_system.return_value = 'Darwin'
        mock_path_exists.return_value = True

        actual = glymur.config._determine_full_path('openjp2')
        expected = pathlib.Path('/opt/local/lib/libopenjp2.dylib')

        self.assertEqual(actual, expected)

    @patch('glymur.config.sys.version', 'not anaconda')
    @patch('glymur.config.find_library')
    @patch('glymur.config.platform.system')
    def test_via_ctypes(self, mock_platform_system, mock_find_library):
        """
        SCENARIO:  the platform is not anaconda and not MacPorts.  The ctypes
        module finds the library.

        EXPECTED RESULT:  the path of the openjp2 library is on standard
        system paths
        """
        mock_platform_system.return_value = 'not darwin'
        mock_find_library.return_value = '/usr/lib/libopenjp2.so'

        actual = glymur.config._determine_full_path('openjp2')
        expected = pathlib.Path('/usr/lib/libopenjp2.so')

        self.assertEqual(actual, expected)

    @patch('glymur.config.sys.version', 'not anaconda')
    @patch('glymur.config.find_library')
    @patch('glymur.config.platform.system')
    def test_not_via_ctypes(self,
                            mock_platform_system,
                            mock_find_library):
        """
        SCENARIO:  the platform is not anaconda and not MacPorts.  The ctypes
        module does NOT find the library.

        EXPECTED RESULT:  the path of the openjp2 library is None
        """
        mock_platform_system.return_value = 'not darwin'
        mock_find_library.return_value = None

        actual = glymur.config.load_openjpeg_library('openjp2')
        self.assertIsNone(actual)

    @unittest.skipIf(platform.system() == 'Windows', 'nonsensical on windows')
    @patch('glymur.config.platform.system')
    @patch('pathlib.Path.home')
    def test_config_dir_on_windows(self,
                                   mock_pathlib_path_home,
                                   mock_platform_system):
        """
        SCENARIO:  the XDG_CONFIG_HOME environment variable is not present, the
        os.name *IS* 'nt'.  Don't bother running on windows because that's what
        we are trying to test from other platforms.

        EXPECTED RESULT:  the path to the configuration directory should be
        under the home directory
        """
        mock_platform_system.return_value = 'Windows'

        expected_path = pathlib.Path('/neither/here/nor/there')
        mock_pathlib_path_home.return_value = expected_path

        with patch.dict('os.environ', values=()):
            # Just make sure XDG_CONFIG_HOME is not present.
            actual = glymur.config.get_configdir()
        self.assertEqual(actual, expected_path / 'glymur')


class TestSuiteOptions(unittest.TestCase):

    def setUp(self):
        glymur.reset_option('all')

    def tearDown(self):
        glymur.reset_option('all')

    def test_reset_single_option(self):
        """
        Verify a single option can be reset.
        """
        glymur.set_option('print.codestream', True)
        glymur.reset_option('print.codestream')
        self.assertTrue(glymur.get_option('print.codestream'))

    def test_bad_reset(self):
        """
        Verify exception when a bad option is given to reset
        """
        with self.assertRaises(KeyError):
            glymur.reset_option('blah')

    def test_bad_deprecated_print_option(self):
        """
        Verify exception when a bad option is given to old set_printoption
        """
        with self.assertRaises(KeyError):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                glymur.set_printoptions(blah='value-blah')


@unittest.skipIf(fixtures.OPENJPEG_NOT_AVAILABLE,
                 fixtures.OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuiteConfigFile(unittest.TestCase):
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
                libloc = glymur.lib.openjp2.OPENJP2._name
                line = 'openjp2: {0}\n'.format(libloc)
                tfile.write(line)
                tfile.flush()
                with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                    imp.reload(glymur.lib.openjp2)
                    Jp2k(self.jp2file)

    def test_config_file_without_library_section(self):
        """
        must ignore if no library section
        """
        with tempfile.TemporaryDirectory() as tdir:
            configdir = os.path.join(tdir, 'glymur')
            os.mkdir(configdir)
            fname = os.path.join(configdir, 'glymurrc')
            with open(fname, 'w') as fptr:
                fptr.write('[testing]\n')
                fptr.write('opj_data_root: blah\n')
                fptr.flush()
                with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                    imp.reload(glymur.lib.openjp2)
                    # It's enough that we did not error out
                    self.assertTrue(True)

    def test_xdg_env_config_file_is_bad(self):
        """A non-existant library location should be rejected."""
        with tempfile.TemporaryDirectory() as tdir:
            configdir = os.path.join(tdir, 'glymur')
            os.mkdir(configdir)

            library_file = os.path.join(tdir, 'libopenjp2.dylib')

            fname = os.path.join(configdir, 'glymurrc')
            with open(fname, 'w') as fptr:
                fptr.write('[library]\n')
                fptr.write(f'openjp2: {library_file}.not.there\n')
                fptr.flush()
                with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                    # Misconfigured new configuration file should
                    # be rejected.
                    with warnings.catch_warnings():
                        # Ignore a wa
                        warnings.simplefilter('ignore')
                        imp.reload(glymur.lib.openjp2)
                    self.assertIsNone(glymur.lib.openjp2.OPENJP2)

    @unittest.skipIf(fixtures.OPENJPEG_NOT_AVAILABLE,
                     fixtures.OPENJPEG_NOT_AVAILABLE_MSG)
    def test_config_dir_but_no_config_file(self):

        with tempfile.TemporaryDirectory() as tdir:
            configdir = os.path.join(tdir, 'glymur')
            os.mkdir(configdir)
            with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                # Should still be able to load openjpeg, despite the
                # configuration file not being there
                imp.reload(glymur.lib.openjp2)
                self.assertIsNotNone(glymur.lib.openjp2.OPENJP2)

    def test_config_file_in_current_directory(self):
        """A configuration file in the current directory should be honored."""
        libloc = glymur.lib.openjp2.OPENJP2._name
        with tempfile.TemporaryDirectory() as tdir1:
            fname = os.path.join(tdir1, 'glymurrc')
            with open(fname, 'w') as fptr:
                fptr.write('[library]\n')
                fptr.write('openjp2: {0}\n'.format(libloc))
                fptr.flush()
                with chdir(tdir1):
                    # Should be able to load openjp2 as before.
                    imp.reload(glymur.lib.openjp2)
                    self.assertEqual(glymur.lib.openjp2.OPENJP2._name, libloc)
