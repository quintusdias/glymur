"""
Test command line interface to JPEG2JP2
"""

# standard library imports
import importlib.resources as ir
from io import StringIO
import platform
import shutil
import sys
import unittest
from unittest.mock import patch

# 3rd party library imports
import lxml.etree as ET

# Local imports
from glymur import JPEG2JP2, Jp2k, jp2box, command_line, reset_option
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(
    platform.system() == 'Linux'
    and platform.freedesktop_os_release()['ID'] == 'fedora',
    'missing importlib.metadata.files ?'
)
class TestSuite(fixtures.TestCommon):

    def test_smoke(self):
        """
        SCENARIO:  no special options

        EXPECTED RESULT:  no errors
        """
        new = ['', str(self.retina), str(self.temp_jp2_filename)]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_verbosity(self):
        """
        SCENARIO:  verbosity is specified on the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--verbosity', 'info'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_tilesize(self):
        """
        SCENARIO:  tilesize is specified on the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--tilesize', '512', '512'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_psnr(self):
        """
        SCENARIO:  specify psnr via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--psnr', '30', '35', '40', '0'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_irreversible(self):
        """
        SCENARIO:  specify the irreversible transform via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--irreversible'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_plt(self):
        """
        SCENARIO:  specify the PLT markers via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--plt'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_eph(self):
        """
        SCENARIO:  specify the EPH markers via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--eph'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_sop(self):
        """
        SCENARIO:  specify the SOP markers via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--sop'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_progression_order(self):
        """
        SCENARIO:  specify the procession order via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--prog', 'rlcp'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_number_of_resolutions(self):
        """
        SCENARIO:  specify resolution

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--numres', '6'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_num_threads(self):
        """
        SCENARIO:  specify number of threads to use

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--num-threads', '4'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

        reset_option('all')

    def test_icc_profile(self):
        """
        SCENARIO:  specify to include an ICC profile

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.rocket), str(self.temp_jp2_filename),
            '--include-icc-profile',
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_layers(self):
        """
        SCENARIO:  specify compression ratios

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--cratio', '200', '50', '10'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_resolution_boxes(self):
        """
        SCENARIO:  specify capture and display resolution

        EXPECTED RESULT:  no errors
        """
        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4

        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--capture-resolution', str(vresc), str(hresc),
            '--display-resolution', str(vresd), str(hresd),
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()


class TestSuiteJP2DUMP(fixtures.TestCommon):
    """
    Tests for verifying how commandline printing works.
    """
    def setUp(self):
        super().setUp()

        # Reset printoptions for every test.
        reset_option('all')

    def tearDown(self):
        super().tearDown()
        reset_option('all')

    def run_jp2dump(self, args):
        sys.argv = args
        with patch('sys.stdout', new=StringIO()) as fake_out:
            command_line.main()
            actual = fake_out.getvalue().strip()
            # Remove the file line, as that is filesystem-dependent.
            lines = actual.split('\n')
            actual = '\n'.join(lines[1:])
        return actual

    def test_default_nemo(self):
        """by default one should get the main header"""
        actual = self.run_jp2dump(['', self.jp2file])

        # shave off the  non-main-header segments
        expected = (
            ir.files('tests.data')
              .joinpath('nemo.txt')
              .read_text()
              .rstrip()
              .split('\n')[:52]
        )
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    def test_jp2_codestream_0(self):
        """Verify dumping with -c 0, supressing all codestream details."""
        actual = self.run_jp2dump(['', '-c', '0', self.jp2file])

        # shave off the codestream details
        expected = (
            ir.files('tests.data')
              .joinpath('nemo.txt')
              .read_text()
              .rstrip()
              .split('\n')[:17]
        )
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    def test_jp2_codestream_1(self):
        """Verify dumping with -c 1, print just the header."""
        actual = self.run_jp2dump(['', '-c', '1', self.jp2file])

        # shave off the  non-main-header segments
        expected = (
            ir.files('tests.data')
              .joinpath('nemo.txt')
              .read_text()
              .rstrip()
              .split('\n')[:52]
        )
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    def test_jp2_codestream_2(self):
        """Verify dumping with -c 2, print entire jp2 jacket, codestream."""
        actual = self.run_jp2dump(['', '-c', '2', self.jp2file])
        expected = (
            ir.files('tests.data')
              .joinpath('nemo.txt')
              .read_text()
              .rstrip()
        )
        self.maxDiff = None
        self.assertEqual(actual, expected)

    def test_j2k_codestream_0(self):
        """-c 0 should print just a single line when used on a codestream."""
        sys.argv = ['', '-c', '0', self.j2kfile]
        with patch('sys.stdout', new=StringIO()) as fake_out:
            command_line.main()
            actual = fake_out.getvalue().strip()
        self.assertRegex(actual, "File:  .*")

    def test_j2k_codestream_1(self):
        """
        SCENARIO:  The jp2dump executable is used with the "-c 1" switch.

        EXPECTED RESULT:  The output should include the codestream header.
        """
        sys.argv = ['', '-c', '1', self.j2kfile]
        with patch('sys.stdout', new=StringIO()) as stdout:
            command_line.main()
            actual = stdout.getvalue().strip()

        expected = (
            ir.files('tests.data')
              .joinpath('goodstuff_codestream_header.txt')
              .read_text()
              .rstrip()
        )
        self.assertEqual(expected, actual)

    def test_j2k_codestream_2(self):
        """Verify dumping with -c 2, full details."""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            sys.argv = ['', '-c', '2', self.j2kfile]
            command_line.main()
            actual = fake_out.getvalue().strip()

        expected = (
            ir.files('tests.data')
              .joinpath('goodstuff_with_full_header.txt')
              .read_text()
              .rstrip()
        )
        self.assertIn(expected, actual)

    def test_codestream_invalid(self):
        """Verify dumping with -c 3, not allowd."""
        with self.assertRaises(ValueError):
            sys.argv = ['', '-c', '3', self.jp2file]
            command_line.main()

    def test_short(self):
        """Verify dumping with -s, short option."""
        actual = self.run_jp2dump(['', '-s', self.jp2file])

        expected = (
            ir.files('tests.data')
              .joinpath('nemo_dump_short.txt')
              .read_text()
              .rstrip()
        )
        self.assertEqual(actual, expected)

    def test_suppress_xml(self):
        """Verify dumping with -x, suppress XML."""

        s = ir.files('tests.data').joinpath('file1_xml.txt').read_text()
        elt = ET.fromstring(s)
        xml = ET.ElementTree(elt)
        box = jp2box.XMLBox(xml=xml, length=439, offset=36)

        shutil.copyfile(self.jp2file, self.temp_jp2_filename)
        jp2 = Jp2k(self.temp_jp2_filename)
        jp2.append(box)

        actual = self.run_jp2dump(['', '-x', str(self.temp_jp2_filename)])

        # shave off the XML and non-main-header segments
        expected = (
            ir.files('tests.data')
              .joinpath('appended_xml_box.txt')
              .read_text()
              .rstrip()
        )

        self.assertEqual(actual, expected)

    def test_suppress_warnings_until_end(self):
        """
        SCENARIO:  JP2DUMP with -x option on file with invalid ftyp box.

        EXPECTED RESULT:  The warning is suppressed until the very end of the
        output.
        """
        path = ir.files('tests.data').joinpath('edf_c2_1178956.jp2')
        actual = self.run_jp2dump(['', '-x', str(path)])
        lines = actual.splitlines()

        for line in lines[:-1]:
            self.assertNotIn('UserWarning', line)

        self.assertIn('UserWarning', lines[-1])
