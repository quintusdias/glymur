"""
Test command line interface to JPEG2JP2
"""

# standard library imports
import importlib.resources as ir
from io import StringIO
import shutil
import sys
from unittest.mock import patch

# 3rd party library imports
import lxml.etree as ET

# Local imports
from glymur import Jp2k, jp2box, command_line, reset_option
from . import fixtures


class TestSuite(fixtures.TestCommon):
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
            ir.files('tests.data.misc')
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
            ir.files('tests.data.misc')
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
            ir.files('tests.data.misc')
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
            ir.files('tests.data.misc')
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
            ir.files('tests.data.misc')
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
            ir.files('tests.data.misc')
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
            ir.files('tests.data.misc')
              .joinpath('nemo_dump_short.txt')
              .read_text()
              .rstrip()
        )
        self.assertEqual(actual, expected)

    def test_suppress_xml(self):
        """Verify dumping with -x, suppress XML."""

        s = ir.files('tests.data.conformance') \
              .joinpath('file1_xml.txt') \
              .read_text()
        elt = ET.fromstring(s)
        xml = ET.ElementTree(elt)
        box = jp2box.XMLBox(xml=xml, length=439, offset=36)

        shutil.copyfile(self.jp2file, self.temp_jp2_filename)
        jp2 = Jp2k(self.temp_jp2_filename)
        jp2.append(box)

        actual = self.run_jp2dump(['', '-x', str(self.temp_jp2_filename)])

        # shave off the XML and non-main-header segments
        expected = (
            ir.files('tests.data.misc')
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
        path = ir.files('tests.data.from-openjpeg') \
                 .joinpath('edf_c2_1178956.jp2')
        actual = self.run_jp2dump(['', '-x', str(path)])
        lines = actual.splitlines()

        for line in lines[:-1]:
            self.assertNotIn('UserWarning', line)

        self.assertIn('UserWarning', lines[-1])
