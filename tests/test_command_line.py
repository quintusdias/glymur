# -*- coding:  utf-8 -*-
"""
Test suite for printing.
"""
# Standard library imports ...
import os
import pkg_resources as pkg
import re
import sys
import unittest

if sys.hexversion < 0x03000000:
    from StringIO import StringIO
else:
    from io import StringIO

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch

import glymur
from glymur import command_line
from . import fixtures


class TestSuite(unittest.TestCase):
    """Tests for verifying how jp2dump console script works."""
    def setUp(self):
        self.jpxfile = glymur.data.jpxfile()
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

        # Reset printoptions for every test.
        glymur.reset_option('all')

    def tearDown(self):
        glymur.reset_option('all')

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
        if 'lxml' not in sys.modules.keys():
            # No lxml, so don't bother verifying.  We know at least that it
            # runs.
            return

        # shave off the  non-main-header segments
        lines = fixtures.nemo.split('\n')
        expected = lines[0:140]
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    @unittest.skipIf('lxml' not in sys.modules.keys(), "No lxml")
    def test_jp2_codestream_0(self):
        """Verify dumping with -c 0, supressing all codestream details."""
        actual = self.run_jp2dump(['', '-c', '0', self.jp2file])

        # shave off the codestream details
        lines = fixtures.nemo.split('\n')
        expected = lines[0:105]
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    @unittest.skipIf('lxml' not in sys.modules.keys(), "No lxml")
    def test_jp2_codestream_1(self):
        """Verify dumping with -c 1, print just the header."""
        actual = self.run_jp2dump(['', '-c', '1', self.jp2file])

        # shave off the  non-main-header segments
        lines = fixtures.nemo.split('\n')
        expected = lines[0:140]
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    @unittest.skipIf('lxml' not in sys.modules.keys(), "No lxml")
    def test_jp2_codestream_2(self):
        """Verify dumping with -c 2, print entire jp2 jacket, codestream."""
        actual = self.run_jp2dump(['', '-c', '2', self.jp2file])
        expected = fixtures.nemo
        self.assertEqual(actual, expected)

    @unittest.skipIf(sys.hexversion < 0x03000000, "assertRegex not in 2.7")
    def test_j2k_codestream_0(self):
        """-c 0 should print just a single line when used on a codestream."""
        sys.argv = ['', '-c', '0', self.j2kfile]
        with patch('sys.stdout', new=StringIO()) as fake_out:
            command_line.main()
            actual = fake_out.getvalue().strip()
        self.assertRegex(actual, "File:  .*")

    def test_j2k_codestream_1(self):
        """-c 1 should print the codestream header"""
        sys.argv = ['', '-c', '1', self.j2kfile]
        with patch('sys.stdout', new=StringIO()) as stdout:
            command_line.main()
            actual = stdout.getvalue().strip()

        expected = fixtures.goodstuff_codestream_header
        self.assertEqual(expected, actual)

    def test_j2k_codestream_2(self):
        """Verify dumping with -c 2, full details."""
        with patch('sys.stdout', new=StringIO()) as stdout:
            sys.argv = ['', '-c', '2', self.j2kfile]
            command_line.main()
            actual = stdout.getvalue().strip()
        expected = fixtures.j2k_codestream_2
        self.assertIn(expected, actual)

    def test_codestream_invalid(self):
        """Verify dumping with -c 3, not allowd."""
        with self.assertRaises(ValueError):
            sys.argv = ['', '-c', '3', self.jp2file]
            command_line.main()

    def test_short(self):
        """Verify dumping with -s, short option."""
        actual = self.run_jp2dump(['', '-s', self.jp2file])

        self.assertEqual(actual, fixtures.nemo_dump_short)

    @unittest.skipIf('lxml' not in sys.modules.keys(), "No lxml")
    def test_suppress_xml(self):
        """Verify dumping with -x, suppress XML."""
        actual = self.run_jp2dump(['', '-x', self.jp2file])

        # shave off the XML and non-main-header segments
        lines = fixtures.nemo.split('\n')
        expected = lines[0:18]
        expected.extend(lines[104:140])
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    @unittest.skipIf(sys.hexversion < 0x03000000, "assertRegex not in 2.7")
    def test_suppress_warnings_until_end(self):
        """
        Warnings about invalid JP2/J2K syntax should be suppressed until end
        """
        file = os.path.join('data', 'edf_c2_1178956.jp2')
        file = pkg.resource_filename(__name__, file)
        actual = self.run_jp2dump(['', '-x', file])

        # The "CME marker segment" part is the last segment in the codestream
        # header.
        pattern = 'JPEG\s2000.*?CME\smarker\ssegment.*?UserWarning'
        r = re.compile(pattern, re.DOTALL)
        self.assertRegex(actual, r)
