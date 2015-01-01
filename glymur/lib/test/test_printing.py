# -*- coding:  utf-8 -*-
"""Test suite for printing.
"""

import re
import sys
import unittest

if sys.hexversion < 0x03000000:
    from mock import patch
    from StringIO import StringIO
else:
    from unittest.mock import patch
    from io import StringIO

import glymur
from . import fixtures


@unittest.skipIf(sys.hexversion < 0x03000000, "do not care about 2.7 here")
@unittest.skipIf(re.match('0|1|2.0', glymur.version.openjpeg_version),
                 "Requires openjpeg 2.1.0 or higher")
class TestPrintingOpenjp2(unittest.TestCase):
    """Tests for verifying how printing works on openjp2 library structures."""
    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_decompression_parameters(self):
        """printing DecompressionParametersType"""
        dparams = glymur.lib.openjp2.set_default_decoder_parameters()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(dparams)
            actual = fake_out.getvalue().strip()
        expected = fixtures.decompression_parameters_type
        self.assertEqual(actual, expected)

    def test_progression_order_changes(self):
        """printing PocType"""
        ptype = glymur.lib.openjp2.PocType()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(ptype)
            actual = fake_out.getvalue().strip()
        expected = fixtures.default_progression_order_changes_type
        self.assertEqual(actual, expected)

    def test_default_compression_parameters(self):
        """printing default compression parameters"""
        cparams = glymur.lib.openjp2.set_default_encoder_parameters()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(cparams)
            actual = fake_out.getvalue().strip()
        expected = fixtures.default_compression_parameters_type
        self.assertEqual(actual, expected)

    def test_default_component_parameters(self):
        """printing default image component parameters"""
        icpt = glymur.lib.openjp2.ImageComptParmType()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(icpt)
            actual = fake_out.getvalue().strip()
        expected = fixtures.default_image_component_parameters
        self.assertEqual(actual, expected)

    def test_default_image_type(self):
        """printing default image type"""
        it = glymur.lib.openjp2.ImageType()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(it)
            actual = fake_out.getvalue().strip()

        expected = fixtures.default_image_type
        self.assertRegex(actual, expected)
