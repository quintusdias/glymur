"""
Tests for OpenJPEG module.
"""
import ctypes
import re
import sys
import unittest

import glymur


@unittest.skipIf(glymur.lib.openjpeg.OPENJPEG is None,
                 "Missing openjpeg library.")
class TestOpenJPEG(unittest.TestCase):
    """Test suite for openjpeg functions we choose to expose."""

    def test_version(self):
        """Only versions 1.3, 1.4, and 1.5 are supported."""
        version = glymur.lib.openjpeg.version()
        regex = re.compile('1.[345].[0-9]')
        if sys.hexversion <= 0x03020000:
            self.assertRegexpMatches(version, regex)
        else:
            self.assertRegex(version, regex)

    def test_default_decoder_parameters(self):
        """Verify that we properly set the default decode parameters."""
        version = glymur.lib.openjpeg.version()
        minor = int(version.split('.')[1])

        dcp = glymur.lib.openjpeg.DecompressionParametersType()
        glymur.lib.openjpeg.set_default_decoder_parameters(ctypes.byref(dcp))

        self.assertEqual(dcp.cp_reduce, 0)
        self.assertEqual(dcp.cp_layer, 0)
        self.assertEqual(dcp.infile, b'')
        self.assertEqual(dcp.outfile, b'')
        self.assertEqual(dcp.decod_format, -1)
        self.assertEqual(dcp.cod_format, -1)
        self.assertEqual(dcp.jpwl_correct, 0)
        self.assertEqual(dcp.jpwl_exp_comps, 0)
        self.assertEqual(dcp.jpwl_max_tiles, 0)
        self.assertEqual(dcp.cp_limit_decoding, 0)
        if minor > 4:
            # Introduced in 1.5.x
            self.assertEqual(dcp.flags, 0)
