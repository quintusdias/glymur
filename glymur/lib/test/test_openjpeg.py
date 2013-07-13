#pylint:  disable-all
import ctypes
import re
import sys

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

import glymur


@unittest.skipIf(glymur.lib._openjpeg.OPENJPEG is None,
                 "Missing openjpeg library.")
class TestOpenJPEG(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_version(self):
        version = glymur.lib._openjpeg.version()
        regex = re.compile('1.[45].[0-9]')
        if sys.hexversion <= 0x03020000:
            self.assertRegexpMatches(version, regex)
        else:
            self.assertRegex(version, regex)

    def test_set_default_decoder_parameters(self):
        # Verify that we properly set the default decode parameters.
        version = glymur.lib._openjpeg.version()
        minor = int(version.split('.')[1])

        dp = glymur.lib._openjpeg.DecompressionParametersType()
        glymur.lib._openjpeg.set_default_decoder_parameters(ctypes.byref(dp))

        self.assertEqual(dp.cp_reduce, 0)
        self.assertEqual(dp.cp_layer, 0)
        self.assertEqual(dp.infile, b'')
        self.assertEqual(dp.outfile, b'')
        self.assertEqual(dp.decod_format, -1)
        self.assertEqual(dp.cod_format, -1)
        self.assertEqual(dp.jpwl_correct, 0)
        self.assertEqual(dp.jpwl_exp_comps, 0)
        self.assertEqual(dp.jpwl_max_tiles, 0)
        self.assertEqual(dp.cp_limit_decoding, 0)
        if minor > 4:
            # Introduced in 1.5.x
            self.assertEqual(dp.flags, 0)
