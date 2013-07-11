#pylint:  disable-all
import ctypes
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
        v = glymur.lib._openjpeg.version()
        parts = v.split('.')
        self.assertEqual(parts[0], '1')
        self.assertEqual(parts[1], '5')

    def test_set_default_decoder_parameters(self):
        # Verify that we properly set the default decode parameters.
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
        self.assertEqual(dp.flags, 0)
