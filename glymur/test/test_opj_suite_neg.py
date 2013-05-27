"""
The tests here do not correspond directly to the OpenJPEG test suite, but
seem like logical negative tests to add.
"""
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np

from ..lib import openjp2 as opj2

# Need some combination of matplotlib, PIL, or scikits-image for reading
# other image formats.
no_read_backend = False
msg = "Either scikit-image with the freeimage backend or matplotlib "
msg += "with the PIL backend must be available in order to run the "
msg += "tests in this suite."
no_read_backend_msg = msg
try:
    import skimage.io
    try:
        skimage.io.use_plugin('freeimage')
        from skimage.io import imread
    except ImportError:
        try:
            skimage.io.use_plugin('PIL')
            from skimage.io import imread
        except ImportError:
            raise
except ImportError:
    try:
        from PIL import Image
        from matplotlib.pyplot import imread
    except ImportError:
        no_read_backend = True

from glymur import Jp2k
import glymur

try:
    data_root = os.environ['OPJ_DATA_ROOT']
except KeyError:
    data_root = None
except:
    raise


def read_image(infile):
    # PIL issues warnings which we do not care about, so suppress them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = imread(infile)
    return data


@unittest.skipIf(no_read_backend, no_read_backend_msg)
@unittest.skipIf(data_root is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuiteNegative(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_negative_psnr_with_cratios(self):
        # Using psnr with cratios options is not allowed.
        # Not an OpenJPEG test, but close.
        infile = os.path.join(data_root, 'input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            with self.assertRaises(RuntimeError):
                j.write(data, psnr=[30, 35, 40], cratios=[2, 3, 4])

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_MarkerIsNotCompliant_j2k_dump(self):
        # SOT marker gives bad offset.
        relpath = 'input/nonregression/MarkerIsNotCompliant.j2k'
        jfile = os.path.join(data_root, relpath)
        jp2k = Jp2k(jfile)
        with self.assertWarns(UserWarning) as cw:
            c = jp2k.get_codestream(header_only=False)

        # Verify that the last segment returned in the codestream is SOD,
        # not EOC.  Codestream parsing should stop when we try to jump to
        # the end of SOT.
        self.assertEqual(c.segment[-1].id, 'SOD')

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_illegalcolortransform_dump(self):
        # SOT marker gives bad offset.
        relpath = 'input/nonregression/illegalcolortransform.j2k'
        jfile = os.path.join(data_root, relpath)
        jp2k = Jp2k(jfile)
        with self.assertWarns(UserWarning) as cw:
            c = jp2k.get_codestream(header_only=False)

        # Verify that the last segment returned in the codestream is SOD,
        # not EOC.  Codestream parsing should stop when we try to jump to
        # the end of SOT.
        self.assertEqual(c.segment[-1].id, 'SOD')

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_NR_Cannotreaddatawithnosizeknown_j2k(self):
        # SOT marker gives bad offset.
        relpath = 'input/nonregression/Cannotreaddatawithnosizeknown.j2k'
        jfile = os.path.join(data_root, relpath)
        jp2k = Jp2k(jfile)
        with self.assertWarns(UserWarning) as cw:
            c = jp2k.get_codestream(header_only=False)

        # Verify that the last segment returned in the codestream is SOD,
        # not EOC.  Codestream parsing should stop when we try to jump to
        # the end of SOT.
        self.assertEqual(c.segment[-1].id, 'SOD')

    def test_code_block_dimensions(self):
        # opj_compress doesn't allow the dimensions of a codeblock
        # to be too small or too big, so neither will we.
        data = np.zeros((256, 256), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')

            # opj_compress doesn't allow code block area to exceed 4096.
            with self.assertRaises(RuntimeError) as cr:
                j.write(data, cbsize=(256, 256))

            # opj_compress doesn't allow either dimension to be less than 4.
            with self.assertRaises(RuntimeError) as cr:
                j.write(data, cbsize=(2048, 2))
            with self.assertRaises(RuntimeError) as cr:
                j.write(data, cbsize=(2, 2048))

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    def test_exceeded_box(self):
        # Verify that a warning is issued if we read past the end of a box
        # This file has a palette (pclr) box whose length is impossibly
        # short.
        infile = os.path.join(data_root,
                              'input/nonregression/mem-b2ace68c-1381.jp2')
        with self.assertWarns(UserWarning) as cw:
            j = Jp2k(infile)

if __name__ == "__main__":
    unittest.main()
