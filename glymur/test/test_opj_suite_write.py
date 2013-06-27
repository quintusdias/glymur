"""
The tests defined here roughly correspond to what is in the OpenJPEG test
suite.
"""
import os
import platform
import sys
import tempfile
import unittest
import warnings

import numpy as np

from glymur.lib import openjp2 as opj2

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


@unittest.skipIf(glymur.lib.openjp2._OPENJP2 is None,
                 "Missing openjp2 library.")
@unittest.skipIf(no_read_backend, no_read_backend_msg)
@unittest.skipIf(data_root is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuiteWrite(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_NR_ENC_Bretagne1_ppm_1_encode(self):
        # NR-ENC-Bretagne1.ppm-1-encode
        infile = os.path.join(data_root, 'input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, cratios=[200, 100, 50])

            # Should be three layers.
            c = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 3)  # layers = 3
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

    def test_NR_ENC_Bretagne1_ppm_2_encode(self):
        # NR-ENC-Bretagne1.ppm-2-encode
        infile = os.path.join(data_root, 'input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, psnr=[30, 35, 40], numres=2)

            # Should be three layers.
            c = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 3)  # layers = 3
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 1)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

    def test_NR_ENC_Bretagne1_ppm_3_encode(self):
        # NR-ENC-Bretagne1.ppm-3-encode
        infile = os.path.join(data_root, 'input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, psnr=[30, 35, 40], cbsize=(16, 16),
                    psizes=[(64, 64)])

            # Should be three layers.
            c = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 3)  # layers = 3
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (16, 16))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(c.segment[2]._precinct_size,
                             [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32),
                              (64, 64)])

    def test_NR_ENC_Bretagne2_ppm_4_encode(self):
        infile = os.path.join(data_root, 'input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data,
                    psizes=[(128, 128)] * 3,
                    cratios=[100, 20, 2],
                    tilesize=(480, 640),
                    cbsize=(32, 32))

            # Should be three layers.
            c = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (data.shape[1], data.shape[0]))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size.  Reported as XY, not RC.
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 3)  # layers = 3
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (32, 32))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(c.segment[2]._precinct_size,
                             [(16, 16), (32, 32), (64, 64)] + [(128, 128)] * 3)

    def test_NR_ENC_Bretagne2_ppm_5_encode(self):
        # NR-ENC-Bretagne2.ppm-4-encode
        infile = os.path.join(data_root, 'input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, tilesize=(127, 127), prog="PCRL")

            c = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (data.shape[1], data.shape[0]))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (127, 127))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.PCRL)
            self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

    def test_NR_ENC_Bretagne2_ppm_6_encode(self):
        # NR-ENC-Bretagne2.ppm-6-encode
        infile = os.path.join(data_root, 'input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, subsam=(2, 2), sop=True)

            c = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (5183, 3887))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (5183, 3887))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(2, 2)] * 3)

            # COD: Coding style default
            self.assertTrue(c.segment[2].Scod & 2)  # sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

            # 18 SOP segments.
            nsops = [x.Nsop for x in c.segment if x.id == 'SOP']
            self.assertEqual(nsops, list(range(18)))

    def test_NR_ENC_Bretagne2_ppm_7_encode(self):
        infile = os.path.join(data_root, 'input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, modesw=38, eph=True)

            c = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (2592, 1944))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (2592, 1944))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertTrue(c.segment[2].Scod & 4)  # eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                            (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertTrue(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertTrue(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertTrue(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

            # 18 EPH segments.
            ephs = [x for x in c.segment if x.id == 'EPH']
            self.assertEqual(len(ephs), 18)

    def test_NR_ENC_Bretagne2_ppm_8_encode(self):
        infile = os.path.join(data_root, 'input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, grid_offset=[300, 150], cratios=[800])

            c = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (2742, 2244))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz),
                             (150, 300))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (2742, 2244))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

    def test_NR_ENC_Cevennes1_bmp_9_encode(self):
        infile = os.path.join(data_root, 'input/nonregression/Cevennes1.bmp')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, cratios=[800])

            c = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (2592, 1944))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (2592, 1944))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

    def test_NR_ENC_Cevennes2_ppm_10_encode(self):
        infile = os.path.join(data_root, 'input/nonregression/Cevennes2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, cratios=[50])

            c = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

    def test_NR_ENC_Rome_bmp_11_encode(self):
        infile = os.path.join(data_root, 'input/nonregression/Rome.bmp')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            jp2 = Jp2k(tfile.name, 'wb')
            jp2.write(data, psnr=[30, 35, 50], prog='LRCP', numres=3)

            ids = [box.id for box in jp2.box]
            lst = ['jP  ', 'ftyp', 'jp2h', 'jp2c']
            self.assertEqual(ids, lst)

            ids = [box.id for box in jp2.box[2].box]
            self.assertEqual(ids, ['ihdr', 'colr'])

            # Signature box.  Check for corruption.
            self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

            # File type box.
            self.assertEqual(jp2.box[1].brand, 'jp2 ')
            self.assertEqual(jp2.box[1].minor_version, 0)
            self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

            # Jp2 Header
            # Image header
            self.assertEqual(jp2.box[2].box[0].height, 480)
            self.assertEqual(jp2.box[2].box[0].width, 640)
            self.assertEqual(jp2.box[2].box[0].num_components, 3)
            self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
            self.assertEqual(jp2.box[2].box[0].signed, False)
            self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
            self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
            self.assertEqual(jp2.box[2].box[0].ip_provided, False)

            # Jp2 Header
            # Colour specification
            self.assertEqual(jp2.box[2].box[1].method, 1)
            self.assertEqual(jp2.box[2].box[1].precedence, 0)
            self.assertEqual(jp2.box[2].box[1].approximation, 0)
            self.assertIsNone(jp2.box[2].box[1].icc_profile)
            self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)

            c = jp2.box[3].main_header

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz),
                             (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(c.segment[1]._signed, (False, False, False))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 3)  # layers = 3
            self.assertEqual(c.segment[2].SPcod[3], 1)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 2)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

    @unittest.skip("Known failure in openjpeg test suite.")
    def test_NR_ENC_random_issue_0005_tif_12_encode(self):
        # opj_decompress has trouble reading it, but that is not an issue here.
        # The nature of the image itself seems to give the compressor trouble.
        infile = os.path.join(data_root,
                              'input/nonregression/random-issue-0005.tif')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data)

            c = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(c.segment[1].Rsiz, 0)
            # Reference grid size
            self.assertEqual((c.segment[1].Xsiz, c.segment[1].Ysiz),
                             (1024, 1024))
            # Reference grid offset
            self.assertEqual((c.segment[1].XOsiz, c.segment[1].YOsiz), (0, 0))
            # Tile size
            self.assertEqual((c.segment[1].XTsiz, c.segment[1].YTsiz),
                             (1024, 1024))
            # Tile offset
            self.assertEqual((c.segment[1].XTOsiz, c.segment[1].YTOsiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(c.segment[1]._bitdepth, (16,))
            # signed
            self.assertEqual(c.segment[1]._signed, (False,))
            # subsampling
            self.assertEqual(list(zip(c.segment[1].XRsiz, c.segment[1].YRsiz)),
                             [(1, 1)])

            # COD: Coding style default
            self.assertFalse(c.segment[2].Scod & 2)  # no sop
            self.assertFalse(c.segment[2].Scod & 4)  # no eph
            self.assertEqual(c.segment[2].SPcod[0], glymur.core.LRCP)
            self.assertEqual(c.segment[2]._layers, 1)  # layers = 1
            self.assertEqual(c.segment[2].SPcod[3], 0)  # mct
            self.assertEqual(c.segment[2].SPcod[4], 5)  # levels
            self.assertEqual(tuple(c.segment[2]._code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(c.segment[2].SPcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(c.segment[2].SPcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(c.segment[2].SPcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(c.segment[2].SPcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(c.segment[2].SPcod[7] & 0x0020)
            self.assertEqual(c.segment[2].SPcod[8],
                             glymur.core.WAVELET_TRANSFORM_5x3_REVERSIBLE)
            self.assertEqual(len(c.segment[2].SPcod), 9)

if __name__ == "__main__":
    unittest.main()
