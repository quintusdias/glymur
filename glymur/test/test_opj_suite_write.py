"""
The tests defined here roughly correspond to what is in the OpenJPEG test
suite.
"""
# C0103:  method names longer that 30 chars are ok in tests, IMHO
# R0904:  Seems like pylint is fooled in this situation
# pylint: disable=R0904,C0103

# unittest2 is python2.6 only (pylint/python-2.7)
# pylint: disable=F0401

import os
import re
import sys
import tempfile

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

from .fixtures import read_image, NO_READ_BACKEND, NO_READ_BACKEND_MSG
from .fixtures import OPJ_DATA_ROOT, opj_data_file

from glymur import Jp2k
import glymur


@unittest.skipIf(os.name == "nt", "no write support on windows, period")
@unittest.skipIf(re.match(r"""1\.[01234]\.\d""",
                          glymur.version.openjpeg_version) is not None,
                 "Writing only supported with openjpeg version 1.5+.")
@unittest.skipIf(NO_READ_BACKEND, NO_READ_BACKEND_MSG)
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuiteWrite(unittest.TestCase):
    """Tests for writing with openjp2 backend.

    These tests roughly correspond with those tests with similar names in the
    OpenJPEG test suite.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_NR_ENC_Bretagne1_ppm_1_encode(self):
        """NR-ENC-Bretagne1.ppm-1-encode"""
        infile = opj_data_file('input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, cratios=[200, 100, 50])

            # Should be three layers.
            codestream = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Bretagne1_ppm_2_encode(self):
        """NR-ENC-Bretagne1.ppm-2-encode"""
        infile = opj_data_file('input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, psnr=[30, 35, 40], numres=2)

            # Should be three layers.
            codestream = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth,
                             (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 1)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Bretagne1_ppm_3_encode(self):
        """NR-ENC-Bretagne1.ppm-3-encode"""
        infile = opj_data_file('input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, psnr=[30, 35, 40], cbsize=(16, 16),
                    psizes=[(64, 64)])

            # Should be three layers.
            codestream = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (16, 16))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32),
                              (64, 64)])

    def test_NR_ENC_Bretagne2_ppm_4_encode(self):
        """NR-ENC-Bretagne2.ppm-4-encode"""
        infile = opj_data_file('input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data,
                    psizes=[(128, 128)] * 3,
                    cratios=[100, 20, 2],
                    tilesize=(480, 640),
                    cbsize=(32, 32))

            # Should be three layers.
            codestream = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (data.shape[1], data.shape[0]))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size.  Reported as XY, not RC.
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (32, 32))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             [(16, 16), (32, 32), (64, 64)] + [(128, 128)] * 3)

    def test_NR_ENC_Bretagne2_ppm_5_encode(self):
        """NR-ENC-Bretagne2.ppm-5-encode"""
        infile = opj_data_file('input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, tilesize=(127, 127), prog="PCRL")

            codestream = j.get_codestream()

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (data.shape[1], data.shape[0]))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (127, 127))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.PCRL)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Bretagne2_ppm_6_encode(self):
        """NR-ENC-Bretagne2.ppm-6-encode"""
        infile = opj_data_file('input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, subsam=(2, 2), sop=True)

            codestream = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (5183, 3887))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (5183, 3887))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(2, 2)] * 3)

            # COD: Coding style default
            self.assertTrue(codestream.segment[2].scod & 2)  # sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding bypass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # Reset context probabilities
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

            # 18 SOP segments.
            nsops = [x.nsop for x in codestream.segment
                     if x.marker_id == 'SOP']
            self.assertEqual(nsops, list(range(18)))

    def test_NR_ENC_Bretagne2_ppm_7_encode(self):
        """NR-ENC-Bretagne2.ppm-7-encode"""
        infile = opj_data_file('input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, modesw=38, eph=True)

            codestream = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (2592, 1944))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (2592, 1944))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertTrue(codestream.segment[2].scod & 4)  # eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                            (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertTrue(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertTrue(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertTrue(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

            # 18 EPH segments.
            ephs = [x for x in codestream.segment if x.marker_id == 'EPH']
            self.assertEqual(len(ephs), 18)

    def test_NR_ENC_Bretagne2_ppm_8_encode(self):
        """NR-ENC-Bretagne2.ppm-8-encode"""
        infile = opj_data_file('input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, grid_offset=[300, 150], cratios=[800])

            codestream = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (2742, 2244))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz),
                             (150, 300))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (2742, 2244))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Cevennes1_bmp_9_encode(self):
        """NR-ENC-Cevennes1.bmp-9-encode"""
        infile = opj_data_file('input/nonregression/Cevennes1.bmp')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, cratios=[800])

            codestream = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (2592, 1944))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (2592, 1944))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Cevennes2_ppm_10_encode(self):
        """NR-ENC-Cevennes2.ppm-10-encode"""
        infile = opj_data_file('input/nonregression/Cevennes2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data, cratios=[50])

            codestream = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Rome_bmp_11_encode(self):
        """NR-ENC-Rome.bmp-11-encode"""
        data = read_image(opj_data_file('input/nonregression/Rome.bmp'))
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            jp2 = Jp2k(tfile.name, 'wb')
            jp2.write(data, psnr=[30, 35, 50], prog='LRCP', numres=3)

            ids = [box.box_id for box in jp2.box]
            self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

            ids = [box.box_id for box in jp2.box[2].box]
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

            codestream = jp2.box[3].main_header

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (640, 480))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz),
                             (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (640, 480))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (8, 8, 8))
            # signed
            self.assertEqual(codestream.segment[1].signed,
                             (False, False, False))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)] * 3)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 2)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    @unittest.skip("Known failure in openjpeg test suite.")
    def test_NR_ENC_random_issue_0005_tif_12_encode(self):
        """NR-ENC-random-issue-0005.tif-12-encode"""
        # opj_decompress has trouble reading it, but that is not an issue here.
        # The nature of the image itself seems to give the compressor trouble.
        infile = opj_data_file('input/nonregression/random-issue-0005.tif')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, 'wb')
            j.write(data)

            codestream = j.get_codestream(header_only=False)

            # SIZ: Image and tile size
            # Profile:  "0" means profile 2
            self.assertEqual(codestream.segment[1].rsiz, 0)
            # Reference grid size
            self.assertEqual((codestream.segment[1].xsiz,
                              codestream.segment[1].ysiz),
                             (1024, 1024))
            # Reference grid offset
            self.assertEqual((codestream.segment[1].xosiz,
                              codestream.segment[1].yosiz), (0, 0))
            # Tile size
            self.assertEqual((codestream.segment[1].xtsiz,
                              codestream.segment[1].ytsiz),
                             (1024, 1024))
            # Tile offset
            self.assertEqual((codestream.segment[1].xtosiz,
                              codestream.segment[1].ytosiz),
                             (0, 0))
            # bitdepth
            self.assertEqual(codestream.segment[1].bitdepth, (16,))
            # signed
            self.assertEqual(codestream.segment[1].signed, (False,))
            # subsampling
            self.assertEqual(list(zip(codestream.segment[1].xrsiz,
                                      codestream.segment[1].yrsiz)),
                             [(1, 1)])

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 0)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            # Selective arithmetic coding BYPASS
            self.assertFalse(codestream.segment[2].spcod[7] & 0x01)
            # RESET context probabilities (RESET)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x02)
            # Termination on each coding pass, RESTART(TERMALL)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x04)
            # Vertically causal context (VSC)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x08)
            # Predictable termination, ERTERM(SEGTERM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0010)
            # Segmentation symbols, SEGMARK(SEGSYSM)
            self.assertFalse(codestream.segment[2].spcod[7] & 0x0020)
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

if __name__ == "__main__":
    unittest.main()
