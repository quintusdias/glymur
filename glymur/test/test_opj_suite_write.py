"""
The tests defined here roughly correspond to what is in the OpenJPEG test
suite.
"""
import os
import re
import sys
import tempfile
import unittest

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch

import numpy as np
try:
    import skimage.io
except ImportError:
    pass

from .fixtures import read_image, NO_READ_BACKEND, NO_READ_BACKEND_MSG
from .fixtures import OPJ_DATA_ROOT, NO_SKIMAGE_FREEIMAGE_SUPPORT
from .fixtures import opj_data_file
from .fixtures import WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG
from . import fixtures

import glymur
from glymur import Jp2k
from glymur.codestream import SIZsegment
from glymur.version import openjpeg_version


class CinemaBase(fixtures.MetadataBase):

    def verify_cinema_cod(self, cod_segment):

        self.assertFalse(cod_segment.scod & 2)  # no sop
        self.assertFalse(cod_segment.scod & 4)  # no eph
        self.assertEqual(cod_segment.spcod[0], glymur.core.CPRL)
        self.assertEqual(cod_segment.layers, 1)
        self.assertEqual(cod_segment.spcod[3], 1)  # mct
        self.assertEqual(cod_segment.spcod[4], 5)  # levels
        self.assertEqual(tuple(cod_segment.code_block_size), (32, 32))

    def check_cinema4k_codestream(self, codestream, image_size):

        kwargs = {'rsiz': 4, 'xysiz': image_size, 'xyosiz': (0, 0),
                  'xytsiz': image_size, 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12), 'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(codestream.segment[1], SIZsegment(**kwargs))

        self.verify_cinema_cod(codestream.segment[2])

    def check_cinema2k_codestream(self, codestream, image_size):

        kwargs = {'rsiz': 3, 'xysiz': image_size, 'xyosiz': (0, 0),
                  'xytsiz': image_size, 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12), 'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(codestream.segment[1], SIZsegment(**kwargs))

        self.verify_cinema_cod(codestream.segment[2])


@unittest.skipIf(NO_SKIMAGE_FREEIMAGE_SUPPORT,
                 "Cannot read input image without scikit-image/freeimage")
@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
@unittest.skipIf(re.match(r'''(1|2.0.0)''',
                          glymur.version.openjpeg_version) is not None,
                 "Uses features not supported until 2.0.1")
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class WriteCinema(CinemaBase):
    """Tests for writing with openjp2 backend.

    These tests either roughly correspond with those tests with similar names
    in the OpenJPEG test suite or are closely associated.
    """
    def test_cinema2K_with_others(self):
        """Can't specify cinema2k with any other options."""
        relfile = 'input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cinema2k=48, cratios=[200, 100, 50])

    def test_cinema4K_with_others(self):
        """Can't specify cinema4k with any other options."""
        relfile = 'input/nonregression/ElephantDream_4K.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cinema4k=True, cratios=[200, 100, 50])


@unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
@unittest.skipIf(NO_SKIMAGE_FREEIMAGE_SUPPORT,
                 "Cannot read input image without scikit-image/freeimage")
@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
@unittest.skipIf(re.match(r'''(1|2.0.0)''',
                          glymur.version.openjpeg_version) is not None,
                 "Uses features not supported until 2.0.1")
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class WriteCinemaWarns(CinemaBase):
    """Tests for writing with openjp2 backend.

    These tests either roughly correspond with those tests with similar names
    in the OpenJPEG test suite or are closely associated.  These tests issue
    warnings.
    """
    def test_NR_ENC_ElephantDream_4K_tif_21_encode(self):
        relfile = 'input/nonregression/ElephantDream_4K.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            regex = 'OpenJPEG library warning:.*'
            with self.assertWarnsRegex(UserWarning, re.compile(regex)):
                j = Jp2k(tfile.name, data=data, cinema4k=True)

            codestream = j.get_codestream()
            self.check_cinema4k_codestream(codestream, (4096, 2160))

    def test_NR_ENC_X_5_2K_24_235_CBR_STEM24_000_tif_19_encode(self):
        relfile = 'input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertWarnsRegex(UserWarning,
                                       'OpenJPEG library warning'):
                j = Jp2k(tfile.name, data=data, cinema2k=48)

            codestream = j.get_codestream()
            self.check_cinema2k_codestream(codestream, (2048, 857))

    def test_NR_ENC_X_6_2K_24_FULL_CBR_CIRCLE_000_tif_20_encode(self):
        relfile = 'input/nonregression/X_6_2K_24_FULL_CBR_CIRCLE_000.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertWarnsRegex(UserWarning,
                                       'OpenJPEG library warning'):
                j = Jp2k(tfile.name, data=data, cinema2k=48)

            codestream = j.get_codestream()
            self.check_cinema2k_codestream(codestream, (2048, 1080))

    def test_NR_ENC_X_6_2K_24_FULL_CBR_CIRCLE_000_tif_17_encode(self):
        relfile = 'input/nonregression/X_6_2K_24_FULL_CBR_CIRCLE_000.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertWarnsRegex(UserWarning,
                                       'OpenJPEG library warning'):
                j = Jp2k(tfile.name, data=data, cinema2k=24)

            codestream = j.get_codestream()
            self.check_cinema2k_codestream(codestream, (2048, 1080))

    def test_NR_ENC_X_5_2K_24_235_CBR_STEM24_000_tif_16_encode(self):
        relfile = 'input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertWarnsRegex(UserWarning,
                                       'OpenJPEG library warning'):
                # OpenJPEG library warning:  The desired maximum codestream
                # size has limited at least one of the desired quality layers
                j = Jp2k(tfile.name, data=data, cinema2k=24)

            codestream = j.get_codestream()
            self.check_cinema2k_codestream(codestream, (2048, 857))

    def test_NR_ENC_X_4_2K_24_185_CBR_WB_000_tif_18_encode(self):
        relfile = 'input/nonregression/X_4_2K_24_185_CBR_WB_000.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            regex = 'OpenJPEG library warning'
            with self.assertWarnsRegex(UserWarning, regex):
                # OpenJPEG library warning:  The desired maximum codestream
                # size has limited at least one of the desired quality layers
                j = Jp2k(tfile.name, data=data, cinema2k=48)

            codestream = j.get_codestream()
            self.check_cinema2k_codestream(codestream, (1998, 1080))


@unittest.skipIf(NO_SKIMAGE_FREEIMAGE_SUPPORT,
                 "Cannot read input image without scikit-image/freeimage")
@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_OPJ_DATA_ROOT environment variable not set")
class TestNegative2pointzero(unittest.TestCase):
    """Feature set not supported for versions less than 2.0.1"""

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

    def tearDown(self):
        pass

    def test_cinema_mode(self):
        """Cinema mode not allowed for anything less than 2.0.1"""
        relfile = 'input/nonregression/X_4_2K_24_185_CBR_WB_000.tif'
        infile = opj_data_file(relfile)
        data = skimage.io.imread(infile)
        versions = ["1.5.0", "2.0.0"]
        for version in versions:
            with patch('glymur.version.openjpeg_version', new=version):
                with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
                    with self.assertRaises(IOError):
                        Jp2k(tfile.name, data=data, cinema2k=48)


@unittest.skipIf(re.match(r'''1.[0-4]''', openjpeg_version) is not None,
                 "Writing not supported until OpenJPEG 1.5")
@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
@unittest.skipIf(NO_READ_BACKEND, NO_READ_BACKEND_MSG)
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuiteWrite(fixtures.MetadataBase):
    """Tests for writing with openjp2 backend.

    These tests either roughly correspond with those tests with similar names
    in the OpenJPEG test suite or are closely associated.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_NR_ENC_issue141_rawl_23_encode(self):
        filename = opj_data_file('input/nonregression/issue141.rawl')
        expdata = np.fromfile(filename, dtype=np.uint16)
        expdata.resize((2816, 2048))
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=expdata, irreversible=True)

            codestream = j.get_codestream()
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

    def test_NR_ENC_Bretagne1_ppm_1_encode(self):
        """NR-ENC-Bretagne1.ppm-1-encode"""
        infile = opj_data_file('input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=data, cratios=[200, 100, 50])

            # Should be three layers.
            c = j.get_codestream()

        kwargs = {'rsiz': 0, 'xysiz': (640, 480), 'xyosiz': (0, 0),
                  'xytsiz': (640, 480), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

    def test_NR_ENC_Bretagne1_ppm_2_encode(self):
        """NR-ENC-Bretagne1.ppm-2-encode"""
        infile = opj_data_file('input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=data, psnr=[30, 35, 40], numres=2)

            # Should be three layers.
            codestream = j.get_codestream()

            kwargs = {'rsiz': 0, 'xysiz': (640, 480), 'xyosiz': (0, 0),
                      'xytsiz': (640, 480), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 1)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False,
                                         False, False, False, False])
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Bretagne1_ppm_3_encode(self):
        """NR-ENC-Bretagne1.ppm-3-encode"""
        infile = opj_data_file('input/nonregression/Bretagne1.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name,
                     data=data,
                     psnr=[30, 35, 40], cbsize=(16, 16), psizes=[(64, 64)])

            # Should be three layers.
            codestream = j.get_codestream()

            kwargs = {'rsiz': 0, 'xysiz': (640, 480), 'xyosiz': (0, 0),
                      'xytsiz': (640, 480), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (16, 16))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False,
                                         False, False, False, False])
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
            j = Jp2k(tfile.name,
                     data=data,
                     psizes=[(128, 128)] * 3,
                     cratios=[100, 20, 2],
                     tilesize=(480, 640),
                     cbsize=(32, 32))

            # Should be three layers.
            codestream = j.get_codestream()

            kwargs = {'rsiz': 0, 'xysiz': (2592, 1944), 'xyosiz': (0, 0),
                      'xytsiz': (640, 480), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (32, 32))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False,
                                         False, False, False, False])
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             [(16, 16), (32, 32), (64, 64)] + [(128, 128)] * 3)

    def test_NR_ENC_Bretagne2_ppm_5_encode(self):
        """NR-ENC-Bretagne2.ppm-5-encode"""
        infile = opj_data_file('input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=data, tilesize=(127, 127), prog="PCRL")

            codestream = j.get_codestream()

            kwargs = {'rsiz': 0, 'xysiz': (2592, 1944), 'xyosiz': (0, 0),
                      'xytsiz': (127, 127), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.PCRL)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False,
                                         False, False, False, False])
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Bretagne2_ppm_6_encode(self):
        """NR-ENC-Bretagne2.ppm-6-encode"""
        infile = opj_data_file('input/nonregression/Bretagne2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=data, subsam=(2, 2), sop=True)

            codestream = j.get_codestream(header_only=False)

            kwargs = {'rsiz': 0, 'xysiz': (5183, 3887), 'xyosiz': (0, 0),
                      'xytsiz': (5183, 3887), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(2, 2, 2), (2, 2, 2)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertTrue(codestream.segment[2].scod & 2)  # sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False, False,
                                         False, False, False])
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
            j = Jp2k(tfile.name, data=data, modesw=38, eph=True)

            codestream = j.get_codestream(header_only=False)

            kwargs = {'rsiz': 0, 'xysiz': (2592, 1944), 'xyosiz': (0, 0),
                      'xytsiz': (2592, 1944), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertTrue(codestream.segment[2].scod & 4)  # eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, True, True,
                                         False, False, True])
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
            j = Jp2k(tfile.name,
                     data=data, grid_offset=[300, 150], cratios=[800])

            codestream = j.get_codestream(header_only=False)

            kwargs = {'rsiz': 0, 'xysiz': (2742, 2244), 'xyosiz': (150, 300),
                      'xytsiz': (2742, 2244), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Cevennes1_bmp_9_encode(self):
        """NR-ENC-Cevennes1.bmp-9-encode"""
        infile = opj_data_file('input/nonregression/Cevennes1.bmp')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=data, cratios=[800])

            codestream = j.get_codestream(header_only=False)

            kwargs = {'rsiz': 0, 'xysiz': (2592, 1944), 'xyosiz': (0, 0),
                      'xytsiz': (2592, 1944), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Cevennes2_ppm_10_encode(self):
        """NR-ENC-Cevennes2.ppm-10-encode"""
        infile = opj_data_file('input/nonregression/Cevennes2.ppm')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=data, cratios=[50])

            codestream = j.get_codestream(header_only=False)

            kwargs = {'rsiz': 0, 'xysiz': (640, 480), 'xyosiz': (0, 0),
                      'xytsiz': (640, 480), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_Rome_bmp_11_encode(self):
        """NR-ENC-Rome.bmp-11-encode"""
        data = read_image(opj_data_file('input/nonregression/Rome.bmp'))
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            jp2 = Jp2k(tfile.name,
                       data=data, psnr=[30, 35, 50], prog='LRCP', numres=3)

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

            codestream = jp2.box[3].codestream

            kwargs = {'rsiz': 0, 'xysiz': (640, 480), 'xyosiz': (0, 0),
                      'xytsiz': (640, 480), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].spcod[3], 1)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 2)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)

    def test_NR_ENC_random_issue_0005_tif_12_encode(self):
        """NR-ENC-random-issue-0005.tif-12-encode"""
        # opj_decompress has trouble reading it, but that is not an issue here.
        # The nature of the image itself seems to give the compressor trouble.
        infile = opj_data_file('input/nonregression/random-issue-0005.tif')
        data = read_image(infile)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=data)

            codestream = j.get_codestream(header_only=False)

            kwargs = {'rsiz': 0, 'xysiz': (1024, 1024), 'xyosiz': (0, 0),
                      'xytsiz': (1024, 1024), 'xytosiz': (0, 0),
                      'bitdepth': (16,), 'signed': (False,),
                      'xyrsiz': [(1,), (1,)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].spcod[0], glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].spcod[3], 0)  # mct
            self.assertEqual(codestream.segment[2].spcod[4], 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].spcod[7],
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].spcod[8],
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(len(codestream.segment[2].spcod), 9)


if __name__ == "__main__":
    unittest.main()
