"""
Tests for general glymur functionality.
"""
# Standard library imports ...
import os
import sys
import tempfile
import unittest
if sys.hexversion >= 0x03030000:
    from unittest.mock import patch
else:
    from mock import patch

# Third party library imports ...
import numpy as np

# Local imports
import glymur
from glymur import Jp2k
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG
from . import fixtures


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(os.name == "nt", fixtures.WINDOWS_TMP_FILE_MSG)
class TestJp2k_write(fixtures.MetadataBase):
    """Write tests, can be run by versions 1.5+"""

    @classmethod
    def setUpClass(cls):
        cls.jp2file = glymur.data.nemo()
        cls.j2kfile = glymur.data.goodstuff()

        cls.j2k_data = glymur.Jp2k(cls.j2kfile)[:]
        cls.jp2_data = glymur.Jp2k(cls.jp2file)[:]

        # Make single channel jp2 and j2k files.
        obj = tempfile.NamedTemporaryFile(delete=False, suffix=".j2k")
        glymur.Jp2k(obj.name, data=cls.j2k_data[:, :, 0])
        cls.single_channel_j2k = obj

        obj = tempfile.NamedTemporaryFile(delete=False, suffix=".jp2")
        glymur.Jp2k(obj.name, data=cls.j2k_data[:, :, 0])
        cls.single_channel_jp2 = obj

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.single_channel_j2k.name)
        os.unlink(cls.single_channel_jp2.name)

    @unittest.skipIf(glymur.version.openjpeg_version_tuple[0] < 2,
                     "Requires as least v2.0")
    def test_null_data(self):
        """
        Verify that we prevent trying to write images with one dimension zero.
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=np.zeros((0, 256), dtype=np.uint8))

    def test_NR_ENC_Bretagne1_ppm_2_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne1.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data,
                     psnr=[30, 35, 40], numres=2)

            codestream = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 1)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self.verify_codeblock_style(codestream.segment[2].cstyle,
                                    [False, False, False, False, False, False])
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(codestream.segment[2].precinct_size,
                         ((32768, 32768)))

    def test_NR_ENC_Bretagne1_ppm_1_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne1.ppm

        """
        data = self.jp2_data
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            # Should be written with 3 layers.
            j = Jp2k(tfile.name, data=data, cratios=[200, 100, 50])
            c = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].mct, 1)  # mct
        self.assertEqual(c.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self.verify_codeblock_style(c.segment[2].cstyle,
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, ((32768, 32768)))

    @unittest.skipIf(glymur.config.load_openjpeg_library('openjpeg') is None,
                     "Needs openjpeg before this test make sense.")
    def test_NR_ENC_Bretagne1_ppm_1_encode_v15(self):
        """
        Test JPEG writing with version 1.5

        Original file tested was

            input/nonregression/Bretagne1.ppm

        """
        data = self.jp2_data
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with patch('glymur.jp2k.version.openjpeg_version_tuple',
                       new=(1, 5, 0)):
                with patch('glymur.jp2k.opj2.OPENJP2', new=None):
                    j = Jp2k(tfile.name, shape=data.shape)
                    j[:] = data
                    c = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 3
        self.assertEqual(c.segment[2].mct, 1)  # mct
        self.assertEqual(c.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblksz
        self.verify_codeblock_style(c.segment[2].cstyle,
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, ((32768, 32768)))

    @unittest.skipIf(fixtures.low_memory_linux_machine(), "Low memory machine")
    def test_NR_ENC_Bretagne1_ppm_3_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne1.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name,
                     data=self.jp2_data,
                     psnr=[30, 35, 40], cbsize=(16, 16), psizes=[(64, 64)])

            codestream = j.get_codestream()

        # COD: Coding style default
        self.assertFalse(codestream.segment[2].scod & 2)  # no sop
        self.assertFalse(codestream.segment[2].scod & 4)  # no eph
        self.assertEqual(codestream.segment[2].prog_order, glymur.core.LRCP)
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
        self.assertEqual(codestream.segment[2].mct, 1)  # mct
        self.assertEqual(codestream.segment[2].num_res, 5)  # levels
        self.assertEqual(tuple(codestream.segment[2].code_block_size),
                         (16, 16))  # cblksz
        self.verify_codeblock_style(codestream.segment[2].cstyle,
                                    [False, False,
                                     False, False, False, False])
        self.assertEqual(codestream.segment[2].xform,
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(codestream.segment[2].precinct_size,
                         ((2, 2), (4, 4), (8, 8), (16, 16), (32, 32),
                          (64, 64)))

    def test_NR_ENC_Bretagne2_ppm_4_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name,
                     data=self.jp2_data,
                     psizes=[(128, 128)] * 3,
                     cratios=[100, 20, 2],
                     tilesize=(480, 640),
                     cbsize=(32, 32))

            # Should be three layers.
            codestream = j.get_codestream()

            # RSIZ
            self.assertEqual(codestream.segment[1].xtsiz, 640)
            self.assertEqual(codestream.segment[1].ytsiz, 480)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (32, 32))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False,
                                         False, False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((16, 16), (32, 32), (64, 64), (128, 128),
                              (128, 128), (128, 128)))

    def test_NR_ENC_Bretagne2_ppm_5_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data,
                     tilesize=(127, 127), prog="PCRL")

            codestream = j.get_codestream()

            # RSIZ
            self.assertEqual(codestream.segment[1].xtsiz, 127)
            self.assertEqual(codestream.segment[1].ytsiz, 127)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.PCRL)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False,
                                         False, False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_Bretagne2_ppm_6_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, subsam=(2, 2), sop=True)

            codestream = j.get_codestream(header_only=False)

            # RSIZ
            self.assertEqual(codestream.segment[1].xrsiz, (2, 2, 2))
            self.assertEqual(codestream.segment[1].yrsiz, (2, 2, 2))

            # COD: Coding style default
            self.assertTrue(codestream.segment[2].scod & 2)  # sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

            # 18 SOP segments.
            nsops = [x.nsop for x in codestream.segment
                     if x.marker_id == 'SOP']
            self.assertEqual(nsops, list(range(18)))

    def test_NR_ENC_Bretagne2_ppm_7_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, modesw=38, eph=True)

            codestream = j.get_codestream(header_only=False)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertTrue(codestream.segment[2].scod & 4)  # eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, True, True,
                                         False, False, True])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

            # 18 EPH segments.
            ephs = [x for x in codestream.segment if x.marker_id == 'EPH']
            self.assertEqual(len(ephs), 18)

    def test_NR_ENC_Bretagne2_ppm_8_encode(self):
        """
        Original file tested was

            input/nonregression/Bretagne2.ppm
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name,
                     data=self.jp2_data, grid_offset=[300, 150], cratios=[800])

            codestream = j.get_codestream(header_only=False)

            # RSIZ
            self.assertEqual(codestream.segment[1].xosiz, 150)
            self.assertEqual(codestream.segment[1].yosiz, 300)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_Cevennes1_bmp_9_encode(self):
        """
        Original file tested was

            input/nonregression/Cevennes1.bmp

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, cratios=[800])

            codestream = j.get_codestream(header_only=False)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_Cevennes2_ppm_10_encode(self):
        """
        Original file tested was

            input/nonregression/Cevennes2.ppm

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, cratios=[50])

            codestream = j.get_codestream(header_only=False)

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_Rome_bmp_11_encode(self):
        """
        Original file tested was

            input/nonregression/Rome.bmp

        """
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            jp2 = Jp2k(tfile.name,
                       data=self.jp2_data,
                       psnr=[30, 35, 50], prog='LRCP', numres=3)

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
            self.assertEqual(jp2.box[2].box[0].height, 1456)
            self.assertEqual(jp2.box[2].box[0].width, 2592)
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

            kwargs = {'rsiz': 0, 'xysiz': (2592, 1456), 'xyosiz': (0, 0),
                      'xytsiz': (2592, 1456), 'xytosiz': (0, 0),
                      'bitdepth': (8, 8, 8), 'signed': (False, False, False),
                      'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
            self.verifySizSegment(codestream.segment[1],
                                  glymur.codestream.SIZsegment(**kwargs))

            # COD: Coding style default
            self.assertFalse(codestream.segment[2].scod & 2)  # no sop
            self.assertFalse(codestream.segment[2].scod & 4)  # no eph
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3
            self.assertEqual(codestream.segment[2].mct, 1)  # mct
            self.assertEqual(codestream.segment[2].num_res, 2)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_random_issue_0005_tif_12_encode(self):
        """
        Original file tested was

            input/nonregression/random-issue-0005.tif
        """
        data = self.jp2_data[:1024, :1024, 0].astype(np.uint16)
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
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.LRCP)
            self.assertEqual(codestream.segment[2].layers, 1)  # layers = 1
            self.assertEqual(codestream.segment[2].mct, 0)
            self.assertEqual(codestream.segment[2].num_res, 5)  # levels
            self.assertEqual(tuple(codestream.segment[2].code_block_size),
                             (64, 64))  # cblksz
            self.verify_codeblock_style(codestream.segment[2].cstyle,
                                        [False, False, False,
                                         False, False, False])
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
            self.assertEqual(codestream.segment[2].precinct_size,
                             ((32768, 32768)))

    def test_NR_ENC_issue141_rawl_23_encode(self):
        """
        Test irreversible option

        Original file tested was

            input/nonregression/issue141.rawl

        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=self.jp2_data, irreversible=True)

            codestream = j.get_codestream()
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

    def test_cblk_size_precinct_size(self):
        """
        code block sizes should never exceed half that of precinct size.
        """
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data,
                     cbsize=(64, 64), psizes=[(64, 64)])

    def test_cblk_size_not_power_of_two(self):
        """
        code block sizes should be powers of two.
        """
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data, cbsize=(13, 12))

    def test_precinct_size_not_p2(self):
        """
        precinct sizes should be powers of two.
        """
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data, psizes=[(173, 173)])

    def test_code_block_dimensions(self):
        """
        don't allow extreme codeblock sizes
        """
        # opj_compress doesn't allow the dimensions of a codeblock
        # to be too small or too big, so neither will we.
        data = self.j2k_data
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            # opj_compress doesn't allow code block area to exceed 4096.
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data, cbsize=(256, 256))

            # opj_compress doesn't allow either dimension to be less than 4.
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data, cbsize=(2048, 2))
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data, cbsize=(2, 2048))

    def test_psnr_with_cratios(self):
        """
        Using psnr with cratios options is not allowed.
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data, psnr=[30, 35, 40],
                     cratios=[2, 3, 4])

    def test_cinema2K_bad_frame_rate(self):
        """
        Cinema2k frame rate must be either 24 or 48.

        Original test input file was
        input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=self.j2k_data, cinema2k=36)

    def test_irreversible(self):
        """
        Verify that the Irreversible option works
        """
        expdata = self.j2k_data
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=expdata, irreversible=True, numres=5)

            codestream = j.get_codestream()
            self.assertEqual(codestream.segment[2].xform,
                             glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

            actdata = j[:]
            self.assertTrue(fixtures.mse(actdata, expdata) < 0.28)

    def test_shape_greyscale_jp2(self):
        """verify shape attribute for greyscale JP2 file
        """
        jp2 = Jp2k(self.single_channel_jp2.name)
        self.assertEqual(jp2.shape, (800, 480))
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_shape_single_channel_j2k(self):
        """verify shape attribute for single channel J2K file
        """
        j2k = Jp2k(self.single_channel_j2k.name)
        self.assertEqual(j2k.shape, (800, 480))

    def test_precinct_size_too_small(self):
        """first precinct size must be >= 2x that of the code block size"""
        data = np.zeros((640, 480), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cbsize=(16, 16), psizes=[(16, 16)])

    def test_precinct_size_not_power_of_two(self):
        """must be power of two"""
        data = np.zeros((640, 480), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cbsize=(16, 16), psizes=[(48, 48)])

    def test_unsupported_int32(self):
        """Should raise a runtime error if trying to write int32"""
        data = np.zeros((128, 128), dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data)

    def test_unsupported_uint32(self):
        """Should raise a runtime error if trying to write uint32"""
        data = np.zeros((128, 128), dtype=np.uint32)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(RuntimeError):
                Jp2k(tfile.name, data=data)

    def test_write_with_version_too_early(self):
        """Should raise a runtime error if trying to write with version 1.3"""
        data = np.zeros((128, 128), dtype=np.uint8)
        versions = ["1.0.0", "1.1.0", "1.2.0", "1.3.0"]
        for version in versions:
            with patch('glymur.version.openjpeg_version', new=version):
                with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
                    with self.assertRaises(RuntimeError):
                        Jp2k(tfile.name, data=data)

    def test_cblkh_different_than_width(self):
        """Verify that we can set a code block size where height does not equal
        width.
        """
        data = np.zeros((128, 128), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            # The code block dimensions are given as rows x columns.
            j = Jp2k(tfile.name, data=data, cbsize=(16, 32))
            codestream = j.get_codestream()

            # Code block size is reported as XY in the codestream.
            self.assertEqual(codestream.segment[2].code_block_size, (16, 32))

    def test_too_many_dimensions(self):
        """OpenJP2 only allows 2D or 3D images."""
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name,
                     data=np.zeros((128, 128, 2, 2), dtype=np.uint8))

    def test_2d_rgb(self):
        """RGB must have at least 3 components."""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name,
                     data=np.zeros((128, 128, 2), dtype=np.uint8),
                     colorspace='rgb')

    def test_colorspace_with_j2k(self):
        """Specifying a colorspace with J2K does not make sense"""
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name,
                     data=np.zeros((128, 128, 3), dtype=np.uint8),
                     colorspace='rgb')

    def test_specify_rgb(self):
        """specify RGB explicitly"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            j = Jp2k(tfile.name,
                     data=np.zeros((128, 128, 3), dtype=np.uint8),
                     colorspace='rgb')
            self.assertEqual(j.box[2].box[1].colorspace, glymur.core.SRGB)

    def test_specify_gray(self):
        """test gray explicitly specified (that's GRAY, not GREY)"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data, colorspace='gray')
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    def test_specify_grey(self):
        """test grey explicitly specified"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data, colorspace='grey')
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    def test_grey_with_two_extra_comps(self):
        """should be able to write gray + two extra components"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            data = np.zeros((128, 128, 3), dtype=np.uint8)
            j = Jp2k(tfile.name, data=data, colorspace='gray')
            self.assertEqual(j.box[2].box[0].height, 128)
            self.assertEqual(j.box[2].box[0].width, 128)
            self.assertEqual(j.box[2].box[0].num_components, 3)
            self.assertEqual(j.box[2].box[1].colorspace,
                             glymur.core.GREYSCALE)

    def test_specify_ycc(self):
        """Should reject YCC"""
        data = np.zeros((128, 128, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data, colorspace='ycc')

    def test_write_with_jp2_in_caps(self):
        """should be able to write with JP2 suffix."""
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        with tempfile.NamedTemporaryFile(suffix='.JP2') as tfile:
            ofile = Jp2k(tfile.name, data=expdata)
            actdata = ofile[:]
            np.testing.assert_array_equal(actdata, expdata)

    def test_write_srgb_without_mct(self):
        """should be able to write RGB without specifying mct"""
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, data=expdata, mct=False)
            actdata = ofile[:]
            np.testing.assert_array_equal(actdata, expdata)

            codestream = ofile.get_codestream()
            self.assertEqual(codestream.segment[2].mct, 0)  # no mct

    def test_write_grayscale_with_mct(self):
        """
        MCT usage makes no sense for grayscale images.
        """
        j2k = Jp2k(self.j2kfile)
        expdata = j2k[:]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=expdata[:, :, 0], mct=True)

    def test_write_cprl(self):
        """Must be able to write a CPRL progression order file"""
        # Issue 17
        j = Jp2k(self.jp2file)
        expdata = j[::2, ::2]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            ofile = Jp2k(tfile.name, data=expdata, prog='CPRL')
            actdata = ofile[:]
            np.testing.assert_array_equal(actdata, expdata)

            codestream = ofile.get_codestream()
            self.assertEqual(codestream.segment[2].prog_order,
                             glymur.core.CPRL)
