"""
The tests defined here roughly correspond to what is in the OpenJPEG test
suite.
"""
import re
import unittest
import warnings

import numpy as np

import glymur
from glymur import Jp2k
from glymur.codestream import CMEsegment, SOTsegment, RGNsegment
from glymur.core import (RCME_ISO_8859_1, RCME_BINARY, SRGB,
                         GREYSCALE, RESTRICTED_ICC_PROFILE,
                         ENUMERATED_COLORSPACE)
from glymur.jp2box import FileTypeBox

from .fixtures import (MetadataBase, OPJ_DATA_ROOT,
                       WARNING_INFRASTRUCTURE_ISSUE,
                       WARNING_INFRASTRUCTURE_MSG,
                       opj_data_file)

comment1 = "Creator: AV-J2K (c) 2000,2001 Algo Vision Technology"


@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuite(MetadataBase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_NR_file409752(self):
        jfile = opj_data_file('input/nonregression/file409752.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(243, 720, num_components=3)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=glymur.core.YCC)
        self.verifyColourSpecificationBox(jp2.box[2].box[1], colr)

        c = jp2.box[3].codestream

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (720, 243), 'xyosiz': (0, 0),
                  'xytsiz': (720, 243), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 2, 2), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 128))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa,
                         [1816, 1792, 1792, 1724, 1770, 1770, 1724, 1868,
                          1868, 1892, 3, 3, 69, 2002, 2002, 1889])
        self.assertEqual(c.segment[3].exponent,
                         [13] * 4 + [12] * 3 + [11] * 3 + [9] * 6)

    def test_NR_p0_01_dump(self):
        jfile = opj_data_file('input/conformance/p0_01.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # Segment IDs.
        actual = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'QCD', 'COD', 'SOT', 'SOD', 'EOC']
        self.assertEqual(actual, expected)

        kwargs = {'rsiz': 1, 'xysiz': (128, 128), 'xyosiz': (0, 0),
                  'xytsiz': (128, 128), 'xytosiz': (0, 0), 'bitdepth': (8,),
                  'signed': (False,), 'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # QCD: Quantization default
        self.assertEqual(c.segment[2].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[2].guard_bits, 2)
        self.assertEqual(c.segment[2].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])
        self.assertEqual(c.segment[2].mantissa,
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # COD: Coding style default
        self.assertFalse(c.segment[3].scod & 2)  # no sop
        self.assertFalse(c.segment[3].scod & 4)  # no eph
        self.assertEqual(c.segment[3].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[3].layers, 1)  # layers = 1
        self.assertEqual(c.segment[3].spcod[3], 0)  # mct
        self.assertEqual(c.segment[3].spcod[4], 3)  # layers
        self.assertEqual(tuple(c.segment[3].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[3].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[3].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        self.verifySOTsegment(c.segment[4], SOTsegment(0, 7314, 0, 1))

    def test_NR_p0_02_dump(self):
        jfile = opj_data_file('input/conformance/p0_02.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (127, 126), 'xyosiz': (0, 0),
                  'xytsiz': (127, 126), 'xytosiz': (0, 0), 'bitdepth': (8,),
                  'signed': (False,), 'xyrsiz': [(2,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # sop
        self.assertTrue(c.segment[2].scod & 4)  # eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 6)  # layers = 6
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, True, False, True, True])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 0)
        self.assertEqual(c.segment[3].spcoc[0], 3)  # levels
        self.assertEqual(tuple(c.segment[3].code_block_size),
                         (32, 32))  # cblk
        self.verify_codeblock_style(c.segment[3].spcoc[3],
                                    [False, False, True, False, True, True])
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4].guard_bits, 3)
        self.assertEqual(c.segment[4].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])
        self.assertEqual(c.segment[4].mantissa,
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[5], CMEsegment(*pargs))

        # One unknown marker
        self.assertEqual(c.segment[6].marker_id, '0xff30')

        self.verifySOTsegment(c.segment[7], SOTsegment(0, 6047, 0, 1))

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[8].marker_id, 'SOD')

        # SOP, EPH
        sop = [x.marker_id for x in c.segment if x.marker_id == 'SOP']
        eph = [x.marker_id for x in c.segment if x.marker_id == 'EPH']
        self.assertEqual(len(sop), 24)
        self.assertEqual(len(eph), 24)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_NR_p0_03_dump(self):
        jfile = opj_data_file('input/conformance/p0_03.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (256, 256), 'xyosiz': (0, 0),
                  'xytsiz': (128, 128), 'xytosiz': (0, 0), 'bitdepth': (4,),
                  'signed': (True,), 'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 8)  # 8
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 1)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 1)  # scalar implicit
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].exponent, [0])
        self.assertEqual(c.segment[3].mantissa, [0])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].cqcc, 0)
        self.assertEqual(c.segment[4].guard_bits, 2)
        # quantization type
        self.assertEqual(c.segment[4].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[4].exponent, [4, 5, 5, 6])
        self.assertEqual(c.segment[4].mantissa, [0, 0, 0, 0])

        # POD: progression order change
        self.assertEqual(c.segment[5].rspod, (0,))
        self.assertEqual(c.segment[5].cspod, (0,))
        self.assertEqual(c.segment[5].lyepod, (8,))
        self.assertEqual(c.segment[5].repod, (33,))
        self.assertEqual(c.segment[5].cdpod, (255,))
        self.assertEqual(c.segment[5].ppod, (glymur.core.LRCP,))

        # CRG:  component registration
        self.assertEqual(c.segment[6].xcrg, (65424,))
        self.assertEqual(c.segment[6].ycrg, (32558,))

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[7], CMEsegment(*pargs))

        pargs = (RCME_ISO_8859_1, comment1.encode())
        self.verifyCMEsegment(c.segment[8], CMEsegment(*pargs))

        pargs = (RCME_BINARY, c.segment[9].ccme)
        self.verifyCMEsegment(c.segment[9], CMEsegment(*pargs))

        # TLM (tile-part length)
        self.assertEqual(c.segment[10].ztlm, 0)
        self.assertEqual(c.segment[10].ttlm, (0, 1, 2, 3))
        self.assertEqual(c.segment[10].ptlm, (4267, 2117, 4080, 2081))

        self.verifySOTsegment(c.segment[11], SOTsegment(0, 4267, 0, 1))
        self.verifyRGNsegment(c.segment[12], RGNsegment(0, 0, 7))

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[13].marker_id, 'SOD')

    def test_NR_p0_04_dump(self):
        jfile = opj_data_file('input/conformance/p0_04.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (640, 480), 'xyosiz': (0, 0),
                  'xytsiz': (640, 480), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 20)  # 20
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, True, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size,
                         [(128, 128), (128, 128), (128, 128), (128, 128),
                          (128, 128), (128, 128), (128, 128)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)  # scalar expounded
        self.assertEqual(c.segment[3].guard_bits, 3)
        self.assertEqual(c.segment[3].exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13,
                          11, 11, 11, 11, 11, 11])
        self.assertEqual(c.segment[3].mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                          1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                          1888])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].cqcc, 1)
        # quantization type
        self.assertEqual(c.segment[4].sqcc & 0x1f, 2)  # none
        self.assertEqual(c.segment[4].guard_bits, 3)
        self.assertEqual(c.segment[4].exponent,
                         [14, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11,
                          9, 9, 9, 9, 9, 9])
        self.assertEqual(c.segment[4].mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845,
                          1845, 1868, 1925, 1925, 2007, 32, 32, 131, 2002,
                          2002, 1888])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].cqcc, 2)
        # quantization type
        self.assertEqual(c.segment[5].sqcc & 0x1f, 2)  # none
        self.assertEqual(c.segment[5].guard_bits, 3)
        self.assertEqual(c.segment[5].exponent,
                         [14, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11,
                          9, 9, 9, 9, 9, 9])
        self.assertEqual(c.segment[5].mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845,
                          1845, 1868, 1925, 1925, 2007, 32, 32, 131, 2002,
                          2002, 1888])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[6], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[7], SOTsegment(0, 264383, 0, 1))

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[8].marker_id, 'SOD')

    def test_NR_p0_05_dump(self):
        jfile = opj_data_file('input/conformance/p0_05.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (1024, 1024), 'xyosiz': (0, 0),
                  'xytsiz': (1024, 1024), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8, 8),
                  'signed': (False, False, False, False),
                  'xyrsiz': [(1, 1, 2, 2), (1, 1, 2, 2)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 7)  # 7
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 32))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 1)
        self.assertEqual(c.segment[3].spcoc[0], 3)  # levels
        self.assertEqual(tuple(c.segment[3].code_block_size),
                         (32, 32))  # cblk
        self.verify_codeblock_style(c.segment[3].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[4].ccoc, 3)
        self.assertEqual(c.segment[4].spcoc[0], 6)  # levels
        self.assertEqual(tuple(c.segment[4].code_block_size),
                         (32, 32))  # cblk
        self.verify_codeblock_style(c.segment[4].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[4].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[5].sqcd & 0x1f, 2)  # scalar expounded
        self.assertEqual(c.segment[5].guard_bits, 3)
        self.assertEqual(c.segment[5].exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13,
                          11, 11, 11, 11, 11, 11])
        self.assertEqual(c.segment[5].mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845,
                          1845, 1868, 1925, 1925, 2007, 32, 32, 131, 2002,
                          2002, 1888])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].cqcc, 0)
        # quantization type
        self.assertEqual(c.segment[6].sqcc & 0x1f, 1)  # scalar derived
        self.assertEqual(c.segment[6].guard_bits, 3)
        self.assertEqual(c.segment[6].exponent, [14])
        self.assertEqual(c.segment[6].mantissa, [0])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[7].cqcc, 3)
        # quantization type
        self.assertEqual(c.segment[7].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[7].guard_bits, 3)
        self.assertEqual(c.segment[7].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10,
                          9, 9, 10])
        self.assertEqual(c.segment[7].mantissa, [0] * 19)

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[8], CMEsegment(*pargs))

        # TLM (tile-part length)
        self.assertEqual(c.segment[9].ztlm, 0)
        self.assertEqual(c.segment[9].ttlm, (0,))
        self.assertEqual(c.segment[9].ptlm, (1310540,))

        self.verifySOTsegment(c.segment[10], SOTsegment(0, 1310540, 0, 1))

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[11].marker_id, 'SOD')

    def test_NR_p0_06_dump(self):
        jfile = opj_data_file('input/conformance/p0_06.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 2, 'xysiz': (513, 129), 'xyosiz': (0, 0),
                  'xytsiz': (513, 129), 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12, 12),
                  'signed': (False, False, False, False),
                  'xyrsiz': [(1, 2, 1, 2), (1, 1, 2, 2)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RPCL)
        self.assertEqual(c.segment[2].layers, 4)  # 4
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)  # scalar expounded
        self.assertEqual(c.segment[3].guard_bits, 3)
        self.assertEqual(c.segment[3].mantissa,
                         [512, 518, 522, 524, 516, 524, 522, 527, 523, 549,
                          557, 561, 853, 852, 700, 163, 78, 1508, 1831])
        self.assertEqual(c.segment[3].exponent,
                         [7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 3, 3, 2, 1, 2,
                          1])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].cqcc, 1)
        # quantization type
        self.assertEqual(c.segment[4].sqcc & 0x1f, 2)  # scalar derived
        self.assertEqual(c.segment[4].guard_bits, 4)
        self.assertEqual(c.segment[4].mantissa,
                         [1527, 489, 665, 506, 487, 502, 493, 493, 500, 485,
                          505, 491, 490, 491, 499, 509, 503, 496, 558])
        self.assertEqual(c.segment[4].exponent,
                         [10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6,
                          5, 5, 5])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].cqcc, 2)
        # quantization type
        self.assertEqual(c.segment[5].sqcc & 0x1f, 2)  # scalar derived
        self.assertEqual(c.segment[5].guard_bits, 5)
        self.assertEqual(c.segment[5].mantissa,
                         [1337, 728, 890, 719, 716, 726, 700, 718, 704, 704,
                          712, 712, 717, 719, 701, 749, 753, 718, 841])
        self.assertEqual(c.segment[5].exponent,
                         [10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6,
                          5, 5, 5])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].cqcc, 3)
        # quantization type
        self.assertEqual(c.segment[6].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[6].guard_bits, 6)
        self.assertEqual(c.segment[6].mantissa, [0] * 19)
        self.assertEqual(c.segment[6].exponent,
                         [12, 13, 13, 14, 13, 13, 14, 13, 13, 14, 13, 13, 14,
                          13, 13, 14, 13, 13, 14])

        # COC: Coding style component
        self.assertEqual(c.segment[7].ccoc, 3)
        self.assertEqual(c.segment[7].spcoc[0], 6)  # levels
        self.assertEqual(tuple(c.segment[7].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[7].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[7].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        self.verifyRGNsegment(c.segment[8], RGNsegment(0, 0, 11))
        self.verifySOTsegment(c.segment[9], SOTsegment(0, 33582, 0, 1))
        self.verifyRGNsegment(c.segment[10], RGNsegment(0, 0, 9))

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[11].marker_id, 'SOD')

    def test_NR_p0_07_dump(self):
        jfile = opj_data_file('input/conformance/p0_07.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (2048, 2048), 'xyosiz': (0, 0),
                  'xytsiz': (128, 128), 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12),
                  'signed': (True, True, True),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertTrue(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 8)  # 8
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa, [0] * 10)
        self.assertEqual(c.segment[3].exponent,
                         [14, 15, 15, 16, 15, 15, 16, 15, 15, 16])

        pargs = (RCME_ISO_8859_1, "Kakadu-3.0.7".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[5], SOTsegment(0, 9951, 0, 0))

        # POD: progression order change
        self.assertEqual(c.segment[6].rspod, (0,))
        self.assertEqual(c.segment[6].cspod, (0,))
        self.assertEqual(c.segment[6].lyepod, (9,))
        self.assertEqual(c.segment[6].repod, (3,))
        self.assertEqual(c.segment[6].cdpod, (3,))
        self.assertEqual(c.segment[6].ppod, (glymur.core.LRCP,))

        # PLT: packet length, tile part
        self.assertEqual(c.segment[7].zplt, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[8].marker_id, 'SOD')

    def test_NR_p0_08_dump(self):
        jfile = opj_data_file('input/conformance/p0_08.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (513, 3072), 'xyosiz': (0, 0),
                  'xytsiz': (513, 3072), 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12),
                  'signed': (True, True, True),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertTrue(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.CPRL)
        self.assertEqual(c.segment[2].layers, 30)  # 30
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 7)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 0)
        self.assertEqual(c.segment[3].spcoc[0], 6)  # levels
        self.assertEqual(tuple(c.segment[3].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[3].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[4].ccoc, 1)
        self.assertEqual(c.segment[4].spcoc[0], 7)  # levels
        self.assertEqual(tuple(c.segment[4].code_block_size),
                         (32, 32))  # cblk
        self.verify_codeblock_style(c.segment[4].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[4].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[5].ccoc, 2)
        self.assertEqual(c.segment[5].spcoc[0], 8)  # levels
        self.assertEqual(tuple(c.segment[5].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[5].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[5].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[6].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[6].guard_bits, 4)
        self.assertEqual(c.segment[6].mantissa, [0] * 22)
        self.assertEqual(c.segment[6].exponent,
                         [11, 12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13,
                          12, 12, 13, 12, 12, 13, 12, 12, 13])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[7].cqcc, 0)
        # quantization type
        self.assertEqual(c.segment[7].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[7].guard_bits, 4)
        self.assertEqual(c.segment[7].mantissa, [0] * 19)
        self.assertEqual(c.segment[7].exponent,
                         [11, 12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13,
                             12, 12, 13, 12, 12, 13])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[8].cqcc, 2)
        # quantization type
        self.assertEqual(c.segment[8].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[8].guard_bits, 4)
        self.assertEqual(c.segment[8].mantissa, [0] * 25)
        self.assertEqual(c.segment[8].exponent,
                         [11, 12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13,
                          12, 12, 13, 12, 12, 13, 12, 12, 13, 12, 12, 13])

        pargs = (RCME_ISO_8859_1, "Kakadu-3.0.7".encode())
        self.verifyCMEsegment(c.segment[9], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[10], SOTsegment(0, 3820593, 0, 1))

    def test_NR_p0_09_dump(self):
        jfile = opj_data_file('input/conformance/p0_09.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 0, 'xysiz': (17, 37), 'xyosiz': (0, 0),
                  'xytsiz': (17, 37), 'xytosiz': (0, 0), 'bitdepth': (8,),
                  'signed': (False,), 'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)  # scalar expounded
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa,
                         [1915, 1884, 1884, 1853, 1884, 1884, 1853, 1962, 1962,
                          1986, 53, 53, 120, 26, 26, 1983])
        self.assertEqual(c.segment[3].exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 12, 12, 12,
                          11, 11, 12])

        pargs = (RCME_ISO_8859_1, "Kakadu-3.0.7".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[5], SOTsegment(0, 478, 0, 1))

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[6].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[7].marker_id, 'EOC')

    def test_NR_p0_10_dump(self):
        jfile = opj_data_file('input/conformance/p0_10.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (256, 256), 'xyosiz': (0, 0),
                  'xytsiz': (128, 128), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(4, 4, 4), (4, 4, 4)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # 2
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3].guard_bits, 0)
        self.assertEqual(c.segment[3].mantissa, [0] * 10)
        self.assertEqual(c.segment[3].exponent,
                         [11, 12, 12, 13, 12, 12, 13, 12, 12, 13])

        self.verifySOTsegment(c.segment[4], SOTsegment(0, 2453, 0, 0))

        self.assertEqual(c.segment[5].marker_id, 'SOD')
        self.verifySOTsegment(c.segment[6], SOTsegment(1, 2403, 0, 0))

        self.assertEqual(c.segment[7].marker_id, 'SOD')
        self.verifySOTsegment(c.segment[8], SOTsegment(2, 2420, 0, 0))

        self.assertEqual(c.segment[9].marker_id, 'SOD')
        self.verifySOTsegment(c.segment[10], SOTsegment(3, 2472, 0, 0))

        self.assertEqual(c.segment[11].marker_id, 'SOD')
        self.verifySOTsegment(c.segment[12], SOTsegment(0, 1043, 1, 2))

        self.assertEqual(c.segment[13].marker_id, 'SOD')
        self.verifySOTsegment(c.segment[14], SOTsegment(1, 1101, 1, 2))

        self.assertEqual(c.segment[15].marker_id, 'SOD')
        self.verifySOTsegment(c.segment[16], SOTsegment(3, 1054, 1, 2))

        self.assertEqual(c.segment[17].marker_id, 'SOD')
        self.verifySOTsegment(c.segment[18], SOTsegment(2, 14, 1, 0))

        self.assertEqual(c.segment[19].marker_id, 'SOD')
        self.verifySOTsegment(c.segment[20], SOTsegment(2, 1089, 2, 0))

        self.assertEqual(c.segment[21].marker_id, 'SOD')
        self.assertEqual(c.segment[22].marker_id, 'EOC')

    def test_NR_p0_11_dump(self):
        jfile = opj_data_file('input/conformance/p0_11.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (128, 1), 'xyosiz': (0, 0),
                  'xytsiz': (128, 128), 'xytosiz': (0, 0), 'bitdepth': (8,),
                  'signed': (False,), 'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertTrue(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 0)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, True])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, [(128, 2)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3].guard_bits, 3)
        self.assertEqual(c.segment[3].mantissa, [0])
        self.assertEqual(c.segment[3].exponent, [8])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[5], SOTsegment(0, 118, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[6].marker_id, 'SOD')

        # SOP, EPH
        sop = [x.marker_id for x in c.segment if x.marker_id == 'SOP']
        eph = [x.marker_id for x in c.segment if x.marker_id == 'EPH']
        self.assertEqual(len(sop), 0)
        self.assertEqual(len(eph), 1)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_NR_p0_12_dump(self):
        jfile = opj_data_file('input/conformance/p0_12.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (3, 5), 'xyosiz': (0, 0),
                  'xytsiz': (3, 5), 'xytosiz': (0, 0), 'bitdepth': (8,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 32))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, True, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3].guard_bits, 3)
        self.assertEqual(c.segment[3].mantissa, [0] * 10)
        self.assertEqual(c.segment[3].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[5], SOTsegment(0, 162, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[6].marker_id, 'SOD')

        # SOP, EPH
        sop = [x.marker_id for x in c.segment if x.marker_id == 'SOP']
        eph = [x.marker_id for x in c.segment if x.marker_id == 'EPH']
        self.assertEqual(len(sop), 4)
        self.assertEqual(len(eph), 0)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_NR_p0_13_dump(self):
        jfile = opj_data_file('input/conformance/p0_13.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (1, 1), 'xyosiz': (0, 0),
                  'xytsiz': (1, 1), 'xytosiz': (0, 0),
                  'bitdepth': tuple([8] * 257),
                  'signed': tuple([False] * 257),
                  'xyrsiz': [tuple([1] * 257), tuple([1] * 257)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 1)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, True, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 2)
        self.assertEqual(c.segment[3].spcoc[0], 1)  # levels
        self.assertEqual(tuple(c.segment[3].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[3].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4].guard_bits, 2)
        self.assertEqual(c.segment[4].mantissa, [0] * 4)
        self.assertEqual(c.segment[4].exponent,
                         [8, 9, 9, 10])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].cqcc, 1)
        self.assertEqual(c.segment[5].guard_bits, 3)
        # quantization type
        self.assertEqual(c.segment[5].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[5].exponent, [9, 10, 10, 11])
        self.assertEqual(c.segment[5].mantissa, [0, 0, 0, 0])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].cqcc, 2)
        self.assertEqual(c.segment[6].guard_bits, 2)
        # quantization type
        self.assertEqual(c.segment[6].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[6].exponent, [9, 10, 10, 11])
        self.assertEqual(c.segment[6].mantissa, [0, 0, 0, 0])

        self.verifyRGNsegment(c.segment[7], RGNsegment(3, 0, 11))

        # POD:  progression order change
        self.assertEqual(c.segment[8].rspod, (0, 0))
        self.assertEqual(c.segment[8].cspod, (0, 128))
        self.assertEqual(c.segment[8].lyepod, (1, 1))
        self.assertEqual(c.segment[8].repod, (33, 33))
        self.assertEqual(c.segment[8].cdpod, (128, 257))
        self.assertEqual(c.segment[8].ppod,
                         (glymur.core.RLCP, glymur.core.CPRL))

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[9], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[10], SOTsegment(0, 1537, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[11].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[12].marker_id, 'EOC')

    def test_NR_p0_14_dump(self):
        jfile = opj_data_file('input/conformance/p0_14.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 0, 'xysiz': (49, 49), 'xyosiz': (0, 0),
                  'xytsiz': (49, 49), 'xytosiz': (0, 0), 'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # 1 layer
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [10, 11, 11, 12, 11, 11, 12, 11, 11, 12, 11, 11, 12,
                          11, 11, 12])

        pargs = (RCME_ISO_8859_1, "Kakadu-3.0.7".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[5], SOTsegment(0, 1528, 0, 1))
        self.assertEqual(c.segment[6].marker_id, 'SOD')
        self.assertEqual(c.segment[7].marker_id, 'EOC')

    def test_NR_p0_15_dump(self):
        jfile = opj_data_file('input/conformance/p0_15.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 1, 'xysiz': (256, 256), 'xyosiz': (0, 0),
                  'xytsiz': (128, 128), 'xytosiz': (0, 0), 'bitdepth': (4,),
                  'signed': (True,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 8)  # layers = 8
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 1)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 1)  # derived
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0])
        self.assertEqual(c.segment[3].exponent, [0])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].cqcc, 0)
        self.assertEqual(c.segment[4].guard_bits, 2)
        # quantization type
        self.assertEqual(c.segment[4].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[4].mantissa, [0] * 4)
        self.assertEqual(c.segment[4].exponent, [4, 5, 5, 6])

        # POD: progression order change
        self.assertEqual(c.segment[5].rspod, (0,))
        self.assertEqual(c.segment[5].cspod, (0,))
        self.assertEqual(c.segment[5].lyepod, (8,))
        self.assertEqual(c.segment[5].repod, (33,))
        self.assertEqual(c.segment[5].cdpod, (255,))
        self.assertEqual(c.segment[5].ppod, (glymur.core.LRCP,))

        # CRG:  component registration
        self.assertEqual(c.segment[6].xcrg, (65424,))
        self.assertEqual(c.segment[6].ycrg, (32558,))

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[7], CMEsegment(*pargs))

        pargs = (RCME_ISO_8859_1, comment1.encode())
        self.verifyCMEsegment(c.segment[8], CMEsegment(*pargs))

        pargs = (RCME_BINARY, c.segment[9].ccme)
        self.verifyCMEsegment(c.segment[9], CMEsegment(*pargs))

        # TLM: tile-part length
        self.assertEqual(c.segment[10].ztlm, 0)
        self.assertEqual(c.segment[10].ttlm, (0, 1, 2, 3))
        self.assertEqual(c.segment[10].ptlm, (4267, 2117, 4080, 2081))

        self.verifySOTsegment(c.segment[11], SOTsegment(0, 4267, 0, 1))

        self.verifyRGNsegment(c.segment[12], RGNsegment(0, 0, 7))

        # SOD:  start of data
        self.assertEqual(c.segment[13].marker_id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        self.verifySOTsegment(c.segment[31], SOTsegment(1, 2117, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[32].marker_id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        self.verifySOTsegment(c.segment[49], SOTsegment(2, 4080, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[50].marker_id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        self.verifySOTsegment(c.segment[67], SOTsegment(3, 2081, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[68].marker_id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        # EOC:  end of codestream
        self.assertEqual(c.segment[85].marker_id, 'EOC')

    def test_NR_p0_16_dump(self):
        jfile = opj_data_file('input/conformance/p0_16.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 0, 'xysiz': (128, 128), 'xyosiz': (0, 0),
                  'xytsiz': (128, 128), 'xytosiz': (0, 0), 'bitdepth': (8,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 10)
        self.assertEqual(c.segment[3].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])

        self.verifySOTsegment(c.segment[4], SOTsegment(0, 7331, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[5].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[6].marker_id, 'EOC')

    def test_NR_p1_01_dump(self):
        jfile = opj_data_file('input/conformance/p1_01.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 2, 'xysiz': (127, 227), 'xyosiz': (5, 128),
                  'xytsiz': (127, 126), 'xytosiz': (1, 101), 'bitdepth': (8,),
                  'signed': (False,),
                  'xyrsiz': [(2,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # SOP
        self.assertTrue(c.segment[2].scod & 4)  # EPH
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 5)  # layers = 5
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, True, False, True, True])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 0)
        self.assertEqual(c.segment[3].spcoc[0], 3)  # level
        self.assertEqual(tuple(c.segment[3].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[3].spcoc[3],
                                    [False, False, True, False, True, True])
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4].guard_bits, 3)
        self.assertEqual(c.segment[4].mantissa, [0] * 10)
        self.assertEqual(c.segment[4].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[5], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[6], SOTsegment(0, 4627, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[7].marker_id, 'SOD')

        # SOP, EPH
        sop = [x.marker_id for x in c.segment if x.marker_id == 'SOP']
        eph = [x.marker_id for x in c.segment if x.marker_id == 'EPH']
        self.assertEqual(len(sop), 20)
        self.assertEqual(len(eph), 20)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_NR_p1_02_dump(self):
        jfile = opj_data_file('input/conformance/p1_02.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 2, 'xysiz': (640, 480), 'xyosiz': (0, 0),
                  'xytsiz': (640, 480), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 19)  # layers = 19
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, True, False, True, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size,
                         [(128, 128), (256, 256), (512, 512), (1024, 1024),
                          (2048, 2048), (4096, 4096), (8192, 8192)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[3].guard_bits, 3)
        self.assertEqual(c.segment[3].mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                          1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                          1888])
        self.assertEqual(c.segment[3].exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13,
                          11, 11, 11, 11, 11, 11])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[4].cqcc, 1)
        self.assertEqual(c.segment[4].guard_bits, 3)
        # quantization type
        self.assertEqual(c.segment[4].sqcc & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[4].mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                          1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                          1888])
        self.assertEqual(c.segment[4].exponent,
                         [14, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11,
                          9, 9, 9, 9, 9, 9])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].cqcc, 2)
        self.assertEqual(c.segment[5].guard_bits, 3)
        # quantization type
        self.assertEqual(c.segment[5].sqcc & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[5].mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                          1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                          1888])
        self.assertEqual(c.segment[5].exponent,
                         [14, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11,
                          9, 9, 9, 9, 9, 9])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[6], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[7], SOTsegment(0, 262838, 0, 1))

        # PPT:  packed packet headers, tile-part header
        self.assertEqual(c.segment[8].marker_id, 'PPT')
        self.assertEqual(c.segment[8].zppt, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[9].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[10].marker_id, 'EOC')

    def test_NR_p1_03_dump(self):
        jfile = opj_data_file('input/conformance/p1_03.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 2, 'xysiz': (1024, 1024), 'xyosiz': (0, 0),
                  'xytsiz': (1024, 1024), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8, 8),
                  'signed': (False, False, False, False),
                  'xyrsiz': [(1, 1, 2, 2), (1, 1, 2, 2)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 10)  # layers = 10
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [True, False, True, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 1)
        self.assertEqual(c.segment[3].spcoc[0], 3)  # level
        self.assertEqual(tuple(c.segment[3].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[3].spcoc[3],
                                    [True, False, True, False, False, False])
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[4].ccoc, 3)
        self.assertEqual(c.segment[4].spcoc[0], 6)  # level
        self.assertEqual(tuple(c.segment[4].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[4].spcoc[3],
                                    [True, False, True, False, False, False])
        self.assertEqual(c.segment[4].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[5].sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[5].guard_bits, 3)
        self.assertEqual(c.segment[5].mantissa,
                         [1814, 1815, 1815, 1817, 1821, 1821, 1827, 1845, 1845,
                             1868, 1925, 1925, 2007, 32, 32, 131, 2002, 2002,
                             1888])
        self.assertEqual(c.segment[5].exponent,
                         [16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13,
                             11, 11, 11, 11, 11, 11])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].cqcc, 0)
        self.assertEqual(c.segment[6].guard_bits, 3)
        # quantization type
        self.assertEqual(c.segment[6].sqcc & 0x1f, 1)  # derived
        self.assertEqual(c.segment[6].mantissa, [0])
        self.assertEqual(c.segment[6].exponent, [14])

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[7].cqcc, 3)
        self.assertEqual(c.segment[7].guard_bits, 3)
        # quantization type
        self.assertEqual(c.segment[7].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[7].mantissa, [0] * 19)
        self.assertEqual(c.segment[7].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10,
                          9, 9, 10])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[8], CMEsegment(*pargs))

        # PPM:  packed packet headers, main header
        self.assertEqual(c.segment[9].marker_id, 'PPM')
        self.assertEqual(c.segment[9].zppm, 0)

        # TLM (tile-part length)
        self.assertEqual(c.segment[10].ztlm, 0)
        self.assertEqual(c.segment[10].ttlm, (0,))
        self.assertEqual(c.segment[10].ptlm, (1366780,))

        self.verifySOTsegment(c.segment[11], SOTsegment(0, 1366780, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[12].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[13].marker_id, 'EOC')

    def test_NR_p1_04_dump(self):
        jfile = opj_data_file('input/conformance/p1_04.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 2, 'xysiz': (1024, 1024), 'xyosiz': (0, 0),
                  'xytsiz': (128, 128), 'xytosiz': (0, 0), 'bitdepth': (12,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa,
                         [84, 423, 408, 435, 450, 435, 470, 549, 520, 618])
        self.assertEqual(c.segment[3].exponent,
                         [8, 10, 10, 10, 9, 9, 9, 8, 8, 8])

        # TLM (tile-part length)
        self.assertEqual(c.segment[4].ztlm, 0)
        self.assertIsNone(c.segment[4].ttlm)
        self.assertEqual(c.segment[4].ptlm,
                         (350, 356, 402, 245, 402, 564, 675, 283, 317, 299,
                          330, 333, 346, 403, 839, 667, 328, 349, 274, 325,
                          501, 561, 756, 710, 779, 620, 628, 675, 600, 66195,
                          721, 719, 565, 565, 546, 586, 574, 641, 713, 634,
                          573, 528, 544, 597, 771, 665, 624, 706, 568, 537,
                          554, 546, 542, 635, 826, 667, 617, 606, 813, 586,
                          641, 654, 669, 623))

        pargs = (RCME_ISO_8859_1, "Created by Aware, Inc.".encode())
        self.verifyCMEsegment(c.segment[5], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[6], SOTsegment(0, 350, 0, 1))

        # SOD:  start of data
        self.assertEqual(c.segment[7].marker_id, 'SOD')

        self.verifySOTsegment(c.segment[8], SOTsegment(1, 356, 0, 1))

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[9].sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[9].guard_bits, 2)
        self.assertEqual(c.segment[9].mantissa,
                         [75, 1093, 1098, 1115, 1157, 1134, 1186, 1217, 1245,
                          1248])
        self.assertEqual(c.segment[9].exponent,
                         [8, 10, 10, 10, 9, 9, 9, 8, 8, 8])

        # SOD:  start of data
        self.assertEqual(c.segment[10].marker_id, 'SOD')

        self.verifySOTsegment(c.segment[11], SOTsegment(2, 402, 0, 1))

        # and so on

        # There should be 64 SOD, SOT, QCD segments.
        ids = [x.marker_id for x in c.segment if x.marker_id == 'SOT']
        self.assertEqual(len(ids), 64)
        ids = [x.marker_id for x in c.segment if x.marker_id == 'SOD']
        self.assertEqual(len(ids), 64)
        ids = [x.marker_id for x in c.segment if x.marker_id == 'QCD']
        self.assertEqual(len(ids), 64)

        # Tiles should be in order, right?
        tiles = [x.isot for x in c.segment if x.marker_id == 'SOT']
        self.assertEqual(tiles, list(range(64)))

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_NR_p1_05_dump(self):
        jfile = opj_data_file('input/conformance/p1_05.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 2, 'xysiz': (529, 524), 'xyosiz': (17, 12),
                  'xytsiz': (37, 37), 'xytosiz': (8, 2),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # sop
        self.assertTrue(c.segment[2].scod & 4)  # eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 2)  # levels = 2
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 7)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 8))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [True, False, False, True, True, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, [(16, 16)] * 8)

        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[3].guard_bits, 3)
        self.assertEqual(c.segment[3].mantissa,
                         [1813, 1814, 1814, 1814, 1815, 1815, 1817, 1821,
                          1821, 1827, 1845, 1845, 1868, 1925, 1925, 2007,
                          32, 32, 131, 2002, 2002, 1888])
        self.assertEqual(c.segment[3].exponent,
                         [17, 17, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14,
                          14, 13, 13, 13, 11, 11, 11, 11, 11, 11])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

        # 225 consecutive PPM segments.
        zppm = [x.zppm for x in c.segment[5:230]]
        self.assertEqual(zppm, list(range(225)))

        self.verifySOTsegment(c.segment[230], SOTsegment(0, 580, 0, 1))

        # 225 total SOT segments
        isot = [x.isot for x in c.segment if x.marker_id == 'SOT']
        self.assertEqual(isot, list(range(225)))

        # scads of SOP, EPH segments
        sop = [x.marker_id for x in c.segment if x.marker_id == 'SOP']
        eph = [x.marker_id for x in c.segment if x.marker_id == 'EPH']
        self.assertEqual(len(sop), 26472)
        self.assertEqual(len(eph), 0)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_NR_p1_06_dump(self):
        jfile = opj_data_file('input/conformance/p1_06.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 2, 'xysiz': (12, 12), 'xyosiz': (0, 0),
                  'xytsiz': (3, 3), 'xytosiz': (0, 0), 'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # sop
        self.assertTrue(c.segment[2].scod & 4)  # eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 4)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, True, False, True])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)  # expounded
        self.assertEqual(c.segment[3].guard_bits, 3)
        self.assertEqual(c.segment[3].mantissa,
                         [1821, 1845, 1845, 1868, 1925, 1925, 2007, 32,
                          32, 131, 2002, 2002, 1888])
        self.assertEqual(c.segment[3].exponent,
                         [14, 14, 14, 14, 13, 13, 13, 11, 11, 11,
                          11, 11, 11])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[5], SOTsegment(0, 349, 0, 1))

        # PPT:  packed packet headers, tile-part header
        self.assertEqual(c.segment[6].marker_id, 'PPT')
        self.assertEqual(c.segment[6].zppt, 0)

        # scads of SOP, EPH segments

        # 16 SOD segments
        sods = [x for x in c.segment if x.marker_id == 'SOD']
        self.assertEqual(len(sods), 16)

        # 16 PPT segments
        ppts = [x for x in c.segment if x.marker_id == 'PPT']
        self.assertEqual(len(ppts), 16)

        # 16 SOT segments
        isots = [x.isot for x in c.segment if x.marker_id == 'SOT']
        self.assertEqual(isots, list(range(16)))

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_NR_p1_07_dump(self):
        jfile = opj_data_file('input/conformance/p1_07.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        kwargs = {'rsiz': 2, 'xysiz': (12, 12), 'xyosiz': (4, 0),
                  'xytsiz': (12, 12), 'xytosiz': (4, 0), 'bitdepth': (8, 8),
                  'signed': (False, False),
                  'xyrsiz': [(4, 1), (1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # sop
        self.assertTrue(c.segment[2].scod & 4)  # eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RPCL)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 1)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, [(1, 1), (2, 2)])

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 1)
        self.assertEqual(c.segment[3].spcoc[0], 1)  # level
        self.assertEqual(tuple(c.segment[3].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[3].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[3].precinct_size, [(2, 2), (4, 4)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4].guard_bits, 2)
        self.assertEqual(c.segment[4].mantissa, [0] * 4)
        self.assertEqual(c.segment[4].exponent, [8, 9, 9, 10])

        pargs = (RCME_ISO_8859_1,
                 "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode())
        self.verifyCMEsegment(c.segment[5], CMEsegment(*pargs))

        self.verifySOTsegment(c.segment[6], SOTsegment(0, 434, 0, 1))

        # scads of SOP, EPH segments

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_NR_00042_j2k_dump(self):
        # Profile 3.
        jfile = opj_data_file('input/nonregression/_00042.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        kwargs = {'rsiz': 3, 'xysiz': (1920, 1080), 'xyosiz': (0, 0),
                  'xytsiz': (1920, 1080), 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.CPRL)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size[0], (128, 128))
        self.assertEqual(c.segment[2].precinct_size[1:], [(256, 256)] * 5)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa,
                         [1824, 1776, 1776, 1728, 1792, 1792, 1760, 1872,
                          1872, 1896, 5, 5, 71, 2003, 2003, 1890])
        self.assertEqual(c.segment[3].exponent,
                         [18, 18, 18, 18, 17, 17, 17, 16,
                          16, 16, 14, 14, 14, 14, 14, 14])

        # COC: Coding style component
        self.assertEqual(c.segment[4].ccoc, 1)
        self.assertEqual(c.segment[4].spcoc[0], 5)  # level
        self.assertEqual(tuple(c.segment[4].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[4].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[4].spcoc[4],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].cqcc, 1)
        self.assertEqual(c.segment[5].guard_bits, 2)
        # quantization type
        self.assertEqual(c.segment[5].sqcc & 0x1f, 2)
        self.assertEqual(c.segment[5].mantissa,
                         [1824, 1776, 1776, 1728, 1792, 1792, 1760, 1872,
                          1872, 1896, 5, 5, 71, 2003, 2003, 1890])
        self.assertEqual(c.segment[5].exponent,
                         [18, 18, 18, 18, 17, 17, 17, 16, 16, 16, 14, 14, 14,
                          14, 14, 14])

        # COC: Coding style component
        self.assertEqual(c.segment[6].ccoc, 2)
        self.assertEqual(c.segment[6].spcoc[0], 5)  # level
        self.assertEqual(tuple(c.segment[6].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[6].spcoc[3],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[6].spcoc[4],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[7].cqcc, 2)
        self.assertEqual(c.segment[7].guard_bits, 2)
        # quantization type
        self.assertEqual(c.segment[7].sqcc & 0x1f, 2)  # none
        self.assertEqual(c.segment[7].mantissa,
                         [1824, 1776, 1776, 1728, 1792, 1792, 1760, 1872,
                          1872, 1896, 5, 5, 71, 2003, 2003, 1890])
        self.assertEqual(c.segment[7].exponent,
                         [18, 18, 18, 18, 17, 17, 17, 16, 16, 16, 14, 14,
                          14, 14, 14, 14])

        pargs = (RCME_ISO_8859_1, "Created by OpenJPEG version 1.3.0".encode())
        self.verifyCMEsegment(c.segment[8], CMEsegment(*pargs))

        # TLM (tile-part length)
        self.assertEqual(c.segment[9].ztlm, 0)
        self.assertEqual(c.segment[9].ttlm, (0, 0, 0))
        self.assertEqual(c.segment[9].ptlm, (45274, 20838, 8909))

        # 3 tiles, one for each component
        idx = [x.isot for x in c.segment if x.marker_id == 'SOT']
        self.assertEqual(idx, [0, 0, 0])
        lens = [x.psot for x in c.segment if x.marker_id == 'SOT']
        self.assertEqual(lens, [45274, 20838, 8909])
        tpsot = [x.tpsot for x in c.segment if x.marker_id == 'SOT']
        self.assertEqual(tpsot, [0, 1, 2])

        sods = [x for x in c.segment if x.marker_id == 'SOD']
        self.assertEqual(len(sods), 3)

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_Bretagne2_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/Bretagne2.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        kwargs = {'rsiz': 0, 'xysiz': (2592, 1944), 'xyosiz': (0, 0),
                  'xytsiz': (640, 480), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size,
                         [(16, 16), (32, 32), (64, 64), (128, 128),
                          (128, 128), (128, 128)])

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        expected += ['SOT', 'COC', 'QCC', 'COC', 'QCC', 'SOD'] * 25
        expected += ['EOC']
        self.assertEqual(ids, expected)

    def test_NR_buxI_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/buxI.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        kwargs = {'rsiz': 0, 'xysiz': (512, 512), 'xyosiz': (0, 0),
                  'xytsiz': (512, 512), 'xytosiz': (0, 0), 'bitdepth': (16,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'SOT', 'SOD', 'EOC']
        self.assertEqual(ids, expected)

    def test_NR_buxR_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/buxR.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        kwargs = {'rsiz': 0, 'xysiz': (512, 512), 'xyosiz': (0, 0),
                  'xytsiz': (512, 512), 'xytosiz': (0, 0), 'bitdepth': (16,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'SOT', 'SOD', 'EOC']
        self.assertEqual(ids, expected)

    def test_NR_Cannotreaddatawithnosizeknown_j2k(self):
        lst = ['input', 'nonregression',
               'Cannotreaddatawithnosizeknown.j2k']
        path = '/'.join(lst)

        jfile = opj_data_file(path)
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (1420, 1416), 'xyosiz': (0, 0),
                  'xytsiz': (1420, 1416), 'xytosiz': (0, 0), 'bitdepth': (16,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 4)
        self.assertEqual(c.segment[3].mantissa, [0] * 34)
        self.assertEqual(c.segment[3].exponent, [16] + [17, 17, 18] * 11)

    def test_NR_CT_Phillips_JPEG2K_Decompr_Problem_dump(self):
        jfile = opj_data_file('input/nonregression/'
                              + 'CT_Phillips_JPEG2K_Decompr_Problem.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (512, 614), 'xyosiz': (0, 0),
                  'xytsiz': (512, 614), 'xytosiz': (0, 0), 'bitdepth': (12,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa,
                         [442, 422, 422, 403, 422, 422, 403, 472, 472, 487,
                          591, 591, 676, 558, 558, 485])
        self.assertEqual(c.segment[3].exponent,
                         [22, 22, 22, 22, 21, 21, 21, 20, 20, 20, 19, 19, 19,
                          18, 18, 18])

        pargs = (RCME_ISO_8859_1, "Kakadu-3.2".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

    def test_NR_cthead1_dump(self):
        jfile = opj_data_file('input/nonregression/cthead1.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'CME']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (256, 256), 'xyosiz': (0, 0),
                  'xytsiz': (256, 256), 'xytosiz': (0, 0), 'bitdepth': (8,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [9, 10, 10, 11, 10, 10, 11, 10, 10, 11, 10, 10, 10,
                          9, 9, 10])

        pargs = (RCME_ISO_8859_1, "Kakadu-v6.3.1".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

    def test_NR_illegalcolortransform_dump(self):
        jfile = opj_data_file('input/nonregression/illegalcolortransform.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (1420, 1416), 'xyosiz': (0, 0),
                  'xytsiz': (1420, 1416), 'xytosiz': (0, 0), 'bitdepth': (16,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 4)
        self.assertEqual(c.segment[3].mantissa, [0] * 34)
        self.assertEqual(c.segment[3].exponent, [16] + [17, 17, 18] * 11)

    def test_NR_j2k32_dump(self):
        jfile = opj_data_file('input/nonregression/j2k32.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (256, 256), 'xyosiz': (0, 0),
                  'xytsiz': (256, 256), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (True, True, True),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent, [8, 9, 9, 10, 9, 9, 10, 9, 9,
                         10, 9, 9, 10, 9, 9, 10])

        pargs = (RCME_BINARY, c.segment[4].ccme)
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

    def test_NR_kakadu_v4_4_openjpegv2_broken_dump(self):
        jfile = opj_data_file('input/nonregression/'
                              + 'kakadu_v4-4_openjpegv2_broken.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        kwargs = {'rsiz': 0, 'xysiz': (2048, 2500), 'xyosiz': (0, 0),
                  'xytsiz': (2048, 2500), 'xytosiz': (0, 0), 'bitdepth': (16,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 12)  # layers = 12
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 8)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa, [0] * 25)
        self.assertEqual(c.segment[3].exponent,
                         [17, 18, 18, 19, 18, 18, 19, 18, 18, 19, 18, 18, 19,
                          18, 18, 19, 18, 18, 19, 18, 18, 19, 18, 18, 19])

        pargs = (RCME_ISO_8859_1, "Kakadu-v4.4".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

        ccme = "Kdu-Layer-Info: log_2{Delta-D(MSE)/[2^16*Delta-L(bytes)]},"
        ccme += " L(bytes)\n"
        ccme += " -65.4, 6.8e+004\n"
        ccme += " -66.3, 1.0e+005\n"
        ccme += " -67.3, 2.0e+005\n"
        ccme += " -68.5, 4.1e+005\n"
        ccme += " -69.0, 5.1e+005\n"
        ccme += " -69.5, 5.9e+005\n"
        ccme += " -69.7, 6.8e+005\n"
        ccme += " -70.3, 8.2e+005\n"
        ccme += " -70.8, 1.0e+006\n"
        ccme += " -71.9, 1.4e+006\n"
        ccme += " -73.8, 2.0e+006\n"
        ccme += "-256.0, 3.7e+006\n"
        pargs = (RCME_ISO_8859_1, ccme.encode())
        self.verifyCMEsegment(c.segment[5], CMEsegment(*pargs))

    def test_NR_MarkerIsNotCompliant_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/MarkerIsNotCompliant.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        kwargs = {'rsiz': 0, 'xysiz': (1420, 1416), 'xyosiz': (0, 0),
                  'xytsiz': (1420, 1416), 'xytosiz': (0, 0), 'bitdepth': (16,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 4)
        self.assertEqual(c.segment[3].mantissa, [0] * 34)
        self.assertEqual(c.segment[3].exponent,
                         [16, 17, 17, 18, 17, 17, 18, 17, 17, 18, 17, 17, 18,
                          17, 17, 18, 17, 17, 18, 17, 17, 18, 17, 17, 18, 17,
                          17, 18, 17, 17, 18, 17, 17, 18])

    def test_NR_movie_00000(self):
        jfile = opj_data_file('input/nonregression/movie_00000.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        kwargs = {'rsiz': 0, 'xysiz': (1920, 1080), 'xyosiz': (0, 0),
                  'xytsiz': (1920, 1080), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_movie_00001(self):
        jfile = opj_data_file('input/nonregression/movie_00001.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        kwargs = {'rsiz': 0, 'xysiz': (1920, 1080), 'xyosiz': (0, 0),
                  'xytsiz': (1920, 1080), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_movie_00002(self):
        jfile = opj_data_file('input/nonregression/movie_00002.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        kwargs = {'rsiz': 0, 'xysiz': (1920, 1080), 'xyosiz': (0, 0),
                  'xytsiz': (1920, 1080), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_orb_blue10_lin_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/orb-blue10-lin-j2k.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (117, 117), 'xyosiz': (0, 0),
                  'xytsiz': (117, 117), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8, 8),
                  'signed': (False, False, False, False),
                  'xyrsiz': [(1, 1, 1, 1), (1, 1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_orb_blue10_win_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/orb-blue10-win-j2k.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (117, 117), 'xyosiz': (0, 0),
                  'xytsiz': (117, 117), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8, 8),
                  'signed': (False, False, False, False),
                  'xyrsiz': [(1, 1, 1, 1), (1, 1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_pacs_ge_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/pacs.ge.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (512, 512), 'xyosiz': (0, 0),
                  'xytsiz': (512, 512), 'xytosiz': (0, 0),
                  'bitdepth': (16,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 16)  # layers = 16
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [18, 19, 19, 20, 19, 19, 20, 19, 19, 20, 19, 19, 20,
                          19, 19, 20])

        pargs = (RCME_ISO_8859_1, "Kakadu-2.0.2".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

    def test_NR_test_lossless_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/test_lossless.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (1024, 1024), 'xyosiz': (0, 0),
                  'xytsiz': (1024, 1024), 'xytosiz': (0, 0),
                  'bitdepth': (12,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [12, 13, 13, 14, 13, 13, 14, 13, 13, 14, 13, 13, 14,
                          13, 13, 14])

        pargs = (RCME_ISO_8859_1, "ClearCanvas DICOM OpenJPEG".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

    def test_NR_123_j2c_dump(self):
        jfile = opj_data_file('input/nonregression/123.j2c')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (1800, 1800), 'xyosiz': (0, 0),
                  'xytsiz': (1800, 1800), 'xytosiz': (0, 0),
                  'bitdepth': (16,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 4)
        self.assertEqual(c.segment[3].mantissa, [0] * 34)
        self.assertEqual(c.segment[3].exponent,
                         [16] + [17, 17, 18] * 11)

    def test_NR_bug_j2c_dump(self):
        jfile = opj_data_file('input/nonregression/bug.j2c')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (1800, 1800), 'xyosiz': (0, 0),
                  'xytsiz': (1800, 1800), 'xytosiz': (0, 0),
                  'bitdepth': (16,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 4)
        self.assertEqual(c.segment[3].mantissa, [0] * 34)
        self.assertEqual(c.segment[3].exponent,
                         [16] + [17, 17, 18] * 11)

    def test_NR_kodak_2layers_lrcp_j2c_dump(self):
        jfile = opj_data_file('input/nonregression/kodak_2layers_lrcp.j2c')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (2048, 1556), 'xyosiz': (0, 0),
                  'xytsiz': (2048, 1556), 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size,
                         [(128, 128)] + [(256, 256)] * 5)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [13, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
                          13, 13, 13])

        pargs = (RCME_ISO_8859_1, "DCP-Werkstatt".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

    def test_NR_issue104_jpxstream_dump(self):
        jfile = opj_data_file('input/nonregression/issue104_jpxstream.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'pclr', 'cmap'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1],
                                 FileTypeBox(compatibility_list=['jp2 ',
                                                                 'jpxb',
                                                                 'jpx ']))

        # Reader requirements talk.
        # unrestricted jpeg 2000 part 1
        self.assertTrue(5 in jp2.box[2].standard_flag)

        ihdr = glymur.jp2box.ImageHeaderBox(203, 479, colorspace_unknown=True)
        self.verifyImageHeaderBox(jp2.box[3].box[0], ihdr)

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=SRGB,
                                                    approximation=1,
                                                    precedence=2)
        self.verifyColourSpecificationBox(jp2.box[3].box[1], colr)

        # Jp2 Header
        # Palette box.
        self.assertEqual(jp2.box[3].box[2].palette.shape, (256, 3))

        # Jp2 Header
        # Component mapping box
        self.assertEqual(jp2.box[3].box[3].component_index, (0, 0, 0))
        self.assertEqual(jp2.box[3].box[3].mapping_type, (1, 1, 1))
        self.assertEqual(jp2.box[3].box[3].palette_index, (0, 1, 2))

        c = jp2.box[4].codestream

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (479, 203), 'xyosiz': (0, 0),
                  'xytsiz': (256, 203), 'xytosiz': (0, 0),
                  'bitdepth': (8,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent, [8] + [9, 9, 10] * 5)

    def test_NR_issue206_image_000_dump(self):
        jfile = opj_data_file('input/nonregression/issue206_image-000.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1],
                                 FileTypeBox(compatibility_list=['jp2 ',
                                                                 'jpxb',
                                                                 'jpx ']))

        # Reader requirements talk.
        # unrestricted jpeg 2000 part 1
        self.assertTrue(5 in jp2.box[2].standard_flag)

        ihdr = glymur.jp2box.ImageHeaderBox(326, 431,
                                            num_components=3,
                                            colorspace_unknown=True)
        self.verifyImageHeaderBox(jp2.box[3].box[0], ihdr)

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=SRGB,
                                                    approximation=1,
                                                    precedence=2)
        self.verifyColourSpecificationBox(jp2.box[3].box[1], colr)

        c = jp2.box[4].codestream

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (431, 326), 'xyosiz': (0, 0),
                  'xytsiz': (256, 256), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent, [8] + [9, 9, 10] * 5)

    def test_NR_Marrin_jp2_dump(self):
        jfile = opj_data_file('input/nonregression/Marrin.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'cdef', 'res '])

        ids = [box.box_id for box in jp2.box[2].box[3].box]
        self.assertEqual(ids, ['resd'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(135, 135, num_components=2,
                                            colorspace_unknown=True)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=GREYSCALE)
        self.verifyColourSpecificationBox(jp2.box[2].box[1], colr)

        # Jp2 Header
        # Channel Definition
        self.assertEqual(jp2.box[2].box[2].index, (0, 1))
        self.assertEqual(jp2.box[2].box[2].channel_type, (0, 1))   # opacity
        self.assertEqual(jp2.box[2].box[2].association, (0, 0))  # both main

        c = jp2.box[3].codestream

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (135, 135), 'xyosiz': (0, 0),
                  'xytsiz': (135, 135), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8),
                  'signed': (False, False),
                  'xyrsiz': [(1, 1), (1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa,
                         [1822, 1770, 1770, 1724, 1792, 1792, 1762, 1868, 1868,
                          1892, 3, 3, 69, 2002, 2002, 1889])
        self.assertEqual(c.segment[3].exponent,
                         [14] * 4 + [13] * 3 + [12] * 3 + [10] * 6)

        pargs = (RCME_ISO_8859_1, "Kakadu-v5.2.1".encode())
        self.verifyCMEsegment(c.segment[4], CMEsegment(*pargs))

    def test_NR_mem_b2b86b74_2753_dump(self):
        jfile = opj_data_file('input/nonregression/mem-b2b86b74-2753.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'pclr', 'cmap'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1],
                                 FileTypeBox(compatibility_list=['jp2 ',
                                                                 'jpxb',
                                                                 'jpx ']))

        # Reader requirements talk.
        # unrestricted jpeg 2000 part 1
        self.assertTrue(5 in jp2.box[2].standard_flag)

        ihdr = glymur.jp2box.ImageHeaderBox(46, 124, bits_per_component=4,
                                            colorspace_unknown=True)
        self.verifyImageHeaderBox(jp2.box[3].box[0], ihdr)

        method = ENUMERATED_COLORSPACE
        colr = glymur.jp2box.ColourSpecificationBox(colorspace=SRGB,
                                                    method=method,
                                                    approximation=1,
                                                    precedence=2)
        self.verifyColourSpecificationBox(jp2.box[3].box[1], colr)

        # Jp2 Header
        # Palette box.
        # 3 columns with 16 entries.
        self.assertEqual(jp2.box[3].box[2].palette.shape, (16, 3))

        # Jp2 Header
        # Component mapping box
        self.assertEqual(jp2.box[3].box[3].component_index, (0, 0, 0))
        self.assertEqual(jp2.box[3].box[3].mapping_type, (1, 1, 1))
        self.assertEqual(jp2.box[3].box[3].palette_index, (0, 1, 2))

        c = jp2.box[4].codestream

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (124, 46), 'xyosiz': (0, 0),
                  'xytsiz': (124, 46), 'xytosiz': (0, 0),
                  'bitdepth': (4,),
                  'signed': (False,),
                  'xyrsiz': [(1,), (1,)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 32))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent, [4] + [5, 5, 6] * 5)

    def test_NR_merged_dump(self):
        jfile = opj_data_file('input/nonregression/merged.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(576, 766, num_components=3)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=glymur.core.YCC)
        self.verifyColourSpecificationBox(jp2.box[2].box[1], colr)

        c = jp2.box[3].codestream

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'POD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (766, 576), 'xyosiz': (0, 0),
                  'xytsiz': (766, 576), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 2, 2), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 128))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 1)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent, [8] + [9, 9, 10] * 5)

        # POD: progression order change
        self.assertEqual(c.segment[4].rspod, (0, 0))
        self.assertEqual(c.segment[4].cspod, (0, 1))
        self.assertEqual(c.segment[4].lyepod, (1, 1))
        self.assertEqual(c.segment[4].repod, (6, 6))
        self.assertEqual(c.segment[4].cdpod, (1, 3))

        podvals = (glymur.core.LRCP, glymur.core.LRCP)
        self.assertEqual(c.segment[4].ppod, podvals)


@unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuiteWarns(MetadataBase):

    @unittest.skipIf(re.match("1.5|2.0.0", glymur.version.openjpeg_version),
                     "Test not passing on 1.5, 2.0:  not introduced until 2.x")
    def test_NR_DEC_issue188_beach_64bitsbox_jp2_41_decode(self):
        """
        Has an 'XML ' box instead of 'xml '.  Just verify we can read it.
        """
        relpath = 'input/nonregression/issue188_beach_64bitsbox.jp2'
        jfile = opj_data_file(relpath)
        with self.assertWarns(UserWarning):
            Jp2k(jfile)[:]
        self.assertTrue(True)

    def test_NR_broken4_jp2_dump(self):
        jfile = opj_data_file('input/nonregression/broken4.jp2')
        with warnings.catch_warnings():
            # Suppress a warning, all we really care is parsing the entire
            # file.
            warnings.simplefilter("ignore")
            with self.assertWarns(UserWarning):
                jp2 = Jp2k(jfile)
                self.assertEqual(jp2.box[-1].codestream.segment[-1].marker_id,
                                 'QCC')

    def test_NR_broken2_jp2_dump(self):
        """
        Invalid marker ID in the codestream.
        """
        jfile = opj_data_file('input/nonregression/broken2.jp2')
        with warnings.catch_warnings():
            # Suppress a warning, all we really care is parsing the entire
            # file.
            warnings.simplefilter("ignore")
            with self.assertWarns(UserWarning):
                # Invalid marker ID on codestream.
                jp2 = Jp2k(jfile)
                self.assertEqual(jp2.box[-1].codestream.segment[-1].marker_id,
                                 'QCC')

    def test_NR_file1_dump(self):
        jfile = opj_data_file('input/conformance/file1.jp2')
        with self.assertWarns(UserWarning):
            # Bad compatibility list item.
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'xml ', 'jp2h', 'xml ',
                               'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        # XML box
        tags = [x.tag for x in jp2.box[2].xml.getroot()]
        self.assertEqual(tags,
                         ['{http://www.jpeg.org/jpx/1.0/xml}'
                          + 'GENERAL_CREATION_INFO'])

        ihdr = glymur.jp2box.ImageHeaderBox(512, 768, num_components=3)
        self.verifyImageHeaderBox(jp2.box[3].box[0], ihdr)

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=SRGB,
                                                    approximation=1)
        self.verifyColourSpecificationBox(jp2.box[3].box[1], colr)

        # XML box
        tags = [x.tag for x in jp2.box[4].xml.getroot()]
        self.assertEqual(tags, ['{http://www.jpeg.org/jpx/1.0/xml}CAPTION',
                                '{http://www.jpeg.org/jpx/1.0/xml}LOCATION',
                                '{http://www.jpeg.org/jpx/1.0/xml}EVENT'])

    def test_NR_file2_dump(self):
        jfile = opj_data_file('input/conformance/file2.jp2')
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'cdef'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(640, 480, num_components=3)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=glymur.core.YCC,
                                                    approximation=1)
        self.verifyColourSpecificationBox(jp2.box[2].box[1], colr)

        # Jp2 Header
        # Channel Definition
        self.assertEqual(jp2.box[2].box[2].index, (0, 1, 2))
        self.assertEqual(jp2.box[2].box[2].channel_type, (0, 0, 0))  # color
        self.assertEqual(jp2.box[2].box[2].association, (3, 2, 1))  # reverse

    def test_NR_file3_dump(self):
        # Three 8-bit components in the sRGB-YCC colourspace, with the Cb and
        # Cr components being subsampled 2x in both the horizontal and
        # vertical directions. The components are stored in the standard
        # order.
        jfile = opj_data_file('input/conformance/file3.jp2')
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(640, 480, num_components=3)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=glymur.core.YCC,
                                                    approximation=1)
        self.verifyColourSpecificationBox(jp2.box[2].box[1], colr)

        # sub-sampling
        codestream = jp2.get_codestream()
        self.assertEqual(codestream.segment[1].xrsiz[0], 1)
        self.assertEqual(codestream.segment[1].yrsiz[0], 1)
        self.assertEqual(codestream.segment[1].xrsiz[1], 2)
        self.assertEqual(codestream.segment[1].yrsiz[1], 2)
        self.assertEqual(codestream.segment[1].xrsiz[2], 2)
        self.assertEqual(codestream.segment[1].yrsiz[2], 2)

    def test_NR_file4_dump(self):
        # One 8-bit component in the grey colourspace.
        jfile = opj_data_file('input/conformance/file4.jp2')
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(512, 768)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=GREYSCALE,
                                                    approximation=1)
        self.verifyColourSpecificationBox(jp2.box[2].box[1], colr)

    def test_NR_file5_dump(self):
        # Three 8-bit components in the ROMM-RGB colourspace, encapsulated in a
        # JPX file. The components have been transformed using
        # the RCT. The colourspace is specified using both a Restricted ICC
        # profile and using the JPX-defined enumerated code for the ROMM-RGB
        # colourspace.
        jfile = opj_data_file('input/conformance/file5.jp2')
        with self.assertWarns(UserWarning):
            # There's a warning for an unknown compatibility entry.
            # Ignore it here.
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        expected = FileTypeBox(brand='jpx ',
                               compatibility_list=['jp2 ', 'jpx ', 'jpxb'])
        self.verify_filetype_box(jp2.box[1], expected)

        ihdr = glymur.jp2box.ImageHeaderBox(512, 768, num_components=3)
        self.verifyImageHeaderBox(jp2.box[3].box[0], ihdr)

        method = RESTRICTED_ICC_PROFILE
        icc_profile = bytes([0] * 546)
        colr = glymur.jp2box.ColourSpecificationBox(method=method,
                                                    approximation=1,
                                                    icc_profile=icc_profile)
        self.verifyColourSpecificationBox(jp2.box[3].box[1], colr)
        self.assertEqual(jp2.box[3].box[1].icc_profile['Size'], 546)

    def test_NR_file6_dump(self):
        jfile = opj_data_file('input/conformance/file6.jp2')
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(512, 768, bits_per_component=12)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        method = ENUMERATED_COLORSPACE
        colr = glymur.jp2box.ColourSpecificationBox(colorspace=GREYSCALE,
                                                    method=method,
                                                    approximation=1)
        self.verifyColourSpecificationBox(jp2.box[2].box[1], colr)

    def test_NR_file7_dump(self):
        # Three 16-bit components in the e-sRGB colourspace, encapsulated in a
        # JP2 compatible JPX file. The components have been transformed using
        # the RCT. The colourspace is specified using both a Restricted ICC
        # profile and using the JPX-defined enumerated code for the e-sRGB
        # colourspace.
        jfile = opj_data_file('input/conformance/file7.jp2')
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'colr'])

        self.verifySignatureBox(jp2.box[0])

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jpx ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        ihdr = glymur.jp2box.ImageHeaderBox(640, 480,
                                            num_components=3,
                                            bits_per_component=16)
        self.verifyImageHeaderBox(jp2.box[3].box[0], ihdr)

        method = RESTRICTED_ICC_PROFILE
        colr = glymur.jp2box.ColourSpecificationBox(method=method,
                                                    approximation=1)
        self.verifyColourSpecificationBox(jp2.box[3].box[1], colr)
        self.assertEqual(jp2.box[3].box[1].icc_profile['Size'], 13332)

    def test_NR_file8_dump(self):
        # One 8-bit component in a gamma 1.8 space. The colourspace is
        # specified using a Restricted ICC profile.
        jfile = opj_data_file('input/conformance/file8.jp2')
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'xml ', 'jp2c',
                               'xml '])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(400, 700)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        method = RESTRICTED_ICC_PROFILE
        colr = glymur.jp2box.ColourSpecificationBox(method=method,
                                                    approximation=1)
        self.verifyColourSpecificationBox(jp2.box[2].box[1], colr)
        self.assertEqual(jp2.box[2].box[1].icc_profile['Size'], 414)

        # XML box
        tags = [x.tag for x in jp2.box[3].xml.getroot()]
        self.assertEqual(tags,
                         ['{http://www.jpeg.org/jpx/1.0/xml}'
                          + 'GENERAL_CREATION_INFO'])

        # XML box
        tags = [x.tag for x in jp2.box[5].xml.getroot()]
        self.assertEqual(tags,
                         ['{http://www.jpeg.org/jpx/1.0/xml}CAPTION',
                          '{http://www.jpeg.org/jpx/1.0/xml}LOCATION',
                          '{http://www.jpeg.org/jpx/1.0/xml}THING',
                          '{http://www.jpeg.org/jpx/1.0/xml}EVENT'])

    def test_NR_file9_dump(self):
        # Colormap
        jfile = opj_data_file('input/conformance/file9.jp2')
        with self.assertWarns(UserWarning):
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'pclr', 'cmap', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(512, 768)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        # Palette box.
        self.assertEqual(jp2.box[2].box[1].palette.shape, (256, 3))
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[0, 0], 0)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[0, 1], 0)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[0, 2], 0)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[128, 0], 73)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[128, 1], 92)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[128, 2], 53)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[255, 0], 245)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[255, 1], 245)
        np.testing.assert_array_equal(jp2.box[2].box[1].palette[255, 2], 245)

        # Component mapping box
        self.assertEqual(jp2.box[2].box[2].component_index, (0, 0, 0))
        self.assertEqual(jp2.box[2].box[2].mapping_type, (1, 1, 1))
        self.assertEqual(jp2.box[2].box[2].palette_index, (0, 1, 2))

        colr = glymur.jp2box.ColourSpecificationBox(colorspace=SRGB,
                                                    approximation=1)
        self.verifyColourSpecificationBox(jp2.box[2].box[3], colr)

    def test_NR_issue188_beach_64bitsbox(self):
        lst = ['input', 'nonregression', 'issue188_beach_64bitsbox.jp2']
        jfile = opj_data_file('/'.join(lst))
        with self.assertWarns(UserWarning):
            # There's a warning for an unknown box.
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', b'XML ', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(200, 200,
                                            num_components=3,
                                            colorspace_unknown=True)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        cspace = glymur.core.SRGB
        colr = glymur.jp2box.ColourSpecificationBox(colorspace=cspace)
        self.verifyColourSpecificationBox(jp2.box[2].box[1], colr)

        # Skip the 4th box, it is uknown.

        c = jp2.box[4].codestream

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'CME']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (200, 200), 'xyosiz': (0, 0),
                  'xytsiz': (200, 200), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8),
                  'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3].guard_bits, 1)

    def test_NR_orb_blue10_lin_jp2_dump(self):
        jfile = opj_data_file('input/nonregression/orb-blue10-lin-jp2.jp2')
        with self.assertWarns(UserWarning):
            # This file has an invalid ICC profile
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(117, 117, num_components=4)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.RESTRICTED_ICC_PROFILE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertIsNone(jp2.box[2].box[1].icc_profile)
        self.assertIsNone(jp2.box[2].box[1].colorspace)

        c = jp2.box[3].codestream

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (117, 117), 'xyosiz': (0, 0),
                  'xytsiz': (117, 117), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8, 8),
                  'signed': (False, False, False, False),
                  'xyrsiz': [(1, 1, 1, 1), (1, 1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])

    def test_NR_orb_blue10_win_jp2_dump(self):
        jfile = opj_data_file('input/nonregression/orb-blue10-win-jp2.jp2')
        with self.assertWarns(UserWarning):
            # This file has an invalid ICC profile
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        self.verifySignatureBox(jp2.box[0])
        self.verify_filetype_box(jp2.box[1], FileTypeBox())

        ihdr = glymur.jp2box.ImageHeaderBox(117, 117, num_components=4)
        self.verifyImageHeaderBox(jp2.box[2].box[0], ihdr)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.RESTRICTED_ICC_PROFILE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertIsNone(jp2.box[2].box[1].icc_profile)
        self.assertIsNone(jp2.box[2].box[1].colorspace)

        c = jp2.box[3].codestream

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        kwargs = {'rsiz': 0, 'xysiz': (117, 117), 'xyosiz': (0, 0),
                  'xytsiz': (117, 117), 'xytosiz': (0, 0),
                  'bitdepth': (8, 8, 8, 8),
                  'signed': (False, False, False, False),
                  'xyrsiz': [(1, 1, 1, 1), (1, 1, 1, 1)]}
        self.verifySizSegment(c.segment[1],
                              glymur.codestream.SIZsegment(**kwargs))

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        self.verify_codeblock_style(c.segment[2].spcod[7],
                                    [False, False, False, False, False, False])
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10, 9, 9, 10])


if __name__ == "__main__":
    unittest.main()
