"""
The tests defined here roughly correspond to what is in the OpenJPEG test
suite.
"""

# Some test names correspond with openjpeg tests.  Long names are ok in this
# case.
# pylint: disable=C0103

# All of these tests correspond to tests in openjpeg, so no docstring is really
# needed.
# pylint: disable=C0111

# This module is very long, cannot be helped.
# pylint: disable=C0302

# unittest fools pylint with "too many public methods"
# pylint: disable=R0904

# Some tests use numpy test infrastructure, which means the tests never
# reference "self", so pylint claims it should be a function.  No, no, no.
# pylint: disable=R0201

# Many tests are pretty long and that can't be helped.
# pylint:  disable=R0915

# asserWarns introduced in python 3.2 (python2.7/pylint issue)
# pylint: disable=E1101

import re
import sys
import unittest

import warnings

import numpy as np

from glymur import Jp2k
import glymur

from .fixtures import OPJ_DATA_ROOT
from .fixtures import mse, peak_tolerance, read_pgx, opj_data_file


@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestSuiteDump(unittest.TestCase):

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

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 243)
        self.assertEqual(jp2.box[2].box[0].width, 720)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.YCC)

        c = jp2.box[3].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 720)
        self.assertEqual(c.segment[1].ysiz, 243)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (720, 243))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1), (2, 1), (2, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 128))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 128)
        self.assertEqual(c.segment[1].ysiz, 128)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

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
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].spcod[7] & 0x0020)
        self.assertEqual(c.segment[3].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # SOT: start of tile part
        self.assertEqual(c.segment[4].isot, 0)
        self.assertEqual(c.segment[4].psot, 7314)
        self.assertEqual(c.segment[4].tpsot, 0)
        self.assertEqual(c.segment[4].tnsot, 1)

    def test_NR_p0_02_dump(self):
        jfile = opj_data_file('input/conformance/p0_02.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 127)
        self.assertEqual(c.segment[1].ysiz, 126)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (127, 126))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(2, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # sop
        self.assertTrue(c.segment[2].scod & 4)  # eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 6)  # layers = 6
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 0)
        self.assertEqual(c.segment[3].spcoc[0], 3)  # levels
        self.assertEqual(tuple(c.segment[3].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[3].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].spcoc[3] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[3].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[3].spcoc[3] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[5].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # One unknown marker
        self.assertEqual(c.segment[6].marker_id, '0xff30')

        # SOT: start of tile part
        self.assertEqual(c.segment[7].isot, 0)
        self.assertEqual(c.segment[7].psot, 6047)
        self.assertEqual(c.segment[7].tpsot, 0)
        self.assertEqual(c.segment[7].tnsot, 1)

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

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 256)
        self.assertEqual(c.segment[1].ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (4,))
        # signed
        self.assertEqual(c.segment[1].signed, (True,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 8)  # 8
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 1)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[7].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[7].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,"
                         + "2001 Algo Vision Technology")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[9].rcme, glymur.core.RCME_BINARY)
        # Comment value
        self.assertEqual(len(c.segment[9].ccme), 62)

        # TLM (tile-part length)
        self.assertEqual(c.segment[10].ztlm, 0)
        self.assertEqual(c.segment[10].ttlm, (0, 1, 2, 3))
        self.assertEqual(c.segment[10].ptlm, (4267, 2117, 4080, 2081))

        # SOT: start of tile part
        self.assertEqual(c.segment[11].isot, 0)
        self.assertEqual(c.segment[11].psot, 4267)
        self.assertEqual(c.segment[11].tpsot, 0)
        self.assertEqual(c.segment[11].tnsot, 1)

        # RGN: region of interest
        self.assertEqual(c.segment[12].crgn, 0)
        self.assertEqual(c.segment[12].srgn, 0)
        self.assertEqual(c.segment[12].sprgn, 7)

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[13].marker_id, 'SOD')

    def test_NR_p0_04_dump(self):
        jfile = opj_data_file('input/conformance/p0_04.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 640)
        self.assertEqual(c.segment[1].ysiz, 480)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (640, 480))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1), (1, 1), (1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 20)  # 20
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[6].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[6].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[7].isot, 0)
        self.assertEqual(c.segment[7].psot, 264383)
        self.assertEqual(c.segment[7].tpsot, 0)
        self.assertEqual(c.segment[7].tnsot, 1)

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[8].marker_id, 'SOD')

    def test_NR_p0_05_dump(self):
        jfile = opj_data_file('input/conformance/p0_05.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1024)
        self.assertEqual(c.segment[1].ysiz, 1024)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1024, 1024))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1), (1, 1), (2, 2), (2, 2)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 7)  # 7
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 1)
        self.assertEqual(c.segment[3].spcoc[0], 3)  # levels
        self.assertEqual(tuple(c.segment[3].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].spcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[4].ccoc, 3)
        self.assertEqual(c.segment[4].spcoc[0], 6)  # levels
        self.assertEqual(tuple(c.segment[4].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[4].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[4].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[4].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[4].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[4].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[4].spcoc[3] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # TLM (tile-part length)
        self.assertEqual(c.segment[9].ztlm, 0)
        self.assertEqual(c.segment[9].ttlm, (0,))
        self.assertEqual(c.segment[9].ptlm, (1310540,))

        # SOT: start of tile part
        self.assertEqual(c.segment[10].isot, 0)
        self.assertEqual(c.segment[10].psot, 1310540)
        self.assertEqual(c.segment[10].tpsot, 0)
        self.assertEqual(c.segment[10].tnsot, 1)

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[11].marker_id, 'SOD')

    def test_NR_p0_06_dump(self):
        jfile = opj_data_file('input/conformance/p0_06.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 513)
        self.assertEqual(c.segment[1].ysiz, 129)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (513, 129))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (12, 12, 12, 12))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1), (2, 1), (1, 2), (2, 2)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RPCL)
        self.assertEqual(c.segment[2].layers, 4)  # 4
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[7].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[7].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[7].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[7].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[7].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[7].spcoc[3] & 0x0020)
        self.assertEqual(c.segment[7].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # RGN: region of interest
        self.assertEqual(c.segment[8].crgn, 0)  # component
        self.assertEqual(c.segment[8].srgn, 0)  # implicit
        self.assertEqual(c.segment[8].sprgn, 11)

        # SOT: start of tile part
        self.assertEqual(c.segment[9].isot, 0)
        self.assertEqual(c.segment[9].psot, 33582)
        self.assertEqual(c.segment[9].tpsot, 0)
        self.assertEqual(c.segment[9].tnsot, 1)

        # RGN: region of interest
        self.assertEqual(c.segment[10].crgn, 0)  # component
        self.assertEqual(c.segment[10].srgn, 0)  # implicit
        self.assertEqual(c.segment[10].sprgn, 9)

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[11].marker_id, 'SOD')

    def test_NR_p0_07_dump(self):
        jfile = opj_data_file('input/conformance/p0_07.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 2048)
        self.assertEqual(c.segment[1].ysiz, 2048)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (12, 12, 12))
        # signed
        self.assertEqual(c.segment[1].signed, (True, True, True))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1), (1, 1), (1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertTrue(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 8)  # 8
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "Kakadu-3.0.7")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].isot, 0)
        self.assertEqual(c.segment[5].psot, 9951)
        self.assertEqual(c.segment[5].tpsot, 0)
        self.assertEqual(c.segment[5].tnsot, 0)  # unknown

        # POD: progression order change
        self.assertEqual(c.segment[6].rspod, (0,))
        self.assertEqual(c.segment[6].cspod, (0,))
        self.assertEqual(c.segment[6].lyepod, (9,))
        self.assertEqual(c.segment[6].repod, (3,))
        self.assertEqual(c.segment[6].cdpod, (3,))
        self.assertEqual(c.segment[6].ppod, (glymur.core.LRCP,))

        # PLT: packet length, tile part
        self.assertEqual(c.segment[7].zplt, 0)
        #self.assertEqual(c.segment[7].iplt), 99)

        # SOD:  start of data
        self.assertEqual(c.segment[8].marker_id, 'SOD')

    def test_NR_p0_08_dump(self):
        jfile = opj_data_file('input/conformance/p0_08.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 513)
        self.assertEqual(c.segment[1].ysiz, 3072)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (513, 3072))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (12, 12, 12))
        # signed
        self.assertEqual(c.segment[1].signed, (True, True, True))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1), (1, 1), (1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertTrue(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.CPRL)
        self.assertEqual(c.segment[2].layers, 30)  # 30
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 7)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 0)
        self.assertEqual(c.segment[3].spcoc[0], 6)  # levels
        self.assertEqual(tuple(c.segment[3].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].spcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[4].ccoc, 1)
        self.assertEqual(c.segment[4].spcoc[0], 7)  # levels
        self.assertEqual(tuple(c.segment[4].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[4].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[4].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[4].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[4].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[4].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[4].spcoc[3] & 0x0020)
        self.assertEqual(c.segment[4].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[5].ccoc, 2)
        self.assertEqual(c.segment[5].spcoc[0], 8)  # levels
        self.assertEqual(tuple(c.segment[5].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[5].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[5].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[5].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[5].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[5].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[5].spcoc[3] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[9].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[9].ccme.decode('latin-1'),
                         "Kakadu-3.0.7")

        # SOT: start of tile part
        self.assertEqual(c.segment[10].isot, 0)
        self.assertEqual(c.segment[10].psot, 3820593)
        self.assertEqual(c.segment[10].tpsot, 0)
        self.assertEqual(c.segment[10].tnsot, 1)  # unknown

    def test_NR_p0_09_dump(self):
        jfile = opj_data_file('input/conformance/p0_09.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "0" means profile 2, or full capabilities
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 17)
        self.assertEqual(c.segment[1].ysiz, 37)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (17, 37))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "Kakadu-3.0.7")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].isot, 0)
        self.assertEqual(c.segment[5].psot, 478)
        self.assertEqual(c.segment[5].tpsot, 0)
        self.assertEqual(c.segment[5].tnsot, 1)  # unknown

        # SOD:  start of data
        # Just one.
        self.assertEqual(c.segment[6].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[7].marker_id, 'EOC')

    def test_NR_p0_10_dump(self):
        jfile = opj_data_file('input/conformance/p0_10.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 256)
        self.assertEqual(c.segment[1].ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(4, 4), (4, 4), (4, 4)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # 2
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SOT: start of tile part
        self.assertEqual(c.segment[4].isot, 0)
        self.assertEqual(c.segment[4].psot, 2453)
        self.assertEqual(c.segment[4].tpsot, 0)
        self.assertEqual(c.segment[4].tnsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[5].marker_id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[6].isot, 1)
        self.assertEqual(c.segment[6].psot, 2403)
        self.assertEqual(c.segment[6].tpsot, 0)
        self.assertEqual(c.segment[6].tnsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[7].marker_id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[8].isot, 2)
        self.assertEqual(c.segment[8].psot, 2420)
        self.assertEqual(c.segment[8].tpsot, 0)
        self.assertEqual(c.segment[8].tnsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[9].marker_id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[10].isot, 3)
        self.assertEqual(c.segment[10].psot, 2472)
        self.assertEqual(c.segment[10].tpsot, 0)
        self.assertEqual(c.segment[10].tnsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[11].marker_id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[12].isot, 0)
        self.assertEqual(c.segment[12].psot, 1043)
        self.assertEqual(c.segment[12].tpsot, 1)
        self.assertEqual(c.segment[12].tnsot, 2)

        # SOD:  start of data
        self.assertEqual(c.segment[13].marker_id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[14].isot, 1)
        self.assertEqual(c.segment[14].psot, 1101)
        self.assertEqual(c.segment[14].tpsot, 1)
        self.assertEqual(c.segment[14].tnsot, 2)

        # SOD:  start of data
        self.assertEqual(c.segment[15].marker_id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[16].isot, 3)
        self.assertEqual(c.segment[16].psot, 1054)
        self.assertEqual(c.segment[16].tpsot, 1)
        self.assertEqual(c.segment[16].tnsot, 2)

        # SOD:  start of data
        self.assertEqual(c.segment[17].marker_id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[18].isot, 2)
        self.assertEqual(c.segment[18].psot, 14)
        self.assertEqual(c.segment[18].tpsot, 1)
        self.assertEqual(c.segment[18].tnsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[19].marker_id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[20].isot, 2)
        self.assertEqual(c.segment[20].psot, 1089)
        self.assertEqual(c.segment[20].tpsot, 2)
        self.assertEqual(c.segment[20].tnsot, 0)

        # SOD:  start of data
        self.assertEqual(c.segment[21].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[22].marker_id, 'EOC')

    def test_NR_p0_11_dump(self):
        jfile = opj_data_file('input/conformance/p0_11.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 128)
        self.assertEqual(c.segment[1].ysiz, 1)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertTrue(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 0)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, [(128, 2)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[3].guard_bits, 3)
        self.assertEqual(c.segment[3].mantissa, [0])
        self.assertEqual(c.segment[3].exponent, [8])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].isot, 0)
        self.assertEqual(c.segment[5].psot, 118)
        self.assertEqual(c.segment[5].tpsot, 0)
        self.assertEqual(c.segment[5].tnsot, 1)

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

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 3)
        self.assertEqual(c.segment[1].ysiz, 5)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (3, 5))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].isot, 0)
        self.assertEqual(c.segment[5].psot, 162)
        self.assertEqual(c.segment[5].tpsot, 0)
        self.assertEqual(c.segment[5].tnsot, 1)

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

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1)
        self.assertEqual(c.segment[1].ysiz, 1)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (1, 1))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, tuple([8] * 257))
        # signed
        self.assertEqual(c.segment[1].signed, tuple([False] * 257))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 257)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 1)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 2)
        self.assertEqual(c.segment[3].spcoc[0], 1)  # levels
        self.assertEqual(tuple(c.segment[3].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].spcoc[3] & 0x0020)
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

        # RGN: region of interest
        self.assertEqual(c.segment[7].crgn, 3)
        self.assertEqual(c.segment[7].srgn, 0)
        self.assertEqual(c.segment[7].sprgn, 11)

        # POD:  progression order change
        self.assertEqual(c.segment[8].rspod, (0, 0))
        self.assertEqual(c.segment[8].cspod, (0, 128))
        self.assertEqual(c.segment[8].lyepod, (1, 1))
        self.assertEqual(c.segment[8].repod, (33, 33))
        self.assertEqual(c.segment[8].cdpod, (128, 257))
        self.assertEqual(c.segment[8].ppod,
                         (glymur.core.RLCP, glymur.core.CPRL))

        # COM: comment
        # Registration
        self.assertEqual(c.segment[9].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[9].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[10].isot, 0)
        self.assertEqual(c.segment[10].psot, 1537)
        self.assertEqual(c.segment[10].tpsot, 0)
        self.assertEqual(c.segment[10].tnsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[11].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[12].marker_id, 'EOC')

    def test_NR_p0_14_dump(self):
        jfile = opj_data_file('input/conformance/p0_14.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "0" means profile 2
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 49)
        self.assertEqual(c.segment[1].ysiz, 49)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (49, 49))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # 1 layer
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "Kakadu-3.0.7")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].isot, 0)
        self.assertEqual(c.segment[5].psot, 1528)
        self.assertEqual(c.segment[5].tpsot, 0)
        self.assertEqual(c.segment[5].tnsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[6].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[7].marker_id, 'EOC')

    def test_NR_p0_15_dump(self):
        jfile = opj_data_file('input/conformance/p0_15.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 0
        self.assertEqual(c.segment[1].rsiz, 1)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 256)
        self.assertEqual(c.segment[1].ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (4,))
        # signed
        self.assertEqual(c.segment[1].signed, (True,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 8)  # layers = 8
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 1)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[7].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[7].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,"
                         + "2001 Algo Vision Technology")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[9].rcme, glymur.core.RCME_BINARY)
        # Comment value
        self.assertEqual(len(c.segment[9].ccme), 62)

        # TLM: tile-part length
        self.assertEqual(c.segment[10].ztlm, 0)
        self.assertEqual(c.segment[10].ttlm, (0, 1, 2, 3))
        self.assertEqual(c.segment[10].ptlm, (4267, 2117, 4080, 2081))

        # SOT: start of tile part
        self.assertEqual(c.segment[11].isot, 0)
        self.assertEqual(c.segment[11].psot, 4267)
        self.assertEqual(c.segment[11].tpsot, 0)
        self.assertEqual(c.segment[11].tnsot, 1)

        # RGN: region of interest
        self.assertEqual(c.segment[12].crgn, 0)
        self.assertEqual(c.segment[12].srgn, 0)
        self.assertEqual(c.segment[12].sprgn, 7)

        # SOD:  start of data
        self.assertEqual(c.segment[13].marker_id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        # SOT: start of tile part
        self.assertEqual(c.segment[31].isot, 1)
        self.assertEqual(c.segment[31].psot, 2117)
        self.assertEqual(c.segment[31].tpsot, 0)
        self.assertEqual(c.segment[31].tnsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[32].marker_id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        # SOT: start of tile part
        self.assertEqual(c.segment[49].isot, 2)
        self.assertEqual(c.segment[49].psot, 4080)
        self.assertEqual(c.segment[49].tpsot, 0)
        self.assertEqual(c.segment[49].tnsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[50].marker_id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        # SOT: start of tile part
        self.assertEqual(c.segment[67].isot, 3)
        self.assertEqual(c.segment[67].psot, 2081)
        self.assertEqual(c.segment[67].tpsot, 0)
        self.assertEqual(c.segment[67].tnsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[68].marker_id, 'SOD')

        # 16 SOP markers would be here if we were looking for them

        # EOC:  end of codestream
        self.assertEqual(c.segment[85].marker_id, 'EOC')

    def test_NR_p0_16_dump(self):
        jfile = opj_data_file('input/conformance/p0_16.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "0" means profile 2
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 128)
        self.assertEqual(c.segment[1].ysiz, 128)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)
        self.assertFalse(c.segment[2].scod & 4)
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # levels
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SOT: start of tile part
        self.assertEqual(c.segment[4].isot, 0)
        self.assertEqual(c.segment[4].psot, 7331)
        self.assertEqual(c.segment[4].tpsot, 0)
        self.assertEqual(c.segment[4].tnsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[5].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[6].marker_id, 'EOC')

    def test_NR_p1_01_dump(self):
        jfile = opj_data_file('input/conformance/p1_01.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 127)
        self.assertEqual(c.segment[1].ysiz, 227)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (5, 128))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (127, 126))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (1, 101))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(2, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # SOP
        self.assertTrue(c.segment[2].scod & 4)  # EPH
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 5)  # layers = 5
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 0)
        self.assertEqual(c.segment[3].spcoc[0], 3)  # level
        self.assertEqual(tuple(c.segment[3].code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[3].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].spcoc[3] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[3].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[3].spcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4].guard_bits, 3)
        self.assertEqual(c.segment[4].mantissa, [0] * 10)
        self.assertEqual(c.segment[4].exponent,
                         [8, 9, 9, 10, 9, 9, 10, 9, 9, 10])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[5].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[6].isot, 0)
        self.assertEqual(c.segment[6].psot, 4627)
        self.assertEqual(c.segment[6].tpsot, 0)
        self.assertEqual(c.segment[6].tnsot, 1)

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

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 640)
        self.assertEqual(c.segment[1].ysiz, 480)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (640, 480))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, tuple([8] * 3))
        # signed
        self.assertEqual(c.segment[1].signed, tuple([False] * 3))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 19)  # layers = 19
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertTrue(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertTrue(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[6].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[6].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[7].isot, 0)
        self.assertEqual(c.segment[7].psot, 262838)
        self.assertEqual(c.segment[7].tpsot, 0)
        self.assertEqual(c.segment[7].tnsot, 1)

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

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1024)
        self.assertEqual(c.segment[1].ysiz, 1024)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1024, 1024))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, tuple([8] * 4))
        # signed
        self.assertEqual(c.segment[1].signed, tuple([False] * 4))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1), (1, 1), (2, 2), (2, 2)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 10)  # layers = 10
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 6)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertTrue(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 1)
        self.assertEqual(c.segment[3].spcoc[0], 3)  # level
        self.assertEqual(tuple(c.segment[3].code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertTrue(c.segment[3].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[3].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].spcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)

        # COC: Coding style component
        self.assertEqual(c.segment[4].ccoc, 3)
        self.assertEqual(c.segment[4].spcoc[0], 6)  # level
        self.assertEqual(tuple(c.segment[4].code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertTrue(c.segment[4].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[4].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertTrue(c.segment[4].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[4].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[4].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[4].spcoc[3] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # PPM:  packed packet headers, main header
        self.assertEqual(c.segment[9].marker_id, 'PPM')
        self.assertEqual(c.segment[9].zppm, 0)

        # TLM (tile-part length)
        self.assertEqual(c.segment[10].ztlm, 0)
        self.assertEqual(c.segment[10].ttlm, (0,))
        self.assertEqual(c.segment[10].ptlm, (1366780,))

        # SOT: start of tile part
        self.assertEqual(c.segment[11].isot, 0)
        self.assertEqual(c.segment[11].psot, 1366780)
        self.assertEqual(c.segment[11].tpsot, 0)
        self.assertEqual(c.segment[11].tnsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[12].marker_id, 'SOD')

        # EOC:  end of codestream
        self.assertEqual(c.segment[13].marker_id, 'EOC')

    def test_NR_p1_04_dump(self):
        jfile = opj_data_file('input/conformance/p1_04.j2k')
        c = Jp2k(jfile).get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1024)
        self.assertEqual(c.segment[1].ysiz, 1024)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (128, 128))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (12,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 3)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[5].ccme.decode('latin-1'),
                         "Created by Aware, Inc.")

        # SOT: start of tile part
        self.assertEqual(c.segment[6].isot, 0)
        self.assertEqual(c.segment[6].psot, 350)
        self.assertEqual(c.segment[6].tpsot, 0)
        self.assertEqual(c.segment[6].tnsot, 1)

        # SOD:  start of data
        self.assertEqual(c.segment[7].marker_id, 'SOD')

        # SOT: start of tile part
        self.assertEqual(c.segment[8].isot, 1)
        self.assertEqual(c.segment[8].psot, 356)
        self.assertEqual(c.segment[8].tpsot, 0)
        self.assertEqual(c.segment[8].tnsot, 1)

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

        # SOT: start of tile part
        self.assertEqual(c.segment[11].isot, 2)
        self.assertEqual(c.segment[11].psot, 402)
        self.assertEqual(c.segment[11].tpsot, 0)
        self.assertEqual(c.segment[11].tnsot, 1)

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

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 529)
        self.assertEqual(c.segment[1].ysiz, 524)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (17, 12))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (37, 37))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (8, 2))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # sop
        self.assertTrue(c.segment[2].scod & 4)  # eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 2)  # levels = 2
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 7)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 8))  # cblk
        # Selective arithmetic coding bypass
        self.assertTrue(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertTrue(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertTrue(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # 225 consecutive PPM segments.
        zppm = [x.zppm for x in c.segment[5:230]]
        self.assertEqual(zppm, list(range(225)))

        # SOT: start of tile part
        self.assertEqual(c.segment[230].isot, 0)
        self.assertEqual(c.segment[230].psot, 580)
        self.assertEqual(c.segment[230].tpsot, 0)
        self.assertEqual(c.segment[230].tnsot, 1)

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

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 12)
        self.assertEqual(c.segment[1].ysiz, 12)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (3, 3))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # sop
        self.assertTrue(c.segment[2].scod & 4)  # eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.PCRL)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 4)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertTrue(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertTrue(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[5].isot, 0)
        self.assertEqual(c.segment[5].psot, 349)
        self.assertEqual(c.segment[5].tpsot, 0)
        self.assertEqual(c.segment[5].tnsot, 1)

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

        # SIZ: Image and tile size
        # Profile:  "1" means profile 1
        self.assertEqual(c.segment[1].rsiz, 2)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 12)
        self.assertEqual(c.segment[1].ysiz, 12)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (4, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (12, 12))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (4, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(4, 1), (1, 1)])

        # COD: Coding style default
        self.assertTrue(c.segment[2].scod & 2)  # sop
        self.assertTrue(c.segment[2].scod & 4)  # eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RPCL)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 1)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[2].precinct_size, [(1, 1), (2, 2)])

        # COC: Coding style component
        self.assertEqual(c.segment[3].ccoc, 1)
        self.assertEqual(c.segment[3].spcoc[0], 1)  # level
        self.assertEqual(tuple(c.segment[3].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[3].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[3].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[3].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[3].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[3].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[3].spcoc[3] & 0x0020)
        self.assertEqual(c.segment[3].spcoc[4],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(c.segment[3].precinct_size, [(2, 2), (4, 4)])

        # QCD: Quantization default
        # quantization type
        self.assertEqual(c.segment[4].sqcd & 0x1f, 0)  # none
        self.assertEqual(c.segment[4].guard_bits, 2)
        self.assertEqual(c.segment[4].mantissa, [0] * 4)
        self.assertEqual(c.segment[4].exponent, [8, 9, 9, 10])

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[5].ccme.decode('latin-1'),
                         "Creator: AV-J2K (c) 2000,2001 Algo Vision")

        # SOT: start of tile part
        self.assertEqual(c.segment[6].isot, 0)
        self.assertEqual(c.segment[6].psot, 434)
        self.assertEqual(c.segment[6].tpsot, 0)
        self.assertEqual(c.segment[6].tnsot, 1)

        # scads of SOP, EPH segments

        # EOC:  end of codestream
        self.assertEqual(c.segment[-1].marker_id, 'EOC')

    def test_NR_file1_dump(self):
        jfile = opj_data_file('input/conformance/file1.jp2')
        with warnings.catch_warnings():
            # Bad compatibility list item.
            warnings.simplefilter("ignore")
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'xml ', 'jp2h', 'xml ',
                               'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # XML box
        tags = [x.tag for x in jp2.box[2].xml.getroot()]
        self.assertEqual(tags,
                         ['{http://www.jpeg.org/jpx/1.0/xml}'
                          + 'GENERAL_CREATION_INFO'])

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 512)
        self.assertEqual(jp2.box[3].box[0].width, 768)
        self.assertEqual(jp2.box[3].box[0].num_components, 3)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[3].box[1].precedence, 0)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact ??
        self.assertEqual(jp2.box[3].box[1].colorspace, glymur.core.SRGB)

        # XML box
        tags = [x.tag for x in jp2.box[4].xml.getroot()]
        self.assertEqual(tags, ['{http://www.jpeg.org/jpx/1.0/xml}CAPTION',
                                '{http://www.jpeg.org/jpx/1.0/xml}LOCATION',
                                '{http://www.jpeg.org/jpx/1.0/xml}EVENT'])

    def test_NR_file2_dump(self):
        jfile = opj_data_file('input/conformance/file2.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'cdef'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 640)
        self.assertEqual(jp2.box[2].box[0].width, 480)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact??
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.YCC)

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
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 640)
        self.assertEqual(jp2.box[2].box[0].width, 480)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.YCC)

        # sub-sampling
        codestream = jp2.get_codestream()
        self.assertEqual(codestream.segment[1].xrsiz[0], 1)
        self.assertEqual(codestream.segment[1].yrsiz[0], 1)
        self.assertEqual(codestream.segment[1].xrsiz[1], 2)
        self.assertEqual(codestream.segment[1].yrsiz[1], 2)
        self.assertEqual(codestream.segment[1].xrsiz[2], 2)
        self.assertEqual(codestream.segment[1].yrsiz[2], 2)

    def test_NR_file4_dump(self):
        # One 8-bit component in the sRGB-grey colourspace.
        jfile = opj_data_file('input/conformance/file4.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 512)
        self.assertEqual(jp2.box[2].box[0].width, 768)
        self.assertEqual(jp2.box[2].box[0].num_components, 1)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact?
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.GREYSCALE)

    def test_NR_file5_dump(self):
        # Three 8-bit components in the ROMM-RGB colourspace, encapsulated in a
        # JPX file. The components have been transformed using
        # the RCT. The colourspace is specified using both a Restricted ICC
        # profile and using the JPX-defined enumerated code for the ROMM-RGB
        # colourspace.
        jfile = opj_data_file('input/conformance/file5.jp2')
        with warnings.catch_warnings():
            # There's a warning for an unknown compatibility entry.
            # Ignore it here.
            warnings.simplefilter("ignore")
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jpx ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 512)
        self.assertEqual(jp2.box[3].box[0].width, 768)
        self.assertEqual(jp2.box[3].box[0].num_components, 3)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method,
                         glymur.core.RESTRICTED_ICC_PROFILE)  # enumerated
        self.assertEqual(jp2.box[3].box[1].precedence, 0)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[3].box[1].icc_profile['Size'], 546)
        self.assertIsNone(jp2.box[3].box[1].colorspace)

    def test_NR_file6_dump(self):
        jfile = opj_data_file('input/conformance/file6.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 512)
        self.assertEqual(jp2.box[2].box[0].width, 768)
        self.assertEqual(jp2.box[2].box[0].num_components, 1)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 12)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact
        self.assertIsNone(jp2.box[2].box[1].icc_profile)
        self.assertEqual(jp2.box[2].box[1].colorspace,
                         glymur.core.GREYSCALE)

    def test_NR_file7_dump(self):
        # Three 16-bit components in the e-sRGB colourspace, encapsulated in a
        # JP2 compatible JPX file. The components have been transformed using
        # the RCT. The colourspace is specified using both a Restricted ICC
        # profile and using the JPX-defined enumerated code for the e-sRGB
        # colourspace.
        jfile = opj_data_file('input/conformance/file7.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jpx ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 640)
        self.assertEqual(jp2.box[3].box[0].width, 480)
        self.assertEqual(jp2.box[3].box[0].num_components, 3)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 16)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method,
                         glymur.core.RESTRICTED_ICC_PROFILE)
        self.assertEqual(jp2.box[3].box[1].precedence, 0)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)
        self.assertEqual(jp2.box[3].box[1].icc_profile['Size'], 13332)
        self.assertIsNone(jp2.box[3].box[1].colorspace)

    def test_NR_file8_dump(self):
        # One 8-bit component in a gamma 1.8 space. The colourspace is
        # specified using a Restricted ICC profile.
        jfile = opj_data_file('input/conformance/file8.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'xml ', 'jp2c',
                               'xml '])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 400)
        self.assertEqual(jp2.box[2].box[0].width, 700)
        self.assertEqual(jp2.box[2].box[0].num_components, 1)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.RESTRICTED_ICC_PROFILE)  # enumerated
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[2].box[1].icc_profile['Size'], 414)
        self.assertIsNone(jp2.box[2].box[1].colorspace)

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
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[2].box]
        self.assertEqual(ids, ['ihdr', 'pclr', 'cmap', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 512)
        self.assertEqual(jp2.box[2].box[0].width, 768)
        self.assertEqual(jp2.box[2].box[0].num_components, 1)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

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

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[3].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[2].box[3].precedence, 0)
        self.assertEqual(jp2.box[2].box[3].approximation, 1)  # JPX exact
        self.assertIsNone(jp2.box[2].box[3].icc_profile)
        self.assertEqual(jp2.box[2].box[3].colorspace, glymur.core.SRGB)

    def test_NR_00042_j2k_dump(self):
        # Profile 3.
        jfile = opj_data_file('input/nonregression/_00042.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "3" means profile 3
        self.assertEqual(c.segment[1].rsiz, 3)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1920)
        self.assertEqual(c.segment[1].ysiz, 1080)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1920, 1080))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (12, 12, 12))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.CPRL)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[4].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[4].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[4].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[4].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[4].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[4].spcoc[3] & 0x0020)
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
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[6].spcoc[3] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[6].spcoc[3] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[6].spcoc[3] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[6].spcoc[3] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[6].spcoc[3] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[6].spcoc[3] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[8].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[8].ccme.decode('latin-1'),
                         "Created by OpenJPEG version 1.3.0")

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
        # Profile 3.
        jfile = opj_data_file('input/nonregression/Bretagne2.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream(header_only=False)

        # SIZ: Image and tile size
        # Profile:  "3" means profile 3
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 2592)
        self.assertEqual(c.segment[1].ysiz, 1944)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (640, 480))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 3)  # layers = 3
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (32, 32))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (512, 512))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (512, 512))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1420)
        self.assertEqual(c.segment[1].ysiz, 1416)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1420, 1416))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 614)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (512, 614))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (12,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'), "Kakadu-3.2")

    def test_NR_cthead1_dump(self):
        jfile = opj_data_file('input/nonregression/cthead1.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 256)
        self.assertEqual(c.segment[1].ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (256, 256))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'), "Kakadu-v6.3.1")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'), "Kakadu-v6.3.1")

    def test_NR_illegalcolortransform_dump(self):
        jfile = opj_data_file('input/nonregression/illegalcolortransform.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1420)
        self.assertEqual(c.segment[1].ysiz, 1416)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1420, 1416))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 256)
        self.assertEqual(c.segment[1].ysiz, 256)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (256, 256))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (True, True, True))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_BINARY)
        # Comment value
        self.assertEqual(len(c.segment[4].ccme), 36)

    def test_NR_kakadu_v4_4_openjpegv2_broken_dump(self):
        jfile = opj_data_file('input/nonregression/'
                              + 'kakadu_v4-4_openjpegv2_broken.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 2048)
        self.assertEqual(c.segment[1].ysiz, 2500)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (2048, 2500))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 12)  # layers = 12
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 8)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'), "Kakadu-v4.4")

        # COM: comment
        # Registration
        self.assertEqual(c.segment[5].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        expected = "Kdu-Layer-Info: log_2{Delta-D(MSE)/[2^16*Delta-L(bytes)]},"
        expected += " L(bytes)\n"
        expected += " -65.4, 6.8e+004\n"
        expected += " -66.3, 1.0e+005\n"
        expected += " -67.3, 2.0e+005\n"
        expected += " -68.5, 4.1e+005\n"
        expected += " -69.0, 5.1e+005\n"
        expected += " -69.5, 5.9e+005\n"
        expected += " -69.7, 6.8e+005\n"
        expected += " -70.3, 8.2e+005\n"
        expected += " -70.8, 1.0e+006\n"
        expected += " -71.9, 1.4e+006\n"
        expected += " -73.8, 2.0e+006\n"
        expected += "-256.0, 3.7e+006\n"
        self.assertEqual(c.segment[5].ccme.decode('latin-1'), expected)

    def test_NR_MarkerIsNotCompliant_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/MarkerIsNotCompliant.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1420)
        self.assertEqual(c.segment[1].ysiz, 1416)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1420, 1416))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1920)
        self.assertEqual(c.segment[1].ysiz, 1080)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1920, 1080))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1920)
        self.assertEqual(c.segment[1].ysiz, 1080)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1920, 1080))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1920)
        self.assertEqual(c.segment[1].ysiz, 1080)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1920, 1080))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 117)
        self.assertEqual(c.segment[1].ysiz, 117)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (117, 117))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 4)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 117)
        self.assertEqual(c.segment[1].ysiz, 117)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (117, 117))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 4)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (512, 512))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1].signed, (True,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 16)  # layers = 16
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "Kakadu-2.0.2")

    def test_NR_test_lossless_j2k_dump(self):
        jfile = opj_data_file('input/nonregression/test_lossless.j2k')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1024)
        self.assertEqual(c.segment[1].ysiz, 1024)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1024, 1024))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (12,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size), (64, 64))
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "ClearCanvas DICOM OpenJPEG")

    def test_NR_123_j2c_dump(self):
        jfile = opj_data_file('input/nonregression/123.j2c')
        jp2k = Jp2k(jfile)
        c = jp2k.get_codestream()

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1800)
        self.assertEqual(c.segment[1].ysiz, 1800)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1800, 1800))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 1800)
        self.assertEqual(c.segment[1].ysiz, 1800)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (1800, 1800))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (16,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 1)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 11)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 2048)
        self.assertEqual(c.segment[1].ysiz, 1556)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz),
                         (2048, 1556))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (12, 12, 12))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "DCP-Werkstatt")

    def test_NR_issue104_jpxstream_dump(self):
        jfile = opj_data_file('input/nonregression/issue104_jpxstream.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'pclr', 'cmap'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jpxb')
        self.assertEqual(jp2.box[1].compatibility_list[2], 'jpx ')

        # Reader requirements talk.
        # unrestricted jpeg 2000 part 1
        self.assertTrue(5 in jp2.box[2].standard_flag)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 203)
        self.assertEqual(jp2.box[3].box[0].width, 479)
        self.assertEqual(jp2.box[3].box[0].num_components, 1)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[3].box[1].precedence, 2)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # exact
        self.assertEqual(jp2.box[3].box[1].colorspace, glymur.core.SRGB)

        # Jp2 Header
        # Palette box.
        self.assertEqual(jp2.box[3].box[2].palette.shape, (256, 3))

        # Jp2 Header
        # Component mapping box
        self.assertEqual(jp2.box[3].box[3].component_index, (0, 0, 0))
        self.assertEqual(jp2.box[3].box[3].mapping_type, (1, 1, 1))
        self.assertEqual(jp2.box[3].box[3].palette_index, (0, 1, 2))

        c = jp2.box[4].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 479)
        self.assertEqual(c.segment[1].ysiz, 203)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (256, 203))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_5X3_REVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[3].guard_bits, 2)
        self.assertEqual(c.segment[3].mantissa, [0] * 16)
        self.assertEqual(c.segment[3].exponent, [8] + [9, 9, 10] * 5)

    def test_NR_issue188_beach_64bitsbox(self):
        lst = ['input', 'nonregression', 'issue188_beach_64bitsbox.jp2']
        jfile = opj_data_file('/'.join(lst))
        with warnings.catch_warnings():
            # There's a warning for an unknown box.  We explicitly test for
            # that down below.
            warnings.simplefilter("ignore")
            jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'jp2h', b'XML ', 'jp2c'])

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
        self.assertEqual(jp2.box[2].box[0].height, 200)
        self.assertEqual(jp2.box[2].box[0].width, 200)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)

        # Skip the 4th box, it is uknown.

        c = jp2.box[4].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 200)
        self.assertEqual(c.segment[1].ysiz, 200)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (200, 200))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
        self.assertEqual(c.segment[2].spcod[8],
                         glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE)
        self.assertEqual(len(c.segment[2].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[3].sqcd & 0x1f, 2)
        self.assertEqual(c.segment[3].guard_bits, 1)

    def test_NR_issue206_image_000_dump(self):
        jfile = opj_data_file('input/nonregression/issue206_image-000.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jpxb')
        self.assertEqual(jp2.box[1].compatibility_list[2], 'jpx ')

        # Reader requirements talk.
        # unrestricted jpeg 2000 part 1
        self.assertTrue(5 in jp2.box[2].standard_flag)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 326)
        self.assertEqual(jp2.box[3].box[0].width, 431)
        self.assertEqual(jp2.box[3].box[0].num_components, 3)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[3].box[1].precedence, 2)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[3].box[1].colorspace, glymur.core.SRGB)

        c = jp2.box[4].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 431)
        self.assertEqual(c.segment[1].ysiz, 326)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (256, 256))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 1)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 135)
        self.assertEqual(jp2.box[2].box[0].width, 135)
        self.assertEqual(jp2.box[2].box[0].num_components, 2)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.GREYSCALE)

        # Jp2 Header
        # Channel Definition
        self.assertEqual(jp2.box[2].box[2].index, (0, 1))
        self.assertEqual(jp2.box[2].box[2].channel_type, (0, 1))   # opacity
        self.assertEqual(jp2.box[2].box[2].association, (0, 0))  # both main

        c = jp2.box[3].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'CME']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 135)
        self.assertEqual(c.segment[1].ysiz, 135)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (135, 135))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 2)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 2)  # layers = 2
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # COM: comment
        # Registration
        self.assertEqual(c.segment[4].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[4].ccme.decode('latin-1'),
                         "Kakadu-v5.2.1")

    def test_NR_mem_b2b86b74_2753_dump(self):
        jfile = opj_data_file('input/nonregression/mem-b2b86b74-2753.jp2')
        jp2 = Jp2k(jfile)

        ids = [box.box_id for box in jp2.box]
        self.assertEqual(ids, ['jP  ', 'ftyp', 'rreq', 'jp2h', 'jp2c'])

        ids = [box.box_id for box in jp2.box[3].box]
        self.assertEqual(ids, ['ihdr', 'colr', 'pclr', 'cmap'])

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')
        self.assertEqual(jp2.box[1].compatibility_list[1], 'jpxb')
        self.assertEqual(jp2.box[1].compatibility_list[2], 'jpx ')

        # Reader requirements talk.
        # unrestricted jpeg 2000 part 1
        self.assertTrue(5 in jp2.box[2].standard_flag)

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[3].box[0].height, 46)
        self.assertEqual(jp2.box[3].box[0].width, 124)
        self.assertEqual(jp2.box[3].box[0].num_components, 1)
        self.assertEqual(jp2.box[3].box[0].bits_per_component, 4)
        self.assertEqual(jp2.box[3].box[0].signed, False)
        self.assertEqual(jp2.box[3].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[3].box[0].colorspace_unknown, True)
        self.assertEqual(jp2.box[3].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[3].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[3].box[1].precedence, 2)
        self.assertEqual(jp2.box[3].box[1].approximation, 1)  # JPX exact
        self.assertEqual(jp2.box[3].box[1].colorspace, glymur.core.SRGB)

        # Jp2 Header
        # Palette box.
        # 3 columns with 16 entries.
        self.assertEqual(jp2.box[3].box[2].palette.shape, (16, 3))

        # Jp2 Header
        # Component mapping box
        self.assertEqual(jp2.box[3].box[3].component_index, (0, 0, 0))
        self.assertEqual(jp2.box[3].box[3].mapping_type, (1, 1, 1))
        self.assertEqual(jp2.box[3].box[3].palette_index, (0, 1, 2))

        c = jp2.box[4].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 124)
        self.assertEqual(c.segment[1].ysiz, 46)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (124, 46))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (4,))
        # signed
        self.assertEqual(c.segment[1].signed, (False,))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.RLCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 32))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

        # Signature box.  Check for corruption.
        self.assertEqual(jp2.box[0].signature, (13, 10, 135, 10))

        # File type box.
        self.assertEqual(jp2.box[1].brand, 'jp2 ')
        self.assertEqual(jp2.box[1].minor_version, 0)
        self.assertEqual(jp2.box[1].compatibility_list[0], 'jp2 ')

        # Jp2 Header
        # Image header
        self.assertEqual(jp2.box[2].box[0].height, 576)
        self.assertEqual(jp2.box[2].box[0].width, 766)
        self.assertEqual(jp2.box[2].box[0].num_components, 3)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.YCC)

        c = jp2.box[3].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD', 'POD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 766)
        self.assertEqual(c.segment[1].ysiz, 576)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (766, 576))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1), (2, 1), (2, 1)])

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (32, 128))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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

    def test_NR_orb_blue10_lin_jp2_dump(self):
        jfile = opj_data_file('input/nonregression/orb-blue10-lin-jp2.jp2')
        with warnings.catch_warnings():
            # This file has an invalid ICC profile
            warnings.simplefilter("ignore")
            jp2 = Jp2k(jfile)

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
        self.assertEqual(jp2.box[2].box[0].height, 117)
        self.assertEqual(jp2.box[2].box[0].width, 117)
        self.assertEqual(jp2.box[2].box[0].num_components, 4)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.RESTRICTED_ICC_PROFILE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertIsNone(jp2.box[2].box[1].icc_profile)
        self.assertIsNone(jp2.box[2].box[1].colorspace)

        c = jp2.box[3].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 117)
        self.assertEqual(c.segment[1].ysiz, 117)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (117, 117))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 4)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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
        with warnings.catch_warnings():
            # This file has an invalid ICC profile
            warnings.simplefilter("ignore")
            jp2 = Jp2k(jfile)

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
        self.assertEqual(jp2.box[2].box[0].height, 117)
        self.assertEqual(jp2.box[2].box[0].width, 117)
        self.assertEqual(jp2.box[2].box[0].num_components, 4)
        self.assertEqual(jp2.box[2].box[0].bits_per_component, 8)
        self.assertEqual(jp2.box[2].box[0].signed, False)
        self.assertEqual(jp2.box[2].box[0].compression, 7)   # wavelet
        self.assertEqual(jp2.box[2].box[0].colorspace_unknown, False)
        self.assertEqual(jp2.box[2].box[0].ip_provided, False)

        # Jp2 Header
        # Colour specification
        self.assertEqual(jp2.box[2].box[1].method,
                         glymur.core.RESTRICTED_ICC_PROFILE)
        self.assertEqual(jp2.box[2].box[1].precedence, 0)
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # JP2
        self.assertIsNone(jp2.box[2].box[1].icc_profile)
        self.assertIsNone(jp2.box[2].box[1].colorspace)

        c = jp2.box[3].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'COD', 'QCD']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 117)
        self.assertEqual(c.segment[1].ysiz, 117)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (117, 117))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 4)

        # COD: Coding style default
        self.assertFalse(c.segment[2].scod & 2)  # no sop
        self.assertFalse(c.segment[2].scod & 4)  # no eph
        self.assertEqual(c.segment[2].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[2].layers, 1)  # layers = 1
        self.assertEqual(c.segment[2].spcod[3], 0)  # mct
        self.assertEqual(c.segment[2].spcod[4], 5)  # level
        self.assertEqual(tuple(c.segment[2].code_block_size),
                         (64, 64))  # cblk
        # Selective arithmetic coding bypass
        self.assertFalse(c.segment[2].spcod[7] & 0x01)
        # Reset context probabilities
        self.assertFalse(c.segment[2].spcod[7] & 0x02)
        # Termination on each coding pass
        self.assertFalse(c.segment[2].spcod[7] & 0x04)
        # Vertically causal context
        self.assertFalse(c.segment[2].spcod[7] & 0x08)
        # Predictable termination
        self.assertFalse(c.segment[2].spcod[7] & 0x0010)
        # Segmentation symbols
        self.assertFalse(c.segment[2].spcod[7] & 0x0020)
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
