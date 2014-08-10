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
class TestSuiteDumpWarnings(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_NR_broken_jp2_dump(self):
        jfile = opj_data_file('input/nonregression/broken.jp2')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            # colr box has bad length.
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
        self.assertEqual(jp2.box[2].box[0].height, 152)
        self.assertEqual(jp2.box[2].box[0].width, 203)
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
        self.assertEqual(jp2.box[2].box[1].approximation, 0)  # not allowed?
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)

        c = jp2.box[3].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'CME', 'COD', 'QCD', 'QCC', 'QCC']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 203)
        self.assertEqual(c.segment[1].ysiz, 152)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (203, 152))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COM: comment
        # Registration
        self.assertEqual(c.segment[2].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[2].ccme.decode('latin-1'),
                         "Creator: JasPer Version 1.701.0")

        # COD: Coding style default
        self.assertFalse(c.segment[3].scod & 2)  # no sop
        self.assertFalse(c.segment[3].scod & 4)  # no eph
        self.assertEqual(c.segment[3].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[3].layers, 1)  # layers = 1
        self.assertEqual(c.segment[3].spcod[3], 1)  # mct
        self.assertEqual(c.segment[3].spcod[4], 5)  # level
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
        self.assertEqual(len(c.segment[3].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[4].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[4].guard_bits, 2)
        self.assertEqual(c.segment[4].mantissa, [0] * 16)
        self.assertEqual(c.segment[4].exponent,
                         [8] + [9, 9, 10] * 5)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].cqcc, 1)
        self.assertEqual(c.segment[5].guard_bits, 2)
        # quantization type
        self.assertEqual(c.segment[5].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[5].mantissa, [0] * 16)
        self.assertEqual(c.segment[5].exponent,
                         [8] + [9, 9, 10] * 5)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].cqcc, 2)
        self.assertEqual(c.segment[6].guard_bits, 2)
        # quantization type
        self.assertEqual(c.segment[6].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[6].mantissa, [0] * 16)
        self.assertEqual(c.segment[6].exponent,
                         [8] + [9, 9, 10] * 5)

    def test_NR_broken2_jp2_dump(self):
        # Invalid marker ID on codestream.
        jfile = opj_data_file('input/nonregression/broken2.jp2')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            jp2 = Jp2k(jfile)

        self.assertEqual(jp2.box[-1].main_header.segment[-1].marker_id, 'QCC')

    @unittest.skipIf(sys.maxsize < 2**32, 'Do not run on 32-bit platforms')
    def test_NR_broken3_jp2_dump(self):
        """
        NR_broken3_jp2_dump

        The file in question here has a colr box with an erroneous box
        length of over 1GB.  Don't run it on 32-bit platforms.
        """
        jfile = opj_data_file('input/nonregression/broken3.jp2')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
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
        self.assertEqual(jp2.box[2].box[0].height, 152)
        self.assertEqual(jp2.box[2].box[0].width, 203)
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
        self.assertEqual(jp2.box[2].box[1].colorspace, glymur.core.SRGB)

        c = jp2.box[3].main_header

        ids = [x.marker_id for x in c.segment]
        expected = ['SOC', 'SIZ', 'CME', 'COD', 'QCD', 'QCC', 'QCC']
        self.assertEqual(ids, expected)

        # SIZ: Image and tile size
        # Profile:
        self.assertEqual(c.segment[1].rsiz, 0)
        # Reference grid size
        self.assertEqual(c.segment[1].xsiz, 203)
        self.assertEqual(c.segment[1].ysiz, 152)
        # Reference grid offset
        self.assertEqual((c.segment[1].xosiz, c.segment[1].yosiz), (0, 0))
        # Tile size
        self.assertEqual((c.segment[1].xtsiz, c.segment[1].ytsiz), (203, 152))
        # Tile offset
        self.assertEqual((c.segment[1].xtosiz, c.segment[1].ytosiz), (0, 0))
        # bitdepth
        self.assertEqual(c.segment[1].bitdepth, (8, 8, 8))
        # signed
        self.assertEqual(c.segment[1].signed, (False, False, False))
        # subsampling
        self.assertEqual(list(zip(c.segment[1].xrsiz, c.segment[1].yrsiz)),
                         [(1, 1)] * 3)

        # COM: comment
        # Registration
        self.assertEqual(c.segment[2].rcme, glymur.core.RCME_ISO_8859_1)
        # Comment value
        self.assertEqual(c.segment[2].ccme.decode('latin-1'),
                         "Creator: JasPer Vers)on 1.701.0")

        # COD: Coding style default
        self.assertFalse(c.segment[3].scod & 2)  # no sop
        self.assertFalse(c.segment[3].scod & 4)  # no eph
        self.assertEqual(c.segment[3].spcod[0], glymur.core.LRCP)
        self.assertEqual(c.segment[3].layers, 1)  # layers = 1
        self.assertEqual(c.segment[3].spcod[3], 1)  # mct
        self.assertEqual(c.segment[3].spcod[4], 5)  # level
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
        self.assertEqual(len(c.segment[3].spcod), 9)

        # QCD: Quantization default
        self.assertEqual(c.segment[4].sqcd & 0x1f, 0)
        self.assertEqual(c.segment[4].guard_bits, 2)
        self.assertEqual(c.segment[4].mantissa, [0] * 16)
        self.assertEqual(c.segment[4].exponent,
                         [8] + [9, 9, 10] * 5)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[5].cqcc, 1)
        self.assertEqual(c.segment[5].guard_bits, 2)
        # quantization type
        self.assertEqual(c.segment[5].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[5].mantissa, [0] * 16)
        self.assertEqual(c.segment[5].exponent,
                         [8] + [9, 9, 10] * 5)

        # QCC: Quantization component
        # associated component
        self.assertEqual(c.segment[6].cqcc, 2)
        self.assertEqual(c.segment[6].guard_bits, 2)
        # quantization type
        self.assertEqual(c.segment[6].sqcc & 0x1f, 0)  # none
        self.assertEqual(c.segment[6].mantissa, [0] * 16)
        self.assertEqual(c.segment[6].exponent,
                         [8] + [9, 9, 10] * 5)

    def test_NR_broken4_jp2_dump(self):
        # Has an invalid marker in the main header
        jfile = opj_data_file('input/nonregression/broken4.jp2')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            jp2 = Jp2k(jfile)

        self.assertEqual(jp2.box[-1].main_header.segment[-1].marker_id, 'QCC')

    def test_NR_gdal_fuzzer_assert_in_opj_j2k_read_SQcd_SQcc_patch_jp2(self):
        lst = ['input', 'nonregression',
               'gdal_fuzzer_assert_in_opj_j2k_read_SQcd_SQcc.patch.jp2']
        jfile = opj_data_file('/'.join(lst))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            Jp2k(jfile)

    def test_NR_gdal_fuzzer_check_comp_dx_dy_jp2_dump(self):
        lst = ['input', 'nonregression', 'gdal_fuzzer_check_comp_dx_dy.jp2']
        jfile = opj_data_file('/'.join(lst))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            Jp2k(jfile)

    def test_NR_gdal_fuzzer_check_number_of_tiles(self):
        # Has an impossible tiling setup.
        lst = ['input', 'nonregression',
               'gdal_fuzzer_check_number_of_tiles.jp2']
        jfile = opj_data_file('/'.join(lst))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            Jp2k(jfile)

    def test_NR_gdal_fuzzer_unchecked_numresolutions_dump(self):
        # Has an invalid number of resolutions.
        lst = ['input', 'nonregression',
               'gdal_fuzzer_unchecked_numresolutions.jp2']
        jfile = opj_data_file('/'.join(lst))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            Jp2k(jfile)

    @unittest.skipIf(re.match("1.5|2.0.0", glymur.version.openjpeg_version),
                     "Test not passing on 1.5.x, not introduced until 2.x")
    def test_NR_DEC_issue188_beach_64bitsbox_jp2_41_decode(self):
        # Has an 'XML ' box instead of 'xml '.  Yes that is pedantic, but it
        # really does deserve a warning.
        relpath = 'input/nonregression/issue188_beach_64bitsbox.jp2'
        jfile = opj_data_file(relpath)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            j = Jp2k(jfile)
        d = j.read()


if __name__ == "__main__":
    unittest.main()
