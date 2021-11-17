# -*- coding:  utf-8 -*-
"""
Test suite for codestream oddities
"""

# Standard library imports ...
import importlib.resources as ir
from io import BytesIO
import struct
import unittest
import warnings

# Local imports ...
import glymur
from glymur import Jp2k
from . import fixtures, data


class TestSuite(unittest.TestCase):
    """Test suite for ICC Profile code."""

    def setUp(self):
        self.p0_03 = ir.path(data, 'p0_03.j2k')
        self.p0_06 = ir.path(data, 'p0_06.j2k')
        self.p1_06 = ir.path(data, 'p1_06.j2k')
        self.issue142 = ir.path(data, 'issue142.j2k')
        self.edf_c2_1178956 = ir.path(data, 'edf_c2_1178956.jp2')

    def test_tlm_segment(self):
        """
        Verify parsing of the TLM segment.

        In this case there's only a single tile.
        """
        with ir.path(data, 'p0_06.j2k') as path:
            j2k = Jp2k(path)

        buffer = b'\xffU\x00\x08\x00@\x00\x00YW'
        b = BytesIO(buffer[2:])
        segment = j2k.codestream._parse_tlm_segment(b)

        self.assertEqual(segment.ztlm, 0)
        self.assertIsNone(segment.ttlm)
        self.assertEqual(segment.ptlm, (22871,))

    def test_ppt_segment(self):
        """
        Verify parsing of the PPT segment
        """
        with ir.path(data, 'p1_06.j2k') as path:
            j2k = Jp2k(path)
        c = j2k.get_codestream(header_only=False)
        self.assertEqual(c.segment[6].zppt, 0)

    def test_plt_segment(self):
        """
        Verify parsing of the PLT segment
        """
        with ir.path(data, 'issue142.j2k') as path:
            c = Jp2k(path).get_codestream(header_only=False)
        self.assertEqual(c.segment[7].zplt, 0)
        self.assertEqual(len(c.segment[7].iplt), 59)

    def test_ppm_segment(self):
        """
        Verify parsing of the PPM segment
        """
        with ir.path(data, 'edf_c2_1178956.jp2') as path:
            with warnings.catch_warnings():
                # Lots of things wrong with this file.
                warnings.simplefilter('ignore')
                jp2 = Jp2k(path)
        c = jp2.get_codestream()
        self.assertEqual(c.segment[2].zppm, 0)
        self.assertEqual(len(c.segment[2].data), 9)

    def test_crg_segment(self):
        """
        Verify parsing of the CRG segment
        """
        with ir.path(data, 'p0_03.j2k') as path:
            j2k = Jp2k(path)
        c = j2k.get_codestream()
        self.assertEqual(c.segment[6].xcrg, (65424,))
        self.assertEqual(c.segment[6].ycrg, (32558,))

    def test_rgn_segment(self):
        """
        Verify parsing of the RGN segment
        """
        with ir.path(data, 'p0_06.j2k') as path:
            j2k = Jp2k(path)
        c = j2k.get_codestream()
        self.assertEqual(c.segment[-1].crgn, 0)
        self.assertEqual(c.segment[-1].srgn, 0)
        self.assertEqual(c.segment[-1].sprgn, 11)


class TestCodestreamRepr(unittest.TestCase):

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_soc(self):
        """Test SOC segment repr"""
        segment = glymur.codestream.SOCsegment()
        newseg = eval(repr(segment))
        self.assertEqual(newseg.marker_id, 'SOC')

    def test_siz(self):
        """Test SIZ segment repr"""
        kwargs = {
            'rsiz': 0,
            'xysiz': (2592, 1456),
            'xyosiz': (0, 0),
            'xytsiz': (2592, 1456),
            'xytosiz': (0, 0),
            'Csiz': 3,
            'bitdepth': (8, 8, 8),
            'signed': (False, False, False),
            'xyrsiz': ((1, 1, 1), (1, 1, 1))
        }
        segment = glymur.codestream.SIZsegment(**kwargs)
        newseg = eval(repr(segment))
        self.assertEqual(newseg.marker_id, 'SIZ')
        self.assertEqual(newseg.xsiz, 2592)
        self.assertEqual(newseg.ysiz, 1456)
        self.assertEqual(newseg.xosiz, 0)
        self.assertEqual(newseg.yosiz, 0)
        self.assertEqual(newseg.xtsiz, 2592)
        self.assertEqual(newseg.ytsiz, 1456)
        self.assertEqual(newseg.xtosiz, 0)
        self.assertEqual(newseg.ytosiz, 0)

        self.assertEqual(newseg.xrsiz, (1, 1, 1))
        self.assertEqual(newseg.yrsiz, (1, 1, 1))
        self.assertEqual(newseg.bitdepth, (8, 8, 8))
        self.assertEqual(newseg.signed, (False, False, False))


class TestCodestream(fixtures.TestCommon):
    """Test suite for unusual codestream cases."""

    def test_reserved_marker_segment(self):
        """
        SCENARIO:  Rewrite a J2K file to include a marker segment with a
        reserved marker 0xff6f (65391).

        EXPECTED RESULT:  The marker segment should be properly parsed.
        """

        with open(self.temp_j2k_filename, 'wb') as tfile:
            with open(self.j2kfile, 'rb') as ifile:
                # Everything up until the first QCD marker.
                read_buffer = ifile.read(65)
                tfile.write(read_buffer)

                # Write the new marker segment, 0xff6f = 65391
                read_buffer = struct.pack('>HHB', int(65391), int(3), int(0))
                tfile.write(read_buffer)

                # Get the rest of the input file.
                read_buffer = ifile.read()
                tfile.write(read_buffer)
                tfile.flush()

        codestream = Jp2k(tfile.name).get_codestream()

        self.assertEqual(codestream.segment[3].marker_id, '0xff6f')
        self.assertEqual(codestream.segment[3].length, 3)
        self.assertEqual(codestream.segment[3].data, b'\x00')

    def test_siz_segment_ssiz_unsigned(self):
        """ssiz attribute to be removed in future release"""
        j = Jp2k(self.jp2file)
        codestream = j.get_codestream()

        # The ssiz attribute was simply a tuple of raw bytes.
        # The first 7 bits are interpreted as the bitdepth, the MSB determines
        # whether or not it is signed.
        self.assertEqual(codestream.segment[1].ssiz, (7, 7, 7))
