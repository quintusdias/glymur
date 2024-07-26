# -*- coding:  utf-8 -*-
"""
Test suite for codestream oddities
"""

# Standard library imports ...
import importlib.resources as ir
from io import BytesIO
import pathlib
import struct
import tempfile
import unittest
import warnings

# Local imports ...
import glymur
from glymur import Jp2k, Jp2kr
from glymur.jp2box import InvalidJp2kError
from . import fixtures


class TestSuite(fixtures.TestCommon):
    """Test suite for codestreams."""

    def setUp(self):
        super().setUp()

        self.p0_03 = ir.files('tests.data').joinpath('p0_03.j2k')
        self.p0_06 = ir.files('tests.data').joinpath('p0_06.j2k')
        self.p1_06 = ir.files('tests.data').joinpath('p1_06.j2k')
        self.issue142 = ir.files('tests.data').joinpath('issue142.j2k')
        self.edf_c2_1178956 = ir.files('tests.data').joinpath('edf_c2_1178956.jp2')  # noqa : E501
        self.htj2k = ir.files('tests.data').joinpath('oj-ht-byte.jph')

    def test_cap_marker_segment(self):
        """
        SCENARIO:  the file has a CAP marker segment for the 3rd segment

        EXPECTED RESULT:  the segment metadata is verified
        """
        j = Jp2k(self.htj2k)
        cap = j.codestream.segment[2]

        self.assertEqual(cap.pcap, 131072)
        self.assertEqual(cap.ccap, (3,))

    def test_unrecognized_marker(self):
        """
        SCENARIO:  There is an unrecognized marker just after an SOT marker but
        before the EOC marker.  All markers must have a leading byte value of
        0xff.

        EXPECTED RESULT:  InvalidJp2kError
        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with open(self.j2kfile, 'rb') as ifile:
                # Everything up until the SOT marker.
                read_buffer = ifile.read(98)
                tfile.write(read_buffer)

                # Write the bad marker 0xd900
                read_buffer = struct.pack('>H', 0xd900)
                tfile.write(read_buffer)

                # Get the rest of the input file.
                read_buffer = ifile.read()
                tfile.write(read_buffer)
                tfile.flush()

            with self.assertRaises(InvalidJp2kError):
                Jp2k(tfile.name).get_codestream(header_only=False)

    def test_bad_tile_part_pointer(self):
        """
        SCENARIO:  A bad SOT marker segment is encountered (Psot value pointing
        far beyond the end of the EOC marker) when requesting a fully parsed
        codestream.

        EXPECTED RESULT:  InvalidJp2kError
        """
        with open(self.temp_jp2_filename, 'wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Copy up until Psot field.
                ofile.write(ifile.read(204))

                # Write a bad Psot value.
                ofile.write(struct.pack('>I', 2000000))

                # copy the rest of the file as-is.
                ifile.seek(208)
                ofile.write(ifile.read())
                ofile.flush()

        j = Jp2kr(self.temp_jp2_filename)
        with self.assertRaises(InvalidJp2kError):
            j.get_codestream(header_only=False)

    def test_tile_height_is_zero(self):
        """
        Scenario:  A tile has height of zero.

        Expected result:  ZeroDivisionError

        Original test file was input/nonregression/2539.pdf.SIGFPE.706.1712.jp2
        """
        fp = BytesIO()

        buffer = struct.pack('>H', 47)  # length

        # kwargs = {'rsiz': 1,
        #           'xysiz': (1000, 1000),
        #           'xyosiz': (0, 0),
        #           'xytsiz': (0, 1000),
        #           'xytosiz': (0, 0),
        #           'Csiz': 3,
        #           'bitdepth': (8, 8, 8),
        #           'signed':  (False, False, False),
        #           'xyrsiz': ((1, 1, 1), (1, 1, 1)),
        #           'length': 47,
        #           'offset': 2}
        buffer += struct.pack('>HIIIIIIIIH', 1, 1000, 1000, 0, 0, 0, 1000,
                              0, 0, 3)
        buffer += struct.pack('>BBBBBBBBB', 7, 1, 1, 7, 1, 1, 7, 1, 1)
        fp.write(buffer)
        fp.seek(0)

        with self.assertRaises(ZeroDivisionError):
            glymur.codestream.Codestream._parse_siz_segment(fp)

    def test_invalid_codestream_past_header(self):
        """
        Scenario:  the codestream is ok thru the header, but invalid after
        that.  The codestream header for the complete test file ends at byte

        Expected result:  InvalidJp2kError
        """
        path = ir.files('tests.data').joinpath('p1_06.j2k')

        with tempfile.TemporaryDirectory() as tdir:
            with open(path, mode='rb') as ifile:
                with open(pathlib.Path(tdir) / 'tmp.j2k', mode='wb') as ofile:
                    ofile.write(ifile.read(555))

                with self.assertRaises(InvalidJp2kError):
                    j = Jp2k(pathlib.Path(tdir) / 'tmp.j2k')
                    j.get_codestream(header_only=False)

    def test_tlm_segment(self):
        """
        Verify parsing of the TLM segment.

        In this case there's only a single tile.
        """
        path = ir.files('tests.data').joinpath('p0_06.j2k')
        j2k = Jp2k(path)

        buffer = b'\xffU\x00\x08\x00@\x00\x00YW'
        b = BytesIO(buffer[2:])
        segment = j2k.codestream._parse_tlm_segment(b)

        self.assertEqual(segment.ztlm, 0)

        # ttlm is an array, but None is the singleton element
        self.assertIsNone(segment.ttlm.item())

        self.assertEqual(segment.ptlm, (22871,))

    def test_ppt_segment(self):
        """
        Verify parsing of the PPT segment
        """
        path = ir.files('tests.data').joinpath('p1_06.j2k')
        j2k = Jp2k(path)
        c = j2k.get_codestream(header_only=False)
        self.assertEqual(c.segment[6].zppt, 0)

    def test_plt_segment(self):
        """
        Verify parsing of the PLT segment
        """
        path = ir.files('tests.data').joinpath('issue142.j2k')
        c = Jp2k(path).get_codestream(header_only=False)
        self.assertEqual(c.segment[7].zplt, 0)
        self.assertEqual(len(c.segment[7].iplt), 59)

    def test_ppm_segment(self):
        """
        Verify parsing of the PPM segment
        """
        with warnings.catch_warnings():
            # Lots of things wrong with this file.
            warnings.simplefilter('ignore')
            jp2 = Jp2k(self.edf_c2_1178956)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[2].zppm, 0)
        self.assertEqual(len(c.segment[2].data), 9)

    def test_crg_segment(self):
        """
        Verify parsing of the CRG segment
        """
        path = ir.files('tests.data').joinpath('p0_03.j2k')
        j2k = Jp2k(path)
        c = j2k.get_codestream()
        self.assertEqual(c.segment[6].xcrg, (65424,))
        self.assertEqual(c.segment[6].ycrg, (32558,))

    def test_rgn_segment(self):
        """
        Verify parsing of the RGN segment
        """
        path = ir.files('tests.data').joinpath('p0_06.j2k')
        j2k = Jp2k(path)
        c = j2k.get_codestream()
        self.assertEqual(c.segment[-1].crgn, 0)
        self.assertEqual(c.segment[-1].srgn, 0)
        self.assertEqual(c.segment[-1].sprgn, 11)

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

    def test_626(self):
        """
        Scenario:  After parsing the SOC and SIZ segments, an unknown segment
        (probably invalid) is hit, and then the file ends, leaving us trying
        to interpret EOF as another marker segment.

        Expected result:  InvalidJp2kError
        """
        path = ir.files('tests.data').joinpath('issue626.j2k')
        with self.assertRaises(InvalidJp2kError):
            Jp2k(path)


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
