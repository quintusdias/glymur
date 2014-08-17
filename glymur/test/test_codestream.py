"""
Test suite for codestream parsing.
"""

# unittest doesn't work well with R0904.
# pylint: disable=R0904

# tempfile.TemporaryDirectory, unittest.assertWarns introduced in 3.2
# pylint: disable=E1101

import os
import struct
import sys
import tempfile
import unittest
import warnings

from glymur import Jp2k
import glymur

from .fixtures import opj_data_file, OPJ_DATA_ROOT

class TestCodestream(unittest.TestCase):
    """Test suite for unusual codestream cases."""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_siz_segment_ssiz_unsigned(self):
        """ssiz attribute to be removed in future release"""
        j = Jp2k(self.jp2file)
        codestream = j.get_codestream()

        # The ssiz attribute was simply a tuple of raw bytes.
        # The first 7 bits are interpreted as the bitdepth, the MSB determines
        # whether or not it is signed.
        self.assertEqual(codestream.segment[1].ssiz, (7, 7, 7))


@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestCodestreamOpjData(unittest.TestCase):
    """Test suite for unusual codestream cases.  Uses OPJ_DATA_ROOT"""

    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_reserved_marker_segment(self):
        """Reserved marker segments are ok."""

        # Some marker segments were reserved in FCD15444-1.  Since that
        # standard is old, some of them may have come into use.
        #
        # Let's inject a reserved marker segment into a file that
        # we know something about to make sure we can still parse it.
        filename = os.path.join(OPJ_DATA_ROOT, 'input/conformance/p0_01.j2k')
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with open(filename, 'rb') as ifile:
                # Everything up until the first QCD marker.
                read_buffer = ifile.read(45)
                tfile.write(read_buffer)

                # Write the new marker segment, 0xff6f = 65391
                read_buffer = struct.pack('>HHB', int(65391), int(3), int(0))
                tfile.write(read_buffer)

                # Get the rest of the input file.
                read_buffer = ifile.read()
                tfile.write(read_buffer)
                tfile.flush()

            codestream = Jp2k(tfile.name).get_codestream()

            self.assertEqual(codestream.segment[2].marker_id, '0xff6f')
            self.assertEqual(codestream.segment[2].length, 3)
            self.assertEqual(codestream.segment[2].data, b'\x00')

    def test_psot_is_zero(self):
        """Psot=0 in SOT is perfectly legal.  Issue #78."""
        filename = os.path.join(OPJ_DATA_ROOT,
                                'input/nonregression/123.j2c')
        j = Jp2k(filename)
        codestream = j.get_codestream(header_only=False)

        # The codestream is valid, so we should be able to get the entire
        # codestream, so the last one is EOC.
        self.assertEqual(codestream.segment[-1].marker_id, 'EOC')


    def test_siz_segment_ssiz_signed(self):
        """ssiz attribute to be removed in future release"""
        filename = os.path.join(OPJ_DATA_ROOT, 'input/conformance/p0_03.j2k')
        j = Jp2k(filename)
        codestream = j.get_codestream()

        # The ssiz attribute was simply a tuple of raw bytes.
        # The first 7 bits are interpreted as the bitdepth, the MSB determines
        # whether or not it is signed.
        self.assertEqual(codestream.segment[1].ssiz, (131,))


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
        kwargs = {'rsiz': 0,
                  'xysiz': (2592, 1456),
                  'xyosiz': (0, 0),
                  'xytsiz': (2592, 1456),
                  'xytosiz': (0, 0),
                  'Csiz': 3,
                  'bitdepth': (8, 8, 8),
                  'signed':  (False, False, False),
                  'xyrsiz': ((1, 1, 1), (1, 1, 1))}
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


if __name__ == "__main__":
    unittest.main()
