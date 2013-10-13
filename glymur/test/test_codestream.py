"""
Test suite for codestream parsing.
"""

# unittest doesn't work well with R0904.
# pylint: disable=R0904

# tempfile.TemporaryDirectory, unittest.assertWarns introduced in 3.2
# pylint: disable=E1101

# unittest2 is python2.6 only (pylint/python-2.7)
# pylint: disable=F0401

import os
import struct
import sys
import tempfile

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

from glymur import Jp2k
import glymur

try:
    DATA_ROOT = os.environ['OPJ_DATA_ROOT']
except KeyError:
    DATA_ROOT = None
except:
    raise


@unittest.skipIf(DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestCodestream(unittest.TestCase):
    """Test suite for unusual codestream cases."""

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
        filename = os.path.join(DATA_ROOT, 'input/conformance/p0_01.j2k')
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

    @unittest.skipIf(sys.hexversion < 0x03020000,
                     "Uses features introduced in 3.2.")
    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_unknown_marker_segment(self):
        """Should warn for an unknown marker."""
        # Let's inject a marker segment whose marker does not appear to
        # be valid.  We still parse the file, but warn about the offending
        # marker.
        filename = os.path.join(DATA_ROOT, 'input/conformance/p0_01.j2k')
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with open(filename, 'rb') as ifile:
                # Everything up until the first QCD marker.
                read_buffer = ifile.read(45)
                tfile.write(read_buffer)

                # Write the new marker segment, 0xff79 = 65401
                read_buffer = struct.pack('>HHB', int(65401), int(3), int(0))
                tfile.write(read_buffer)

                # Get the rest of the input file.
                read_buffer = ifile.read()
                tfile.write(read_buffer)
                tfile.flush()

                with self.assertWarns(UserWarning):
                    codestream = Jp2k(tfile.name).get_codestream()

                self.assertEqual(codestream.segment[2].marker_id, '0xff79')
                self.assertEqual(codestream.segment[2].length, 3)
                self.assertEqual(codestream.segment[2].data, b'\x00')

    def test_psot_is_zero(self):
        """Psot=0 in SOT is perfectly legal.  Issue #78."""
        filename = os.path.join(DATA_ROOT,
                                'input/nonregression/123.j2c')
        j = Jp2k(filename)
        codestream = j.get_codestream(header_only=False)

        # The codestream is valid, so we should be able to get the entire
        # codestream, so the last one is EOC.
        self.assertEqual(codestream.segment[-1].marker_id, 'EOC')

if __name__ == "__main__":
    unittest.main()
