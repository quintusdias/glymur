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

@unittest.skipIf(sys.platform.startswith('linux'), 'warnings failing on linux')
@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
class TestCodestreamOpjDataWarnings(unittest.TestCase):
    """Test suite for unusual codestream cases.  Uses OPJ_DATA_ROOT"""

    def test_bad_rsiz(self):
        """Should warn if RSIZ is bad.  Issue196"""
        filename = opj_data_file('input/nonregression/edf_c2_1002767.jp2')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            j = Jp2k(filename)
            self.assertEqual(len(w), 3)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue('Invalid profile' in str(w[0].message))

    def test_bad_wavelet_transform(self):
        """Should warn if wavelet transform is bad.  Issue195"""
        filename = opj_data_file('input/nonregression/edf_c2_10025.jp2')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            j = Jp2k(filename)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue('Invalid wavelet transform' in str(w[0].message))

    def test_invalid_progression_order(self):
        """Should still be able to parse even if prog order is invalid."""
        jfile = opj_data_file('input/nonregression/2977.pdf.asan.67.2198.jp2')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Jp2k(jfile)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue('Invalid progression order' in str(w[0].message))

    def test_tile_height_is_zero(self):
        """Zero tile height should not cause an exception."""
        filename = opj_data_file('input/nonregression/2539.pdf.SIGFPE.706.1712.jp2')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Jp2k(filename)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue('Invalid tile dimensions' in str(w[0].message))

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_unknown_marker_segment(self):
        """Should warn for an unknown marker."""
        # Let's inject a marker segment whose marker does not appear to
        # be valid.  We still parse the file, but warn about the offending
        # marker.
        filename = os.path.join(OPJ_DATA_ROOT, 'input/conformance/p0_01.j2k')
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

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    codestream = Jp2k(tfile.name).get_codestream()
                    self.assertTrue(issubclass(w[0].category, UserWarning))
                    self.assertTrue('Unrecognized marker' in str(w[0].message))

                self.assertEqual(codestream.segment[2].marker_id, '0xff79')
                self.assertEqual(codestream.segment[2].length, 3)
                self.assertEqual(codestream.segment[2].data, b'\x00')


if __name__ == "__main__":
    unittest.main()
