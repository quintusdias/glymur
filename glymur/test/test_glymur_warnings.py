"""
Test suite for warnings issued by glymur.
"""
import os
import re
import struct
import tempfile
import unittest

from glymur import Jp2k
import glymur

from .fixtures import opj_data_file, OPJ_DATA_ROOT
from .fixtures import WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG


@unittest.skipIf(OPJ_DATA_ROOT is None,
                 "OPJ_DATA_ROOT environment variable not set")
@unittest.skipIf(WARNING_INFRASTRUCTURE_ISSUE, WARNING_INFRASTRUCTURE_MSG)
class TestWarnings(unittest.TestCase):
    """Test suite for warnings issued by glymur."""

    def test_invalid_compatibility_list_entry(self):
        """should not error out with invalid compatibility list entry"""
        filename = opj_data_file('input/nonregression/issue397.jp2')
        with self.assertWarns(UserWarning):
            Jp2k(filename)
        self.assertTrue(True)

    def test_exceeded_box_length(self):
        """
        should warn if reading past end of a box

        Verify that a warning is issued if we read past the end of a box
        This file has a palette (pclr) box whose length is impossibly
        short.
        """
        infile = os.path.join(OPJ_DATA_ROOT,
                              'input/nonregression/mem-b2ace68c-1381.jp2')
        regex = re.compile(r'''Encountered\san\sunrecoverable\sValueError\s
                               while\sparsing\sa\sPalette\sbox\sat\sbyte\s
                               offset\s\d+\.\s+The\soriginal\serror\smessage\s
                               was\s"total\ssize\sof\snew\sarray\smust\sbe\s
                               unchanged"''',
                           re.VERBOSE)
        with self.assertWarnsRegex(UserWarning, regex):
            Jp2k(infile)

    def test_NR_DEC_issue188_beach_64bitsbox_jp2_41_decode(self):
        """
        Has an 'XML ' box instead of 'xml '.  Yes that is pedantic, but it
        really does deserve a warning.
        """
        relpath = 'input/nonregression/issue188_beach_64bitsbox.jp2'
        jfile = opj_data_file(relpath)
        pattern = r"""Unrecognized\sbox\s\(b'XML\s'\)\sencountered."""
        regex = re.compile(pattern, re.VERBOSE)
        with self.assertWarnsRegex(UserWarning, regex):
            Jp2k(jfile)

    def test_NR_gdal_fuzzer_unchecked_numresolutions_dump(self):
        """
        Has an invalid number of resolutions.
        """
        lst = ['input', 'nonregression',
               'gdal_fuzzer_unchecked_numresolutions.jp2']
        jfile = opj_data_file('/'.join(lst))
        regex = re.compile(r"""Invalid\snumber\sof\sresolutions\s
                               \(\d+\)\.""",
                           re.VERBOSE)
        with self.assertWarnsRegex(UserWarning, regex):
            Jp2k(jfile).get_codestream()

    @unittest.skipIf(re.match("1.5|2.0.0", glymur.version.openjpeg_version),
                     "Test not passing on 1.5.x, not introduced until 2.x")
    def test_NR_gdal_fuzzer_check_number_of_tiles(self):
        """
        Has an impossible tiling setup.
        """
        lst = ['input', 'nonregression',
               'gdal_fuzzer_check_number_of_tiles.jp2']
        jfile = opj_data_file('/'.join(lst))
        regex = re.compile(r"""Invalid\snumber\sof\stiles\s
                               \(\d+\)\.""",
                           re.VERBOSE)
        with self.assertWarnsRegex(UserWarning, regex):
            Jp2k(jfile).get_codestream()

    def test_NR_gdal_fuzzer_check_comp_dx_dy_jp2_dump(self):
        """
        Invalid subsampling value.
        """
        lst = ['input', 'nonregression', 'gdal_fuzzer_check_comp_dx_dy.jp2']
        jfile = opj_data_file('/'.join(lst))
        regex = re.compile(r"""Invalid\ssubsampling\svalue\sfor\scomponent\s
                               \d+:\s+
                               dx=\d+,\s*dy=\d+""",
                           re.VERBOSE)
        with self.assertWarnsRegex(UserWarning, regex):
            Jp2k(jfile).get_codestream()

    def test_NR_gdal_fuzzer_assert_in_opj_j2k_read_SQcd_SQcc_patch_jp2(self):
        lst = ['input', 'nonregression',
               'gdal_fuzzer_assert_in_opj_j2k_read_SQcd_SQcc.patch.jp2']
        jfile = opj_data_file('/'.join(lst))
        regex = re.compile(r"""Invalid\scomponent\snumber\s\(\d+\),\s
                               number\sof\scomponents\sis\sonly\s\d+""",
                           re.VERBOSE)
        with self.assertWarnsRegex(UserWarning, regex):
            Jp2k(jfile).get_codestream()

    def test_bad_rsiz(self):
        """Should warn if RSIZ is bad.  Issue196"""
        filename = opj_data_file('input/nonregression/edf_c2_1002767.jp2')
        with self.assertWarnsRegex(UserWarning, 'Invalid profile'):
            Jp2k(filename).get_codestream()

    def test_bad_wavelet_transform(self):
        """Should warn if wavelet transform is bad.  Issue195"""
        filename = opj_data_file('input/nonregression/edf_c2_10025.jp2')
        with self.assertWarnsRegex(UserWarning, 'Invalid wavelet transform'):
            Jp2k(filename).get_codestream()

    def test_invalid_progression_order(self):
        """Should still be able to parse even if prog order is invalid."""
        jfile = opj_data_file('input/nonregression/2977.pdf.asan.67.2198.jp2')
        with self.assertWarnsRegex(UserWarning, 'Invalid progression order'):
            Jp2k(jfile).get_codestream()

    def test_tile_height_is_zero(self):
        """Zero tile height should not cause an exception."""
        filename = 'input/nonregression/2539.pdf.SIGFPE.706.1712.jp2'
        filename = opj_data_file(filename)
        with self.assertWarnsRegex(UserWarning, 'Invalid tile dimensions'):
            Jp2k(filename).get_codestream()

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

                with self.assertWarnsRegex(UserWarning, 'Unrecognized marker'):
                    Jp2k(tfile.name).get_codestream()


if __name__ == "__main__":
    unittest.main()
