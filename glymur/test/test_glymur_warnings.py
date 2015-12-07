"""
Test suite for warnings issued by glymur.
"""
from io import BytesIO
import os
import struct
import sys
import tempfile
import unittest
import warnings

from glymur import Jp2k
import glymur
from glymur.jp2k import InvalidJP2ColourspaceMethodWarning
from glymur.jp2box import InvalidColourspaceMethod
from glymur.jp2box import InvalidICCProfileLengthWarning


@unittest.skipIf(sys.hexversion < 0x03000000, 'Do not bother on python2')
class TestSuite(unittest.TestCase):

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()
        self.jpxfile = glymur.data.jpxfile()

    def tearDown(self):
        warnings.resetwarnings()

    def test_unrecognized_marker(self):
        """
        EOC marker is not retrieved because there is an unrecognized marker

        Original file tested was input/nonregression/illegalcolortransform.j2k
        """
        exp_warning = glymur.codestream.UnrecognizedMarkerWarning
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
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

            exp_warning = glymur.codestream.UnrecognizedMarkerWarning
            if sys.hexversion < 0x03000000:
                with warnings.catch_warnings(record=True) as w:
                    c = Jp2k(tfile.name).get_codestream(header_only=False)
                assert issubclass(w[-1].category, exp_warning)
            else:
                with self.assertWarns(exp_warning):
                    c = Jp2k(tfile.name).get_codestream(header_only=False)

        # Verify that the last segment returned in the codestream is SOT,
        # not EOC.  It was after SOT that the invalid marker was inserted.
        self.assertEqual(c.segment[-1].marker_id, 'SOT')

    def test_unrecoverable_xml(self):
        """
        Bad byte sequence in XML that cannot be parsed.

        Original test file was
        26ccf3651020967f7778238ef5af08af.SIGFPE.d25.527.jp2
        """
        fptr = BytesIO()

        payload = b'\xees'
        fptr.write(payload)
        fptr.seek(0)

        exp_warning = glymur.jp2box.UnrecoverableXMLWarning
        if sys.hexversion < 0x03000000:
            pass
            with warnings.catch_warnings(record=True) as w:
                box = glymur.jp2box.XMLBox.parse(fptr, 0, 8 + len(payload))
            assert issubclass(w[-1].category, exp_warning)
        else:
            with self.assertWarns(exp_warning):
                box = glymur.jp2box.XMLBox.parse(fptr, 0, 8 + len(payload))

        self.assertIsNone(box.xml)

    def test_bom(self):
        """
        Byte order markers are illegal in UTF-8.  Issue 185

        Original test file was input/nonregression/issue171.jp2
        """
        fptr = BytesIO()

        s = "<?xpacket begin='\ufeff' id='W5M0MpCehiHzreSzNTczkc9d'?>"
        s += "<stuff>goes here</stuff>"
        s += "<?xpacket end='w'?>"
        data = s.encode('utf-8')
        fptr.write(data)
        fptr.seek(0)

        exp_warning = glymur.jp2box.ByteOrderMarkerWarning
        if sys.hexversion < 0x03000000:
            pass
            # with warnings.catch_warnings(record=True) as w:
            #     glymur.jp2box.XMLBox.parse(fptr, 0, 8 + len(data))
            # assert issubclass(w[-1].category, exp_warning)
        else:
            with self.assertWarns(exp_warning):
                glymur.jp2box.XMLBox.parse(fptr, 0, 8 + len(data))

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_unknown_marker_segment(self):
        """
        Should warn for an unknown marker.

        Let's inject a marker segment whose marker does not appear to
        be valid.  We still parse the file, but warn about the offending
        marker.
        """
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with open(self.j2kfile, 'rb') as ifile:
                # Everything up until the first QCD marker.
                read_buffer = ifile.read(65)
                tfile.write(read_buffer)

                # Write the new marker segment, 0xff79 = 65401
                read_buffer = struct.pack('>HHB', int(65401), int(3), int(0))
                tfile.write(read_buffer)

                # Get the rest of the input file.
                read_buffer = ifile.read()
                tfile.write(read_buffer)
                tfile.flush()

            exp_warning = glymur.codestream.UnrecognizedMarkerWarning
            if sys.hexversion < 0x03000000:
                with warnings.catch_warnings(record=True) as w:
                    Jp2k(tfile.name).get_codestream()
                assert issubclass(w[-1].category, exp_warning)
            else:
                with self.assertWarns(exp_warning):
                    Jp2k(tfile.name).get_codestream()

    def test_tile_height_is_zero(self):
        """
        Zero tile height should not cause an exception.

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

        exp_warning = glymur.codestream.InvalidTileSpecificationWarning
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.codestream.Codestream._parse_siz_segment(fp)
            assert issubclass(w[-1].category, exp_warning)
        else:
            with self.assertWarns(exp_warning):
                glymur.codestream.Codestream._parse_siz_segment(fp)

    def test_invalid_progression_order(self):
        """
        Should still be able to parse even if prog order is invalid.

        Original test file was input/nonregression/2977.pdf.asan.67.2198.jp2
        """
        fp = BytesIO()
        buffer = struct.pack('>HBBBBBBBBBB', 12, 3, 33, 1, 1, 3, 3, 0, 0, 1, 1)
        fp.write(buffer)
        fp.seek(0)

        exp_warning = glymur.codestream.InvalidProgressionOrderWarning
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.codestream.Codestream._parse_cod_segment(fp)
            assert issubclass(w[-1].category, exp_warning)
        else:
            with self.assertWarns(exp_warning):
                glymur.codestream.Codestream._parse_cod_segment(fp)

    def test_bad_wavelet_transform(self):
        """
        Should warn if wavelet transform is bad.  Issue195

        Original file tested was input/nonregression/edf_c2_10025.jp2
        """
        fp = BytesIO()
        buffer = struct.pack('>HBHBBBBBBB', 12, 0, 1, 1, 1, 3, 3, 0, 0, 10)
        fp.write(buffer)
        fp.seek(0)

        exp_warning = glymur.codestream.InvalidWaveletTransformWarning
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.codestream.Codestream._parse_cod_segment(fp)
            assert issubclass(w[-1].category, exp_warning)
        else:
            with self.assertWarns(exp_warning):
                glymur.codestream.Codestream._parse_cod_segment(fp)

    def test_NR_gdal_fuzzer_assert_in_opj_j2k_read_SQcd_SQcc_patch_jp2(self):
        """
        validate the QCC component number against Csiz

        The original test file was
        gdal_fuzzer_assert_in_opj_j2k_read_SQcd_SQcc.patch.jp2
        """
        fp = BytesIO()

        buffer = struct.pack('>HBB', 4, 64, 64)
        fp.write(buffer)
        fp.seek(0)

        exp_warning = glymur.codestream.InvalidQCCComponentNumber
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.codestream.Codestream._parse_qcc_segment(fp)
            assert issubclass(w[-1].category, exp_warning)
        else:
            with self.assertWarns(exp_warning):
                glymur.codestream.Codestream._parse_qcc_segment(fp)

    def test_NR_gdal_fuzzer_check_comp_dx_dy_jp2_dump(self):
        """
        Invalid subsampling value.

        Original test file was gdal_fuzzer_check_comp_dx_dy.jp2
        """
        fp = BytesIO()

        buffer = struct.pack('>H', 47)  # length

        # kwargs = {'rsiz': 1,
        #           'xysiz': (1000, 1000),
        #           'xyosiz': (0, 0),
        #           'xytsiz': (1000, 1000),
        #           'xytosiz': (0, 0),
        #           'Csiz': 3,
        #           'bitdepth': (8, 8, 8),
        #           'signed':  (False, False, False),
        #           'xyrsiz': ((1, 1, 1), (1, 1, 1)),
        #           'length': 47,
        #           'offset': 2}
        buffer += struct.pack('>HIIIIIIIIH', 1, 1000, 1000, 0, 0, 1000, 1000,
                              0, 0, 3)
        buffer += struct.pack('>BBBBBBBBB', 7, 1, 1, 7, 1, 1, 7, 1, 0)
        fp.write(buffer)
        fp.seek(0)

        exp_warning = glymur.codestream.InvalidSubsamplingWarning
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.codestream.Codestream._parse_siz_segment(fp)
            assert issubclass(w[-1].category, exp_warning)
        else:
            with self.assertWarns(exp_warning):
                glymur.codestream.Codestream._parse_siz_segment(fp)

    def test_read_past_end_of_box(self):
        """
        should warn if reading past end of a box

        Verify that a warning is issued if we read past the end of a box
        This file has a palette (pclr) box whose length is short.

        The original file tested was input/nonregression/mem-b2ace68c-1381.jp2
        """
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jp2') as ofile:
            with open(self.jpxfile, 'rb') as ifile:
                ofile.write(ifile.read(93))

                # Rewrite the ncols, nrows portion.  Increase the number of
                # rows.  This causes python to think there are more rows
                # than there actually are when resizing the palette.
                buffer = struct.pack('>HB', 257, 3)
                ofile.write(buffer)

                ifile.seek(96)
                ofile.write(ifile.read())
                ofile.flush()

            exp_warning = glymur.jp2box.UnrecoverableBoxParsingWarning
            if sys.hexversion < 0x03000000:
                with warnings.catch_warnings(record=True) as w:
                    Jp2k(ofile.name)
                assert issubclass(w[-1].category, exp_warning)
            else:
                with self.assertWarns(exp_warning):
                    Jp2k(ofile.name)

    def test_NR_gdal_fuzzer_check_number_of_tiles(self):
        """
        Has an impossible tiling setup.

        Original test file was input/nonregression
                               /gdal_fuzzer_check_number_of_tiles.jp2
        """
        fp = BytesIO()

        buffer = struct.pack('>H', 47)  # length

        # kwargs = {'rsiz': 1,
        #           'xysiz': (20, 16777236),
        #           'xyosiz': (0, 0),
        #           'xytsiz': (20, 20),
        #           'xytosiz': (0, 0),
        #           'Csiz': 3,
        #           'bitdepth': (8, 8, 8),
        #           'signed':  (False, False, False),
        #           'xyrsiz': ((1, 1, 1), (1, 1, 1)),
        #           'length': 47,
        #           'offset': 2}
        buffer += struct.pack('>HIIIIIIIIH', 1, 20, 16777236, 0, 0, 20, 20,
                              0, 0, 3)
        buffer += struct.pack('>BBBBBBBBB', 7, 1, 1, 7, 1, 1, 7, 1, 1)
        fp.write(buffer)
        fp.seek(0)

        exp_warning = glymur.codestream.InvalidNumberOfTilesWarning
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.codestream.Codestream._parse_siz_segment(fp)
            assert issubclass(w[-1].category, exp_warning)
        else:
            with self.assertWarns(exp_warning):
                glymur.codestream.Codestream._parse_siz_segment(fp)

    def test_NR_gdal_fuzzer_unchecked_numresolutions_dump(self):
        """
        Has an invalid number of resolutions.

        Original test file was input/nonregression/
                               gdal_fuzzer_unchecked_numresolutions.jp2
        """
        pargs = (0, 0, 1, 1, 64, 3, 3, 0, 0, None)
        spcod = struct.pack('>BHBBBBBB', 0, 1, 1, 64, 3, 3, 0, 0)
        spcod = bytearray(spcod)
        exp_warning = glymur.codestream.InvalidNumberOfResolutionsWarning
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.codestream.CODsegment(*pargs, length=12, offset=174)
            assert issubclass(w[-1].category, exp_warning)
        else:
            with self.assertWarns(exp_warning):
                glymur.codestream.CODsegment(*pargs, length=12, offset=174)

    def test_file_pointer_badly_positioned(self):
        """
        The file pointer should not be positioned beyond end of superbox

        Make a superbox too long by making a sub box too long.

        Original file tested was nput/nonregression/broken1.jp2
        """
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jp2') as ofile:
            with open(self.jp2file, 'rb') as ifile:

                # Write up to the colr box
                ofile.write(ifile.read(62))

                # Write a too-long color box
                buffer = struct.pack('>I4sBBBI',
                                     4194319, b'colr', 1, 0, 0, 0)
                ofile.write(buffer)

                # Write everything past the colr box.
                ifile.seek(77)
                ofile.write(ifile.read())
                ofile.flush()

            exp_warning = glymur.jp2box.FilePointerPositioningWarning
            if sys.hexversion < 0x03000000:
                with warnings.catch_warnings(record=True) as w:
                    Jp2k(ofile.name)
                assert issubclass(w[-1].category, exp_warning)
            else:
                with self.assertWarns(exp_warning):
                    Jp2k(ofile.name)

    def test_NR_DEC_issue188_beach_64bitsbox_jp2_41_decode(self):
        """
        Has an 'XML ' box instead of 'xml '.  Yes that is pedantic, but it
        really does deserve a warning.

        Original file tested was nonregression/issue188_beach_64bitsbox.jp2

        The best way to test this really is to tack a new box onto the end of
        an existing file.
        """
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.jp2') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                ofile.write(ifile.read())

                buffer = struct.pack('>I4s', 32, b'XML ')
                s = "<stuff>goes here</stuff>"
                buffer += s.encode('utf-8')
                ofile.write(buffer)
                ofile.flush()

            if sys.hexversion < 0x03000000:
                with warnings.catch_warnings(record=True) as w:
                    Jp2k(ofile.name)
                assert issubclass(w[-1].category,
                                  glymur.jp2box.UnrecognizedBoxWarning)
            else:
                with self.assertWarns(glymur.jp2box.UnrecognizedBoxWarning):
                    Jp2k(ofile.name)

    def test_truncated_icc_profile(self):
        """
        Validate a warning for a truncated ICC profile
        """
        obj = BytesIO()
        obj.write(b'\x00' * 66)

        # Write a colr box with a truncated ICC profile.
        # profile.
        buffer = struct.pack('>I4s', 47, b'colr')
        buffer += struct.pack('>BBB', 2, 0, 0)

        buffer += b'\x00' * 12 + b'scnr' + b'XYZ ' + b'Lab '
        # Need a date in bytes 24:36
        buffer += struct.pack('>HHHHHH', 1966, 2, 15, 0, 0, 0)
        obj.write(buffer)
        obj.seek(74)

        # Should be able to read the colr box now
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.jp2box.ColourSpecificationBox.parse(obj, 66, 47)
                assert issubclass(w[-1].category,
                                  InvalidICCProfileLengthWarning)
        else:
            with self.assertWarns(InvalidICCProfileLengthWarning):
                glymur.jp2box.ColourSpecificationBox.parse(obj, 66, 47)

    def test_invalid_colour_specification_method(self):
        """
        should not error out with invalid colour specification method
        """
        obj = BytesIO()
        obj.write(b'\x00' * 66)

        # Write a colr box with a bad method (254).  This requires an ICC
        # profile.
        buffer = struct.pack('>I4s', 143, b'colr')
        buffer += struct.pack('>BBB', 254, 0, 0)

        buffer += b'\x00' * 12 + b'scnr' + b'XYZ ' + b'Lab '
        # Need a date in bytes 24:36
        buffer += struct.pack('>HHHHHH', 1966, 2, 15, 0, 0, 0)
        buffer += b'\x00' * 92
        obj.write(buffer)
        obj.seek(74)

        # Should be able to read the colr box now
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.jp2box.ColourSpecificationBox.parse(obj, 66, 143)
                assert issubclass(w[-1].category, InvalidColourspaceMethod)
        else:
            with self.assertWarns(glymur.jp2box.InvalidColourspaceMethod):
                glymur.jp2box.ColourSpecificationBox.parse(obj, 66, 143)

    def test_bad_color_space_specification(self):
        """
        Verify that a warning is issued if the color space method is invalid.

        For JP2, the method must be either 1 or 2.
        """
        jp2 = glymur.Jp2k(self.jp2file)
        jp2.box[2].box[1].method = 3
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                jp2._validate()
                assert issubclass(w[-1].category,
                                  InvalidJP2ColourspaceMethodWarning)
        else:
            with self.assertWarns(InvalidJP2ColourspaceMethodWarning):
                jp2._validate()

    def test_unknown_superbox(self):
        """Verify warning for an unknown superbox."""

        with tempfile.NamedTemporaryFile(suffix='.jpx') as tfile:
            with open(self.jpxfile, 'rb') as ifile:
                tfile.write(ifile.read())

            # Add the header for an unknown superbox.
            write_buffer = struct.pack('>I4s', 20, 'grp '.encode())
            tfile.write(write_buffer)

            # Add a free box inside of it.  We won't be able to identify it,
            # but it's there.
            write_buffer = struct.pack('>I4sI', 12, 'free'.encode(), 0)
            tfile.write(write_buffer)
            tfile.flush()

            with self.assertWarns(glymur.jp2box.UnrecognizedBoxWarning):
                jpx = Jp2k(tfile.name)


if __name__ == "__main__":
    unittest.main()
