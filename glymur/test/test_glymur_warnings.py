"""
Test suite for warnings issued by glymur.
"""
import imp
from io import BytesIO
import os
import struct
import sys
import tempfile
import unittest
import warnings
import numpy as np

from glymur import Jp2k
import glymur
from glymur.jp2k import InvalidJP2ColourspaceMethodWarning
from glymur.jp2box import InvalidColourspaceMethod
from glymur.core import COLOR, RED, GREEN, BLUE

from .fixtures import WINDOWS_TMP_FILE_MSG

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch


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
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
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
                Jp2k(tfile.name)

    def test_brand_unknown(self):
        """A ftyp box brand must be 'jp2 ' or 'jpx '."""
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.jp2box.FileTypeBox(brand='jp3')
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                glymur.jp2box.FileTypeBox(brand='jp3')

    def test_bad_type(self):
        """Channel types are limited to 0, 1, 2, 65535
        Should reject if not all of index, channel_type, association the
        same length.
        """
        channel_type = (COLOR, COLOR, 3)
        association = (RED, GREEN, BLUE)
        with self.assertWarns(UserWarning):
            glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                               association=association)

    def test_wrong_lengths(self):
        """Should reject if not all of index, channel_type, association the
        same length.
        """
        channel_type = (COLOR, COLOR)
        association = (RED, GREEN, BLUE)
        with self.assertWarns(UserWarning):
            glymur.jp2box.ChannelDefinitionBox(channel_type=channel_type,
                                               association=association)

    def test_cl_entry_unknown(self):
        """A ftyp box cl list can only contain 'jp2 ', 'jpx ', or 'jpxb'."""
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True):
                # Bad compatibility list item.
                glymur.jp2box.FileTypeBox(compatibility_list=['jp3'])
        else:
            with self.assertWarns(UserWarning):
                # Bad compatibility list item.
                glymur.jp2box.FileTypeBox(compatibility_list=['jp3'])

    def test_colr_with_cspace_and_icc(self):
        """Colour specification boxes can't have both."""
        with self.assertWarns(UserWarning):
            colorspace = glymur.core.SRGB
            rawb = b'\x01\x02\x03\x04'
            glymur.jp2box.ColourSpecificationBox(colorspace=colorspace,
                                                 icc_profile=rawb)

    def test_colr_with_bad_method(self):
        """colr must have a valid method field"""
        colorspace = glymur.core.SRGB
        method = -1
        with self.assertWarns(UserWarning):
            glymur.jp2box.ColourSpecificationBox(colorspace=colorspace,
                                                 method=method)

    def test_colr_with_bad_approx(self):
        """colr should have a valid approximation field"""
        colorspace = glymur.core.SRGB
        approx = -1
        with self.assertWarns(UserWarning):
            glymur.jp2box.ColourSpecificationBox(colorspace=colorspace,
                                                 approximation=approx)

    def test_mismatched_bitdepth_signed(self):
        """bitdepth and signed arguments must have equal length"""
        palette = np.array([[255, 0, 255], [0, 255, 0]], dtype=np.uint8)
        bps = (8, 8, 8)
        signed = (False, False)
        with self.assertWarns(UserWarning):
            glymur.jp2box.PaletteBox(palette, bits_per_component=bps,
                                     signed=signed)

    def test_mismatched_signed_palette(self):
        """bitdepth and signed arguments must have equal length"""
        palette = np.array([[255, 0, 255], [0, 255, 0]], dtype=np.uint8)
        bps = (8, 8, 8, 8)
        signed = (False, False, False, False)
        with self.assertWarns(UserWarning):
            glymur.jp2box.PaletteBox(palette, bits_per_component=bps,
                                     signed=signed)

    def test_invalid_xml_box(self):
        """
        Should be able to recover info from xml box with bad xml.
        """
        jp2file = glymur.data.nemo()
        with tempfile.NamedTemporaryFile(suffix='.jp2', delete=False) as tfile:
            bad_xml_file = tfile.name
            with open(jp2file, 'rb') as ifile:
                # Everything up until the UUID box.
                write_buffer = ifile.read(77)
                tfile.write(write_buffer)

                # Write the xml box with bad xml
                # Length = 28, id is 'xml '.
                write_buffer = struct.pack('>I4s', int(28), b'xml ')
                tfile.write(write_buffer)

                write_buffer = '<test>this is a test'
                write_buffer = write_buffer.encode()
                tfile.write(write_buffer)

                # Get the rest of the input file.
                write_buffer = ifile.read()
                tfile.write(write_buffer)
                tfile.flush()

            with self.assertWarns(UserWarning):
                Jp2k(bad_xml_file)

    def test_deurl_child_of_dtbl(self):
        """
        Data reference boxes can only contain data entry url boxes.

        It's just a warning here because we haven't tried to write it.
        """
        ftyp = glymur.jp2box.FileTypeBox()
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.jp2box.DataReferenceBox([ftyp])
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                glymur.jp2box.DataReferenceBox([ftyp])

    def test_flst_lens_not_the_same(self):
        """A fragment list box items must be the same length."""
        offset = [89]
        length = [1132288]
        reference = [0, 0]
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.jp2box.FragmentListBox(offset, length, reference)
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                glymur.jp2box.FragmentListBox(offset, length, reference)

    def test_flst_offsets_not_positive(self):
        """A fragment list box offsets must be positive."""
        offset = [0]
        length = [1132288]
        reference = [0]
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.jp2box.FragmentListBox(offset, length, reference)
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                glymur.jp2box.FragmentListBox(offset, length, reference)

    def test_flst_lengths_not_positive(self):
        """A fragment list box lengths must be positive."""
        offset = [89]
        length = [0]
        reference = [0]
        if sys.hexversion < 0x03000000:
            with warnings.catch_warnings(record=True) as w:
                glymur.jp2box.FragmentListBox(offset, length, reference)
                assert issubclass(w[-1].category, UserWarning)
        else:
            with self.assertWarns(UserWarning):
                glymur.jp2box.FragmentListBox(offset, length, reference)

    def test_unrecognized_exif_tag(self):
        """Verify warning in case of unrecognized tag."""
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as tfile:

            with open(self.jp2file, 'rb') as ifptr:
                tfile.write(ifptr.read())

            # Write L, T, UUID identifier.
            tfile.write(struct.pack('>I4s', 52, b'uuid'))
            tfile.write(b'JpgTiffExif->JP2')

            tfile.write(b'Exif\x00\x00')
            xbuffer = struct.pack('<BBHI', 73, 73, 42, 8)
            tfile.write(xbuffer)

            # We will write just a single tag.
            tfile.write(struct.pack('<H', 1))

            # The "Make" tag is tag no. 271.  Corrupt it to 171.
            tfile.write(struct.pack('<HHI4s', 171, 2, 3, b'HTC\x00'))
            tfile.flush()

            with self.assertWarns(UserWarning):
                glymur.Jp2k(tfile.name)

    def test_bad_tag_datatype(self):
        """Only certain datatypes are allowable"""
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as tfile:

            with open(self.jp2file, 'rb') as ifptr:
                tfile.write(ifptr.read())

            # Write L, T, UUID identifier.
            tfile.write(struct.pack('>I4s', 52, b'uuid'))
            tfile.write(b'JpgTiffExif->JP2')

            tfile.write(b'Exif\x00\x00')
            xbuffer = struct.pack('<BBHI', 73, 73, 42, 8)
            tfile.write(xbuffer)

            # We will write just a single tag.
            tfile.write(struct.pack('<H', 1))

            # 2000 is not an allowable TIFF datatype.
            tfile.write(struct.pack('<HHI4s', 271, 2000, 3, b'HTC\x00'))
            tfile.flush()

            with self.assertWarns(UserWarning):
                glymur.Jp2k(tfile.name)

    def test_bad_tiff_header_byte_order_indication(self):
        """Only b'II' and b'MM' are allowed."""
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as tfile:

            with open(self.jp2file, 'rb') as ifptr:
                tfile.write(ifptr.read())

            # Write L, T, UUID identifier.
            tfile.write(struct.pack('>I4s', 52, b'uuid'))
            tfile.write(b'JpgTiffExif->JP2')

            tfile.write(b'Exif\x00\x00')
            xbuffer = struct.pack('<BBHI', 74, 73, 42, 8)
            tfile.write(xbuffer)

            # We will write just a single tag.
            tfile.write(struct.pack('<H', 1))

            # 271 is the Make.
            tfile.write(struct.pack('<HHI4s', 271, 2, 3, b'HTC\x00'))
            tfile.flush()

            with self.assertWarns(UserWarning):
                glymur.Jp2k(tfile.name)


@unittest.skipIf(os.name == "nt", WINDOWS_TMP_FILE_MSG)
@unittest.skipIf(sys.hexversion < 0x03020000,
                 "TemporaryDirectory introduced in 3.2.")
@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Needs openjp2 library first before these tests make sense.")
class TestConfigurationWarnings(unittest.TestCase):
    """Test suite for configuration file warnings."""

    @classmethod
    def setUpClass(cls):
        imp.reload(glymur)
        imp.reload(glymur.lib.openjp2)

    @classmethod
    def tearDownClass(cls):
        imp.reload(glymur)
        imp.reload(glymur.lib.openjp2)

    def test_xdg_env_config_file_is_bad(self):
        """A non-existant library location should be rejected."""
        with tempfile.TemporaryDirectory() as tdir:
            configdir = os.path.join(tdir, 'glymur')
            os.mkdir(configdir)
            fname = os.path.join(configdir, 'glymurrc')
            with open(fname, 'w') as fptr:
                with tempfile.NamedTemporaryFile(suffix='.dylib') as tfile:
                    fptr.write('[library]\n')
                    fptr.write('openjp2: {0}.not.there\n'.format(tfile.name))
                    fptr.flush()
                    with patch.dict('os.environ', {'XDG_CONFIG_HOME': tdir}):
                        # Warn about a bad library being rejected.
                        with self.assertWarns(UserWarning):
                            imp.reload(glymur.lib.openjp2)


if __name__ == "__main__":
    unittest.main()
