"""
Test suite for warnings issued by glymur.
"""
# Standard library imports
import codecs
import importlib.resources as ir
from io import BytesIO
import struct
import unittest
import warnings

# 3rd party library imports
import numpy as np

# Local imports
from glymur import Jp2k
import glymur
from glymur.core import COLOR, RED, GREEN, BLUE
from glymur.jp2box import InvalidJp2kError

from . import fixtures, data
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


class TestSuite(fixtures.TestCommon):

    def setUp(self):
        super(TestSuite, self).setUp()

        # Reset printoptions for every test.
        glymur.reset_option('all')

    def tearDown(self):
        super(TestSuite, self).tearDown()

        warnings.resetwarnings()
        glymur.reset_option('all')

    def test_parsing_bad_fptr_box(self):
        """
        SCENARIO: An ftyp box advertises too many bytes to be read.

        EXPECTED RESULT:  A warning is issued.  In this case we also end up
        erroring out anyway since we don't get a valid FileType box.
        """
        with ir.path(data, 'issue438.jp2') as path:
            with self.assertWarns(UserWarning):
                with self.assertRaises(InvalidJp2kError):
                    Jp2k(path)

    def test_siz_ihdr_mismatch(self):
        """
        SCENARIO:  The dimensions reported by the IHDR box don't match what is
        reported by the SIZ marker.

        EXPECTED RESULT: A warning is issued.
        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the IHDR payload
                read_buffer = ifile.read(48)
                tfile.write(read_buffer)

                # Write the bad IHDR.  The correct sequence of values read
                # should be
                # (1456, 2592, 3, 7, 7, 0, 0)
                bad_ihdr = (1600, 2592, 3, 7, 7, 0, 0)
                buffer = struct.pack('>IIHBBBB', *bad_ihdr)
                tfile.write(buffer)

                # Get the rest of the input file.
                ifile.seek(62)
                read_buffer = ifile.read()
                tfile.write(read_buffer)
                tfile.flush()

            with self.assertWarns(UserWarning):
                # c = Jp2k(tfile.name).get_codestream(header_only=False)
                Jp2k(tfile.name)

    def test_unrecognized_marker(self):
        """
        SCENARIO:  There is an unrecognized marker just after an SOT marker but
        before the EOC marker.  All markers must have a leading byte value of
        0xff.

        EXPECTED RESULT:  The SOT marker is the last one retrieved from the
        codestream.
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

            with self.assertRaises(ValueError):
                Jp2k(tfile.name).get_codestream(header_only=False)

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

        with self.assertWarns(UserWarning):
            box = glymur.jp2box.XMLBox.parse(fptr, 0, 8 + len(payload))

        self.assertIsNone(box.xml)

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

        with self.assertWarns(UserWarning):
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

        with self.assertWarns(UserWarning):
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

        with self.assertWarns(UserWarning):
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

        with self.assertWarns(UserWarning):
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

        with self.assertWarns(UserWarning):
            glymur.codestream.Codestream._parse_siz_segment(fp)

    def test_read_past_end_of_box(self):
        """
        SCENARIO:  A pclr box has more rows specified than can fit inside the
        given box length.

        EXPECTED RESULT:  A warning is issued for an attempt to read past the
        end of the box.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
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

            with self.assertWarns(UserWarning):
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

        with self.assertWarns(UserWarning):
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
        with self.assertWarns(UserWarning):
            glymur.codestream.CODsegment(*pargs, length=12, offset=174)

    def test_file_pointer_badly_positioned(self):
        """
        SCENARIO:  A colr box has an impossibly too long box length.   Since
        the colr box is the last one in the jp2h super box, this results in
        an attempt to read past the end of the super box.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
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

            with self.assertWarns(UserWarning):
                Jp2k(ofile.name)

    def test_NR_DEC_issue188_beach_64bitsbox_jp2_41_decode(self):
        """
        SCENARIO:  A JP2 file has a box with label 'XML ' instead of 'xml '.

        EXPECTED RESULT:  A warning is issued about the box being unrecognized.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                ofile.write(ifile.read())

                buffer = struct.pack('>I4s', 32, b'XML ')
                s = "<stuff>goes here</stuff>"
                buffer += s.encode('utf-8')
                ofile.write(buffer)
                ofile.flush()

            with self.assertWarns(UserWarning):
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
        with self.assertWarns(UserWarning):
            glymur.jp2box.ColourSpecificationBox.parse(obj, 66, 143)

    def test_bad_color_space_specification(self):
        """
        Verify that a warning is issued if the color space method is invalid.

        For JP2, the method must be either 1 or 2.
        """
        jp2 = glymur.Jp2k(self.jp2file)
        jp2.box[2].box[1].method = 3
        with self.assertWarns(UserWarning):
            jp2._validate()

    def test_unknown_superbox(self):
        """
        SCENARIO:  There is a superbox with an unrecognized label.

        EXPECTED RESULT:  A warning is issued.
        """

        with open(self.temp_jpx_filename, mode='wb') as tfile:
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

            with self.assertWarns(UserWarning):
                Jp2k(tfile.name)

    def test_brand_unknown(self):
        """A ftyp box brand must be 'jp2 ' or 'jpx '."""
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
        with self.assertWarns(UserWarning):
            # Bad compatibility list item.
            glymur.jp2box.FileTypeBox(compatibility_list=['jp3'])

    def test_colr_with_cspace_and_icc(self):
        """Colour specification boxes can't have both."""
        buffer = ir.read_binary(data, 'sgray.icc')

        with self.assertWarns(UserWarning):
            colorspace = glymur.core.SRGB
            glymur.jp2box.ColourSpecificationBox(colorspace=colorspace,
                                                 icc_profile=buffer)

    def test_colr_with_bad_method(self):
        """colr must have a valid method field"""
        colorspace = glymur.core.SRGB
        method = -1
        with self.assertWarns(UserWarning):
            glymur.jp2box.ColourSpecificationBox(colorspace=colorspace,
                                                 method=method)

    def test_colr_with_bad_approx(self):
        """
        SCENARIO:  An ColourSpecificationBox is given an invalid approximation
        value.

        EXPECTED RESULT:  A warning is issued.
        """
        with self.assertWarns(UserWarning):
            glymur.jp2box.ColourSpecificationBox(colorspace=glymur.core.SRGB,
                                                 approximation=-1)

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
        SCENARIO:  An xml box has invalid XML.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            bad_xml_file = tfile.name
            with open(self.jp2file, 'rb') as ifile:
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
        with self.assertWarns(UserWarning):
            glymur.jp2box.DataReferenceBox([ftyp])

    def test_flst_lens_not_the_same(self):
        """A fragment list box items must be the same length."""
        offset = [89]
        length = [1132288]
        reference = [0, 0]
        with self.assertWarns(UserWarning):
            glymur.jp2box.FragmentListBox(offset, length, reference)

    def test_flst_offsets_not_positive(self):
        """A fragment list box offsets must be positive."""
        offset = [0]
        length = [1132288]
        reference = [0]
        with self.assertWarns(UserWarning):
            glymur.jp2box.FragmentListBox(offset, length, reference)

    def test_flst_lengths_not_positive(self):
        """A fragment list box lengths must be positive."""
        offset = [89]
        length = [0]
        reference = [0]
        with self.assertWarns(UserWarning):
            glymur.jp2box.FragmentListBox(offset, length, reference)

    def test_unrecognized_exif_tag(self):
        """
        SCENARIO:  An Exif UUID box has an unrecognized tag.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as tfile:

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
        """
        SCENARIO:  A tag with an unrecognized numeric datatype field is found
        in an Exif UUID box.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as tfile:

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
        """
        SCENARIO:  An invalid TIFF header byte order marker is encountered
        in an Exif UUID box.  Only b'II' and b'MM' are allowed.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as tfile:

            with open(self.jp2file, 'rb') as ifptr:
                tfile.write(ifptr.read())

            # Write L, T, UUID identifier.
            tfile.write(struct.pack('>I4s', 52, b'uuid'))
            tfile.write(b'JpgTiffExif->JP2')

            tfile.write(b'Exif\x00\x00')

            # Here's the bad byte order.
            tfile.write(b'JI')

            # Write the rest of the header.
            xbuffer = struct.pack('<HI', 42, 8)
            tfile.write(xbuffer)

            # We will write just a single tag.
            tfile.write(struct.pack('<H', 1))

            # 271 is the Make.
            tfile.write(struct.pack('<HHI4s', 271, 2, 3, b'HTC\x00'))
            tfile.flush()

            with self.assertWarns(UserWarning):
                glymur.Jp2k(tfile.name)

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_warn_if_using_read_method(self):
        """Should warn if deprecated read method is called"""
        with self.assertWarns(DeprecationWarning):
            Jp2k(self.jp2file).read()

    def test_bad_rsiz(self):
        """
        SCENARIO:  The SIZ value parsed from the SIZ segment is invalid.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Copy up until the RSIZ value.
                ofile.write(ifile.read(3237))

                # Write the bad RSIZ value.
                buffer = struct.pack('>H', 32)
                ofile.write(buffer)
                ifile.seek(3239)

                # Get the rest of the file.
                ofile.write(ifile.read())

                ofile.seek(0)

            with self.assertWarns(UserWarning):
                Jp2k(ofile.name)

    def test_undecodeable_box_id(self):
        """
        SCENARIO:  an unknown box ID is encountered

        EXPECTED RESULT:  Should warn but not error out.
        """
        bad_box_id = b'abcd'
        with open(self.temp_jp2_filename, mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                ofile.write(ifile.read())

                # Tack an unrecognized box onto the end of nemo.
                buffer = struct.pack('>I4s', 8, bad_box_id)
                ofile.write(buffer)
                ofile.flush()

            with self.assertWarns(UserWarning):
                jp2 = Jp2k(ofile.name)

            # Now make sure we got all of the boxes.
            box_ids = [box.box_id for box in jp2.box]
            self.assertEqual(box_ids, ['jP  ', 'ftyp', 'jp2h', 'uuid', 'jp2c',
                                       'xxxx'])
            self.assertEqual(jp2.box[5].claimed_box_id, b'abcd')

    def test_bad_ftyp_brand(self):
        """
        SCENARIO:  The ftyp box has an invalid brand field.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Write the JPEG2000 signature box
                ofile.write(ifile.read(12))

                # Write a bad version of the file type box.  'jp  ' is not
                # allowed as a brand.
                buffer = struct.pack('>I4s4sI4s', 20, b'ftyp', b'jp  ', 0,
                                     b'jp2 ')
                ofile.write(buffer)

                # Write the rest of the boxes as-is.
                ifile.seek(32)
                ofile.write(ifile.read())
                ofile.flush()

            with self.assertWarns(UserWarning):
                Jp2k(ofile.name)

    def test_bad_ftyp_compatibility_list_item(self):
        """
        SCENARIO:  The ftyp box has an invalid compatibility list item.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Write the JPEG2000 signature box
                ofile.write(ifile.read(12))

                # Write a bad compatibility list item.  'jp3' is not valid.
                buffer = struct.pack('>I4s4sI4s', 20, b'ftyp', b'jp2 ', 0,
                                     b'jp3 ')
                ofile.write(buffer)

                # Write the rest of the boxes as-is.
                ifile.seek(32)
                ofile.write(ifile.read())
                ofile.flush()

            with self.assertWarns(UserWarning):
                Jp2k(ofile.name)

    def test_invalid_approximation(self):
        """
        SCENARIO:  The colr box has an invalid approximation field.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Copy the signature, file type, and jp2 header, image header
                # box as-is.
                ofile.write(ifile.read(62))

                # Write a bad version of the color specification box.  32 is an
                # invalid approximation value.
                buffer = struct.pack('>I4sBBBI', 15, b'colr', 1, 2, 32, 16)
                ofile.write(buffer)

                # Write the rest of the boxes as-is.
                ifile.seek(77)
                ofile.write(ifile.read())
                ofile.flush()

            with self.assertWarns(UserWarning):
                Jp2k(ofile.name)

    def test_invalid_colorspace(self):
        """
        SCENARIO:  A colr box has an invalid colorspace field.

        EXPECTED RESULT:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Copy the signature, file type, and jp2 header, image header
                # box as-is.
                ofile.write(ifile.read(62))

                # Write a bad version of the color specification box.  276 is
                # an invalid colorspace.
                buffer = struct.pack('>I4sBBBI', 15, b'colr', 1, 2, 0, 276)
                ofile.write(buffer)

                # Write the rest of the boxes as-is.
                ifile.seek(77)
                ofile.write(ifile.read())
                ofile.flush()

        with self.assertWarns(UserWarning):
            Jp2k(ofile.name)

    def test_stupid_windows_eol_at_end(self):
        """
        SCENARIO:  An otherwise valid JP2 file has invalid bytes appended to
        the end of the file.  The number of bytes is less than 8 because any
        more than that would be interpreted as a box.

        SCENARIO:  A warning is issued.
        """
        with open(self.temp_jp2_filename, mode='wb') as ofile:
            with open(self.jp2file, 'rb') as ifile:
                # Copy the file all the way until the end.
                ofile.write(ifile.read())

                # then append a few extra bytes
                ofile.write(b'\0')
                ofile.flush()

            with self.assertWarns(UserWarning):
                Jp2k(ofile.name)

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_NR_ENC_X_6_2K_24_FULL_CBR_CIRCLE_000_tif_17_encode(self):
        """
        SCENARIO:  Too much data is written as a Cinema2K file.

        EXPECTED RESULT:  A warning from the openjpeg library is issued.
        """
        # Need to provide the proper size image
        data = glymur.Jp2k(self.jp2file)[:]
        data = np.concatenate((data, data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:1080, :2048, :]

        with self.assertWarns(UserWarning):
            Jp2k(self.temp_jp2_filename, data=data, cinema2k=24)

    def test_deprecated_set_get_printoptions(self):
        """
        Verify deprecated get_printoptions and set_printoptions
        """
        with self.assertWarns(DeprecationWarning):
            glymur.set_printoptions(short=True)
        with self.assertWarns(DeprecationWarning):
            glymur.set_printoptions(xml=True)
        with self.assertWarns(DeprecationWarning):
            glymur.set_printoptions(codestream=True)
        with self.assertWarns(DeprecationWarning):
            glymur.get_printoptions()

    def test_deprecated_set_get_parseoption(self):
        """
        Verify deprecated get_parseoptions and set_parseoptions
        """
        with self.assertWarns(DeprecationWarning):
            glymur.set_parseoptions(full_codestream=True)
        with self.assertWarns(DeprecationWarning):
            glymur.get_parseoptions()


class TestSuiteXML(unittest.TestCase):
    """
    This test should be run on both python2 and python3.
    """
    def test_bom(self):
        """
        Byte order markers are illegal in UTF-8.  Issue 185

        Original test file was input/nonregression/issue171.jp2
        """
        fptr = BytesIO()

        buffer = b"<?xpacket "
        buffer += b"begin='" + codecs.BOM_UTF8 + b"' "
        buffer += b"id='W5M0MpCehiHzreSzNTczkc9d'?>"
        buffer += b"<stuff>goes here</stuff>"
        buffer += b"<?xpacket end='w'?>"

        fptr.write(buffer)
        num_bytes = fptr.tell()
        fptr.seek(0)

        with warnings.catch_warnings(record=True) as w:
            glymur.jp2box.XMLBox.parse(fptr, 0, 8 + num_bytes)
            self.assertEqual(len(w), 0)
