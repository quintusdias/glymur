# -*- coding:  utf-8 -*-
"""
Test suite for printing.
"""
# Standard library imports ...
import importlib.resources as ir
from io import BytesIO, StringIO
import struct
import sys
import unittest
from unittest.mock import patch
from uuid import UUID
import warnings

# Third party imports ...
import numpy as np
import lxml.etree as ET

import glymur
from glymur.codestream import LRCP, WAVELET_XFORM_5X3_REVERSIBLE
from glymur.core import COLOR, RED, GREEN, BLUE, RESTRICTED_ICC_PROFILE
from glymur.jp2box import BitsPerComponentBox, ColourSpecificationBox
from glymur.jp2box import LabelBox
from glymur import Jp2k, command_line
from glymur.lib import openjp2 as opj2
from . import fixtures, data
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


class TestPrinting(fixtures.TestCommon):
    """
    Tests for verifying how printing works.
    """
    def setUp(self):
        super(TestPrinting, self).setUp()

        # Reset printoptions for every test.
        glymur.reset_option('all')

    def tearDown(self):
        super(TestPrinting, self).tearDown()
        glymur.reset_option('all')

    def test_empty_file(self):
        """
        SCENARIO:  Print the file after with object is constructed, but
        before data is written to it.

        EXPECTED RESULT:  Just the single line.
        """
        filename = self.test_dir_path / 'a.jp2'
        actual = str(Jp2k(filename))
        expected = 'File:  a.jp2'
        self.assertEqual(actual, expected)

    def test_bad_color_specification(self):
        """
        Invalid channel type should not prevent printing.
        """
        with ir.path(data, 'issue392.jp2') as path:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                str(Jp2k(path))

    def test_palette(self):
        """
        verify printing of pclr box

        Original file tested was input/conformance/file9.jp2
        """
        palette = np.array([[0, 0, 0] for _ in range(256)], dtype=np.uint8)
        bps = (8, 8, 8)
        signed = (False, False, False)
        box = glymur.jp2box.PaletteBox(palette, bits_per_component=bps,
                                       signed=signed, length=782, offset=66)
        actual = str(box)
        expected = ('Palette Box (pclr) @ (66, 782)\n'
                    '    Size:  (256 x 3)')
        self.assertEqual(actual, expected)

        glymur.set_option('print.short', True)
        actual = str(box)
        expected = ('Palette Box (pclr) @ (66, 782)')
        self.assertEqual(actual, expected)

    def test_component_mapping_palette(self):
        """
        verify printing of cmap box tied to a palette

        Original file tested was input/conformance/file9.jp2
        """
        cmap = glymur.jp2box.ComponentMappingBox(component_index=(0, 0, 0),
                                                 mapping_type=(1, 1, 1),
                                                 palette_index=(0, 1, 2),
                                                 length=20, offset=848)
        actual = str(cmap)
        expected = ('Component Mapping Box (cmap) @ (848, 20)\n'
                    '    Component 0 ==> palette column 0\n'
                    '    Component 0 ==> palette column 1\n'
                    '    Component 0 ==> palette column 2')
        self.assertEqual(actual, expected)

    def test_component_mapping_non_palette(self):
        """
        verify printing of cmap box where there is no palette
        """
        cmap = glymur.jp2box.ComponentMappingBox(component_index=(0, 1, 2),
                                                 mapping_type=(0, 0, 0),
                                                 palette_index=(0, 0, 0),
                                                 length=20, offset=848)
        actual = str(cmap)
        expected = ('Component Mapping Box (cmap) @ (848, 20)\n'
                    '    Component 0 ==> 0\n'
                    '    Component 1 ==> 1\n'
                    '    Component 2 ==> 2')
        self.assertEqual(actual, expected)

    def test_channel_definition(self):
        """
        verify printing of cdef box

        Original file tested was input/conformance/file2.jp2
        """
        channel_type = [COLOR, COLOR, COLOR]
        association = [BLUE, GREEN, RED]
        cdef = glymur.jp2box.ChannelDefinitionBox(index=[0, 1, 2],
                                                  channel_type=channel_type,
                                                  association=association,
                                                  length=28, offset=81)
        actual = str(cdef)
        expected = ('Channel Definition Box (cdef) @ (81, 28)\n'
                    '    Channel 0 (color) ==> (3)\n'
                    '    Channel 1 (color) ==> (2)\n'
                    '    Channel 2 (color) ==> (1)')
        self.assertEqual(actual, expected)

        glymur.set_option('print.short', True)
        actual = str(cdef)
        expected = ('Channel Definition Box (cdef) @ (81, 28)')
        self.assertEqual(actual, expected)

    def test_xml(self):
        """
        SCENARIO:  JP2 file has an XML box.

        The original test file was input/conformance/file1.jp2

        EXPECTED RESULT:  The string representation of the XML box matches
        expectations.
        """
        elt = ET.fromstring(fixtures.FILE1_XML)
        xml = ET.ElementTree(elt)
        box = glymur.jp2box.XMLBox(xml=xml, length=439, offset=36)
        actual = str(box)
        expected = fixtures.FILE1_XML_BOX
        self.assertEqual(actual, expected)

    def test_xml_short_option(self):
        """
        verify printing of XML box when print.xml option set to false
        """
        elt = ET.fromstring(fixtures.FILE1_XML)
        xml = ET.ElementTree(elt)
        box = glymur.jp2box.XMLBox(xml=xml, length=439, offset=36)
        glymur.set_option('print.short', True)

        actual = str(box)
        expected = fixtures.FILE1_XML_BOX.splitlines()[0]
        self.assertEqual(actual, expected)

    def test_xml_no_xml_option(self):
        """
        verify printing of XML box when print.xml option set to false
        """
        elt = ET.fromstring(fixtures.FILE1_XML)
        xml = ET.ElementTree(elt)
        box = glymur.jp2box.XMLBox(xml=xml, length=439, offset=36)

        glymur.set_option('print.xml', False)
        actual = str(box)
        expected = fixtures.FILE1_XML_BOX.splitlines()[0]
        self.assertEqual(actual, expected)

    def test_xml_no_xml(self):
        """
        verify printing of XML box when there is no XML
        """
        box = glymur.jp2box.XMLBox()

        actual = str(box)
        expected = ("XML Box (xml ) @ (-1, 0)\n"
                    "    None")
        self.assertEqual(actual, expected)

    def test_uuid(self):
        """
        verify printing of UUID box

        Original test file was text_GBR.jp2
        """
        buuid = UUID('urn:uuid:3a0d0218-0ae9-4115-b376-4bca41ce0e71')
        box = glymur.jp2box.UUIDBox(buuid, b'\x00', 25, 1544)
        actual = str(box)
        expected = (
            'UUID Box (uuid) @ (1544, 25)\n'
            '    UUID:  3a0d0218-0ae9-4115-b376-4bca41ce0e71 (unknown)\n'
            '    UUID Data:  1 bytes')
        self.assertEqual(actual, expected)

    def test_invalid_progression_order(self):
        """
        Should still be able to print even if prog order is invalid.

        Original test file was 2977.pdf.asan.67.2198.jp2
        """
        pargs = (0, 33, 1, 1, 5, 3, 3, 0, 0, None)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            segment = glymur.codestream.CODsegment(*pargs, length=12,
                                                   offset=174)
        actual = str(segment)
        expected = fixtures.ISSUE186_PROGRESSION_ORDER
        self.assertEqual(actual, expected)

    def test_bad_wavelet_transform(self):
        """
        Should still be able to print if wavelet xform is bad, issue195

        Original test file was edf_c2_10025.jp2
        """
        pargs = (0, 0, 0, 0, 0, 0, 0, 0, 2, None)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            segment = glymur.codestream.CODsegment(*pargs, length=0, offset=0)
        str(segment)

    def test_bad_rsiz(self):
        """
        Should still be able to print if rsiz is bad, issue196

        Original test file was edf_c2_1002767.jp2
        """
        kwargs = {'rsiz': 33,
                  'xysiz': (1920, 1080),
                  'xyosiz': (0, 0),
                  'xytsiz': (1920, 1080),
                  'xytosiz': (0, 0),
                  'Csiz': 3,
                  'bitdepth': (12, 12, 12),
                  'signed': (False, False, False),
                  'xyrsiz': ((1, 1, 1), (1, 1, 1)),
                  'length': 47,
                  'offset': 2}
        segment = glymur.codestream.SIZsegment(**kwargs)
        str(segment)

    def test_invalid_approximation(self):
        """
        An invalid approximation value shouldn't cause a printing error.

        Original test file was edf_c2_1015644.jp2
        """
        kwargs = {
            'colorspace': 1,
            'precedence': 2,
            'approximation': 32,
        }
        with warnings.catch_warnings():
            # Get a warning for the bad approximation value when parsing.
            warnings.simplefilter("ignore")
            colr = ColourSpecificationBox(**kwargs)
        actual = str(colr)
        expected = ("Colour Specification Box (colr) @ (-1, 0)\n"
                    "    Method:  enumerated colorspace\n"
                    "    Precedence:  2\n"
                    "    Approximation:  invalid (32)\n"
                    "    Colorspace:  1 (unrecognized)")
        self.assertEqual(actual, expected)

    def test_invalid_colorspace(self):
        """
        SCENARIO:  An invalid colorspace shouldn't cause an error when
        printing.

        EXPECTED RESULT:  No error, although there is a warning.
        """
        with self.assertWarns(UserWarning):
            colr = ColourSpecificationBox(colorspace=276)
        str(colr)

    def test_label_box_short(self):
        """
        Test the short option for the LabelBox
        """
        box = LabelBox('test')
        glymur.set_option('print.short', True)
        actual = str(box)
        expected = "Label Box (lbl ) @ (-1, 0)"
        self.assertEqual(actual, expected)

    def test_bpcc(self):
        """
        BPCC boxes are rare :-)
        """
        bpcc = (5, 5, 5, 1)
        signed = (False, False, True, False)
        box = BitsPerComponentBox(bpcc, signed, length=12, offset=62)
        actual = str(box)

        expected = ("Bits Per Component Box (bpcc) @ (62, 12)\n"
                    "    Bits per component:  (5, 5, 5, 1)\n"
                    "    Signed:  (False, False, True, False)")

        self.assertEqual(actual, expected)

        glymur.set_option('print.short', True)
        actual = str(box)
        self.assertEqual(actual, expected.splitlines()[0])

    def test_cinema_profile(self):
        """
        Should print Cinema 2K when the profile is 3.
        """
        kwargs = {'rsiz': 3,
                  'xysiz': (1920, 1080),
                  'xyosiz': (0, 0),
                  'xytsiz': (1920, 1080),
                  'xytosiz': (0, 0),
                  'Csiz': 3,
                  'bitdepth': (12, 12, 12),
                  'signed': (False, False, False),
                  'xyrsiz': ((1, 1, 1), (1, 1, 1)),
                  'length': 47,
                  'offset': 2}
        segment = glymur.codestream.SIZsegment(**kwargs)
        actual = str(segment)

        expected = (
            "SIZ marker segment @ (2, 47)\n"
            "    Profile:  2K cinema\n"
            "    Reference Grid Height, Width:  (1080 x 1920)\n"
            "    Vertical, Horizontal Reference Grid Offset:  (0 x 0)\n"
            "    Reference Tile Height, Width:  (1080 x 1920)\n"
            "    Vertical, Horizontal Reference Tile Offset:  (0 x 0)\n"
            "    Bitdepth:  (12, 12, 12)\n"
            "    Signed:  (False, False, False)\n"
            "    Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))"
        )

        self.assertEqual(actual, expected)

    def test_version_info(self):
        """Should be able to print(glymur.version.info)"""
        str(glymur.version.info)

        self.assertTrue(True)

    def test_unknown_superbox(self):
        """
        SCENARIO:  An unknown superbox is encountered.

        EXPECTED RESULT:  str should produce a predictable result.
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

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jpx = Jp2k(tfile.name)

            glymur.set_option('print.short', True)
            actual = str(jpx.box[-1])
            expected = ("Unknown Box (xxxx) @ (1399071, 20)\n"
                        "    Claimed ID:  b'grp '")
            self.assertEqual(actual, expected)

    def test_printoptions_bad_argument(self):
        """Verify error when bad parameter to set_printoptions"""
        with self.assertRaises(KeyError):
            glymur.set_option('hi', 'low')

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_asoc_label_box(self):
        """
        SCENARIO:  A JPX file has both asoc and labl boxes.

        EXPECTED RESULT:  str representations validate
        """
        # Construct a fake file with an asoc and a label box, as
        # OpenJPEG doesn't have such a file.
        data = glymur.Jp2k(self.jp2file)[::2, ::2]

        # Create a JP2 file with only the basic JP2 boxes.
        vanilla_jp2_file = self.test_dir_path / 'tmp_test.jp2'
        glymur.Jp2k(vanilla_jp2_file, data=data)

        with open(vanilla_jp2_file, mode='rb') as tfile:
            with open(self.temp_jp2_filename, mode='wb') as tfile2:

                # Offset of the codestream is where we start.
                wbuffer = tfile.read(77)
                tfile2.write(wbuffer)

                # read the rest of the file, it's the codestream.
                codestream = tfile.read()

                # Write the asoc superbox.
                # Length = 36, id is 'asoc'.
                wbuffer = struct.pack('>I4s', int(56), b'asoc')
                tfile2.write(wbuffer)

                # Write the contained label box
                wbuffer = struct.pack('>I4s', int(13), b'lbl ')
                tfile2.write(wbuffer)
                tfile2.write('label'.encode())

                # Write the xml box
                # Length = 36, id is 'xml '.
                wbuffer = struct.pack('>I4s', int(35), b'xml ')
                tfile2.write(wbuffer)

                wbuffer = '<test>this is a test</test>'
                wbuffer = wbuffer.encode()
                tfile2.write(wbuffer)

                # Now append the codestream.
                tfile2.write(codestream)
                tfile2.flush()

                jasoc = glymur.Jp2k(tfile2.name)
                actual = str(jasoc.box[3])
                expected = ('Association Box (asoc) @ (77, 56)\n'
                            '    Label Box (lbl ) @ (85, 13)\n'
                            '        Label:  label\n'
                            '    XML Box (xml ) @ (98, 35)\n'
                            '        <test>this is a test</test>')
                self.assertEqual(actual, expected)

    def test_coc_segment(self):
        """verify printing of COC segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        actual = str(codestream.segment[6])

        exp = ('COC marker segment @ (3356, 9)\n'
               '    Associated component:  1\n'
               '    Coding style for this component:  '
               'Entropy coder, PARTITION = 0\n'
               '    Coding style parameters:\n'
               '        Number of decomposition levels:  1\n'
               '        Code block height, width:  (64 x 64)\n'
               '        Wavelet transform:  5-3 reversible\n'
               '        Precinct size:  (32768, 32768)\n'
               '        Code block context:\n'
               '            Selective arithmetic coding bypass:  False\n'
               '            Reset context probabilities '
               'on coding pass boundaries:  False\n'
               '            Termination on each coding pass:  False\n'
               '            Vertically stripe causal context:  False\n'
               '            Predictable termination:  False\n'
               '            Segmentation symbols:  False')

        self.assertEqual(actual, exp)

    def test_cod_segment_unknown(self):
        """
        Verify printing of transform when it's actually unknown
        """
        scod = 0
        prog_order = LRCP
        num_layers = 2
        mct = 4
        nr = 1
        xcb = ycb = 4
        cstyle = 0
        xform = WAVELET_XFORM_5X3_REVERSIBLE
        precinct_size = None
        length = 12
        offset = 3282
        pargs = (scod, prog_order, num_layers, mct, nr, xcb, ycb, cstyle,
                 xform, precinct_size, length, offset)
        segment = glymur.codestream.CODsegment(*pargs)
        actual = str(segment)
        exp = ('COD marker segment @ (3282, 12)\n'
               '    Coding style:\n'
               '        Entropy coder, without partitions\n'
               '        SOP marker segments:  False\n'
               '        EPH marker segments:  False\n'
               '    Coding style parameters:\n'
               '        Progression order:  LRCP\n'
               '        Number of layers:  2\n'
               '        Multiple component transformation usage:  unknown\n'
               '        Number of decomposition levels:  1\n'
               '        Code block height, width:  (64 x 64)\n'
               '        Wavelet transform:  5-3 reversible\n'
               '        Precinct size:  (32768, 32768)\n'
               '        Code block context:\n'
               '            Selective arithmetic coding bypass:  False\n'
               '            Reset context probabilities on coding '
               'pass boundaries:  False\n'
               '            Termination on each coding pass:  False\n'
               '            Vertically stripe causal context:  False\n'
               '            Predictable termination:  False\n'
               '            Segmentation symbols:  False')

        self.assertEqual(actual, exp)

    def test_cod_segment_irreversible(self):
        """
        Verify printing of irreversible transform
        """
        scod = 0
        prog_order = LRCP
        num_layers = 2
        mct = 2
        nr = 1
        xcb = ycb = 4
        cstyle = 0
        xform = WAVELET_XFORM_5X3_REVERSIBLE
        precinct_size = None
        length = 12
        offset = 3282
        pargs = (scod, prog_order, num_layers, mct, nr, xcb, ycb, cstyle,
                 xform, precinct_size, length, offset)
        segment = glymur.codestream.CODsegment(*pargs)
        actual = str(segment)
        exp = ('COD marker segment @ (3282, 12)\n'
               '    Coding style:\n'
               '        Entropy coder, without partitions\n'
               '        SOP marker segments:  False\n'
               '        EPH marker segments:  False\n'
               '    Coding style parameters:\n'
               '        Progression order:  LRCP\n'
               '        Number of layers:  2\n'
               '        Multiple component transformation usage:  '
               'irreversible\n'
               '        Number of decomposition levels:  1\n'
               '        Code block height, width:  (64 x 64)\n'
               '        Wavelet transform:  5-3 reversible\n'
               '        Precinct size:  (32768, 32768)\n'
               '        Code block context:\n'
               '            Selective arithmetic coding bypass:  False\n'
               '            Reset context probabilities on coding '
               'pass boundaries:  False\n'
               '            Termination on each coding pass:  False\n'
               '            Vertically stripe causal context:  False\n'
               '            Predictable termination:  False\n'
               '            Segmentation symbols:  False')

        self.assertEqual(actual, exp)

    def test_cod_segment(self):
        """verify printing of COD segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        actual = str(codestream.segment[2])

        exp = ('COD marker segment @ (3282, 12)\n'
               '    Coding style:\n'
               '        Entropy coder, without partitions\n'
               '        SOP marker segments:  False\n'
               '        EPH marker segments:  False\n'
               '    Coding style parameters:\n'
               '        Progression order:  LRCP\n'
               '        Number of layers:  2\n'
               '        Multiple component transformation usage:  '
               'reversible\n'
               '        Number of decomposition levels:  1\n'
               '        Code block height, width:  (64 x 64)\n'
               '        Wavelet transform:  5-3 reversible\n'
               '        Precinct size:  (32768, 32768)\n'
               '        Code block context:\n'
               '            Selective arithmetic coding bypass:  False\n'
               '            Reset context probabilities on coding '
               'pass boundaries:  False\n'
               '            Termination on each coding pass:  False\n'
               '            Vertically stripe causal context:  False\n'
               '            Predictable termination:  False\n'
               '            Segmentation symbols:  False')

        self.assertEqual(actual, exp)

    def test_eoc_segment(self):
        """verify printing of eoc segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        actual = str(codestream.segment[-1])

        expected = 'EOC marker segment @ (1135517, 0)'
        self.assertEqual(actual, expected)

    def test_qcc_segment(self):
        """verify printing of qcc segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        actual = str(codestream.segment[7])

        expected = ('QCC marker segment @ (3367, 8)\n'
                    '    Associated Component:  1\n'
                    '    Quantization style:  no quantization, 2 guard bits\n'
                    '    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]')

        self.assertEqual(actual, expected)

    def test_qcd_segment_5x3_transform(self):
        """verify printing of qcd segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        actual = str(codestream.segment[3])

        expected = ('QCD marker segment @ (3296, 7)\n'
                    '    Quantization style:  no quantization, 2 guard bits\n'
                    '    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]')

        self.assertEqual(actual, expected)

    def test_siz_segment(self):
        """verify printing of SIZ segment"""
        j = glymur.Jp2k(self.jp2file)
        actual = str(j.codestream.segment[1])

        exp = ('SIZ marker segment @ (3233, 47)\n'
               '    Profile:  no profile\n'
               '    Reference Grid Height, Width:  (1456 x 2592)\n'
               '    Vertical, Horizontal Reference Grid Offset:  (0 x 0)\n'
               '    Reference Tile Height, Width:  (1456 x 2592)\n'
               '    Vertical, Horizontal Reference Tile Offset:  (0 x 0)\n'
               '    Bitdepth:  (8, 8, 8)\n'
               '    Signed:  (False, False, False)\n'
               '    Vertical, Horizontal Subsampling:  '
               '((1, 1), (1, 1), (1, 1))')

        self.assertEqual(actual, exp)

    def test_soc_segment(self):
        """verify printing of SOC segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        actual = str(codestream.segment[0])

        expected = 'SOC marker segment @ (3231, 0)'
        self.assertEqual(actual, expected)

    def test_sod_segment(self):
        """verify printing of SOD segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        actual = str(codestream.segment[10])

        expected = 'SOD marker segment @ (3398, 0)'
        self.assertEqual(actual, expected)

    def test_sot_segment(self):
        """verify printing of SOT segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        actual = str(codestream.segment[5])

        expected = ('SOT marker segment @ (3344, 10)\n'
                    '    Tile part index:  0\n'
                    '    Tile part length:  1132173\n'
                    '    Tile part instance:  0\n'
                    '    Number of tile parts:  1')

        self.assertEqual(actual, expected)

    def test_xmp(self):
        """
        Verify the printing of a UUID/XMP box.
        """
        j = glymur.Jp2k(self.jp2file)
        actual = str(j.box[3])

        expected = fixtures.NEMO_XMP_BOX
        self.assertEqual(actual, expected)

    def test_codestream(self):
        """
        verify printing of entire codestream
        """
        j = glymur.Jp2k(self.jp2file)
        actual = str(j.get_codestream())
        exp = ('Codestream:\n'
               '    SOC marker segment @ (3231, 0)\n'
               '    SIZ marker segment @ (3233, 47)\n'
               '        Profile:  no profile\n'
               '        Reference Grid Height, Width:  (1456 x 2592)\n'
               '        Vertical, Horizontal Reference Grid Offset:  (0 x 0)\n'
               '        Reference Tile Height, Width:  (1456 x 2592)\n'
               '        Vertical, Horizontal Reference Tile Offset:  (0 x 0)\n'
               '        Bitdepth:  (8, 8, 8)\n'
               '        Signed:  (False, False, False)\n'
               '        Vertical, Horizontal Subsampling:  '
               '((1, 1), (1, 1), (1, 1))\n'
               '    COD marker segment @ (3282, 12)\n'
               '        Coding style:\n'
               '            Entropy coder, without partitions\n'
               '            SOP marker segments:  False\n'
               '            EPH marker segments:  False\n'
               '        Coding style parameters:\n'
               '            Progression order:  LRCP\n'
               '            Number of layers:  2\n'
               '            Multiple component transformation usage:  '
               'reversible\n'
               '            Number of decomposition levels:  1\n'
               '            Code block height, width:  (64 x 64)\n'
               '            Wavelet transform:  5-3 reversible\n'
               '            Precinct size:  (32768, 32768)\n'
               '            Code block context:\n'
               '                Selective arithmetic coding bypass:  False\n'
               '                Reset context probabilities on '
               'coding pass boundaries:  False\n'
               '                Termination on each coding pass:  False\n'
               '                Vertically stripe causal context:  False\n'
               '                Predictable termination:  False\n'
               '                Segmentation symbols:  False\n'
               '    QCD marker segment @ (3296, 7)\n'
               '        Quantization style:  no quantization, '
               '2 guard bits\n'
               '        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]\n'
               '    CME marker segment @ (3305, 37)\n'
               '        "Created by OpenJPEG version 2.0.0"')
        self.assertEqual(actual, exp)

    def test_xml_latin1(self):
        """Should be able to print an XMLBox with utf-8 encoding (latin1)."""
        # Seems to be inconsistencies between different versions of python2.x
        # as to what gets printed.
        #
        # 2.7.5 (fedora 19) prints xml entities.
        # 2.7.3 seems to want to print hex escapes.
        text = u"""<flow>Strömung</flow>"""
        xml = ET.parse(StringIO(text))

        xmlbox = glymur.jp2box.XMLBox(xml=xml)
        actual = str(xmlbox)
        expected = ("XML Box (xml ) @ (-1, 0)\n"
                    "    <flow>Strömung</flow>")
        self.assertEqual(actual, expected)

    def test_xml_cyrrilic(self):
        """Should be able to print XMLBox with utf-8 encoding (cyrrillic)."""
        # Seems to be inconsistencies between different versions of python2.x
        # as to what gets printed.
        #
        # 2.7.5 (fedora 19) prints xml entities.
        # 2.7.3 seems to want to print hex escapes.
        text = u"""<country>Россия</country>"""
        xml = ET.parse(StringIO(text))

        xmlbox = glymur.jp2box.XMLBox(xml=xml)
        actual = str(xmlbox)
        expected = ("XML Box (xml ) @ (-1, 0)\n"
                    "    <country>Россия</country>")

        self.assertEqual(actual, expected)

    def test_less_common_boxes(self):
        """verify uinf, ulst, url, res, resd, resc box printing"""
        with open(self.temp_jp2_filename, mode='wb') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                wbuffer = ifile.read(77)
                tfile.write(wbuffer)

                # Write the UINF superbox
                # Length = 50, id is uinf.
                wbuffer = struct.pack('>I4s', int(50), b'uinf')
                tfile.write(wbuffer)

                # Write the ULST box.
                # Length is 26, 1 UUID, hard code that UUID as zeros.
                wbuffer = struct.pack('>I4sHIIII', int(26), b'ulst', int(1),
                                      int(0), int(0), int(0), int(0))
                tfile.write(wbuffer)

                # Write the URL box.
                # Length is 16, version is one byte, flag is 3 bytes, url
                # is the rest.
                wbuffer = struct.pack('>I4sBBBB',
                                      int(16), b'url ',
                                      int(0), int(0), int(0), int(0))
                tfile.write(wbuffer)

                wbuffer = struct.pack('>ssss', b'a', b'b', b'c', b'd')
                tfile.write(wbuffer)

                # Start the resolution superbox.
                wbuffer = struct.pack('>I4s', int(44), b'res ')
                tfile.write(wbuffer)

                # Write the capture resolution box.
                wbuffer = struct.pack('>I4sHHHHBB',
                                      int(18), b'resc',
                                      int(1), int(1), int(1), int(1),
                                      int(0), int(1))
                tfile.write(wbuffer)

                # Write the display resolution box.
                wbuffer = struct.pack('>I4sHHHHBB',
                                      int(18), b'resd',
                                      int(1), int(1), int(1), int(1),
                                      int(1), int(0))
                tfile.write(wbuffer)

                # Get the rest of the input file.
                wbuffer = ifile.read()
                tfile.write(wbuffer)
                tfile.flush()

            jp2k = glymur.Jp2k(tfile.name)
            with patch('sys.stdout', new=StringIO()) as stdout:
                print(jp2k.box[3])
                print(jp2k.box[4])
                actual = stdout.getvalue().strip()
            exp = ('UUIDInfo Box (uinf) @ (77, 50)\n'
                   '    UUID List Box (ulst) @ (85, 26)\n'
                   '        UUID[0]:  00000000-0000-0000-0000-000000000000\n'
                   '    Data Entry URL Box (url ) @ (111, 16)\n'
                   '        Version:  0\n'
                   '        Flag:  0 0 0\n'
                   '        URL:  "abcd"\n'
                   'Resolution Box (res ) @ (127, 44)\n'
                   '    Capture Resolution Box (resc) @ (135, 18)\n'
                   '        VCR:  1.0\n'
                   '        HCR:  10.0\n'
                   '    Display Resolution Box (resd) @ (153, 18)\n'
                   '        VDR:  10.0\n'
                   '        HDR:  1.0')

            self.assertEqual(actual, exp)

            glymur.set_option('print.short', True)
            with patch('sys.stdout', new=StringIO()) as stdout:
                print(jp2k.box[3])
                print(jp2k.box[4])
                actual = stdout.getvalue().strip()
            exp = ('UUIDInfo Box (uinf) @ (77, 50)\n'
                   '    UUID List Box (ulst) @ (85, 26)\n'
                   '    Data Entry URL Box (url ) @ (111, 16)\n'
                   'Resolution Box (res ) @ (127, 44)\n'
                   '    Capture Resolution Box (resc) @ (135, 18)\n'
                   '    Display Resolution Box (resd) @ (153, 18)')

            self.assertEqual(actual, exp)

    def test_flst(self):
        """Verify printing of fragment list box."""
        flst = glymur.jp2box.FragmentListBox([89], [1132288], [0])
        actual = str(flst)
        expected = ("Fragment List Box (flst) @ (-1, 0)\n"
                    "    Offset 0:  89\n"
                    "    Fragment Length 0:  1132288\n"
                    "    Data Reference 0:  0")
        self.assertEqual(actual, expected)

        glymur.set_option('print.short', True)
        actual = str(flst)
        self.assertEqual(actual, expected.splitlines()[0])

    def test_dref(self):
        """Verify printing of data reference box."""

        version = 0
        flag = (0, 0, 0)
        url = "http://readthedocs.glymur.org"
        deu = glymur.jp2box.DataEntryURLBox(version, flag, url)

        dref = glymur.jp2box.DataReferenceBox([deu])
        actual = str(dref)
        expected = ("Data Reference Box (dtbl) @ (-1, 0)\n"
                    "    Data Entry URL Box (url ) @ (-1, 0)\n"
                    "        Version:  0\n"
                    "        Flag:  0 0 0\n"
                    '        URL:  "http://readthedocs.glymur.org"')
        self.assertEqual(actual, expected)

        # Test the short version.
        glymur.set_option('print.short', True)
        actual = str(dref)
        self.assertEqual(actual, 'Data Reference Box (dtbl) @ (-1, 0)')

    def test_empty_dref(self):
        """Verify printing of data reference box with no content."""

        dref = glymur.jp2box.DataReferenceBox()
        actual = str(dref)
        expected = "Data Reference Box (dtbl) @ (-1, 0)"
        self.assertEqual(actual, expected)

    def test_jplh_cgrp(self):
        """Verify printing of compositing layer header box, color group box."""
        jpx = glymur.Jp2k(self.jpxfile)
        actual = str(jpx.box[7])

        expected = (
            "Compositing Layer Header Box (jplh) @ (314227, 31)\n"
            "    Colour Group Box (cgrp) @ (314235, 23)\n"
            "        Colour Specification Box (colr) @ (314243, 15)\n"
            "            Method:  enumerated colorspace\n"
            "            Precedence:  0\n"
            "            Colorspace:  sRGB"
        )

        self.assertEqual(actual, expected)

    def test_free(self):
        """Verify printing of Free box."""
        free = glymur.jp2box.FreeBox()
        actual = str(free)
        self.assertEqual(actual, 'Free Box (free) @ (-1, 0)')

    def test_nlst(self):
        """Verify printing of number list box."""
        assn = (0, 16777216, 33554432, 50331648)
        nlst = glymur.jp2box.NumberListBox(assn)

        actual = str(nlst)
        expected = ("Number List Box (nlst) @ (-1, 0)\n"
                    "    Association[0]:  the rendered result\n"
                    "    Association[1]:  codestream 0\n"
                    "    Association[2]:  compositing layer 0\n"
                    "    Association[3]:  unrecognized")

        self.assertEqual(actual, expected)

    def test_nlst_short(self):
        glymur.set_option('print.short', True)

        assn = (0, 16777216, 33554432)
        nlst = glymur.jp2box.NumberListBox(assn)

        actual = str(nlst)
        expected = "Number List Box (nlst) @ (-1, 0)"
        self.assertEqual(actual, expected)

    def test_ftbl(self):
        """Verify printing of fragment table box."""
        flst = glymur.jp2box.FragmentListBox([89], [1132288], [0])
        ftbl = glymur.jp2box.FragmentTableBox([flst])
        actual = str(ftbl)

        expected = ("Fragment Table Box (ftbl) @ (-1, 0)\n"
                    "    Fragment List Box (flst) @ (-1, 0)\n"
                    "        Offset 0:  89\n"
                    "        Fragment Length 0:  1132288\n"
                    "        Data Reference 0:  0")
        self.assertEqual(actual, expected)

    def test_jpch(self):
        """Verify printing of JPCH box."""
        jpx = glymur.Jp2k(self.jpxfile)
        actual = str(jpx.box[3])
        self.assertEqual(actual, 'Codestream Header Box (jpch) @ (887, 8)')

    def test_exif_uuid(self):
        """
        SCENARIO:  A JP2 file has an Exif UUID box.

        EXPECTED RESULT:  Verify printing of Exif information.
        """
        with open(self.temp_jp2_filename, mode='wb') as tfile:

            with open(self.jp2file, 'rb') as ifptr:
                tfile.write(ifptr.read())

            # Write L, T, UUID identifier.
            tfile.write(struct.pack('>I4s', 76, b'uuid'))
            tfile.write(b'JpgTiffExif->JP2')

            tfile.write(b'Exif\x00\x00')
            xbuffer = struct.pack('<BBHI', 73, 73, 42, 8)
            tfile.write(xbuffer)

            # We will write just three tags.
            tfile.write(struct.pack('<H', 3))

            # The "Make" tag is tag no. 271.
            tfile.write(struct.pack('<HHII', 256, 4, 1, 256))
            tfile.write(struct.pack('<HHII', 257, 4, 1, 512))
            tfile.write(struct.pack('<HHI4s', 271, 2, 3, b'HTC\x00'))
            tfile.flush()

            j = glymur.Jp2k(tfile.name)

            actual = str(j.box[5])

        expected = (
            "UUID Box (uuid) @ (1135519, 76)\n"
            "    UUID:  4a706754-6966-6645-7869-662d3e4a5032 (EXIF)\n"
            "    UUID Data:  OrderedDict([('ImageWidth', 256), "
            "('ImageLength', 512), ('Make', 'HTC')])"
        )
        self.assertEqual(actual, expected)

    def test_crg(self):
        """verify printing of CRG segment"""
        crg = glymur.codestream.CRGsegment((65535,), (32767,), 6, 87)
        actual = str(crg)
        expected = ('CRG marker segment @ (87, 6)\n'
                    '    Vertical, Horizontal offset:  (0.50, 1.00)')
        self.assertEqual(actual, expected)

    def test_rgn(self):
        """
        verify printing of RGN segment
        """
        segment = glymur.codestream.RGNsegment(0, 0, 7, 5, 310)
        actual = str(segment)
        expected = ('RGN marker segment @ (310, 5)\n'
                    '    Associated component:  0\n'
                    '    ROI style:  0\n'
                    '    Parameter:  7')
        self.assertEqual(actual, expected)

    def test_sop(self):
        """
        verify printing of SOP segment
        """
        segment = glymur.codestream.SOPsegment(15, 4, 12836)
        actual = str(segment)
        expected = ('SOP marker segment @ (12836, 4)\n'
                    '    Nsop:  15')
        self.assertEqual(actual, expected)

    def test_cme(self):
        """
        Test printing a CME or comment marker segment.

        Originally tested with input/conformance/p0_02.j2k
        """
        buffer = "Creator: AV-J2K (c) 2000,2001 Algo Vision".encode('latin-1')
        segment = glymur.codestream.CMEsegment(1, buffer, 45, 85)
        actual = str(segment)
        expected = ('CME marker segment @ (85, 45)\n'
                    '    "Creator: AV-J2K (c) 2000,2001 Algo Vision"')
        self.assertEqual(actual, expected)

    def test_plt_segment(self):
        """
        verify printing of PLT segment

        Originally tested with input/conformance/p0_07.j2k
        """
        pkt_lengths = [9, 122, 19, 30, 27, 9, 41, 62, 18, 29, 261,
                       55, 82, 299, 93, 941, 951, 687, 1729, 1443, 1008, 2168,
                       2188, 2223]
        segment = glymur.codestream.PLTsegment(0, pkt_lengths, 38, 7871146)

        actual = str(segment)

        exp = ('PLT marker segment @ (7871146, 38)\n'
               '    Index:  0\n'
               '    Iplt:  [9, 122, 19, 30, 27, 9, 41, 62, 18, 29, 261,'
               ' 55, 82, 299, 93, 941, 951, 687, 1729, 1443, 1008, 2168,'
               ' 2188, 2223]')
        self.assertEqual(actual, exp)

    def test_invalid_pod_segment(self):
        """
        SCENARIO:  A progression order parameter is out of range.

        EXPECTED RESPONSE:  Should not error out.  The invalid progression
        order should be clearly displayed.
        """
        params = (0, 0, 1, 33, 128, 1, 0, 128, 1, 33, 257, 16)
        segment = glymur.codestream.PODsegment(params, 20, 878)
        actual = str(segment)

        expected = (
            'POD marker segment @ (878, 20)\n'
            '    Progression change 0:\n'
            '        Resolution index start:  0\n'
            '        Component index start:  0\n'
            '        Layer index end:  1\n'
            '        Resolution index end:  33\n'
            '        Component index end:  128\n'
            '        Progression order:  RLCP\n'
            '    Progression change 1:\n'
            '        Resolution index start:  0\n'
            '        Component index start:  128\n'
            '        Layer index end:  1\n'
            '        Resolution index end:  33\n'
            '        Component index end:  257\n'
            '        Progression order:  invalid value: 16'
        )

        self.assertEqual(actual, expected)

    def test_pod_segment(self):
        """
        verify printing of POD segment

        Original test file was input/conformance/p0_13.j2k
        """
        params = (0, 0, 1, 33, 128, 1, 0, 128, 1, 33, 257, 4)
        segment = glymur.codestream.PODsegment(params, 20, 878)
        actual = str(segment)

        expected = ('POD marker segment @ (878, 20)\n'
                    '    Progression change 0:\n'
                    '        Resolution index start:  0\n'
                    '        Component index start:  0\n'
                    '        Layer index end:  1\n'
                    '        Resolution index end:  33\n'
                    '        Component index end:  128\n'
                    '        Progression order:  RLCP\n'
                    '    Progression change 1:\n'
                    '        Resolution index start:  0\n'
                    '        Component index start:  128\n'
                    '        Layer index end:  1\n'
                    '        Resolution index end:  33\n'
                    '        Component index end:  257\n'
                    '        Progression order:  CPRL')

        self.assertEqual(actual, expected)

    def test_ppm_segment(self):
        """
        verify printing of PPM segment

        Original file tested was input/conformance/p1_03.j2k
        """
        segment = glymur.codestream.PPMsegment(0, b'\0' * 43709, 43712, 213)
        actual = str(segment)

        expected = ('PPM marker segment @ (213, 43712)\n'
                    '    Index:  0\n'
                    '    Data:  43709 uninterpreted bytes')

        self.assertEqual(actual, expected)

    def test_ppt_segment(self):
        """
        verify printing of ppt segment

        Original file tested was input/conformance/p1_06.j2k
        """
        segment = glymur.codestream.PPTsegment(0, b'\0' * 106, 109, 155)
        actual = str(segment)

        expected = ('PPT marker segment @ (155, 109)\n'
                    '    Index:  0\n'
                    '    Packet headers:  106 uninterpreted bytes')

        self.assertEqual(actual, expected)

    def test_tlm_segment(self):
        """
        verify printing of TLM segment

        Original file tested was input/conformance/p0_15.j2k
        """
        segment = glymur.codestream.TLMsegment(0,
                                               (0, 1, 2, 3),
                                               (4267, 2117, 4080, 2081),
                                               28, 268)
        actual = str(segment)

        expected = ('TLM marker segment @ (268, 28)\n'
                    '    Index:  0\n'
                    '    Tile number:  (0, 1, 2, 3)\n'
                    '    Length:  (4267, 2117, 4080, 2081)')

        self.assertEqual(actual, expected)

    def test_differing_subsamples(self):
        """
        verify printing of SIZ with different subsampling... Issue 86.
        """
        kwargs = {'rsiz': 1,
                  'xysiz': (1024, 1024),
                  'xyosiz': (0, 0),
                  'xytsiz': (1024, 1024),
                  'xytosiz': (0, 0),
                  'Csiz': 4,
                  'bitdepth': (8, 8, 8, 8),
                  'signed': (False, False, False, False),
                  'xyrsiz': ((1, 1, 2, 2), (1, 1, 2, 2)),
                  'length': 50,
                  'offset': 2}
        segment = glymur.codestream.SIZsegment(**kwargs)
        actual = str(segment)
        exp = ('SIZ marker segment @ (2, 50)\n'
               '    Profile:  0\n'
               '    Reference Grid Height, Width:  (1024 x 1024)\n'
               '    Vertical, Horizontal Reference Grid Offset:  (0 x 0)\n'
               '    Reference Tile Height, Width:  (1024 x 1024)\n'
               '    Vertical, Horizontal Reference Tile Offset:  (0 x 0)\n'
               '    Bitdepth:  (8, 8, 8, 8)\n'
               '    Signed:  (False, False, False, False)\n'
               '    Vertical, Horizontal Subsampling:  '
               '((1, 1), (1, 1), (2, 2), (2, 2))')
        self.assertEqual(actual, exp)

    def test_issue182(self):
        """
        SCENARIO: Print a component mapping box.

        This is a regression test.

        Original file tested was input/nonregression/mem-b2ace68c-1381.jp2

        EXPECTED RESULT:  Format strings like %d should not appear in the
        output.
        """
        cmap = glymur.jp2box.ComponentMappingBox(component_index=(0, 0, 0, 0),
                                                 mapping_type=(1, 1, 1, 1),
                                                 palette_index=(0, 1, 2, 3),
                                                 length=24, offset=130)
        actual = str(cmap)
        expected = ("Component Mapping Box (cmap) @ (130, 24)\n"
                    "    Component 0 ==> palette column 0\n"
                    "    Component 0 ==> palette column 1\n"
                    "    Component 0 ==> palette column 2\n"
                    "    Component 0 ==> palette column 3")
        self.assertEqual(actual, expected)

        glymur.set_option('print.short', True)
        actual = str(cmap)
        expected = expected.splitlines()[0]
        self.assertEqual(actual, expected)

    def test_issue183(self):
        """
        SCENARIO:  An ICC profile is present in a ColourSpecificationBox, but
        it is invalid.

        EXPECTED RESULT:  The printed representation validates.
        """
        colr = ColourSpecificationBox(method=RESTRICTED_ICC_PROFILE,
                                      icc_profile=None, length=12, offset=62)

        actual = str(colr)
        expected = ("Colour Specification Box (colr) @ (62, 12)\n"
                    "    Method:  restricted ICC profile\n"
                    "    Precedence:  0\n"
                    "    ICC Profile:  None")

        self.assertEqual(actual, expected)

    def test_icc_profile(self):
        """
        SCENARIO:  print a colr box with an ICC profile

        EXPECTED RESULT:  validate the string representation
        """
        with ir.path(data, 'text_GBR.jp2') as path:
            with self.assertWarns(UserWarning):
                # The brand is wrong, this is JPX, not JP2.
                j = Jp2k(path)
        box = j.box[3].box[1]
        actual = str(box)
        # Don't bother verifying the OrderedDict part of the colr box.
        # OrderedDicts are brittle print-wise.
        actual = actual.split('\n')[:5]
        actual = '\n'.join(actual)
        expected = (
            "Colour Specification Box (colr) @ (179, 1339)\n"
            "    Method:  any ICC profile\n"
            "    Precedence:  2\n"
            "    Approximation:  "
            "accurately represents correct colorspace definition\n"
            "    ICC Profile:"
        )
        self.assertEqual(actual, expected)

    def test_rreq(self):
        """
        verify printing of reader requirements box

        Original file tested was text_GBR.jp2
        """

        fuam = 0xffff
        dcm = 0xf8f0
        standard_flag = 1, 5, 12, 18, 44
        standard_mask = 0x8000, 0x4080, 0x2040, 0x1020, 0x810
        vendor_feature = [UUID('{3a0d0218-0ae9-4115-b376-4bca41ce0e71}')]
        vendor_feature.append(UUID('{47c92ccc-d1a1-4581-b904-38bb5467713b}'))
        vendor_feature.append(UUID('{bc45a774-dd50-4ec6-a9f6-f3a137f47e90}'))
        vendor_feature.append(UUID('{d7c8c5ef-951f-43b2-8757-042500f538e8}'))
        vendor_mask = 0,
        box = glymur.jp2box.ReaderRequirementsBox(fuam, dcm, standard_flag,
                                                  standard_mask,
                                                  vendor_feature, vendor_mask,
                                                  length=109, offset=40)
        actual = str(box)
        self.assertEqual(actual, fixtures.TEXT_GBR_RREQ)

        glymur.set_option('print.short', True)
        actual = str(box)
        self.assertEqual(actual, fixtures.TEXT_GBR_RREQ.splitlines()[0])

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            box = glymur.jp2box.XMLBox.parse(fptr, 0, 8 + len(data))
            # No need to verify, it's enough that we don't error out.
            str(box)

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_precincts(self):
        """
        SCENARIO:  print the first COD segment

        EXPECTED RESULT:  the precinct information validates predetermined
        values
        """
        data = Jp2k(self.jp2file)[:]
        j = Jp2k(self.temp_j2k_filename, data=data, psizes=[(128, 128)] * 3)

        # Should be three layers.
        codestream = j.get_codestream()

        actual = str(codestream.segment[2])
        expected = fixtures.MULTIPLE_PRECINCT_SIZE
        self.assertEqual(actual, expected)

    def test_old_short_option(self):
        """
        Verify printing with deprecated set_printoptions "short"
        """
        jp2 = Jp2k(self.jp2file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            glymur.set_printoptions(short=True)

        actual = str(jp2)

        # Get rid of leading "File" line, as that is volatile.
        actual = '\n'.join(actual.splitlines()[1:])

        expected = fixtures.NEMO_DUMP_SHORT
        self.assertEqual(actual, expected)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            opt = glymur.get_printoptions()['short']
        self.assertTrue(opt)

    def test_suppress_xml_old_option(self):
        """
        Verify printing with xml suppressed, deprecated method
        """
        jp2 = Jp2k(self.jp2file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            glymur.set_printoptions(xml=False)

        actual = str(jp2)

        # Get rid of leading "File" line, as that is volatile.
        actual = '\n'.join(actual.splitlines()[1:])

        # shave off the XML and non-main-header segments
        expected = fixtures.NEMO_DUMP_NO_XML
        self.assertEqual(actual, expected)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            opt = glymur.get_printoptions()['xml']
        self.assertFalse(opt)

    def test_suppress_xml(self):
        """
        Verify printing with xml suppressed
        """
        jp2 = Jp2k(self.jp2file)
        glymur.set_option('print.xml', False)

        actual = str(jp2)

        # Get rid of the file line, that's kind of volatile.
        actual = '\n'.join(actual.splitlines()[1:])

        # shave off the XML and non-main-header segments
        expected = fixtures.NEMO_DUMP_NO_XML
        self.assertEqual(actual, expected)

        opt = glymur.get_option('print.xml')
        self.assertFalse(opt)

    def test_suppress_codestream_old_option(self):
        """
        Verify printing with codestream suppressed, deprecated
        """
        jp2 = Jp2k(self.jp2file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            glymur.set_printoptions(codestream=False)

        actual = str(jp2)

        # Get rid of the file line, that's kind of volatile.
        actual = '\n'.join(actual.splitlines()[1:])

        expected = fixtures.NEMO_DUMP_NO_CODESTREAM
        self.assertEqual(actual, expected)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            opt = glymur.get_printoptions()['codestream']
        self.assertFalse(opt)

    def test_suppress_codestream(self):
        """
        Verify printing with codestream suppressed
        """
        jp2 = Jp2k(self.jp2file)
        glymur.set_option('print.codestream', False)

        # Get rid of the file line
        actual = '\n'.join(str(jp2).splitlines()[1:])

        expected = fixtures.NEMO_DUMP_NO_CODESTREAM
        self.assertEqual(actual, expected)

        opt = glymur.get_option('print.codestream')
        self.assertFalse(opt)

    def test_full_codestream(self):
        """
        Verify printing with the full blown codestream
        """
        jp2 = Jp2k(self.jp2file)
        glymur.set_option('parse.full_codestream', True)

        # Get rid of the file line
        actual = '\n'.join(str(jp2).splitlines()[1:])

        expected = fixtures.NEMO
        self.assertEqual(actual, expected)

        opt = glymur.get_option('print.codestream')
        self.assertTrue(opt)

    def test_reserved_marker(self):
        """
        SCENARIO:  print a marker segment with a reserver marker value

        EXPECTED RESULT:  validate the string representation
        """
        with ir.path(data, 'p0_02.j2k') as path:
            j = Jp2k(path)
        actual = str(j.codestream.segment[6])
        expected = '0xff30 marker segment @ (132, 0)'
        self.assertEqual(actual, expected)

    def test_scalar_implicit_quantization_file(self):
        with ir.path(data, 'p0_03.j2k') as path:
            j = Jp2k(path)
        actual = str(j.codestream.segment[3])
        self.assertIn('scalar implicit', actual)

    def test_scalar_explicit_quantization_file(self):
        with ir.path(data, 'p0_06.j2k') as path:
            j = Jp2k(path)
        actual = str(j.codestream.segment[3])
        self.assertIn('scalar explicit', actual)

    def test_non_default_precinct_size(self):
        with ir.path(data, 'p1_07.j2k') as path:
            j = Jp2k(path)
        actual = str(j.codestream.segment[3])
        expected = fixtures.P1_07
        self.assertEqual(actual, expected)


class TestJp2dump(unittest.TestCase):
    """Tests for verifying how jp2dump console script works."""
    def setUp(self):
        self.jpxfile = glymur.data.jpxfile()
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

        # Reset printoptions for every test.
        glymur.reset_option('all')

    def tearDown(self):
        glymur.reset_option('all')

    def run_jp2dump(self, args):
        sys.argv = args
        with patch('sys.stdout', new=StringIO()) as fake_out:
            command_line.main()
            actual = fake_out.getvalue().strip()
            # Remove the file line, as that is filesystem-dependent.
            lines = actual.split('\n')
            actual = '\n'.join(lines[1:])
        return actual

    def test_default_nemo(self):
        """by default one should get the main header"""
        actual = self.run_jp2dump(['', self.jp2file])

        # shave off the  non-main-header segments
        lines = fixtures.NEMO.split('\n')
        expected = lines[0:140]
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    def test_jp2_codestream_0(self):
        """Verify dumping with -c 0, supressing all codestream details."""
        actual = self.run_jp2dump(['', '-c', '0', self.jp2file])

        # shave off the codestream details
        lines = fixtures.NEMO.split('\n')
        expected = lines[0:105]
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    def test_jp2_codestream_1(self):
        """Verify dumping with -c 1, print just the header."""
        actual = self.run_jp2dump(['', '-c', '1', self.jp2file])

        # shave off the  non-main-header segments
        lines = fixtures.NEMO.split('\n')
        expected = lines[0:140]
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    def test_jp2_codestream_2(self):
        """Verify dumping with -c 2, print entire jp2 jacket, codestream."""
        actual = self.run_jp2dump(['', '-c', '2', self.jp2file])
        expected = fixtures.NEMO
        self.assertEqual(actual, expected)

    def test_j2k_codestream_0(self):
        """-c 0 should print just a single line when used on a codestream."""
        sys.argv = ['', '-c', '0', self.j2kfile]
        with patch('sys.stdout', new=StringIO()) as fake_out:
            command_line.main()
            actual = fake_out.getvalue().strip()
        self.assertRegex(actual, "File:  .*")

    def test_j2k_codestream_1(self):
        """
        SCENARIO:  The jp2dump executable is used with the "-c 1" switch.

        EXPECTED RESULT:  The output should include the codestream header.
        """
        sys.argv = ['', '-c', '1', self.j2kfile]
        with patch('sys.stdout', new=StringIO()) as stdout:
            command_line.main()
            actual = stdout.getvalue().strip()

        expected = fixtures.GOODSTUFF_CODESTREAM_HEADER
        self.assertEqual(expected, actual)

    def test_j2k_codestream_2(self):
        """Verify dumping with -c 2, full details."""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            sys.argv = ['', '-c', '2', self.j2kfile]
            command_line.main()
            actual = fake_out.getvalue().strip()

        expected = fixtures.GOODSTUFF_WITH_FULL_HEADER
        self.assertIn(expected, actual)

    def test_codestream_invalid(self):
        """Verify dumping with -c 3, not allowd."""
        with self.assertRaises(ValueError):
            sys.argv = ['', '-c', '3', self.jp2file]
            command_line.main()

    def test_short(self):
        """Verify dumping with -s, short option."""
        actual = self.run_jp2dump(['', '-s', self.jp2file])

        self.assertEqual(actual, fixtures.NEMO_DUMP_SHORT)

    def test_suppress_xml(self):
        """Verify dumping with -x, suppress XML."""
        actual = self.run_jp2dump(['', '-x', self.jp2file])

        # shave off the XML and non-main-header segments
        lines = fixtures.NEMO.split('\n')
        expected = lines[0:18]
        expected.extend(lines[104:140])
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    def test_suppress_warnings_until_end(self):
        """
        SCENARIO:  JP2DUMP with -x option on file with invalid ftyp box.

        EXPECTED RESULT:  The warning is suppressed until the very end of the
        output.
        """
        with ir.path(data, 'edf_c2_1178956.jp2') as path:
            actual = self.run_jp2dump(['', '-x', str(path)])
        lines = actual.splitlines()

        for line in lines[:-1]:
            self.assertNotIn('UserWarning', line)

        self.assertIn('UserWarning', lines[-1])

    def test_default_component_parameters(self):
        """printing default image component parameters"""
        icpt = glymur.lib.openjp2.ImageComptParmType()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(icpt)
            actual = fake_out.getvalue().strip()
        expected = ("<class 'glymur.lib.openjp2.ImageComptParmType'>:\n"
                    "    dx: 0\n"
                    "    dy: 0\n"
                    "    w: 0\n"
                    "    h: 0\n"
                    "    x0: 0\n"
                    "    y0: 0\n"
                    "    prec: 0\n"
                    "    bpp: 0\n"
                    "    sgnd: 0")
        self.assertEqual(actual, expected)

    def test_default_image_type(self):
        """printing default image type"""
        it = glymur.lib.openjp2.ImageType()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(it)
            actual = fake_out.getvalue().strip()

        expected = (
            "<class 'glymur.lib.openjp2.ImageType'>:\n"
            "    x0: 0\n"
            "    y0: 0\n"
            "    x1: 0\n"
            "    y1: 0\n"
            "    numcomps: 0\n"
            "    color_space: 0\n"
            "    icc_profile_buf: "
            "<glymur.lib.openjp2.LP_c_ubyte object at 0x[0-9A-Fa-f]*>\n"
            "    icc_profile_len: 0")
        self.assertRegex(actual, expected)

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_image_comp_type(self):
        obj = opj2.ImageCompType()
        actual = str(obj)
        expected = (
            r'''<class 'glymur.lib.openjp2.ImageCompType'>:\n'''
            '''    dx: 0\n'''
            '''    dy: 0\n'''
            '''    w: 0\n'''
            '''    h: 0\n'''
            '''    x0: 0\n'''
            '''    y0: 0\n'''
            '''    prec: 0\n'''
            '''    bpp: 0\n'''
            '''    sgnd: 0\n'''
            '''    resno_decoded: 0\n'''
            '''    factor: 0\n'''
            '''    data: '''
            '''<(glymur.lib.openjp2|ctypes.wintypes).LP_c_(int|long) '''
            '''object at 0x[a-fA-F0-9]+>\n'''
            '''    alpha: 0\n'''
        )
        self.assertRegex(actual, expected)
