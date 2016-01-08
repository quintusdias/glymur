# -*- coding:  utf-8 -*-
"""
Test suite for printing.
"""
from io import BytesIO
import os
import re
import struct
import sys
import tempfile
import unittest
from uuid import UUID
import warnings

if sys.hexversion < 0x03000000:
    from StringIO import StringIO
else:
    from io import StringIO

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch

import numpy as np
import lxml.etree as ET

import glymur
from glymur.core import RESTRICTED_ICC_PROFILE, ANY_ICC_PROFILE
from glymur.core import COLOR, RED, GREEN, BLUE
from glymur.jp2box import BitsPerComponentBox
from glymur.jp2box import ColourSpecificationBox
from glymur import Jp2k, command_line
from . import fixtures
from .fixtures import (WINDOWS_TMP_FILE_MSG,
                       OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)


@unittest.skipIf(os.name == "nt", WINDOWS_TMP_FILE_MSG)
class TestPrinting(unittest.TestCase):
    """Tests for verifying how printing works."""
    def setUp(self):
        self.jpxfile = glymur.data.jpxfile()
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

        # Reset printoptions for every test.
        glymur.reset_option('all')

    def tearDown(self):
        glymur.reset_option('all')

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

    def test_component_mapping(self):
        """
        verify printing of cmap box

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

    def test_xml(self):
        """
        verify printing of XML box

        Original test file was input/conformance/file1.jp2
        """
        elt = ET.fromstring(fixtures.file1_xml)
        xml = ET.ElementTree(elt)
        box = glymur.jp2box.XMLBox(xml=xml, length=439, offset=36)
        actual = str(box)
        expected = fixtures.file1_xml_box
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
        expected = fixtures.issue_186_progression_order
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

    def test_invalid_colorspace(self):
        """
        An invalid colorspace shouldn't cause an error.

        Original test file was edf_c2_1103421.jp2
        """
        colr = ColourSpecificationBox(colorspace=276)
        colr.colorspace = 276
        str(colr)

    def test_bpcc(self):
        """
        BPCC boxes are rare :-)
        """
        box = BitsPerComponentBox((5, 5, 5, 1),
                                  (False, False, False, False),
                                  length=12, offset=62)
        actual = str(box)
        self.assertEqual(actual, fixtures.bpcc)

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
        self.assertEqual(actual, fixtures.cinema2k_profile)

    def test_version_info(self):
        """Should be able to print(glymur.version.info)"""
        str(glymur.version.info)

        self.assertTrue(True)

    def test_unknown_superbox(self):
        """Verify that we can handle an unknown superbox."""
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

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jpx = Jp2k(tfile.name)

            glymur.set_option('print.short', True)
            actual = str(jpx.box[-1])
            if sys.hexversion < 0x03000000:
                expected = "Unknown Box (grp ) @ (1399071, 20)"
            else:
                expected = "Unknown Box (b'grp ') @ (1399071, 20)"
            self.assertEqual(actual, expected)

    def test_printoptions_bad_argument(self):
        """Verify error when bad parameter to set_printoptions"""
        with self.assertRaises(KeyError):
            glymur.set_option('hi', 'low')

    @unittest.skipIf(re.match("1.5|2",
                              glymur.version.openjpeg_version) is None,
                     "Must have openjpeg 1.5 or higher to run")
    def test_asoc_label_box(self):
        """verify printing of asoc, label boxes"""
        # Construct a fake file with an asoc and a label box, as
        # OpenJPEG doesn't have such a file.
        data = glymur.Jp2k(self.jp2file)[::2, ::2]
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile2:
                glymur.Jp2k(tfile.name, data=data)

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
               '        Number of resolutions:  2\n'
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
               '        Number of resolutions:  2\n'
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
        codestream = j.get_codestream()
        actual = str(codestream.segment[1])

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
        """Verify the printing of a UUID/XMP box."""
        j = glymur.Jp2k(self.jp2file)
        actual = str(j.box[3])

        expected = fixtures.nemo_xmp_box
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
               '            Number of resolutions:  2\n'
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

    @unittest.skipIf(sys.hexversion < 0x03000000,
                     "Only trusting python3 for printing non-ascii chars")
    def test_xml_latin1(self):
        """Should be able to print an XMLBox with utf-8 encoding (latin1)."""
        # Seems to be inconsistencies between different versions of python2.x
        # as to what gets printed.
        #
        # 2.7.5 (fedora 19) prints xml entities.
        # 2.7.3 seems to want to print hex escapes.
        text = u"""<flow>Strömung</flow>"""
        if sys.hexversion < 0x03000000:
            xml = ET.parse(StringIO(text.encode('utf-8')))
        else:
            xml = ET.parse(StringIO(text))

        xmlbox = glymur.jp2box.XMLBox(xml=xml)
        actual = str(xmlbox)
        if sys.hexversion < 0x03000000:
            expected = ("XML Box (xml ) @ (-1, 0)\n"
                        "    <flow>Str\xc3\xb6mung</flow>")
        else:
            expected = ("XML Box (xml ) @ (-1, 0)\n"
                        "    <flow>Strömung</flow>")
        self.assertEqual(actual, expected)

    @unittest.skipIf(sys.hexversion < 0x03000000,
                     "Only trusting python3 for printing non-ascii chars")
    def test_xml_cyrrilic(self):
        """Should be able to print XMLBox with utf-8 encoding (cyrrillic)."""
        # Seems to be inconsistencies between different versions of python2.x
        # as to what gets printed.
        #
        # 2.7.5 (fedora 19) prints xml entities.
        # 2.7.3 seems to want to print hex escapes.
        text = u"""<country>Россия</country>"""
        if sys.hexversion < 0x03000000:
            xml = ET.parse(StringIO(text.encode('utf-8')))
        else:
            xml = ET.parse(StringIO(text))

        xmlbox = glymur.jp2box.XMLBox(xml=xml)
        actual = str(xmlbox)
        if sys.hexversion < 0x03000000:
            expected = ("XML Box (xml ) @ (-1, 0)\n"
                        ("    <country>&#1056;&#1086;&#1089;&#1089;"
                         "&#1080;&#1103;</country>"))
        else:
            expected = ("XML Box (xml ) @ (-1, 0)\n"
                        "    <country>Россия</country>")

        self.assertEqual(actual, expected)

    @unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
    def test_less_common_boxes(self):
        """verify uinf, ulst, url, res, resd, resc box printing"""
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
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
            with patch('sys.stdout', new=StringIO()) as fake_out:
                print(jp2k.box[3])
                print(jp2k.box[4])
                actual = fake_out.getvalue().strip()
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

    def test_flst(self):
        """Verify printing of fragment list box."""
        flst = glymur.jp2box.FragmentListBox([89], [1132288], [0])
        actual = str(flst)
        self.assertEqual(actual, fixtures.fragment_list_box)

    def test_dref(self):
        """Verify printing of data reference box."""
        dref = glymur.jp2box.DataReferenceBox()
        actual = str(dref)
        self.assertEqual(actual, 'Data Reference Box (dtbl) @ (-1, 0)')

    def test_jplh_cgrp(self):
        """Verify printing of compositing layer header box, color group box."""
        jpx = glymur.Jp2k(self.jpxfile)
        actual = str(jpx.box[7])
        self.assertEqual(actual, fixtures.jplh_color_group_box)

    def test_free(self):
        """Verify printing of Free box."""
        free = glymur.jp2box.FreeBox()
        actual = str(free)
        self.assertEqual(actual, 'Free Box (free) @ (-1, 0)')

    def test_nlst(self):
        """Verify printing of number list box."""
        assn = (0, 16777216, 33554432)
        nlst = glymur.jp2box.NumberListBox(assn)
        actual = str(nlst)
        self.assertEqual(actual, fixtures.number_list_box)

    def test_ftbl(self):
        """Verify printing of fragment table box."""
        ftbl = glymur.jp2box.FragmentTableBox()
        actual = str(ftbl)
        self.assertEqual(actual, 'Fragment Table Box (ftbl) @ (-1, 0)')

    def test_jpch(self):
        """Verify printing of JPCH box."""
        jpx = glymur.Jp2k(self.jpxfile)
        actual = str(jpx.box[3])
        self.assertEqual(actual, 'Codestream Header Box (jpch) @ (887, 8)')

    @unittest.skipIf(sys.hexversion < 0x03000000,
                     "Ordered dicts not printing well in 2.7")
    def test_exif_uuid(self):
        """Verify printing of exif information"""
        with tempfile.NamedTemporaryFile(suffix='.jp2', mode='wb') as tfile:

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

        expected = ("UUID Box (uuid) @ (1135519, 76)\n"
                    "    UUID:  4a706754-6966-6645-7869-662d3e4a5032 (EXIF)\n"
                    "    UUID Data:  OrderedDict([('ImageWidth', 256), "
                    "('ImageLength', 512), ('Make', 'HTC')])")
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
        Should not show the format string in output.

        The cmap box is wildly broken, but printing was still wrong.
        Format strings like %d were showing up in the output.

        Original file tested was input/nonregression/mem-b2ace68c-1381.jp2
        """
        cmap = glymur.jp2box.ComponentMappingBox(component_index=(0, 0, 0, 0),
                                                 mapping_type=(1, 1, 1, 1),
                                                 palette_index=(0, 1, 2, 3),
                                                 length=24, offset=130)
        actual = str(cmap)
        self.assertEqual(actual, fixtures.issue_182_cmap)

    def test_issue183(self):
        """
        Broken ICC profile

        Original file tested was input/nonregression/orb-blue10-lin-jp2.jp2
        """
        colr = ColourSpecificationBox(method=RESTRICTED_ICC_PROFILE,
                                      icc_profile=None, length=12, offset=62)

        actual = str(colr)
        self.assertEqual(actual, fixtures.issue_183_colr)

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
        self.assertEqual(actual, fixtures.text_GBR_rreq)

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

    def test_icc_profile(self):
        """
        verify icc profile printing with a jpx

        2.7, 3.3, 3.4, and 3.5 all print ordered dicts differently
        Original file tested was input/nonregression/text_GBR.jp2.
        """
        fp = BytesIO()
        fp.write(b'\x00' * 179)

        # Write the colr box header.
        buffer = struct.pack('>I4s', 1339, b'colr')
        buffer += struct.pack('>BBB', ANY_ICC_PROFILE, 2, 1)

        buffer += struct.pack('>IIBB', 1328, 1634758764, 2, 32)
        buffer += b'\x00' * 2 + b'mntr' + b'RGB ' + b'XYZ '
        # Need a date in bytes 24:36
        buffer += struct.pack('>HHHHHH', 2009, 2, 25, 11, 26, 11)
        buffer += 'acsp'.encode('utf-8')
        buffer += 'APPL'.encode('utf-8')
        buffer += b'\x00' * 4
        buffer += 'appl'.encode('utf-8')  # 48 - 52
        buffer += b'\x00' * 16
        buffer += struct.pack('>III', 63190, 65536, 54061)  # 68 - 80
        buffer += 'appl'.encode('utf-8')  # 80 - 84
        buffer += b'\x00' * 44
        fp.write(buffer)
        fp.seek(179 + 8)

        # Should be able to read the colr box now
        if sys.hexversion < 0x03000000:
            box = glymur.jp2box.ColourSpecificationBox.parse(fp, 179, 1339)
        else:
            box = glymur.jp2box.ColourSpecificationBox.parse(fp, 179, 1339)

        actual = str(box)

        if sys.hexversion < 0x03000000:
            expected = fixtures.text_gbr_27
        elif sys.hexversion < 0x03040000:
            expected = fixtures.text_gbr_33
        elif sys.hexversion < 0x03050000:
            expected = fixtures.text_gbr_34
        else:
            expected = fixtures.text_gbr_35

        self.assertEqual(actual, expected)

    @unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
    def test_precincts(self):
        """
        verify that we print precincts correctly
        """
        data = Jp2k(self.jp2file)[:]
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            j = Jp2k(tfile.name, data=data, psizes=[(128, 128)] * 3)

            # Should be three layers.
            codestream = j.get_codestream()

        actual = str(codestream.segment[2])

        self.assertEqual(actual, fixtures.multiple_precinct_size)

    def test_old_short_option(self):
        """
        Verify printing with deprecated set_printoptions "short"
        """
        jp2 = Jp2k(self.jp2file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            glymur.set_printoptions(short=True)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(jp2)
            actual = fake_out.getvalue().strip()
            # Get rid of the file line, that's kind of volatile.
            actual = '\n'.join(actual.splitlines()[1:])

        expected = fixtures.nemo_dump_short
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
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(jp2)
            actual = fake_out.getvalue().strip()
            # Get rid of the file line, that's kind of volatile.
            actual = '\n'.join(actual.splitlines()[1:])

        # shave off the XML and non-main-header segments
        expected = fixtures.nemo_dump_no_xml
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
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(jp2)
            actual = fake_out.getvalue().strip()
            # Get rid of the file line, that's kind of volatile.
            actual = '\n'.join(actual.splitlines()[1:])

        # shave off the XML and non-main-header segments
        expected = fixtures.nemo_dump_no_xml
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
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(jp2)
            actual = fake_out.getvalue().strip()
            # Get rid of the file line, that's kind of volatile.
            actual = '\n'.join(actual.splitlines()[1:])

        expected = fixtures.nemo_dump_no_codestream
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

        expected = fixtures.nemo_dump_no_codestream
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

        expected = fixtures.nemo
        self.assertEqual(actual, expected)

        opt = glymur.get_option('print.codestream')
        self.assertTrue(opt)

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
        lines = fixtures.nemo.split('\n')
        expected = lines[0:140]
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    def test_jp2_codestream_0(self):
        """Verify dumping with -c 0, supressing all codestream details."""
        actual = self.run_jp2dump(['', '-c', '0', self.jp2file])

        # shave off the codestream details
        lines = fixtures.nemo.split('\n')
        expected = lines[0:105]
        expected = '\n'.join(expected)
        self.assertEqual(actual, expected)

    def test_jp2_codestream_1(self):
        """Verify dumping with -c 1, print just the header."""
        actual = self.run_jp2dump(['', '-c', '1', self.jp2file])

        # shave off the  non-main-header segments
        lines = fixtures.nemo.split('\n')
        expected = lines[0:140]
        expected = '\n'.join(expected)
        self.maxDiff = None
        self.assertEqual(actual, expected)

    def test_jp2_codestream_2(self):
        """Verify dumping with -c 2, print entire jp2 jacket, codestream."""
        actual = self.run_jp2dump(['', '-c', '2', self.jp2file])
        expected = fixtures.nemo
        self.assertEqual(actual, expected)

    @unittest.skipIf(sys.hexversion < 0x03000000, "assertRegex not in 2.7")
    def test_j2k_codestream_0(self):
        """-c 0 should print just a single line when used on a codestream."""
        sys.argv = ['', '-c', '0', self.j2kfile]
        with patch('sys.stdout', new=StringIO()) as fake_out:
            command_line.main()
            actual = fake_out.getvalue().strip()
        self.assertRegex(actual, "File:  .*")

    def test_j2k_codestream_2(self):
        """Verify dumping with -c 2, full details."""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            sys.argv = ['', '-c', '2', self.j2kfile]
            command_line.main()
            actual = fake_out.getvalue().strip()

        expected = fixtures.goodstuff_with_full_header
        self.assertIn(expected, actual)

    def test_codestream_invalid(self):
        """Verify dumping with -c 3, not allowd."""
        with self.assertRaises(ValueError):
            sys.argv = ['', '-c', '3', self.jp2file]
            command_line.main()

    def test_short(self):
        """Verify dumping with -s, short option."""
        actual = self.run_jp2dump(['', '-s', self.jp2file])

        self.assertEqual(actual, fixtures.nemo_dump_short)

    def test_suppress_xml(self):
        """Verify dumping with -x, suppress XML."""
        actual = self.run_jp2dump(['', '-x', self.jp2file])

        # shave off the XML and non-main-header segments
        lines = fixtures.nemo.split('\n')
        expected = lines[0:18]
        expected.extend(lines[104:140])
        expected = '\n'.join(expected)
        self.maxDiff = None
        self.assertEqual(actual, expected)
