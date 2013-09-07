"""Test suite for printing.
"""
# C0302:  don't care too much about having too many lines in a test module
# pylint: disable=C0302

# E061:  unittest.mock introduced in 3.3 (python-2.7/pylint issue)
# pylint: disable=E0611,F0401

# R0904:  Not too many methods in unittest.
# pylint: disable=R0904

import os
import struct
import sys
import tempfile
import warnings

if sys.hexversion < 0x02070000:
    import unittest2 as unittest
else:
    import unittest

if sys.hexversion < 0x03000000:
    from StringIO import StringIO
else:
    from io import StringIO

if sys.hexversion <= 0x03030000:
    from mock import patch
else:
    from unittest.mock import patch

import glymur
from glymur import Jp2k
from .fixtures import OPJ_DATA_ROOT, opj_data_file


@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
@unittest.skipIf(glymur.lib.openjp2.OPENJP2 is None,
                 "Missing openjp2 library.")
class TestPrintingNeedsLib(unittest.TestCase):
    """These tests require the library, mostly in order to just setup the test.
    """

    @classmethod
    def setUpClass(cls):
        # Setup a plain JP2 file without the two UUID boxes.
        jp2file = glymur.data.nemo()
        with tempfile.NamedTemporaryFile(suffix='.jp2', delete=False) as tfile:
            cls._plain_nemo_file = tfile.name
            ijfile = Jp2k(jp2file)
            data = ijfile.read(rlevel=1)
            ojfile = Jp2k(cls._plain_nemo_file, 'wb')
            ojfile.write(data)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._plain_nemo_file)

    def setUp(self):
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()

        # Save the output of dumping nemo.jp2 for more than one test.
        lines = ['JPEG 2000 Signature Box (jP  ) @ (0, 12)',
                 '    Signature:  0d0a870a',
                 'File Type Box (ftyp) @ (12, 20)',
                 '    Brand:  jp2 ',
                 "    Compatibility:  ['jp2 ']",
                 'JP2 Header Box (jp2h) @ (32, 45)',
                 '    Image Header Box (ihdr) @ (40, 22)',
                 '        Size:  [728 1296 3]',
                 '        Bitdepth:  8',
                 '        Signed:  False',
                 '        Compression:  wavelet',
                 '        Colorspace Unknown:  False',
                 '    Colour Specification Box (colr) @ (62, 15)',
                 '        Method:  enumerated colorspace',
                 '        Precedence:  0',
                 '        Colorspace:  sRGB',
                 'Contiguous Codestream Box (jp2c) @ (77, 1632355)',
                 '    Main header:',
                 '        SOC marker segment @ (85, 0)',
                 '        SIZ marker segment @ (87, 47)',
                 '            Profile:  2',
                 '            Reference Grid Height, Width:  (728 x 1296)',
                 '            Vertical, Horizontal Reference Grid Offset:  '
                 + '(0 x 0)',
                 '            Reference Tile Height, Width:  (728 x 1296)',
                 '            Vertical, Horizontal Reference Tile Offset:  '
                 + '(0 x 0)',
                 '            Bitdepth:  (8, 8, 8)',
                 '            Signed:  (False, False, False)',
                 '            Vertical, Horizontal Subsampling:  '
                 + '((1, 1), (1, 1), (1, 1))',
                 '        COD marker segment @ (136, 12)',
                 '            Coding style:',
                 '                Entropy coder, without partitions',
                 '                SOP marker segments:  False',
                 '                EPH marker segments:  False',
                 '            Coding style parameters:',
                 '                Progression order:  LRCP',
                 '                Number of layers:  1',
                 '                Multiple component transformation usage:  '
                 + 'reversible',
                 '                Number of resolutions:  6',
                 '                Code block height, width:  (64 x 64)',
                 '                Wavelet transform:  5-3 reversible',
                 '                Precinct size:  default, 2^15 x 2^15',
                 '                Code block context:',
                 '                    Selective arithmetic coding bypass:  '
                 + 'False',
                 '                    Reset context probabilities on '
                 + 'coding pass boundaries:  False',
                 '                    Termination on each coding pass:  False',
                 '                    Vertically stripe causal context:  '
                 + 'False',
                 '                    Predictable termination:  False',
                 '                    Segmentation symbols:  False',
                 '        QCD marker segment @ (150, 19)',
                 '            Quantization style:  no quantization, '
                 + '2 guard bits',
                 '            Step size:  [(0, 8), (0, 9), (0, 9), '
                 + '(0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), '
                 + '(0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), '
                 + '(0, 10)]']
        self.expected_plain = '\n'.join(lines)

    def tearDown(self):
        pass

    def test_asoc_label_box(self):
        """verify printing of asoc, label boxes"""
        # Construct a fake file with an asoc and a label box, as
        # OpenJPEG doesn't have such a file.
        data = glymur.Jp2k(self.jp2file).read(rlevel=1)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            j = glymur.Jp2k(tfile.name, 'wb')
            j.write(data)

            with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile2:

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
                with patch('sys.stdout', new=StringIO()) as fake_out:
                    print(jasoc.box[3])
                    actual = fake_out.getvalue().strip()
                lines = ['Association Box (asoc) @ (77, 56)',
                         '    Label Box (lbl ) @ (85, 13)',
                         '        Label:  label',
                         '    XML Box (xml ) @ (98, 35)',
                         '        <test>this is a test</test>']
                expected = '\n'.join(lines)
                self.assertEqual(actual, expected)

    def test_jp2dump(self):
        """basic jp2dump test"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            glymur.jp2dump(self._plain_nemo_file)
            actual = fake_out.getvalue().strip()

        # Get rid of the filename line, as it is not set in stone.
        lst = actual.split('\n')
        lst = lst[1:]
        actual = '\n'.join(lst)
        self.assertEqual(actual, self.expected_plain)

    def test_entire_file(self):
        """verify output from printing entire file"""
        j = glymur.Jp2k(self._plain_nemo_file)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j)
            actual = fake_out.getvalue().strip()

        # Get rid of the filename line, as it is not set in stone.
        lst = actual.split('\n')
        lst = lst[1:]
        actual = '\n'.join(lst)

        self.assertEqual(actual, self.expected_plain)


class TestPrinting(unittest.TestCase):
    """Test suite for printing where the libraries are not needed"""

    def setUp(self):
        # Save sys.stdout.
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_coc_segment(self):
        """verify printing of COC segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[6])
            actual = fake_out.getvalue().strip()

        lines = ['COC marker segment @ (3260, 9)',
                 '    Associated component:  1',
                 '    Coding style for this component:  '
                 + 'Entropy coder, PARTITION = 0',
                 '    Coding style parameters:',
                 '        Number of resolutions:  2',
                 '        Code block height, width:  (64 x 64)',
                 '        Wavelet transform:  5-3 reversible',
                 '        Code block context:',
                 '            Selective arithmetic coding bypass:  False',
                 '            Reset context probabilities '
                 + 'on coding pass boundaries:  False',
                 '            Termination on each coding pass:  False',
                 '            Vertically stripe causal context:  False',
                 '            Predictable termination:  False',
                 '            Segmentation symbols:  False']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_cod_segment(self):
        """verify printing of COD segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[2])
            actual = fake_out.getvalue().strip()

        lines = ['COD marker segment @ (3186, 12)',
                 '    Coding style:',
                 '        Entropy coder, without partitions',
                 '        SOP marker segments:  False',
                 '        EPH marker segments:  False',
                 '    Coding style parameters:',
                 '        Progression order:  LRCP',
                 '        Number of layers:  2',
                 '        Multiple component transformation usage:  '
                 + 'reversible',
                 '        Number of resolutions:  2',
                 '        Code block height, width:  (64 x 64)',
                 '        Wavelet transform:  5-3 reversible',
                 '        Precinct size:  default, 2^15 x 2^15',
                 '        Code block context:',
                 '            Selective arithmetic coding bypass:  False',
                 '            Reset context probabilities on coding '
                 + 'pass boundaries:  False',
                 '            Termination on each coding pass:  False',
                 '            Vertically stripe causal context:  False',
                 '            Predictable termination:  False',
                 '            Segmentation symbols:  False']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_icc_profile(self):
        """verify printing of colr box with ICC profile"""
        filename = opj_data_file('input/nonregression/text_GBR.jp2')
        with warnings.catch_warnings():
            # brand is 'jp2 ', but has any icc profile.
            warnings.simplefilter("ignore")
            jp2 = Jp2k(filename)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(jp2.box[3].box[1])
            actual = fake_out.getvalue().strip()
        lin27 = ["Colour Specification Box (colr) @ (179, 1339)",
                 "    Method:  any ICC profile",
                 "    Precedence:  2",
                 "    Approximation:  accurately represents correct "
                 + "colorspace definition",
                 "    ICC Profile:",
                 "        {'Color Space': 'RGB',",
                 "         'Connection Space': 'XYZ',",
                 "         'Creator': u'appl',",
                 "         'Datetime': "
                 + "datetime.datetime(2009, 2, 25, 11, 26, 11),",
                 "         'Device Attributes': 'reflective, glossy, "
                 + "positive media polarity, color media',",
                 "         'Device Class': 'display device profile',",
                 "         'Device Manufacturer': u'appl',",
                 "         'Device Model': '',",
                 "         'File Signature': u'acsp',",
                 "         'Flags': "
                 + "'not embedded, can be used independently',",
                 "         'Illuminant': "
                 + "array([ 0.96420288,  1.        ,  0.8249054 ]),",
                 "         'Platform': u'APPL',",
                 "         'Preferred CMM Type': 1634758764,",
                 "         'Rendering Intent': 'perceptual',",
                 "         'Size': 1328,",
                 "         'Version': '2.2.0'}"]
        lin33 = ["Colour Specification Box (colr) @ (179, 1339)",
                 "    Method:  any ICC profile",
                 "    Precedence:  2",
                 "    Approximation:  accurately represents correct "
                 + "colorspace definition",
                 "    ICC Profile:",
                 "        {'Size': 1328,",
                 "         'Preferred CMM Type': 1634758764,",
                 "         'Version': '2.2.0',",
                 "         'Device Class': 'display device profile',",
                 "         'Color Space': 'RGB',",
                 "         'Connection Space': 'XYZ',",
                 "         'Datetime': "
                 + "datetime.datetime(2009, 2, 25, 11, 26, 11),",
                 "         'File Signature': 'acsp',",
                 "         'Platform': 'APPL',",
                 "         'Flags': 'not embedded, can be used "
                 + "independently',",
                 "         'Device Manufacturer': 'appl',",
                 "         'Device Model': '',",
                 "         'Device Attributes': 'reflective, glossy, "
                 + "positive media polarity, color media',",
                 "         'Rendering Intent': 'perceptual',",
                 "         'Illuminant': "
                 + "array([ 0.96420288,  1.        ,  0.8249054 ]),",
                 "         'Creator': 'appl'}"]

        lines = lin27 if sys.hexversion < 0x03000000 else lin33
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_crg(self):
        """verify printing of CRG segment"""
        filename = opj_data_file('input/conformance/p0_03.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[-5])
            actual = fake_out.getvalue().strip()
        lines = ['CRG marker segment @ (87, 6)',
                 '    Vertical, Horizontal offset:  (0.50, 1.00)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_rgn(self):
        """verify printing of RGN segment"""
        filename = opj_data_file('input/conformance/p0_03.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream(header_only=False)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[12])
            actual = fake_out.getvalue().strip()
        lines = ['RGN marker segment @ (310, 5)',
                 '    Associated component:  0',
                 '    ROI style:  0',
                 '    Parameter:  7']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_sop(self):
        """verify printing of SOP segment"""
        filename = opj_data_file('input/conformance/p0_03.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream(header_only=False)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[-2])
            actual = fake_out.getvalue().strip()
        lines = ['SOP marker segment @ (12836, 4)',
                 '    Nsop:  15']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_cme(self):
        """Test printing a CME or comment marker segment."""
        filename = opj_data_file('input/conformance/p0_02.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        # 2nd to last segment in the main header
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[-2])
            actual = fake_out.getvalue().strip()
        lines = ['CME marker segment @ (85, 45)',
                 '    "Creator: AV-J2K (c) 2000,2001 Algo Vision"']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_eoc_segment(self):
        """verify printing of eoc segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[-1])
            actual = fake_out.getvalue().strip()

        lines = ['EOC marker segment @ (1135421, 0)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_plt_segment(self):
        """verify printing of PLT segment"""
        filename = opj_data_file('input/conformance/p0_07.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream(header_only=False)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[49935])
            actual = fake_out.getvalue().strip()

        lines = ['PLT marker segment @ (7871146, 38)',
                 '    Index:  0',
                 '    Iplt:  [9, 122, 19, 30, 27, 9, 41, 62, 18, 29, 261,'
                 + ' 55, 82, 299, 93, 941, 951, 687, 1729, 1443, 1008, 2168,'
                 + ' 2188, 2223]']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_pod_segment(self):
        """verify printing of POD segment"""
        filename = opj_data_file('input/conformance/p0_13.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[8])
            actual = fake_out.getvalue().strip()

        lines = ['POD marker segment @ (878, 20)',
                 '    Progression change 0:',
                 '        Resolution index start:  0',
                 '        Component index start:  0',
                 '        Layer index end:  1',
                 '        Resolution index end:  33',
                 '        Component index end:  128',
                 '        Progression order:  RLCP',
                 '    Progression change 1:',
                 '        Resolution index start:  0',
                 '        Component index start:  128',
                 '        Layer index end:  1',
                 '        Resolution index end:  33',
                 '        Component index end:  257',
                 '        Progression order:  CPRL']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_ppm_segment(self):
        """verify printing of PPM segment"""
        filename = opj_data_file('input/conformance/p1_03.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[9])
            actual = fake_out.getvalue().strip()

        lines = ['PPM marker segment @ (213, 43712)',
                 '    Index:  0',
                 '    Data:  43709 uninterpreted bytes']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_ppt_segment(self):
        """verify printing of ppt segment"""
        filename = opj_data_file('input/conformance/p1_06.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream(header_only=False)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[6])
            actual = fake_out.getvalue().strip()

        lines = ['PPT marker segment @ (155, 109)',
                 '    Index:  0',
                 '    Packet headers:  106 uninterpreted bytes']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_qcc_segment(self):
        """verify printing of qcc segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[7])
            actual = fake_out.getvalue().strip()

        lines = ['QCC marker segment @ (3271, 8)',
                 '    Associated Component:  1',
                 '    Quantization style:  no quantization, 2 guard bits',
                 '    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_qcd_segment_5x3_transform(self):
        """verify printing of qcd segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[3])
            actual = fake_out.getvalue().strip()

        lines = ['QCD marker segment @ (3200, 7)',
                 '    Quantization style:  no quantization, 2 guard bits',
                 '    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_siz_segment(self):
        """verify printing of SIZ segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[1])
            actual = fake_out.getvalue().strip()

        lines = ['SIZ marker segment @ (3137, 47)',
                 '    Profile:  2',
                 '    Reference Grid Height, Width:  (1456 x 2592)',
                 '    Vertical, Horizontal Reference Grid Offset:  (0 x 0)',
                 '    Reference Tile Height, Width:  (1456 x 2592)',
                 '    Vertical, Horizontal Reference Tile Offset:  (0 x 0)',
                 '    Bitdepth:  (8, 8, 8)',
                 '    Signed:  (False, False, False)',
                 '    Vertical, Horizontal Subsampling:  '
                 + '((1, 1), (1, 1), (1, 1))']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_soc_segment(self):
        """verify printing of SOC segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[0])
            actual = fake_out.getvalue().strip()

        lines = ['SOC marker segment @ (3135, 0)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_sod_segment(self):
        """verify printing of SOD segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[10])
            actual = fake_out.getvalue().strip()

        lines = ['SOD marker segment @ (3302, 0)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_sot_segment(self):
        """verify printing of SOT segment"""
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[5])
            actual = fake_out.getvalue().strip()

        lines = ['SOT marker segment @ (3248, 10)',
                 '    Tile part index:  0',
                 '    Tile part length:  1132173',
                 '    Tile part instance:  0',
                 '    Number of tile parts:  1']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_tlm_segment(self):
        """verify printing of TLM segment"""
        filename = opj_data_file('input/conformance/p0_15.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[10])
            actual = fake_out.getvalue().strip()

        lines = ['TLM marker segment @ (268, 28)',
                 '    Index:  0',
                 '    Tile number:  (0, 1, 2, 3)',
                 '    Length:  (4267, 2117, 4080, 2081)']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(sys.hexversion < 0x02070000,
                     "Differences in XML printing between 2.6 and 2.7")
    def test_xmp(self):
        """Verify the printing of a UUID/XMP box."""
        j = glymur.Jp2k(self.jp2file)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j.box[4])
            actual = fake_out.getvalue().strip()

        lst = ['UUID Box (uuid) @ (715, 2412)',
               '    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)',
               '    UUID Data:  ',
               '    <ns0:xmpmeta xmlns:ns0="adobe:ns:meta/" '
               + 'xmlns:ns2="http://ns.adobe.com/xap/1.0/" '
               + 'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
               + 'ns0:xmptk="XMP Core 4.4.0-Exiv2">',
               '      <rdf:RDF>',
               '        <rdf:Description ns2:CreatorTool="glymur" '
               + 'rdf:about="" />',
               '      </rdf:RDF>',
               '    </ns0:xmpmeta>']
        expected = '\n'.join(lst)
        self.assertEqual(actual, expected)

    def test_codestream(self):
        """verify printing of entire codestream"""
        j = glymur.Jp2k(self.jp2file)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j.get_codestream())
            actual = fake_out.getvalue().strip()
        lst = ['Codestream:',
               '    SOC marker segment @ (3135, 0)',
               '    SIZ marker segment @ (3137, 47)',
               '        Profile:  2',
               '        Reference Grid Height, Width:  (1456 x 2592)',
               '        Vertical, Horizontal Reference Grid Offset:  (0 x 0)',
               '        Reference Tile Height, Width:  (1456 x 2592)',
               '        Vertical, Horizontal Reference Tile Offset:  (0 x 0)',
               '        Bitdepth:  (8, 8, 8)',
               '        Signed:  (False, False, False)',
               '        Vertical, Horizontal Subsampling:  '
               + '((1, 1), (1, 1), (1, 1))',
               '    COD marker segment @ (3186, 12)',
               '        Coding style:',
               '            Entropy coder, without partitions',
               '            SOP marker segments:  False',
               '            EPH marker segments:  False',
               '        Coding style parameters:',
               '            Progression order:  LRCP',
               '            Number of layers:  2',
               '            Multiple component transformation usage:  '
               + 'reversible',
               '            Number of resolutions:  2',
               '            Code block height, width:  (64 x 64)',
               '            Wavelet transform:  5-3 reversible',
               '            Precinct size:  default, 2^15 x 2^15',
               '            Code block context:',
               '                Selective arithmetic coding bypass:  False',
               '                Reset context probabilities on '
               + 'coding pass boundaries:  False',
               '                Termination on each coding pass:  False',
               '                Vertically stripe causal context:  False',
               '                Predictable termination:  False',
               '                Segmentation symbols:  False',
               '    QCD marker segment @ (3200, 7)',
               '        Quantization style:  no quantization, '
               + '2 guard bits',
               '        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]',
               '    CME marker segment @ (3209, 37)',
               '        "Created by OpenJPEG version 2.0.0"']
        expected = '\n'.join(lst)
        self.assertEqual(actual, expected)

    @unittest.skipIf(sys.hexversion < 0x02070000,
                     "Differences in XML printing between 2.6 and 2.7")
    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_xml(self):
        """verify printing of XML box"""
        filename = opj_data_file('input/conformance/file1.jp2')
        j = glymur.Jp2k(filename)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j.box[2])
            actual = fake_out.getvalue().strip()

        lines = ['XML Box (xml ) @ (36, 439)',
                 '    <ns0:IMAGE_CREATION '
                 + 'xmlns:ns0="http://www.jpeg.org/jpx/1.0/xml" '
                 + 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                 + 'xsi:schemaLocation="http://www.jpeg.org/jpx/1.0/xml '
                 + 'http://www.jpeg.org/metadata/15444-2.xsd">',

                 '      <ns0:GENERAL_CREATION_INFO>',
                 '        <ns0:CREATION_TIME>'
                 + '2001-11-01T13:45:00.000-06:00'
                 + '</ns0:CREATION_TIME>',

                 '        <ns0:IMAGE_SOURCE>'
                 + 'Professional 120 Image'
                 + '</ns0:IMAGE_SOURCE>',

                 '      </ns0:GENERAL_CREATION_INFO>',
                 '    </ns0:IMAGE_CREATION>']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_channel_definition(self):
        """verify printing of cdef box"""
        filename = opj_data_file('input/conformance/file2.jp2')
        j = glymur.Jp2k(filename)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j.box[2].box[2])
            actual = fake_out.getvalue().strip()
        lines = ['Channel Definition Box (cdef) @ (81, 28)',
                 '    Channel 0 (color) ==> (3)',
                 '    Channel 1 (color) ==> (2)',
                 '    Channel 2 (color) ==> (1)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_component_mapping(self):
        """verify printing of cmap box"""
        filename = opj_data_file('input/conformance/file9.jp2')
        j = glymur.Jp2k(filename)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j.box[2].box[2])
            actual = fake_out.getvalue().strip()
        lines = ['Component Mapping Box (cmap) @ (848, 20)',
                 '    Component 0 ==> palette column 0',
                 '    Component 0 ==> palette column 1',
                 '    Component 0 ==> palette column 2']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_palette7(self):
        """verify printing of pclr box"""
        filename = opj_data_file('input/conformance/file9.jp2')
        j = glymur.Jp2k(filename)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j.box[2].box[1])
            actual = fake_out.getvalue().strip()
        lines = ['Palette Box (pclr) @ (66, 782)',
                 '    Size:  (256 x 3)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_rreq(self):
        """verify printing of reader requirements box"""
        filename = opj_data_file('input/conformance/file7.jp2')
        j = glymur.Jp2k(filename)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j.box[2])
            actual = fake_out.getvalue().strip()
        lines = ['Reader Requirements Box (rreq) @ (44, 24)',
                 '    Standard Features:',
                 '        Feature 005:  '
                 + 'Unrestricted JPEG 2000 Part 1 codestream, '
                 + 'ITU-T Rec. T.800 | ISO/IEC 15444-1',
                 '        Feature 060:  e-sRGB enumerated colorspace',

                 '        Feature 043:  '
                 + '(Deprecated) '
                 + 'compositing layer uses restricted ICC profile',

                 '    Vendor Features:']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_differing_subsamples(self):
        """verify printing of SIZ with different subsampling... Issue 86."""
        filename = opj_data_file('input/conformance/p0_05.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(codestream.segment[1])
            actual = fake_out.getvalue().strip()
        lines = ['SIZ marker segment @ (2, 50)',
                 '    Profile:  0',
                 '    Reference Grid Height, Width:  (1024 x 1024)',
                 '    Vertical, Horizontal Reference Grid Offset:  (0 x 0)',
                 '    Reference Tile Height, Width:  (1024 x 1024)',
                 '    Vertical, Horizontal Reference Tile Offset:  (0 x 0)',
                 '    Bitdepth:  (8, 8, 8, 8)',
                 '    Signed:  (False, False, False, False)',
                 '    Vertical, Horizontal Subsampling:  '
                 + '((1, 1), (1, 1), (2, 2), (2, 2))']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_palette_box(self):
        """Verify that palette (pclr) boxes are printed without error."""
        filename = opj_data_file('input/conformance/file9.jp2')
        j = glymur.Jp2k(filename)
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j.box[2].box[1])
            actual = fake_out.getvalue().strip()
        lines = ['Palette Box (pclr) @ (66, 782)',
                 '    Size:  (256 x 3)']
        expected = '\n'.join(lines)
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
            lines = ['UUIDInfo Box (uinf) @ (77, 50)',
                     '    UUID List Box (ulst) @ (85, 26)',
                     '        UUID[0]:  00000000-0000-0000-0000-000000000000',
                     '    Data Entry URL Box (url ) @ (111, 16)',
                     '        Version:  0',
                     '        Flag:  0 0 0',
                     '        URL:  "abcd"',
                     'Resolution Box (res ) @ (127, 44)',
                     '    Capture Resolution Box (resc) @ (135, 18)',
                     '        VCR:  1.0',
                     '        HCR:  10.0',
                     '    Display Resolution Box (resd) @ (153, 18)',
                     '        VDR:  10.0',
                     '        HDR:  1.0']

            expected = '\n'.join(lines)
            self.assertEqual(actual, expected)

    @unittest.skipIf(sys.hexversion < 0x03000000,
                     "Ordered dicts not printing well in 2.7")
    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_jpx_approx_icc_profile(self):
        """verify jpx with approx field equal to zero"""
        # ICC profiles may be used in JP2, but the approximation field should
        # be zero unless we have jpx.  This file does both.
        filename = opj_data_file('input/nonregression/text_GBR.jp2')
        with warnings.catch_warnings():
            # brand is 'jp2 ', but has any icc profile.
            warnings.simplefilter("ignore")
            jp2 = Jp2k(filename)

        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(jp2.box[3].box[1])
            actual = fake_out.getvalue().strip()
        lines = ["Colour Specification Box (colr) @ (179, 1339)",
                 "    Method:  any ICC profile",
                 "    Precedence:  2",
                 "    Approximation:  accurately represents "
                 + "correct colorspace definition",
                 "    ICC Profile:",
                 "        {'Size': 1328,",
                 "         'Preferred CMM Type': 1634758764,",
                 "         'Version': '2.2.0',",
                 "         'Device Class': 'display device profile',",
                 "         'Color Space': 'RGB',",
                 "         'Connection Space': 'XYZ',",
                 "         'Datetime': "
                 + "datetime.datetime(2009, 2, 25, 11, 26, 11),",
                 "         'File Signature': 'acsp',",
                 "         'Platform': 'APPL',",
                 "         'Flags': 'not embedded, "
                 + "can be used independently',",
                 "         'Device Manufacturer': 'appl',",
                 "         'Device Model': '',",
                 "         'Device Attributes': 'reflective, glossy, "
                 + "positive media polarity, color media',",
                 "         'Rendering Intent': 'perceptual',",
                 "         'Illuminant': array([ 0.96420288,  1.        ,"
                 + "  0.8249054 ]),",
                 "         'Creator': 'appl'}"]

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(OPJ_DATA_ROOT is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_uuid(self):
        """verify printing of UUID box"""
        filename = opj_data_file('input/nonregression/text_GBR.jp2')
        with warnings.catch_warnings():
            # brand is 'jp2 ', but has any icc profile.
            warnings.simplefilter("ignore")
            jp2 = Jp2k(filename)

        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(jp2.box[4])
            actual = fake_out.getvalue().strip()
        lines = ['UUID Box (uuid) @ (1544, 25)',
                 '    UUID:  3a0d0218-0ae9-4115-b376-4bca41ce0e71',
                 '    UUID Data:  1 bytes']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(sys.hexversion < 0x03000000,
                     "Ordered dicts not printing well in 2.7")
    def test_exif_uuid(self):
        """Verify printing of exif information"""
        j = glymur.Jp2k(self.jp2file)

        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(j.box[3])
            actual = fake_out.getvalue().strip()

        lines = ["UUID Box (uuid) @ (77, 638)",
                 "    UUID:  4a706754-6966-6645-7869-662d3e4a5032 (Exif)",
                 "    UUID Data:  ",
                 "{'Image': {'Make': 'HTC',",
                 "           'Model': 'HTC Glacier',",
                 "           'XResolution': 72.0,",
                 "           'YResolution': 72.0,",
                 "           'ResolutionUnit': 2,",
                 "           'YCbCrPositioning': 1,",
                 "           'ExifTag': 138,",
                 "           'GPSTag': 354},",
                 " 'Photo': {'ISOSpeedRatings': 76,",
                 "           'ExifVersion': (48, 50, 50, 48),",
                 "           'DateTimeOriginal': '2013:02:09 14:47:53',",
                 "           'DateTimeDigitized': '2013:02:09 14:47:53',",
                 "           'ComponentsConfiguration': (1, 2, 3, 0),",
                 "           'FocalLength': 3.53,",
                 "           'FlashpixVersion': (48, 49, 48, 48),",
                 "           'ColorSpace': 1,",
                 "           'PixelXDimension': 2528,",
                 "           'PixelYDimension': 1424,",
                 "           'InteroperabilityTag': 324},",
                 " 'GPSInfo': {'GPSVersionID': (2, 2, 0),",
                 "             'GPSLatitudeRef': 'N',",
                 "             'GPSLatitude': [42.0, 20.0, 33.61],",
                 "             'GPSLongitudeRef': 'W',",
                 "             'GPSLongitude': [71.0, 5.0, 17.32],",
                 "             'GPSAltitudeRef': 0,",
                 "             'GPSAltitude': 0.0,",
                 "             'GPSTimeStamp': [19.0, 47.0, 53.0],",
                 "             'GPSMapDatum': 'WGS-84',",
                 "             'GPSProcessingMethod': (65,",
                 "                                     83,",
                 "                                     67,",
                 "                                     73,",
                 "                                     73,",
                 "                                     0,",
                 "                                     0,",
                 "                                     0,",
                 "                                     78,",
                 "                                     69,",
                 "                                     84,",
                 "                                     87,",
                 "                                     79,",
                 "                                     82,",
                 "                                     75),",
                 "             'GPSDateStamp': '2013:02:09'},",
                 " 'Iop': None}"]

        expected = '\n'.join(lines)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
