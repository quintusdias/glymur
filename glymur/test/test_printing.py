import os
import pkg_resources
import struct
import sys
import tempfile
import unittest

if sys.hexversion < 0x03000000:
    from StringIO import StringIO
else:
    from io import StringIO

import glymur
from glymur import Jp2k

try:
    data_root = os.environ['OPJ_DATA_ROOT']
except KeyError:
    data_root = None
except:
    raise


class TestPrinting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup a plain JP2 file without the two UUID boxes.
        jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                  "data/nemo.jp2")
        with tempfile.NamedTemporaryFile(suffix='.jp2', delete=False) as tfile:
            cls._plain_nemo_file = tfile.name
            ijfile = Jp2k(jp2file)
            data = ijfile.read(reduce=3)
            ojfile = Jp2k(cls._plain_nemo_file, 'wb')
            ojfile.write(data)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls._plain_nemo_file)

    def setUp(self):
        # Save sys.stdout.
        self.stdout = sys.stdout
        sys.stdout = StringIO()
        self.jp2file = pkg_resources.resource_filename(glymur.__name__,
                                                       "data/nemo.jp2")

        # Save the output of dumping nemo.jp2 for more than one test.
        lines = ['JPEG 2000 Signature Box (jP  ) @ (0, 12)',
                 '    Signature:  0d0a870a',
                 'File Type Box (ftyp) @ (12, 20)',
                 '    Brand:  jp2 ',
                 "    Compatibility:  ['jp2 ']",
                 'JP2 Header Box (jp2h) @ (32, 45)',
                 '    Image Header Box (ihdr) @ (40, 22)',
                 '        Size:  [182 324 3]',
                 '        Bitdepth:  8',
                 '        Signed:  False',
                 '        Compression:  wavelet',
                 '        Colorspace Unknown:  False',
                 '    Colour Specification Box (colr) @ (62, 15)',
                 '        Method:  enumerated colorspace',
                 '        Precedence:  0',
                 '        Colorspace:  sRGB',
                 'Contiguous Codestream Box (jp2c) @ (77, 112814)',
                 '    Main header:',
                 '        SOC marker segment @ (85, 0)',
                 '        SIZ marker segment @ (87, 47)',
                 '            Profile:  2',
                 '            Reference Grid Height, Width:  (182 x 324)',
                 '            Vertical, Horizontal Reference Grid Offset:  '
                 + '(0 x 0)',
                 '            Reference Tile Height, Width:  (182 x 324)',
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
        self.expectedPlain = '\n'.join(lines)

    def tearDown(self):
        # Restore stdout.
        sys.stdout = self.stdout

    def test_jp2dump(self):
        glymur.jp2dump(self._plain_nemo_file)
        actual = sys.stdout.getvalue().strip()

        # Get rid of the filename line, as it is not set in stone.
        lst = actual.split('\n')
        lst = lst[1:]
        actual = '\n'.join(lst)

        self.assertEqual(actual, self.expectedPlain)

    def test_COC_segment(self):
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        print(codestream.segment[5])
        actual = sys.stdout.getvalue().strip()

        lines = ['COC marker segment @ (3233, 9)',
                 '    Associated component:  1',
                 '    Coding style for this component:  '
                 + 'Entropy coder, PARTITION = 0',
                 '    Coding style parameters:',
                 '        Number of resolutions:  6',
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

    def test_COD_segment(self):
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        print(codestream.segment[2])
        actual = sys.stdout.getvalue().strip()

        lines = ['COD marker segment @ (3186, 12)',
                 '    Coding style:',
                 '        Entropy coder, without partitions',
                 '        SOP marker segments:  False',
                 '        EPH marker segments:  False',
                 '    Coding style parameters:',
                 '        Progression order:  LRCP',
                 '        Number of layers:  3',
                 '        Multiple component transformation usage:  '
                 + 'reversible',
                 '        Number of resolutions:  6',
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
        self.actual = actual
        self.expected = expected
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_CRG(self):
        filename = os.path.join(data_root, 'input/conformance/p0_03.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        print(codestream.segment[-5])
        actual = sys.stdout.getvalue().strip()
        lines = ['CRG marker segment at (87, 6)',
                 '    Vertical, Horizontal offset:  (0.50, 1.00)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_RGN(self):
        filename = os.path.join(data_root, 'input/conformance/p0_03.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream(header_only=False)
        print(codestream.segment[12])
        actual = sys.stdout.getvalue().strip()
        lines = ['RGN marker segment @ (310, 5)',
                 '    Associated component:  0',
                 '    ROI style:  0',
                 '    Parameter:  7']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_SOP(self):
        filename = os.path.join(data_root, 'input/conformance/p0_03.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream(header_only=False)
        print(codestream.segment[-2])
        actual = sys.stdout.getvalue().strip()
        lines = ['SOP marker segment @ (12836, 4)',
                 '    Nsop:  15']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_CME(self):
        # Test printing a CME or comment marker segment.
        filename = os.path.join(data_root, 'input/conformance/p0_02.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        # 2nd to last segment in the main header
        print(codestream.segment[-2])
        actual = sys.stdout.getvalue().strip()
        lines = ['CME marker segment @ (85, 45)',
                 '    "Creator: AV-J2K (c) 2000,2001 Algo Vision"']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_EOC_segment(self):
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        print(codestream.segment[-1])
        actual = sys.stdout.getvalue().strip()

        lines = ['EOC marker segment @ (1136552, 0)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_PLT_segment(self):
        filename = os.path.join(data_root, 'input/conformance/p0_07.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream(header_only=False)
        print(codestream.segment[49935])
        actual = sys.stdout.getvalue().strip()

        lines = ['PLT marker segment @ (7871146, 38)',
                 '    Index:  0',
                 '    Iplt:  [9, 122, 19, 30, 27, 9, 41, 62, 18, 29, 261,'
                 + ' 55, 82, 299, 93, 941, 951, 687, 1729, 1443, 1008, 2168,'
                 + ' 2188, 2223]']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_POD_segment(self):
        filename = os.path.join(data_root, 'input/conformance/p0_13.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        print(codestream.segment[8])
        actual = sys.stdout.getvalue().strip()

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

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_PPM_segment(self):
        filename = os.path.join(data_root, 'input/conformance/p1_03.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        print(codestream.segment[9])
        actual = sys.stdout.getvalue().strip()

        lines = ['PPM marker segment @ (213, 43712)',
                 '    Index:  0',
                 '    Data:  43709 uninterpreted bytes']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_PPT_segment(self):
        filename = os.path.join(data_root, 'input/conformance/p1_06.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream(header_only=False)
        print(codestream.segment[6])
        actual = sys.stdout.getvalue().strip()

        lines = ['PPT marker segment @ (155, 109)',
                 '    Index:  0',
                 '    Packet headers:  106 uninterpreted bytes']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_QCC_segment(self):
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        print(codestream.segment[6])
        actual = sys.stdout.getvalue().strip()

        lines = ['QCC marker segment @ (3244, 20)',
                 '    Associated Component:  1',
                 '    Quantization style:  no quantization, 2 guard bits',
                 '    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), '
                 + '(0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), '
                 + '(0, 10), (0, 9), (0, 9), (0, 10)]']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_QCD_segment_5x3_transform(self):
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        print(codestream.segment[3])
        actual = sys.stdout.getvalue().strip()

        lines = ['QCD marker segment @ (3200, 19)',
                 '    Quantization style:  no quantization, 2 guard bits',
                 '    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), '
                 + '(0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), '
                 + '(0, 9), (0, 10), (0, 9), (0, 9), (0, 10)]']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_SIZ_segment(self):
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        print(codestream.segment[1])
        actual = sys.stdout.getvalue().strip()

        lines = ['SIZ marker segment @ (3137, 47)',
                 '    Profile:  2',
                 '    Reference Grid Height, Width:  (1456 x 2592)',
                 '    Vertical, Horizontal Reference Grid Offset:  (0 x 0)',
                 '    Reference Tile Height, Width:  (512 x 512)',
                 '    Vertical, Horizontal Reference Tile Offset:  (0 x 0)',
                 '    Bitdepth:  (8, 8, 8)',
                 '    Signed:  (False, False, False)',
                 '    Vertical, Horizontal Subsampling:  '
                 + '((1, 1), (1, 1), (1, 1))']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_SOC_segment(self):
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream()
        print(codestream.segment[0])
        actual = sys.stdout.getvalue().strip()

        lines = ['SOC marker segment @ (3135, 0)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_SOD_segment(self):
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        print(codestream.segment[9])
        actual = sys.stdout.getvalue().strip()

        lines = ['SOD marker segment @ (3299, 0)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_SOT_segment(self):
        j = glymur.Jp2k(self.jp2file)
        codestream = j.get_codestream(header_only=False)
        print(codestream.segment[4])
        actual = sys.stdout.getvalue().strip()

        lines = ['SOT marker segment @ (3221, 10)',
                 '    Tile part index:  0',
                 '    Tile part length:  78629',
                 '    Tile part instance:  0',
                 '    Number of tile parts:  1']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_TLM_segment(self):
        filename = os.path.join(data_root, 'input/conformance/p0_15.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        print(codestream.segment[10])
        actual = sys.stdout.getvalue().strip()

        lines = ['TLM marker segment @ (268, 28)',
                 '    Index:  0',
                 '    Tile number:  (0, 1, 2, 3)',
                 '    Length:  (4267, 2117, 4080, 2081)']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_xmp(self):
        # Verify the printing of a UUID/XMP box.
        j = glymur.Jp2k(self.jp2file)
        print(j.box[4])
        actual = sys.stdout.getvalue().strip()
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

    def test_entire_file(self):
        j = glymur.Jp2k(self._plain_nemo_file)
        print(j)
        actual = sys.stdout.getvalue().strip()

        # Get rid of the filename line, as it is not set in stone.
        lst = actual.split('\n')
        lst = lst[1:]
        actual = '\n'.join(lst)

        self.assertEqual(actual, self.expectedPlain)

    def test_codestream(self):
        j = glymur.Jp2k(self.jp2file)
        print(j.get_codestream())
        actual = sys.stdout.getvalue().strip()
        lst = ['Codestream:',
               '    SOC marker segment @ (3135, 0)',
               '    SIZ marker segment @ (3137, 47)',
               '        Profile:  2',
               '        Reference Grid Height, Width:  (1456 x 2592)',
               '        Vertical, Horizontal Reference Grid Offset:  (0 x 0)',
               '        Reference Tile Height, Width:  (512 x 512)',
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
               '            Number of layers:  3',
               '            Multiple component transformation usage:  '
               + 'reversible',
               '            Number of resolutions:  6',
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
               '    QCD marker segment @ (3200, 19)',
               '        Quantization style:  no quantization, '
               + '2 guard bits',
               '        Step size:  [(0, 8), (0, 9), (0, 9), '
               + '(0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), '
               + '(0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), '
               + '(0, 10)]']
        expected = '\n'.join(lst)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_xml(self):
        filename = os.path.join(data_root, 'input/conformance/file1.jp2')
        j = glymur.Jp2k(filename)
        print(j.box[2])
        actual = sys.stdout.getvalue().strip()

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

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_component_definition(self):
        filename = os.path.join(data_root, 'input/conformance/file2.jp2')
        j = glymur.Jp2k(filename)
        print(j.box[2].box[2])
        actual = sys.stdout.getvalue().strip()
        lines = ['Component Definition Box (cdef) @ (81, 28)',
                 '    Component 0 (color) ==> (3)',
                 '    Component 1 (color) ==> (2)',
                 '    Component 2 (color) ==> (1)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_component_mapping(self):
        filename = os.path.join(data_root, 'input/conformance/file9.jp2')
        j = glymur.Jp2k(filename)
        print(j.box[2].box[2])
        actual = sys.stdout.getvalue().strip()
        lines = ['Component Mapping Box (cmap) @ (848, 20)',
                 '    Component 0 ==> palette column 0',
                 '    Component 0 ==> palette column 1',
                 '    Component 0 ==> palette column 2']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_palette(self):
        filename = os.path.join(data_root, 'input/conformance/file9.jp2')
        j = glymur.Jp2k(filename)
        print(j.box[2].box[1])
        actual = sys.stdout.getvalue().strip()
        lines = ['Palette Box (pclr) @ (66, 782)',
                 '    Size:  (256 x 3)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_palette(self):
        filename = os.path.join(data_root, 'input/conformance/file7.jp2')
        j = glymur.Jp2k(filename)
        print(j.box[2])
        actual = sys.stdout.getvalue().strip()
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

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_CRG(self):
        filename = os.path.join(data_root, 'input/conformance/p0_03.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        print(codestream.segment[6])
        actual = sys.stdout.getvalue().strip()
        lines = ['CRG marker segment @ (87, 6)',
                 '    Vertical, Horizontal offset:  (0.50, 1.00)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_differing_subsamples(self):
        # Issue 86.
        filename = os.path.join(data_root, 'input/conformance/p0_05.j2k')
        j = glymur.Jp2k(filename)
        codestream = j.get_codestream()
        print(codestream.segment[1])
        actual = sys.stdout.getvalue().strip()
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

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_palette_box(self):
        # Verify that palette (pclr) boxes are printed without error.
        filename = os.path.join(data_root, 'input/conformance/file9.jp2')
        j = glymur.Jp2k(filename)
        print(j.box[2].box[1])
        actual = sys.stdout.getvalue().strip()
        lines = ['Palette Box (pclr) @ (66, 782)',
                 '    Size:  (256 x 3)']
        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_asoc_label_box(self):
        # Construct a fake file with an asoc and a label box, as
        # OpenJPEG doesn't have such a file.
        data = glymur.Jp2k(self.jp2file).read(reduce=3)
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            j = glymur.Jp2k(tfile.name, 'wb')
            j.write(data)

            with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile2:

                # Offset of the codestream is where we start.
                buffer = tfile.read(77)
                tfile2.write(buffer)

                # read the rest of the file, it's the codestream.
                codestream = tfile.read()

                # Write the asoc superbox.
                # Length = 36, id is 'asoc'.
                buffer = struct.pack('>I4s', int(56), b'asoc')
                tfile2.write(buffer)

                # Write the contained label box
                buffer = struct.pack('>I4s', int(13), b'lbl ')
                tfile2.write(buffer)
                tfile2.write('label'.encode())

                # Write the xml box
                # Length = 36, id is 'xml '.
                buffer = struct.pack('>I4s', int(35), b'xml ')
                tfile2.write(buffer)

                buffer = '<test>this is a test</test>'
                buffer = buffer.encode()
                tfile2.write(buffer)

                # Now append the codestream.
                tfile2.write(codestream)
                tfile2.flush()

                jasoc = glymur.Jp2k(tfile2.name)
                print(jasoc.box[3])
                actual = sys.stdout.getvalue().strip()
                lines = ['Association Box (asoc) @ (77, 56)',
                         '    Label Box (lbl ) @ (85, 13)',
                         '        Label:  label',
                         '    XML Box (xml ) @ (98, 35)',
                         '        <test>this is a test</test>']
                expected = '\n'.join(lines)
                self.assertEqual(actual, expected)

    def test_less_common_boxes(self):
        with tempfile.NamedTemporaryFile(suffix='.jp2') as tfile:
            with open(self.jp2file, 'rb') as ifile:
                # Everything up until the jp2c box.
                buffer = ifile.read(77)
                tfile.write(buffer)

                # Write the UINF superbox
                # Length = 50, id is uinf.
                buffer = struct.pack('>I4s', int(50), b'uinf')
                tfile.write(buffer)

                # Write the ULST box.
                # Length is 26, 1 UUID, hard code that UUID as zeros.
                buffer = struct.pack('>I4sHIIII', int(26), b'ulst', int(1),
                                     int(0), int(0), int(0), int(0))
                tfile.write(buffer)

                # Write the URL box.
                # Length is 16, version is one byte, flag is 3 bytes, url
                # is the rest.
                buffer = struct.pack('>I4sBBBB',
                                     int(16), b'url ',
                                     int(0), int(0), int(0), int(0))
                tfile.write(buffer)
                buffer = struct.pack('>ssss', b'a', b'b', b'c', b'd')
                tfile.write(buffer)

                # Start the resolution superbox.
                buffer = struct.pack('>I4s', int(44), b'res ')
                tfile.write(buffer)

                # Write the capture resolution box.
                buffer = struct.pack('>I4sHHHHBB',
                                     int(18), b'resc',
                                     int(1), int(1), int(1), int(1),
                                     int(0), int(1))
                tfile.write(buffer)

                # Write the display resolution box.
                buffer = struct.pack('>I4sHHHHBB',
                                     int(18), b'resd',
                                     int(1), int(1), int(1), int(1),
                                     int(1), int(0))
                tfile.write(buffer)

                # Get the rest of the input file.
                buffer = ifile.read()
                tfile.write(buffer)
                tfile.flush()

            jp2k = glymur.Jp2k(tfile.name)
            print(jp2k.box[3])
            print(jp2k.box[4])
            actual = sys.stdout.getvalue().strip()
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

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_jpx_approximation_with_icc_profile(self):
        # ICC profiles may be used in JP2, but the approximation field should
        # be zero unless we have jpx.  This file does both.
        filename = os.path.join(data_root, 'input/nonregression/text_GBR.jp2')
        j = glymur.Jp2k(filename)

        print(j.box[3].box[1])
        actual = sys.stdout.getvalue().strip()
        lines = ['Colour Specification Box (colr) @ (179, 1339)',
                 '    Method:  any ICC profile',
                 '    Precedence:  2',
                 '    Approximation:  accurately represents '
                 + 'correct colorspace definition',
                 '    ICC Profile:  1328 bytes']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    @unittest.skipIf(data_root is None,
                     "OPJ_DATA_ROOT environment variable not set")
    def test_uuid(self):
        # UUID box
        filename = os.path.join(data_root, 'input/nonregression/text_GBR.jp2')
        j = glymur.Jp2k(filename)

        print(j.box[4])
        actual = sys.stdout.getvalue().strip()
        lines = ['UUID Box (uuid) @ (1544, 25)',
                 '    UUID:  3a0d0218-0ae9-4115-b376-4bca41ce0e71',
                 '    UUID Data:  1 bytes']

        expected = '\n'.join(lines)
        self.assertEqual(actual, expected)

    def test_exif_uuid(self):
        j = glymur.Jp2k(self.jp2file)

        print(j.box[3])
        actual = sys.stdout.getvalue().strip()

        lines = ["UUID Box (uuid) @ (77, 638)",
                 "    UUID:  4a706754-6966-6645-7869-662d3e4a5032 (Exif)",
                 "    UUID Data:  ",
                 "{'GPSInfo': {'GPSAltitude': 0.0,",
                 "             'GPSAltitudeRef': 0,",
                 "             'GPSDateStamp': '2013:02:09',",
                 "             'GPSLatitude': [42.0, 20.0, 33.61],",
                 "             'GPSLatitudeRef': 'N',",
                 "             'GPSLongitude': [71.0, 5.0, 17.32],",
                 "             'GPSLongitudeRef': 'W',",
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
                 "             'GPSTimeStamp': [19.0, 47.0, 53.0],",
                 "             'GPSVersionID': (2, 2, 0)},",
                 " 'Image': {'ExifTag': 138,",
                 "           'GPSTag': 354,",
                 "           'Make': 'HTC',",
                 "           'Model': 'HTC Glacier',",
                 "           'ResolutionUnit': 2,",
                 "           'XResolution': 72.0,",
                 "           'YCbCrPositioning': 1,",
                 "           'YResolution': 72.0},",
                 " 'Iop': None,",
                 " 'Photo': {'ColorSpace': 1,",
                 "           'ComponentsConfiguration': (1, 2, 3, 0),",
                 "           'DateTimeDigitized': '2013:02:09 14:47:53',",
                 "           'DateTimeOriginal': '2013:02:09 14:47:53',",
                 "           'ExifVersion': (48, 50, 50, 48),",
                 "           'FlashpixVersion': (48, 49, 48, 48),",
                 "           'FocalLength': 3.53,",
                 "           'ISOSpeedRatings': 76,",
                 "           'InteroperabilityTag': 324,",
                 "           'PixelXDimension': 2528,",
                 "           'PixelYDimension': 1424}}"]

        expected = '\n'.join(lines)

        self.assertEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()
