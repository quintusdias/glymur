"""
Test fixtures common to more than one test point.
"""
import os
import re
import subprocess
import sys
import textwrap
import unittest
import warnings

import numpy as np

import glymur

# If openjpeg is not installed, many tests cannot be run.
if glymur.version.openjpeg_version < '1.5.0':
    OPENJPEG_NOT_AVAILABLE = True
    OPENJPEG_NOT_AVAILABLE_MSG = 'OpenJPEG library not installed'
else:
    OPENJPEG_NOT_AVAILABLE = False
    OPENJPEG_NOT_AVAILABLE_MSG = None

# Cannot reopen a named temporary file in windows.
WINDOWS_TMP_FILE_MSG = "cannot use NamedTemporaryFile like this in windows"


def low_memory_linux_machine():
    """
    Detect if the current machine is low-memory (< 2.5GB)

    This is primarily aimed at Digital Ocean VMs running linux.  Don't bother
    on mac or windows.

    Returns
    -------
    bool
        True if <2GB, False otherwise
    """
    if not sys.platform.startswith('linux'):
        return False
    cmd1 = "free -m"
    cmd2 = "tail -n +2"
    cmd3 = "awk '{sum += $2} END {print sum}'"
    p1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, shell=True,
                          stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(cmd3, shell=True,
                          stdin=p2.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    stdout, stderr = p3.communicate()
    nbytes = int(stdout.decode('utf-8').strip())
    return nbytes < 2000

class MetadataBase(unittest.TestCase):
    """
    Base class for testing metadata.

    This class has helper routines defined for testing metadata so that it can
    be subclassed and used easily.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def verify_codeblock_style(self, actual, style):
        """
        Verify the code-block style for SPcod and SPcoc parameters.

        This information is stored in a single byte.  Please reference
        Table A-17 in FCD15444-1
        """
        expected = 0
        if style[0]:
            # Selective arithmetic coding bypass
            expected |= 0x01
        if style[1]:
            # Reset context probabilities
            expected |= 0x02
        if style[2]:
            # Termination on each coding pass
            expected |= 0x04
        if style[3]:
            # Vertically causal context
            expected |= 0x08
        if style[4]:
            # Predictable termination
            expected |= 0x10
        if style[5]:
            # Segmentation symbols
            expected |= 0x20
        self.assertEqual(actual, expected)

    def verifySignatureBox(self, box):
        """
        The signature box is a constant.
        """
        self.assertEqual(box.signature, (13, 10, 135, 10))

    def verify_filetype_box(self, actual, expected):
        """
        All JP2 files should have a brand reading 'jp2 ' and just a single
        entry in the compatibility list, also 'jp2 '.  JPX files can have more
        compatibility items.
        """
        self.assertEqual(actual.brand, expected.brand)
        self.assertEqual(actual.minor_version, expected.minor_version)
        self.assertEqual(actual.minor_version, 0)
        for cl in expected.compatibility_list:
            self.assertIn(cl, actual.compatibility_list)

    def verifyRGNsegment(self, actual, expected):
        """
        verify the fields of a RGN segment
        """
        self.assertEqual(actual.crgn, expected.crgn)  # 0 = component
        self.assertEqual(actual.srgn, expected.srgn)  # 0 = implicit
        self.assertEqual(actual.sprgn, expected.sprgn)

    def verifySOTsegment(self, actual, expected):
        """
        verify the fields of a SOT (start of tile) segment
        """
        self.assertEqual(actual.isot, expected.isot)
        self.assertEqual(actual.psot, expected.psot)
        self.assertEqual(actual.tpsot, expected.tpsot)
        self.assertEqual(actual.tnsot, expected.tnsot)

    def verifyCMEsegment(self, actual, expected):
        """
        verify the fields of a CME (comment) segment
        """
        self.assertEqual(actual.rcme, expected.rcme)
        self.assertEqual(actual.ccme, expected.ccme)

    def verifySizSegment(self, actual, expected):
        """
        Verify the fields of the SIZ segment.
        """
        for field in ['rsiz', 'xsiz', 'ysiz', 'xosiz', 'yosiz', 'xtsiz',
                      'ytsiz', 'xtosiz', 'ytosiz', 'bitdepth',
                      'xrsiz', 'yrsiz']:
            self.assertEqual(getattr(actual, field), getattr(expected, field))

    def verifyImageHeaderBox(self, box1, box2):
        self.assertEqual(box1.height, box2.height)
        self.assertEqual(box1.width, box2.width)
        self.assertEqual(box1.num_components, box2.num_components)
        self.assertEqual(box1.bits_per_component, box2.bits_per_component)
        self.assertEqual(box1.signed, box2.signed)
        self.assertEqual(box1.compression, box2.compression)
        self.assertEqual(box1.colorspace_unknown, box2.colorspace_unknown)
        self.assertEqual(box1.ip_provided, box2.ip_provided)

    def verifyColourSpecificationBox(self, actual, expected):
        """
        Does not currently check icc profiles.
        """
        self.assertEqual(actual.method, expected.method)
        self.assertEqual(actual.precedence, expected.precedence)
        self.assertEqual(actual.approximation, expected.approximation)

        if expected.colorspace is None:
            self.assertIsNone(actual.colorspace)
            self.assertIsNotNone(actual.icc_profile)
        else:
            self.assertEqual(actual.colorspace, expected.colorspace)
            self.assertIsNone(actual.icc_profile)


NO_READ_BACKEND_MSG = "Matplotlib with the PIL backend must be available in "
NO_READ_BACKEND_MSG += "order to run the tests in this suite."

# The Cinema2K/4K tests seem to need the freeimage backend to skimage.io
# in order to work.  Unfortunately, scikit-image/freeimage is about as wonky as
# it gets.  Anaconda can get totally weirded out on versions up through 3.6.4
# on Python3 with scikit-image up through version 0.10.0.
NO_SKIMAGE_FREEIMAGE_SUPPORT = False
try:
    import skimage
    import skimage.io
    if (((sys.hexversion >= 0x03000000) and
         ('Anaconda' in sys.version) and
         (re.match('0.10', skimage.__version__)))):
        NO_SKIMAGE_FREEIMAGE_SUPPORT = True
    else:
        skimage.io.use_plugin('freeimage', 'imread')
except ((ImportError, RuntimeError)):
    NO_SKIMAGE_FREEIMAGE_SUPPORT = True

# Do we have gdal?
try:
    import gdal
    HAVE_GDAL = True
except ImportError:
    HAVE_GDAL = False

def _indent(textstr):
    """
    Indent a string.

    Textwrap's indent method only exists for 3.3 or above.  In 2.7 we have
    to fake it.

    Parameters
    ----------
    textstring : str
        String to be indented.
    indent_level : str
        Number of spaces of indentation to add.

    Returns
    -------
    indented_string : str
        Possibly multi-line string indented a certain bit.
    """
    if sys.hexversion >= 0x03030000:
        return textwrap.indent(textstr, '    ')
    else:
        lst = [('    ' + x) for x in textstr.split('\n')]
        return '\n'.join(lst)


try:
    import matplotlib
    if not re.match('[1-9]\.[3-9]', matplotlib.__version__):
        # Probably too old.  On Ubuntu 12.04.5, the old PIL
        # is still used for the backend, and it can't read
        # the images we need.
        raise ImportError('MPL is too old')  
    from matplotlib.pyplot import imread

    # The whole point of trying to import PIL is to determine if it's there
    # or not.  We won't use it directly.
    import PIL

    NO_READ_BACKEND = False
except ImportError:
    NO_READ_BACKEND = True


def read_image(infile):
    """Read image using matplotlib backend.

    Hopefully PIL(low) is installed as matplotlib's backend.  It issues
    warnings which we do not care about, so suppress them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = imread(infile)
    return data


def mse(amat, bmat):
    """Mean Square Error"""
    diff = amat.astype(np.double) - bmat.astype(np.double)
    err = np.mean(diff**2)
    return err


nemo_xmp = """<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<ns0:xmpmeta xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:ns0="adobe:ns:meta/" xmlns:ns2="http://ns.adobe.com/xap/1.0/" xmlns:ns3="http://ns.adobe.com/tiff/1.0/" xmlns:ns4="http://ns.adobe.com/exif/1.0/" xmlns:ns5="http://ns.adobe.com/photoshop/1.0/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" ns0:xmptk="Exempi + XMP Core 5.1.2">
 <rdf:RDF>
  <rdf:Description rdf:about="">
   <ns2:CreatorTool>Google</ns2:CreatorTool>
   <ns2:CreateDate>2013-02-09T14:47:53</ns2:CreateDate>
  </rdf:Description>
  <rdf:Description rdf:about="">
   <ns3:YCbCrPositioning>1</ns3:YCbCrPositioning>
   <ns3:XResolution>72/1</ns3:XResolution>
   <ns3:YResolution>72/1</ns3:YResolution>
   <ns3:ResolutionUnit>2</ns3:ResolutionUnit>
   <ns3:Make>HTC</ns3:Make>
   <ns3:Model>HTC Glacier</ns3:Model>
   <ns3:ImageWidth>2592</ns3:ImageWidth>
   <ns3:ImageLength>1456</ns3:ImageLength>
   <ns3:BitsPerSample>
    <rdf:Seq>
     <rdf:li>8</rdf:li>
     <rdf:li>8</rdf:li>
     <rdf:li>8</rdf:li>
    </rdf:Seq>
   </ns3:BitsPerSample>
   <ns3:PhotometricInterpretation>2</ns3:PhotometricInterpretation>
   <ns3:SamplesPerPixel>3</ns3:SamplesPerPixel>
   <ns3:WhitePoint>
    <rdf:Seq>
     <rdf:li>1343036288/4294967295</rdf:li>
     <rdf:li>1413044224/4294967295</rdf:li>
    </rdf:Seq>
   </ns3:WhitePoint>
   <ns3:PrimaryChromaticities>
    <rdf:Seq>
     <rdf:li>2748779008/4294967295</rdf:li>
     <rdf:li>1417339264/4294967295</rdf:li>
     <rdf:li>1288490240/4294967295</rdf:li>
     <rdf:li>2576980480/4294967295</rdf:li>
     <rdf:li>644245120/4294967295</rdf:li>
     <rdf:li>257698032/4294967295</rdf:li>
    </rdf:Seq>
   </ns3:PrimaryChromaticities>
  </rdf:Description>
  <rdf:Description rdf:about="">
   <ns4:ColorSpace>1</ns4:ColorSpace>
   <ns4:PixelXDimension>2528</ns4:PixelXDimension>
   <ns4:PixelYDimension>1424</ns4:PixelYDimension>
   <ns4:FocalLength>353/100</ns4:FocalLength>
   <ns4:GPSAltitudeRef>0</ns4:GPSAltitudeRef>
   <ns4:GPSAltitude>0/1</ns4:GPSAltitude>
   <ns4:GPSMapDatum>WGS-84</ns4:GPSMapDatum>
   <ns4:DateTimeOriginal>2013-02-09T14:47:53</ns4:DateTimeOriginal>
   <ns4:ISOSpeedRatings>
    <rdf:Seq>
     <rdf:li>76</rdf:li>
    </rdf:Seq>
   </ns4:ISOSpeedRatings>
   <ns4:ExifVersion>0220</ns4:ExifVersion>
   <ns4:FlashpixVersion>0100</ns4:FlashpixVersion>
   <ns4:ComponentsConfiguration>
    <rdf:Seq>
     <rdf:li>1</rdf:li>
     <rdf:li>2</rdf:li>
     <rdf:li>3</rdf:li>
     <rdf:li>0</rdf:li>
    </rdf:Seq>
   </ns4:ComponentsConfiguration>
   <ns4:GPSLatitude>42,20.56N</ns4:GPSLatitude>
   <ns4:GPSLongitude>71,5.29W</ns4:GPSLongitude>
   <ns4:GPSTimeStamp>2013-02-09T19:47:53Z</ns4:GPSTimeStamp>
   <ns4:GPSProcessingMethod>NETWORK</ns4:GPSProcessingMethod>
  </rdf:Description>
  <rdf:Description rdf:about="">
   <ns5:DateCreated>2013-02-09T14:47:53</ns5:DateCreated>
  </rdf:Description>
  <rdf:Description rdf:about="">
   <dc:Creator>
    <rdf:Seq>
     <rdf:li>Glymur</rdf:li>
     <rdf:li>Python XMP Toolkit</rdf:li>
    </rdf:Seq>
   </dc:Creator>
  </rdf:Description>
 </rdf:RDF>
</ns0:xmpmeta>
<?xpacket end="w"?>"""

nemo_xmp_box = """UUID Box (uuid) @ (77, 3146)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
    UUID Data:
{0}""".format(_indent(nemo_xmp))

SimpleRDF = """<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
  <rdf:Description rdf:about='Test:XMPCoreCoverage/kSimpleRDF'
                   xmlns:ns1='ns:test1/' xmlns:ns2='ns:test2/'>

    <ns1:SimpleProp>Simple value</ns1:SimpleProp>

    <ns1:Distros>
      <rdf:Bag>
        <rdf:li>Suse</rdf:li>
        <rdf:li>Fedora</rdf:li>
      </rdf:Bag>
    </ns1:Distros>

  </rdf:Description>
</rdf:RDF>"""

text_gbr_34 = """Colour Specification Box (colr) @ (179, 1339)
    Method:  any ICC profile
    Precedence:  2
    Approximation:  accurately represents correct colorspace definition
    ICC Profile:
        {'Size': 1328,
         'Preferred CMM Type': 1634758764,
         'Version': '2.2.0',
         'Device Class': 'display device profile',
         'Color Space': 'RGB',
         'Connection Space': 'XYZ',
         'Datetime': datetime.datetime(2009, 2, 25, 11, 26, 11),
         'File Signature': 'acsp',
         'Platform': 'APPL',
         'Flags': 'not embedded, can be used independently',
         'Device Manufacturer': 'appl',
         'Device Model': '',
         'Device Attributes': 'reflective, glossy, positive media polarity, color '
                              'media',
         'Rendering Intent': 'perceptual',
         'Illuminant': array([ 0.96420288,  1.        ,  0.8249054 ]),
         'Creator': 'appl'}"""

text_gbr_35 = """Colour Specification Box (colr) @ (179, 1339)
    Method:  any ICC profile
    Precedence:  2
    Approximation:  accurately represents correct colorspace definition
    ICC Profile:
        OrderedDict([('Size', 1328),
                     ('Preferred CMM Type', 1634758764),
                     ('Version', '2.2.0'),
                     ('Device Class', 'display device profile'),
                     ('Color Space', 'RGB'),
                     ('Connection Space', 'XYZ'),
                     ('Datetime', datetime.datetime(2009, 2, 25, 11, 26, 11)),
                     ('File Signature', 'acsp'),
                     ('Platform', 'APPL'),
                     ('Flags', 'not embedded, can be used independently'),
                     ('Device Manufacturer', 'appl'),
                     ('Device Model', ''),
                     ('Device Attributes',
                      'reflective, glossy, positive media polarity, color media'),
                     ('Rendering Intent', 'perceptual'),
                     ('Illuminant', array([ 0.96420288,  1.        ,  0.8249054 ])),
                     ('Creator', 'appl')])"""


# Metadata dump of nemo.
nemo_fmt = r'''JPEG 2000 Signature Box (jP  ) @ (0, 12)
    Signature:  0d0a870a
File Type Box (ftyp) @ (12, 20)
    Brand:  jp2 
    Compatibility:  ['jp2 ']
JP2 Header Box (jp2h) @ (32, 45)
    Image Header Box (ihdr) @ (40, 22)
        Size:  [1456 2592 3]
        Bitdepth:  8
        Signed:  False
        Compression:  wavelet
        Colorspace Unknown:  False
    Colour Specification Box (colr) @ (62, 15)
        Method:  enumerated colorspace
        Precedence:  0
        Colorspace:  sRGB
UUID Box (uuid) @ (77, 3146)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
    UUID Data:
{xmp}
Contiguous Codestream Box (jp2c) @ (3223, 1132296)
{codestream}'''

codestream_header = '''SOC marker segment @ (3231, 0)
SIZ marker segment @ (3233, 47)
    Profile:  no profile
    Reference Grid Height, Width:  (1456 x 2592)
    Vertical, Horizontal Reference Grid Offset:  (0 x 0)
    Reference Tile Height, Width:  (1456 x 2592)
    Vertical, Horizontal Reference Tile Offset:  (0 x 0)
    Bitdepth:  (8, 8, 8)
    Signed:  (False, False, False)
    Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
COD marker segment @ (3282, 12)
    Coding style:
        Entropy coder, without partitions
        SOP marker segments:  False
        EPH marker segments:  False
    Coding style parameters:
        Progression order:  LRCP
        Number of layers:  2
        Multiple component transformation usage:  reversible
        Number of resolutions:  2
        Code block height, width:  (64 x 64)
        Wavelet transform:  5-3 reversible
        Precinct size:  (32768, 32768)
        Code block context:
            Selective arithmetic coding bypass:  False
            Reset context probabilities on coding pass boundaries:  False
            Termination on each coding pass:  False
            Vertically stripe causal context:  False
            Predictable termination:  False
            Segmentation symbols:  False
QCD marker segment @ (3296, 7)
    Quantization style:  no quantization, 2 guard bits
    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
CME marker segment @ (3305, 37)
    "Created by OpenJPEG version 2.0.0"'''

codestream_trailer = """SOT marker segment @ (3344, 10)
    Tile part index:  0
    Tile part length:  1132173
    Tile part instance:  0
    Number of tile parts:  1
COC marker segment @ (3356, 9)
    Associated component:  1
    Coding style for this component:  Entropy coder, PARTITION = 0
    Coding style parameters:
        Number of resolutions:  2
        Code block height, width:  (64 x 64)
        Wavelet transform:  5-3 reversible
        Precinct size:  (32768, 32768)
        Code block context:
            Selective arithmetic coding bypass:  False
            Reset context probabilities on coding pass boundaries:  False
            Termination on each coding pass:  False
            Vertically stripe causal context:  False
            Predictable termination:  False
            Segmentation symbols:  False
QCC marker segment @ (3367, 8)
    Associated Component:  1
    Quantization style:  no quantization, 2 guard bits
    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
COC marker segment @ (3377, 9)
    Associated component:  2
    Coding style for this component:  Entropy coder, PARTITION = 0
    Coding style parameters:
        Number of resolutions:  2
        Code block height, width:  (64 x 64)
        Wavelet transform:  5-3 reversible
        Precinct size:  (32768, 32768)
        Code block context:
            Selective arithmetic coding bypass:  False
            Reset context probabilities on coding pass boundaries:  False
            Termination on each coding pass:  False
            Vertically stripe causal context:  False
            Predictable termination:  False
            Segmentation symbols:  False
QCC marker segment @ (3388, 8)
    Associated Component:  2
    Quantization style:  no quantization, 2 guard bits
    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
SOD marker segment @ (3398, 0)
EOC marker segment @ (1135517, 0)"""

codestream = '\n'.join([codestream_header, codestream_trailer])

_kwargs = {
    'xmp': _indent(nemo_xmp),
    'codestream': _indent(codestream_header)
}
nemo_with_codestream_header = nemo_fmt.format(**_kwargs)

nemo_dump_short = r"""JPEG 2000 Signature Box (jP  ) @ (0, 12)
File Type Box (ftyp) @ (12, 20)
JP2 Header Box (jp2h) @ (32, 45)
    Image Header Box (ihdr) @ (40, 22)
    Colour Specification Box (colr) @ (62, 15)
UUID Box (uuid) @ (77, 3146)
Contiguous Codestream Box (jp2c) @ (3223, 1132296)"""

_fmt = """JPEG 2000 Signature Box (jP  ) @ (0, 12)
    Signature:  0d0a870a
File Type Box (ftyp) @ (12, 20)
    Brand:  jp2 
    Compatibility:  ['jp2 ']
JP2 Header Box (jp2h) @ (32, 45)
    Image Header Box (ihdr) @ (40, 22)
        Size:  [1456 2592 3]
        Bitdepth:  8
        Signed:  False
        Compression:  wavelet
        Colorspace Unknown:  False
    Colour Specification Box (colr) @ (62, 15)
        Method:  enumerated colorspace
        Precedence:  0
        Colorspace:  sRGB
UUID Box (uuid) @ (77, 3146)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
    UUID Data:
{xmp}
Contiguous Codestream Box (jp2c) @ (3223, 1132296)"""
nemo_dump_no_codestream = _fmt.format(xmp=_indent(nemo_xmp))


nemo_dump_no_codestream_no_xml = r"""JPEG 2000 Signature Box (jP  ) @ (0, 12)
    Signature:  0d0a870a
File Type Box (ftyp) @ (12, 20)
    Brand:  jp2
    Compatibility:  ['jp2 ']
JP2 Header Box (jp2h) @ (32, 45)
    Image Header Box (ihdr) @ (40, 22)
        Size:  [1456 2592 3]
        Bitdepth:  8
        Signed:  False
        Compression:  wavelet
        Colorspace Unknown:  False
    Colour Specification Box (colr) @ (62, 15)
        Method:  enumerated colorspace
        Precedence:  0
        Colorspace:  sRGB
UUID Box (uuid) @ (77, 3146)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
Contiguous Codestream Box (jp2c) @ (3223, 1132296)"""

_kwargs = {
    'xmp': _indent(nemo_xmp),
    'codestream': _indent(codestream)
}
nemo = nemo_fmt.format(**_kwargs)

_fmt = '''JPEG 2000 Signature Box (jP  ) @ (0, 12)
    Signature:  0d0a870a
File Type Box (ftyp) @ (12, 20)
    Brand:  jp2 
    Compatibility:  ['jp2 ']
JP2 Header Box (jp2h) @ (32, 45)
    Image Header Box (ihdr) @ (40, 22)
        Size:  [1456 2592 3]
        Bitdepth:  8
        Signed:  False
        Compression:  wavelet
        Colorspace Unknown:  False
    Colour Specification Box (colr) @ (62, 15)
        Method:  enumerated colorspace
        Precedence:  0
        Colorspace:  sRGB
UUID Box (uuid) @ (77, 3146)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
Contiguous Codestream Box (jp2c) @ (3223, 1132296)
{codestream}'''
nemo_dump_no_xml = _fmt.format(codestream=_indent(codestream_header))

# Output of reader requirements printing for text_GBR.jp2
text_GBR_rreq = r"""Reader Requirements Box (rreq) @ (40, 109)
    Fully Understands Aspect Mask:  0xffff
    Display Completely Mask:  0xf8f0
    Standard Features and Masks:
        Feature 001:  0x8000 Deprecated - contains no extensions
        Feature 005:  0x4080 Unrestricted JPEG 2000 Part 1 codestream, ITU-T Rec. T.800 | ISO/IEC 15444-1
        Feature 012:  0x2040 Deprecated - codestream is contiguous
        Feature 018:  0x1020 Deprecated - support for compositing is not required
        Feature 044:  0x810 Compositing layer uses Any ICC profile
    Vendor Features:
        UUID 3a0d0218-0ae9-4115-b376-4bca41ce0e71
        UUID 47c92ccc-d1a1-4581-b904-38bb5467713b
        UUID bc45a774-dd50-4ec6-a9f6-f3a137f47e90
        UUID d7c8c5ef-951f-43b2-8757-042500f538e8"""

file1_xml = """<IMAGE_CREATION xmlns="http://www.jpeg.org/jpx/1.0/xml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.jpeg.org/jpx/1.0/xml http://www.jpeg.org/metadata/15444-2.xsd">
    <GENERAL_CREATION_INFO>
        <CREATION_TIME>2001-11-01T13:45:00.000-06:00</CREATION_TIME>
        <IMAGE_SOURCE>Professional 120 Image</IMAGE_SOURCE>
    </GENERAL_CREATION_INFO>
</IMAGE_CREATION>"""

file1_xml_box = """XML Box (xml ) @ (36, 439)
    <IMAGE_CREATION xmlns="http://www.jpeg.org/jpx/1.0/xml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.jpeg.org/jpx/1.0/xml http://www.jpeg.org/metadata/15444-2.xsd">
        <GENERAL_CREATION_INFO>
            <CREATION_TIME>2001-11-01T13:45:00.000-06:00</CREATION_TIME>
            <IMAGE_SOURCE>Professional 120 Image</IMAGE_SOURCE>
        </GENERAL_CREATION_INFO>
    </IMAGE_CREATION>"""

issue_182_cmap = """Component Mapping Box (cmap) @ (130, 24)
    Component 0 ==> palette column 0
    Component 0 ==> palette column 1
    Component 0 ==> palette column 2
    Component 0 ==> palette column 3"""

issue_183_colr = """Colour Specification Box (colr) @ (62, 12)
    Method:  restricted ICC profile
    Precedence:  0
    ICC Profile:  None"""


# Progression order is invalid.
issue_186_progression_order = """COD marker segment @ (174, 12)
    Coding style:
        Entropy coder, without partitions
        SOP marker segments:  False
        EPH marker segments:  False
    Coding style parameters:
        Progression order:  33 (invalid)
        Number of layers:  1
        Multiple component transformation usage:  reversible
        Number of resolutions:  6
        Code block height, width:  (32 x 32)
        Wavelet transform:  9-7 irreversible
        Precinct size:  (32768, 32768)
        Code block context:
            Selective arithmetic coding bypass:  False
            Reset context probabilities on coding pass boundaries:  False
            Termination on each coding pass:  False
            Vertically stripe causal context:  False
            Predictable termination:  False
            Segmentation symbols:  False"""

# Cinema 2K profile
cinema2k_profile = """SIZ marker segment @ (2, 47)
    Profile:  Cinema 2K
    Reference Grid Height, Width:  (1080 x 1920)
    Vertical, Horizontal Reference Grid Offset:  (0 x 0)
    Reference Tile Height, Width:  (1080 x 1920)
    Vertical, Horizontal Reference Tile Offset:  (0 x 0)
    Bitdepth:  (12, 12, 12)
    Signed:  (False, False, False)
    Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))"""

jplh_color_group_box = r"""Compositing Layer Header Box (jplh) @ (314227, 31)
    Colour Group Box (cgrp) @ (314235, 23)
        Colour Specification Box (colr) @ (314243, 15)
            Method:  enumerated colorspace
            Precedence:  0
            Colorspace:  sRGB"""

goodstuff_codestream_header = r"""File:  goodstuff.j2k
Codestream:
    SOC marker segment @ (0, 0)
    SIZ marker segment @ (2, 47)
        Profile:  no profile
        Reference Grid Height, Width:  (800 x 480)
        Vertical, Horizontal Reference Grid Offset:  (0 x 0)
        Reference Tile Height, Width:  (800 x 480)
        Vertical, Horizontal Reference Tile Offset:  (0 x 0)
        Bitdepth:  (8, 8, 8)
        Signed:  (False, False, False)
        Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
    COD marker segment @ (51, 12)
        Coding style:
            Entropy coder, without partitions
            SOP marker segments:  False
            EPH marker segments:  False
        Coding style parameters:
            Progression order:  LRCP
            Number of layers:  1
            Multiple component transformation usage:  reversible
            Number of resolutions:  6
            Code block height, width:  (64 x 64)
            Wavelet transform:  5-3 reversible
            Precinct size:  (32768, 32768)
            Code block context:
                Selective arithmetic coding bypass:  False
                Reset context probabilities on coding pass boundaries:  False
                Termination on each coding pass:  False
                Vertically stripe causal context:  False
                Predictable termination:  False
                Segmentation symbols:  False
    QCD marker segment @ (65, 19)
        Quantization style:  no quantization, 2 guard bits
        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10)]"""

goodstuff_with_full_header = r"""Codestream:
    SOC marker segment @ (0, 0)
    SIZ marker segment @ (2, 47)
        Profile:  no profile
        Reference Grid Height, Width:  (800 x 480)
        Vertical, Horizontal Reference Grid Offset:  (0 x 0)
        Reference Tile Height, Width:  (800 x 480)
        Vertical, Horizontal Reference Tile Offset:  (0 x 0)
        Bitdepth:  (8, 8, 8)
        Signed:  (False, False, False)
        Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
    COD marker segment @ (51, 12)
        Coding style:
            Entropy coder, without partitions
            SOP marker segments:  False
            EPH marker segments:  False
        Coding style parameters:
            Progression order:  LRCP
            Number of layers:  1
            Multiple component transformation usage:  reversible
            Number of resolutions:  6
            Code block height, width:  (64 x 64)
            Wavelet transform:  5-3 reversible
            Precinct size:  (32768, 32768)
            Code block context:
                Selective arithmetic coding bypass:  False
                Reset context probabilities on coding pass boundaries:  False
                Termination on each coding pass:  False
                Vertically stripe causal context:  False
                Predictable termination:  False
                Segmentation symbols:  False
    QCD marker segment @ (65, 19)
        Quantization style:  no quantization, 2 guard bits
        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10)]
    SOT marker segment @ (86, 10)
        Tile part index:  0
        Tile part length:  115132
        Tile part instance:  0
        Number of tile parts:  1
    COC marker segment @ (98, 9)
        Associated component:  1
        Coding style for this component:  Entropy coder, PARTITION = 0
        Coding style parameters:
            Number of resolutions:  6
            Code block height, width:  (64 x 64)
            Wavelet transform:  5-3 reversible
            Precinct size:  (32768, 32768)
            Code block context:
                Selective arithmetic coding bypass:  False
                Reset context probabilities on coding pass boundaries:  False
                Termination on each coding pass:  False
                Vertically stripe causal context:  False
                Predictable termination:  False
                Segmentation symbols:  False
    QCC marker segment @ (109, 20)
        Associated Component:  1
        Quantization style:  no quantization, 2 guard bits
        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10)]
    COC marker segment @ (131, 9)
        Associated component:  2
        Coding style for this component:  Entropy coder, PARTITION = 0
        Coding style parameters:
            Number of resolutions:  6
            Code block height, width:  (64 x 64)
            Wavelet transform:  5-3 reversible
            Precinct size:  (32768, 32768)
            Code block context:
                Selective arithmetic coding bypass:  False
                Reset context probabilities on coding pass boundaries:  False
                Termination on each coding pass:  False
                Vertically stripe causal context:  False
                Predictable termination:  False
                Segmentation symbols:  False
    QCC marker segment @ (142, 20)
        Associated Component:  2
        Quantization style:  no quantization, 2 guard bits
        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10)]
    SOD marker segment @ (164, 0)
    EOC marker segment @ (115218, 0)"""

# manually verified via gdalinfo
geotiff_uuid = """UUID Box (uuid) @ (149, 523)
    UUID:  b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03 (GeoTIFF)
    UUID Data:  Coordinate System =
        PROJCS["Equirectangular MARS",
            GEOGCS["GCS_MARS",
                DATUM["unknown",
                    SPHEROID["unnamed",3396190,0]],
                PRIMEM["Greenwich",0],
                UNIT["degree",0.0174532925199433]],
            PROJECTION["Equirectangular"],
            PARAMETER["latitude_of_origin",0],
            PARAMETER["central_meridian",180],
            PARAMETER["standard_parallel_1",0],
            PARAMETER["false_easting",0],
            PARAMETER["false_northing",0],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]]]
    Origin = (-2523306.125000000000000,-268608.875000000000000)
    Pixel Size = (0.250000000000000,-0.250000000000000)
    Corner Coordinates:
    Upper Left  (-2523306.125, -268608.875) (137d25'49.08"E,  4d31'53.74"S
    Lower Left  (-2523306.125, -268609.125) (137d25'49.08"E,  4d31'53.75"S
    Upper Right (-2523305.875, -268608.875) (137d25'49.09"E,  4d31'53.74"S
    Lower Right (-2523305.875, -268609.125) (137d25'49.09"E,  4d31'53.75"S
    Center      (-2523306.000, -268609.000) (137d25'49.09"E,  4d31'53.75"S"""

geotiff_uuid_without_gdal = """UUID Box (uuid) @ (149, 523)
    UUID:  b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03 (GeoTIFF)
    UUID Data:  OrderedDict([('ImageWidth', 1), ('ImageLength', 1), ('BitsPerSample', 8), ('Compression', 1), ('PhotometricInterpretation', 1), ('StripOffsets', 8), ('SamplesPerPixel', 1), ('RowsPerStrip', 1), ('StripByteCounts', 1), ('PlanarConfiguration', 1), ('ModelPixelScale', (0.25, 0.25, 0.0)), ('ModelTiePoint', (0.0, 0.0, 0.0, -2523306.125, -268608.875, 0.0)), ('GeoKeyDirectory', (1, 1, 0, 18, 1024, 0, 1, 1, 1025, 0, 1, 1, 1026, 34737, 21, 0, 2048, 0, 1, 32767, 2049, 34737, 9, 21, 2050, 0, 1, 32767, 2054, 0, 1, 9102, 2056, 0, 1, 32767, 2057, 34736, 1, 4, 2058, 34736, 1, 5, 3072, 0, 1, 32767, 3074, 0, 1, 32767, 3075, 0, 1, 17, 3076, 0, 1, 9001, 3082, 34736, 1, 2, 3083, 34736, 1, 3, 3088, 34736, 1, 1, 3089, 34736, 1, 0)), ('GeoDoubleParams', (0.0, 180.0, 0.0, 0.0, 3396190.0, 3396190.0)), ('GeoAsciiParams', 'Equirectangular MARS|GCS_MARS|')])"""

multiple_precinct_size = """COD marker segment @ (51, 18)
    Coding style:
        Entropy coder, with partitions
        SOP marker segments:  False
        EPH marker segments:  False
    Coding style parameters:
        Progression order:  LRCP
        Number of layers:  1
        Multiple component transformation usage:  reversible
        Number of resolutions:  6
        Code block height, width:  (64 x 64)
        Wavelet transform:  5-3 reversible
        Precinct size:  ((16, 16), (32, 32), (64, 64), (128, 128), (128, 128), (128, 128))
        Code block context:
            Selective arithmetic coding bypass:  False
            Reset context probabilities on coding pass boundaries:  False
            Termination on each coding pass:  False
            Vertically stripe causal context:  False
            Predictable termination:  False
            Segmentation symbols:  False"""

decompression_parameters_type = """<class 'glymur.lib.openjp2.DecompressionParametersType'>:
    cp_reduce: 0
    cp_layer: 0
    infile: b''
    outfile: b''
    decod_format: -1
    cod_format: -1
    DA_x0: 0
    DA_x1: 0
    DA_y0: 0
    DA_y1: 0
    m_verbose: 0
    tile_index: 0
    nb_tile_to_decode: 0
    jpwl_correct: 0
    jpwl_exp_comps: 0
    jpwl_max_tiles: 0
    flags: 0"""

default_progression_order_changes_type = """<class 'glymur.lib.openjp2.PocType'>:
    resno0: 0
    compno0: 0
    layno1: 0
    resno1: 0
    compno1: 0
    layno0: 0
    precno0: 0
    precno1: 0
    prg1: 0
    prg: 0
    progorder: b''
    tile: 0
    tx0: 0
    tx1: 0
    ty0: 0
    ty1: 0
    layS: 0
    resS: 0
    compS: 0
    prcS: 0
    layE: 0
    resE: 0
    compE: 0
    prcE: 0
    txS: 0
    txE: 0
    tyS: 0
    tyE: 0
    dx: 0
    dy: 0
    lay_t: 0
    res_t: 0
    comp_t: 0
    prec_t: 0
    tx0_t: 0
    ty0_t: 0"""

default_compression_parameters_type = """<class 'glymur.lib.openjp2.CompressionParametersType'>:
    tile_size_on: 0
    cp_tx0: 0
    cp_ty0: 0
    cp_tdx: 0
    cp_tdy: 0
    cp_disto_alloc: 0
    cp_fixed_alloc: 0
    cp_fixed_quality: 0
    cp_matrice: None
    cp_comment: None
    csty: 0
    prog_order: 0
    numpocs: 0
    numpocs: 0
    tcp_numlayers: 0
    tcp_rates: []
    tcp_distoratio: []
    numresolution: 6
    cblockw_init: 64
    cblockh_init: 64
    mode: 0
    irreversible: 0
    roi_compno: -1
    roi_shift: 0
    res_spec: 0
    prch_init: []
    prcw_init: []
    infile: b''
    outfile: b''
    index_on: 0
    index: b''
    image_offset_x0: 0
    image_offset_y0: 0
    subsampling_dx: 1
    subsampling_dy: 1
    decod_format: -1
    cod_format: -1
    jpwl_epc_on: 0
    jpwl_hprot_mh: 0
    jpwl_hprot_tph_tileno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_hprot_tph: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_pprot_tileno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_pprot_packno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_pprot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_sens_size: 0
    jpwl_sens_addr: 0
    jpwl_sens_range: 0
    jpwl_sens_mh: 0
    jpwl_sens_tph_tileno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_sens_tph: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cp_cinema: 0
    max_comp_size: 0
    cp_rsiz: 0
    tp_on: 0
    tp_flag: 0
    tcp_mct: 0
    jpip_on: 0
    mct_data: None
    max_cs_size: 0
    rsiz: 0"""

default_image_component_parameters = """<class 'glymur.lib.openjp2.ImageComptParmType'>:
    dx: 0
    dy: 0
    w: 0
    h: 0
    x0: 0
    y0: 0
    prec: 0
    bpp: 0
    sgnd: 0"""

# The "icc_profile_buf" field is problematic as it is a pointer value, i.e.
#
#     icc_profile_buf: <glymur.lib.openjp2.LP_c_ubyte object at 0x7f28cd5d5d90>
#
# Have to treat it as a regular expression.
default_image_type = """<class 'glymur.lib.openjp2.ImageType'>:
    x0: 0
    y0: 0
    x1: 0
    y1: 0
    numcomps: 0
    color_space: 0
    icc_profile_buf: <glymur.lib.openjp2.LP_c_ubyte object at 0x[0-9A-Fa-f]*>
    icc_profile_len: 0"""
