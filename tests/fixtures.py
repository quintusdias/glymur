"""
Test fixtures common to more than one test point.
"""
import re
import subprocess
import sys
import textwrap
import unittest

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

    def verifySizSegment(self, actual, expected):
        """
        Verify the fields of the SIZ segment.
        """
        for field in ['rsiz', 'xsiz', 'ysiz', 'xosiz', 'yosiz', 'xtsiz',
                      'ytsiz', 'xtosiz', 'ytosiz', 'bitdepth',
                      'xrsiz', 'yrsiz']:
            self.assertEqual(getattr(actual, field), getattr(expected, field))

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
x = ("Reader Requirements Box (rreq) @ (40, 109)\n"
     "    Fully Understands Aspect Mask:  0xffff\n"
     "    Display Completely Mask:  0xf8f0\n"
     "    Standard Features and Masks:\n"
     "        Feature 001:  0x8000 Deprecated - contains no extensions\n"
     "        Feature 005:  0x4080 Unrestricted JPEG 2000 Part 1 codestream, "
     "ITU-T Rec. T.800 | ISO/IEC 15444-1\n"
     "        Feature 012:  0x2040 Deprecated - codestream is contiguous\n"
     "        Feature 018:  0x1020 Deprecated - "
     "support for compositing is not required\n"
     "        Feature 044:  0x810 Compositing layer uses Any ICC profile\n"
     "    Vendor Features:\n"
     "        UUID 3a0d0218-0ae9-4115-b376-4bca41ce0e71\n"
     "        UUID 47c92ccc-d1a1-4581-b904-38bb5467713b\n"
     "        UUID bc45a774-dd50-4ec6-a9f6-f3a137f47e90\n"
     "        UUID d7c8c5ef-951f-43b2-8757-042500f538e8")
text_GBR_rreq = x

x = ('XML Box (xml ) @ (36, 439)\n'
     '<IMAGE_CREATION xmlns="http://www.jpeg.org/jpx/1.0/xml" '
     'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
     'xsi:schemaLocation="http://www.jpeg.org/jpx/1.0/xml '
     'http://www.jpeg.org/metadata/15444-2.xsd">\n'
     '    <GENERAL_CREATION_INFO>\n'
     '        <CREATION_TIME>2001-11-01T13:45:00.000-06:00</CREATION_TIME>\n'
     '        <IMAGE_SOURCE>Professional 120 Image</IMAGE_SOURCE>\n'
     '    </GENERAL_CREATION_INFO>\n'
     '</IMAGE_CREATION>')
file1_xml_box = x

x = ("File:  goodstuff.j2k\n"
     "Codestream:\n"
     "    SOC marker segment @ (0, 0)\n"
     "    SIZ marker segment @ (2, 47)\n"
     "        Profile:  no profile\n"
     "        Reference Grid Height, Width:  (800 x 480)\n"
     "        Vertical, Horizontal Reference Grid Offset:  (0 x 0)\n"
     "        Reference Tile Height, Width:  (800 x 480)\n"
     "        Vertical, Horizontal Reference Tile Offset:  (0 x 0)\n"
     "        Bitdepth:  (8, 8, 8)\n"
     "        Signed:  (False, False, False)\n"
     "        Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))\n"
     "    COD marker segment @ (51, 12)\n"
     "        Coding style:\n"
     "            Entropy coder, without partitions\n"
     "            SOP marker segments:  False\n"
     "            EPH marker segments:  False\n"
     "        Coding style parameters:\n"
     "            Progression order:  LRCP\n"
     "            Number of layers:  1\n"
     "            Multiple component transformation usage:  reversible\n"
     "            Number of resolutions:  6\n"
     "            Code block height, width:  (64 x 64)\n"
     "            Wavelet transform:  5-3 reversible\n"
     "            Precinct size:  (32768, 32768)\n"
     "            Code block context:\n"
     "                Selective arithmetic coding bypass:  False\n"
     "                Reset context probabilities "
     "on coding pass boundaries:  False\n"
     "                Termination on each coding pass:  False\n"
     "                Vertically stripe causal context:  False\n"
     "                Predictable termination:  False\n"
     "                Segmentation symbols:  False\n"
     "    QCD marker segment @ (65, 19)\n"
     "        Quantization style:  no quantization, 2 guard bits\n"
     "        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), "
     "(0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), "
     "(0, 9), (0, 10)]")
goodstuff_codestream_header = x

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

x = ("UUID Box (uuid) @ (149, 523)\n"
     "    UUID:  b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03 (GeoTIFF)\n"
     "    UUID Data:  "
     "OrderedDict([('ImageWidth', 1), "
     "('ImageLength', 1), "
     "('BitsPerSample', 8), "
     "('Compression', 1), "
     "('PhotometricInterpretation', 1), "
     "('StripOffsets', 8), "
     "('SamplesPerPixel', 1), "
     "('RowsPerStrip', 1), "
     "('StripByteCounts', 1), "
     "('PlanarConfiguration', 1), "
     "('ModelPixelScale', (0.25, 0.25, 0.0)), "
     "('ModelTiePoint', (0.0, 0.0, 0.0, -2523306.125, -268608.875, 0.0)), "
     "('GeoKeyDirectory', (1, 1, 0, 18, "
     "1024, 0, 1, 1, "
     "1025, 0, 1, 1, "
     "1026, 34737, 21, 0, "
     "2048, 0, 1, 32767, "
     "2049, 34737, 9, 21, "
     "2050, 0, 1, 32767, "
     "2054, 0, 1, 9102, "
     "2056, 0, 1, 32767, "
     "2057, 34736, 1, 4, "
     "2058, 34736, 1, 5, "
     "3072, 0, 1, 32767, "
     "3074, 0, 1, 32767, "
     "3075, 0, 1, 17, "
     "3076, 0, 1, 9001, "
     "3082, 34736, 1, 2, "
     "3083, 34736, 1, 3, "
     "3088, 34736, 1, 1, "
     "3089, 34736, 1, 0)), "
     "('GeoDoubleParams', (0.0, 180.0, 0.0, 0.0, 3396190.0, 3396190.0)), "
     "('GeoAsciiParams', 'Equirectangular MARS|GCS_MARS|')])")
geotiff_uuid_without_gdal = x

x = ("Codestream:\n"
     "    SOC marker segment @ (0, 0)\n"
     "    SIZ marker segment @ (2, 47)\n"
     "        Profile:  no profile\n"
     "        Reference Grid Height, Width:  (800 x 480)\n"
     "        Vertical, Horizontal Reference Grid Offset:  (0 x 0)\n"
     "        Reference Tile Height, Width:  (800 x 480)\n"
     "        Vertical, Horizontal Reference Tile Offset:  (0 x 0)\n"
     "        Bitdepth:  (8, 8, 8)\n"
     "        Signed:  (False, False, False)\n"
     "        Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))\n"
     "    COD marker segment @ (51, 12)\n"
     "        Coding style:\n"
     "            Entropy coder, without partitions\n"
     "            SOP marker segments:  False\n"
     "            EPH marker segments:  False\n"
     "        Coding style parameters:\n"
     "            Progression order:  LRCP\n"
     "            Number of layers:  1\n"
     "            Multiple component transformation usage:  reversible\n"
     "            Number of resolutions:  6\n"
     "            Code block height, width:  (64 x 64)\n"
     "            Wavelet transform:  5-3 reversible\n"
     "            Precinct size:  (32768, 32768)\n"
     "            Code block context:\n"
     "                Selective arithmetic coding bypass:  False\n"
     "                Reset context probabilities "
     "on coding pass boundaries:  False\n"
     "                Termination on each coding pass:  False\n"
     "                Vertically stripe causal context:  False\n"
     "                Predictable termination:  False\n"
     "                Segmentation symbols:  False\n"
     "    QCD marker segment @ (65, 19)\n"
     "        Quantization style:  no quantization, 2 guard bits\n"
     "        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), "
     "(0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), "
     "(0, 10), (0, 9), (0, 9), (0, 10)]\n"
     "    SOT marker segment @ (86, 10)\n"
     "        Tile part index:  0\n"
     "        Tile part length:  115132\n"
     "        Tile part instance:  0\n"
     "        Number of tile parts:  1\n"
     "    COC marker segment @ (98, 9)\n"
     "        Associated component:  1\n"
     "        Coding style for this component:  "
     "Entropy coder, PARTITION = 0\n"
     "        Coding style parameters:\n"
     "            Number of resolutions:  6\n"
     "            Code block height, width:  (64 x 64)\n"
     "            Wavelet transform:  5-3 reversible\n"
     "            Precinct size:  (32768, 32768)\n"
     "            Code block context:\n"
     "                Selective arithmetic coding bypass:  False\n"
     "                Reset context probabilities "
     "on coding pass boundaries:  False\n"
     "                Termination on each coding pass:  False\n"
     "                Vertically stripe causal context:  False\n"
     "                Predictable termination:  False\n"
     "                Segmentation symbols:  False\n"
     "    QCC marker segment @ (109, 20)\n"
     "        Associated Component:  1\n"
     "        Quantization style:  no quantization, 2 guard bits\n"
     "        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), "
     "(0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), "
     "(0, 10), (0, 9), (0, 9), (0, 10)]\n"
     "    COC marker segment @ (131, 9)\n"
     "        Associated component:  2\n"
     "        Coding style for this component:  "
     "Entropy coder, PARTITION = 0\n"
     "        Coding style parameters:\n"
     "            Number of resolutions:  6\n"
     "            Code block height, width:  (64 x 64)\n"
     "            Wavelet transform:  5-3 reversible\n"
     "            Precinct size:  (32768, 32768)\n"
     "            Code block context:\n"
     "                Selective arithmetic coding bypass:  False\n"
     "                Reset context probabilities "
     "on coding pass boundaries:  False\n"
     "                Termination on each coding pass:  False\n"
     "                Vertically stripe causal context:  False\n"
     "                Predictable termination:  False\n"
     "                Segmentation symbols:  False\n"
     "    QCC marker segment @ (142, 20)\n"
     "        Associated Component:  2\n"
     "        Quantization style:  no quantization, 2 guard bits\n"
     "        Step size:  [(0, 8), (0, 9), (0, 9), (0, 10), (0, 9), "
     "(0, 9), (0, 10), (0, 9), (0, 9), (0, 10), (0, 9), (0, 9), "
     "(0, 10), (0, 9), (0, 9), (0, 10)]\n"
     "    SOD marker segment @ (164, 0)\n"
     "    EOC marker segment @ (115218, 0)")
j2k_codestream_2 = x
