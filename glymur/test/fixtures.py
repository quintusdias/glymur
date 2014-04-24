"""
Test fixtures common to more than one test point.
"""
import os
import re
import sys
import textwrap
import warnings

import numpy as np

import glymur


# The Python XMP Toolkit may be used for XMP UUIDs, but only if available and
# if the version is at least 2.0.0.
try:
    import libxmp
    if hasattr(libxmp, 'version') and re.match(r'''[2-9].\d*.\d*''',
                                               libxmp.version.VERSION):
        from libxmp import XMPMeta
        HAS_PYTHON_XMP_TOOLKIT = True
    else:
        HAS_PYTHON_XMP_TOOLKIT = False
except:
    HAS_PYTHON_XMP_TOOLKIT = False


NO_READ_BACKEND_MSG = "Matplotlib with the PIL backend must be available in "
NO_READ_BACKEND_MSG += "order to run the tests in this suite."

try:
    OPJ_DATA_ROOT = os.environ['OPJ_DATA_ROOT']
except KeyError:
    OPJ_DATA_ROOT = None
except:
    raise


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


def opj_data_file(relative_file_name):
    """Compact way of forming a full filename from OpenJPEG's test suite."""
    jfile = os.path.join(OPJ_DATA_ROOT, relative_file_name)
    return jfile

try:
    from matplotlib.pyplot import imread

    # The whole point of trying to import PIL is to determine if it's there
    # or not.  We won't use it directly.
    # pylint:  disable=F0401,W0611
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


def peak_tolerance(amat, bmat):
    """Peak Tolerance"""
    diff = np.abs(amat.astype(np.double) - bmat.astype(np.double))
    ptol = diff.max()
    return ptol


def read_pgx(pgx_file):
    """Helper function for reading the PGX comparison files.
    """
    header, pos = read_pgx_header(pgx_file)

    tokens = re.split(r'\s', header)

    if (tokens[1][0] == 'M') and (sys.byteorder == 'little'):
        swapbytes = True
    elif (tokens[1][0] == 'L') and (sys.byteorder == 'big'):
        swapbytes = True
    else:
        swapbytes = False

    if (len(tokens) == 6):
        bitdepth = int(tokens[3])
        signed = bitdepth < 0
        if signed:
            bitdepth = -1 * bitdepth
        nrows = int(tokens[5])
        ncols = int(tokens[4])
    else:
        bitdepth = int(tokens[2])
        signed = bitdepth < 0
        if signed:
            bitdepth = -1 * bitdepth
        nrows = int(tokens[4])
        ncols = int(tokens[3])

    dtype = determine_pgx_datatype(signed, bitdepth)

    shape = [nrows, ncols]

    # Reopen the file in binary mode and seek to the start of the binary
    # data
    with open(pgx_file, 'rb') as fptr:
        fptr.seek(pos)
        data = np.fromfile(file=fptr, dtype=dtype).reshape(shape)

    return(data.byteswap(swapbytes))


def determine_pgx_datatype(signed, bitdepth):
    """Determine the datatype of the PGX file.

    Parameters
    ----------
    signed : bool
        True if the datatype is signed, false otherwise
    bitdepth : int
        How many bits are used to make up an image plane.  Should be 8 or 16.
    """
    if signed:
        if bitdepth <= 8:
            dtype = np.int8
        elif bitdepth <= 16:
            dtype = np.int16
        else:
            raise RuntimeError("unhandled bitdepth")
    else:
        if bitdepth <= 8:
            dtype = np.uint8
        elif bitdepth <= 16:
            dtype = np.uint16
        else:
            raise RuntimeError("unhandled bitdepth")

    return dtype


def read_pgx_header(pgx_file):
    """Open the file in ascii mode (not really) and read the header line.
    Will look something like

    PG ML + 8 128 128
    PG%[ \t]%c%c%[ \t+-]%d%[ \t]%d%[ \t]%d"
    """
    header = ''
    with open(pgx_file, 'rb') as fptr:
        while True:
            char = fptr.read(1)
            if char[0] == 10 or char == '\n':
                pos = fptr.tell()
                break
            else:
                if sys.hexversion < 0x03000000:
                    header += char
                else:
                    header += chr(char[0])

    header = header.rstrip()
    return header, pos

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

text_gbr_27 = """Colour Specification Box (colr) @ (179, 1339)
    Method:  any ICC profile
    Precedence:  2
    Approximation:  accurately represents correct colorspace definition
    ICC Profile:
        {'Color Space': 'RGB',
         'Connection Space': 'XYZ',
         'Creator': u'appl',
         'Datetime': datetime.datetime(2009, 2, 25, 11, 26, 11),
         'Device Attributes': 'reflective, glossy, positive media polarity, color media',
         'Device Class': 'display device profile',
         'Device Manufacturer': u'appl',
         'Device Model': '',
         'File Signature': u'acsp',
         'Flags': 'not embedded, can be used independently',
         'Illuminant': array([ 0.96420288,  1.        ,  0.8249054 ]),
         'Platform': u'APPL',
         'Preferred CMM Type': 1634758764,
         'Rendering Intent': 'perceptual',
         'Size': 1328,
         'Version': '2.2.0'}"""

text_gbr_33 = """Colour Specification Box (colr) @ (179, 1339)
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
         'Device Attributes': 'reflective, glossy, positive media polarity, color media',
         'Rendering Intent': 'perceptual',
         'Illuminant': array([ 0.96420288,  1.        ,  0.8249054 ]),
         'Creator': 'appl'}"""

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


# Metadata dump of nemo.
dump = r'''JPEG 2000 Signature Box (jP  ) @ (0, 12)
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
{0}
Contiguous Codestream Box (jp2c) @ (3223, 1132296)
    Main header:
        SOC marker segment @ (3231, 0)
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
                Precinct size:  default, 2^15 x 2^15
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
nemo_dump_full = dump.format(_indent(nemo_xmp))

nemo_dump_short = r"""JPEG 2000 Signature Box (jP  ) @ (0, 12)
File Type Box (ftyp) @ (12, 20)
JP2 Header Box (jp2h) @ (32, 45)
    Image Header Box (ihdr) @ (40, 22)
    Colour Specification Box (colr) @ (62, 15)
UUID Box (uuid) @ (77, 3146)
Contiguous Codestream Box (jp2c) @ (3223, 1132296)"""

nemo_dump_no_xml = '''JPEG 2000 Signature Box (jP  ) @ (0, 12)
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
    Main header:
        SOC marker segment @ (3231, 0)
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
                Precinct size:  default, 2^15 x 2^15
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

dump = r"""JPEG 2000 Signature Box (jP  ) @ (0, 12)
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
{0}
Contiguous Codestream Box (jp2c) @ (3223, 1132296)"""
nemo_dump_no_codestream = dump.format(_indent(nemo_xmp))

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

file1_xml = """XML Box (xml ) @ (36, 439)
    <IMAGE_CREATION xmlns="http://www.jpeg.org/jpx/1.0/xml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.jpeg.org/jpx/1.0/xml http://www.jpeg.org/metadata/15444-2.xsd">
    \t<GENERAL_CREATION_INFO>
    \t\t<CREATION_TIME>2001-11-01T13:45:00.000-06:00</CREATION_TIME>
    \t\t<IMAGE_SOURCE>Professional 120 Image</IMAGE_SOURCE>
    \t</GENERAL_CREATION_INFO>
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
        Precinct size:  default, 2^15 x 2^15
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

fragment_list_box = r"""Fragment List Box (flst) @ (-1, 0)
    Offset 0:  89
    Fragment Length 0:  1132288
    Data Reference 0:  0"""

number_list_box = r"""Number List Box (nlst) @ (-1, 0)
    Association[0]:  the rendered result
    Association[1]:  codestream 0
    Association[2]:  compositing layer 0"""


goodstuff = r"""Codestream:
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
            Precinct size:  default, 2^15 x 2^15
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

