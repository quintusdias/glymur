"""
Test fixtures common to more than one test point.
"""
import os
import re
import sys
import warnings

import numpy as np

import glymur


# Need to know of the libopenjp2 version is the official 2.0.0 release and NOT
# the 2.0+ development version.
OPENJP2_IS_V2_OFFICIAL = False
if glymur.lib.openjp2.OPENJP2 is not None:
    if not hasattr(glymur.lib.openjp2.OPENJP2,
                   'opj_stream_create_default_file_stream_v3'):
        OPENJP2_IS_V2_OFFICIAL = True


NO_READ_BACKEND_MSG = "Matplotlib with the PIL backend must be available in "
NO_READ_BACKEND_MSG += "order to run the tests in this suite."

try:
    OPJ_DATA_ROOT = os.environ['OPJ_DATA_ROOT']
except KeyError:
    OPJ_DATA_ROOT = None
except:
    raise


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

nemo_xmp_box = """UUID Box (uuid) @ (77, 3146)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
    UUID Data:  <?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Exempi + XMP Core 4.4.0">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/">
   <xmp:CreatorTool>Google</xmp:CreatorTool>
   <xmp:CreateDate>2013-02-09T14:47:53</xmp:CreateDate>
  </rdf:Description>
  <rdf:Description rdf:about=""
    xmlns:tiff="http://ns.adobe.com/tiff/1.0/">
   <tiff:YCbCrPositioning>1</tiff:YCbCrPositioning>
   <tiff:XResolution>72/1</tiff:XResolution>
   <tiff:YResolution>72/1</tiff:YResolution>
   <tiff:ResolutionUnit>2</tiff:ResolutionUnit>
   <tiff:Make>HTC</tiff:Make>
   <tiff:Model>HTC Glacier</tiff:Model>
   <tiff:ImageWidth>2592</tiff:ImageWidth>
   <tiff:ImageLength>1456</tiff:ImageLength>
   <tiff:BitsPerSample>
    <rdf:Seq>
     <rdf:li>8</rdf:li>
     <rdf:li>8</rdf:li>
     <rdf:li>8</rdf:li>
    </rdf:Seq>
   </tiff:BitsPerSample>
   <tiff:PhotometricInterpretation>2</tiff:PhotometricInterpretation>
   <tiff:SamplesPerPixel>3</tiff:SamplesPerPixel>
   <tiff:WhitePoint>
    <rdf:Seq>
     <rdf:li>1343036288/4294967295</rdf:li>
     <rdf:li>1413044224/4294967295</rdf:li>
    </rdf:Seq>
   </tiff:WhitePoint>
   <tiff:PrimaryChromaticities>
    <rdf:Seq>
     <rdf:li>2748779008/4294967295</rdf:li>
     <rdf:li>1417339264/4294967295</rdf:li>
     <rdf:li>1288490240/4294967295</rdf:li>
     <rdf:li>2576980480/4294967295</rdf:li>
     <rdf:li>644245120/4294967295</rdf:li>
     <rdf:li>257698032/4294967295</rdf:li>
    </rdf:Seq>
   </tiff:PrimaryChromaticities>
  </rdf:Description>
  <rdf:Description rdf:about=""
    xmlns:exif="http://ns.adobe.com/exif/1.0/">
   <exif:ColorSpace>1</exif:ColorSpace>
   <exif:PixelXDimension>2528</exif:PixelXDimension>
   <exif:PixelYDimension>1424</exif:PixelYDimension>
   <exif:FocalLength>353/100</exif:FocalLength>
   <exif:GPSAltitudeRef>0</exif:GPSAltitudeRef>
   <exif:GPSAltitude>0/1</exif:GPSAltitude>
   <exif:GPSMapDatum>WGS-84</exif:GPSMapDatum>
   <exif:DateTimeOriginal>2013-02-09T14:47:53</exif:DateTimeOriginal>
   <exif:ISOSpeedRatings>
    <rdf:Seq>
     <rdf:li>76</rdf:li>
    </rdf:Seq>
   </exif:ISOSpeedRatings>
   <exif:ExifVersion>0220</exif:ExifVersion>
   <exif:FlashpixVersion>0100</exif:FlashpixVersion>
   <exif:ComponentsConfiguration>
    <rdf:Seq>
     <rdf:li>1</rdf:li>
     <rdf:li>2</rdf:li>
     <rdf:li>3</rdf:li>
     <rdf:li>0</rdf:li>
    </rdf:Seq>
   </exif:ComponentsConfiguration>
   <exif:GPSLatitude>42,20.56N</exif:GPSLatitude>
   <exif:GPSLongitude>71,5.29W</exif:GPSLongitude>
   <exif:GPSTimeStamp>2013-02-09T19:47:53Z</exif:GPSTimeStamp>
   <exif:GPSProcessingMethod>NETWORK</exif:GPSProcessingMethod>
  </rdf:Description>
  <rdf:Description rdf:about=""
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/">
   <photoshop:DateCreated>2013-02-09T14:47:53</photoshop:DateCreated>
  </rdf:Description>
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:Creator>
    <rdf:Seq>
     <rdf:li>Glymur</rdf:li>
     <rdf:li>Python XMP Toolkit</rdf:li>
    </rdf:Seq>
   </dc:Creator>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

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
