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
nemo_dump_full_opj2 = r'''JPEG 2000 Signature Box (jP  ) @ (0, 12)
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
UUID Box (uuid) @ (77, 638)
    UUID:  4a706754-6966-6645-7869-662d3e4a5032 (Exif)
    UUID Data:  
{'Image': {'Make': 'HTC',
           'Model': 'HTC Glacier',
           'XResolution': 72.0,
           'YResolution': 72.0,
           'ResolutionUnit': 2,
           'YCbCrPositioning': 1,
           'ExifTag': 138,
           'GPSTag': 354},
 'Photo': {'ISOSpeedRatings': 76,
           'ExifVersion': (48, 50, 50, 48),
           'DateTimeOriginal': '2013:02:09 14:47:53',
           'DateTimeDigitized': '2013:02:09 14:47:53',
           'ComponentsConfiguration': (1, 2, 3, 0),
           'FocalLength': 3.53,
           'FlashpixVersion': (48, 49, 48, 48),
           'ColorSpace': 1,
           'PixelXDimension': 2528,
           'PixelYDimension': 1424,
           'InteroperabilityTag': 324},
 'GPSInfo': {'GPSVersionID': (2, 2, 0),
             'GPSLatitudeRef': 'N',
             'GPSLatitude': [42.0, 20.0, 33.61],
             'GPSLongitudeRef': 'W',
             'GPSLongitude': [71.0, 5.0, 17.32],
             'GPSAltitudeRef': 0,
             'GPSAltitude': 0.0,
             'GPSTimeStamp': [19.0, 47.0, 53.0],
             'GPSMapDatum': 'WGS-84',
             'GPSProcessingMethod': (65,
                                     83,
                                     67,
                                     73,
                                     73,
                                     0,
                                     0,
                                     0,
                                     78,
                                     69,
                                     84,
                                     87,
                                     79,
                                     82,
                                     75),
             'GPSDateStamp': '2013:02:09'},
 'Iop': None}
UUID Box (uuid) @ (715, 2412)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
    UUID Data:  
    <ns0:xmpmeta xmlns:ns0="adobe:ns:meta/" xmlns:ns2="http://ns.adobe.com/xap/1.0/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" ns0:xmptk="XMP Core 4.4.0-Exiv2">
      <rdf:RDF>
        <rdf:Description ns2:CreatorTool="glymur" rdf:about="" />
      </rdf:RDF>
    </ns0:xmpmeta>
    
Contiguous Codestream Box (jp2c) @ (3127, 1132296)
    Main header:
        SOC marker segment @ (3135, 0)
        SIZ marker segment @ (3137, 47)
            Profile:  2
            Reference Grid Height, Width:  (1456 x 2592)
            Vertical, Horizontal Reference Grid Offset:  (0 x 0)
            Reference Tile Height, Width:  (1456 x 2592)
            Vertical, Horizontal Reference Tile Offset:  (0 x 0)
            Bitdepth:  (8, 8, 8)
            Signed:  (False, False, False)
            Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
        COD marker segment @ (3186, 12)
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
        QCD marker segment @ (3200, 7)
            Quantization style:  no quantization, 2 guard bits
            Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
        CME marker segment @ (3209, 37)
            "Created by OpenJPEG version 2.0.0"'''
nemo_dump_full_p27 = r'''JPEG 2000 Signature Box (jP  ) @ (0, 12)
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
UUID Box (uuid) @ (77, 638)
    UUID:  4a706754-6966-6645-7869-662d3e4a5032 (Exif)
    UUID Data:  
{'GPSInfo': OrderedDict([('GPSVersionID', (2, 2, 0)), ('GPSLatitudeRef', 'N'), ('GPSLatitude', [42.0, 20.0, 33.61]), ('GPSLongitudeRef', 'W'), ('GPSLongitude', [71.0, 5.0, 17.32]), ('GPSAltitudeRef', 0), ('GPSAltitude', 0.0), ('GPSTimeStamp', [19.0, 47.0, 53.0]), ('GPSMapDatum', 'WGS-84'), ('GPSProcessingMethod', (65, 83, 67, 73, 73, 0, 0, 0, 78, 69, 84, 87, 79, 82, 75)), ('GPSDateStamp', '2013:02:09')]),
 'Image': OrderedDict([('Make', 'HTC'), ('Model', 'HTC Glacier'), ('XResolution', 72.0), ('YResolution', 72.0), ('ResolutionUnit', 2), ('YCbCrPositioning', 1), ('ExifTag', 138), ('GPSTag', 354)]),
 'Iop': None,
 'Photo': OrderedDict([('ISOSpeedRatings', 76), ('ExifVersion', (48, 50, 50, 48)), ('DateTimeOriginal', '2013:02:09 14:47:53'), ('DateTimeDigitized', '2013:02:09 14:47:53'), ('ComponentsConfiguration', (1, 2, 3, 0)), ('FocalLength', 3.53), ('FlashpixVersion', (48, 49, 48, 48)), ('ColorSpace', 1), ('PixelXDimension', 2528), ('PixelYDimension', 1424), ('InteroperabilityTag', 324)])}
UUID Box (uuid) @ (715, 2412)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
    UUID Data:  
    <ns0:xmpmeta xmlns:ns0="adobe:ns:meta/" xmlns:ns2="http://ns.adobe.com/xap/1.0/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" ns0:xmptk="XMP Core 4.4.0-Exiv2">
      <rdf:RDF>
        <rdf:Description ns2:CreatorTool="glymur" rdf:about="" />
      </rdf:RDF>
    </ns0:xmpmeta>
    
Contiguous Codestream Box (jp2c) @ (3127, 1132296)
    Main header:
        SOC marker segment @ (3135, 0)
        SIZ marker segment @ (3137, 47)
            Profile:  2
            Reference Grid Height, Width:  (1456 x 2592)
            Vertical, Horizontal Reference Grid Offset:  (0 x 0)
            Reference Tile Height, Width:  (1456 x 2592)
            Vertical, Horizontal Reference Tile Offset:  (0 x 0)
            Bitdepth:  (8, 8, 8)
            Signed:  (False, False, False)
            Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
        COD marker segment @ (3186, 12)
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
        QCD marker segment @ (3200, 7)
            Quantization style:  no quantization, 2 guard bits
            Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
        CME marker segment @ (3209, 37)
            "Created by OpenJPEG version 2.0.0"'''
