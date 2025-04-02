# standard library imports
from collections import OrderedDict
import ctypes
from enum import IntEnum
import os
import queue
import re
import struct
import warnings

# 3rd party library imports
import numpy as np

# Local imports
from ..config import glymur_config

# The error messages queue
EQ = queue.Queue()

loader = ctypes.windll.LoadLibrary if os.name == "nt" else ctypes.CDLL
_LIBTIFF = glymur_config("tiff")
_LIBC = glymur_config("c")


class LibTIFFError(RuntimeError):
    """Raise this exception if we detect a generic error from libtiff."""

    pass


class Compression(IntEnum):
    """Compression scheme used on the image data.

    See Also
    --------
    Photometric : The color space of the image data.
    """

    NONE = 1
    CCITTRLE = 2  # CCITT modified Huffman RLE
    CCITTFAX3 = 3  # CCITT Group 3 fax encoding
    CCITT_T4 = 3  # CCITT T.4 (TIFF 6 name)
    CCITTFAX4 = 4  # CCITT Group 4 fax encoding
    CCITT_T6 = 4  # CCITT T.6 (TIFF 6 name)
    LZW = 5  # Lempel-Ziv  & Welch
    OJPEG = 6  # 6.0 JPEG
    JPEG = 7  # JPEG DCT compression
    T85 = 9  # TIFF/FX T.85 JBIG compression
    T43 = 10  # TIFF/FX T.43 colour by layered JBIG compression
    NEXT = 32766  # NeXT 2-bit RLE
    CCITTRLEW = 32771  # #1 w/ word alignment
    PACKBITS = 32773  # Macintosh RLE
    THUNDERSCAN = 32809  # ThunderScan RLE
    PIXARFILM = 32908  # companded 10bit LZW
    PIXARLOG = 32909  # companded 11bit ZIP
    DEFLATE = 32946  # compression
    ADOBE_DEFLATE = 8  # compression, as recognized by Adobe
    DCS = 32947  # DCS encoding
    JBIG = 34661  # JBIG
    SGILOG = 34676  # Log Luminance RLE
    SGILOG24 = 34677  # Log 24-bit packed
    JP2000 = 34712  # JPEG2000
    LZMA = 34925  # LZMA2


class InkSet(IntEnum):
    """
    The set of inks used in a separated (PhotometricInterpretation=5) image.
    """

    CMYK = 1
    MULTIINK = 2


class JPEGColorMode(IntEnum):
    """When writing images with photometric interpretation equal to YCbCr and
    compression equal to JPEG, the pseudo tag JPEGColorMode should usually be
    set to RGB, unless the image values truly are in YCbCr.

    See Also
    --------
    Photometric : The color space of the image data.
    """

    RAW = 0
    RGB = 1


class PlanarConfig(IntEnum):
    """How the components of each pixel are stored.

    Writing images with a PlanarConfig value of PlanarConfig.SEPARATE is not
    currently supported.
    """

    CONTIG = 1  # single image plane
    SEPARATE = 2  # separate planes of data


class Orientation(IntEnum):
    """The orientation of the image with respect to the rows and columns."""

    TOPLEFT = 1  # row 0 top, col 0 lhs */
    TOPRIGHT = 2  # row 0 top, col 0 rhs */
    BOTRIGHT = 3  # row 0 bottom, col 0 rhs */
    BOTLEFT = 4  # row 0 bottom, col 0 lhs */
    LEFTTOP = 5  # row 0 lhs, col 0 top */
    RIGHTTOP = 6  # row 0 rhs, col 0 top */
    RIGHTBOT = 7  # row 0 rhs, col 0 bottom */
    LEFTBOT = 8  # row 0 lhs, col 0 bottom */


class Photometric(IntEnum):
    """The color space of the image data."""

    MINISWHITE = 0  # value is white
    MINISBLACK = 1  # value is black
    RGB = 2  # color model
    PALETTE = 3  # map indexed
    MASK = 4  # holdout mask
    SEPARATED = 5  # color separations
    YCBCR = 6  # CCIR 601
    CIELAB = 8  # 1976 CIE L*a*b*
    ICCLAB = 9  # L*a*b* [Adobe TIFF Technote 4]
    ITULAB = 10  # L*a*b*
    CFA = 32803  # filter array
    LOGL = 32844  # Log2(L)
    LOGLUV = 32845  # Log2(L) (u',v')


class SampleFormat(IntEnum):
    """Specifies how to interpret each data sample in a pixel."""

    UINT = 1
    INT = 2
    IEEEFP = 3
    VOID = 4
    COMPLEXINT = 5
    COMPLEXIEEEP = 6


def _handle_error(module, fmt, ap):
    # Use VSPRINTF in the C library to put together the error message.
    # int vsprintf(char * buffer, const char * restrict format, va_list ap);
    buffer = ctypes.create_string_buffer(1000)

    argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    _LIBC.vsprintf.argtypes = argtypes
    _LIBC.vsprintf.restype = ctypes.c_int32
    _LIBC.vsprintf(buffer, fmt, ap)

    module = module.decode("utf-8")
    error_str = buffer.value.decode("utf-8")

    message = f"{module}: {error_str}"
    EQ.put(message)
    return None


def _handle_warning(module, fmt, ap):
    # Use VSPRINTF in the C library to put together the warning message.
    # int vsprintf(char * buffer, const char * restrict format, va_list ap);
    buffer = ctypes.create_string_buffer(1000)

    argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
    _LIBC.vsprintf.argtypes = argtypes
    _LIBC.vsprintf.restype = ctypes.c_int32
    _LIBC.vsprintf(buffer, fmt, ap)

    module = module.decode("utf-8")
    warning_str = buffer.value.decode("utf-8")

    message = f"{module}: {warning_str}"
    warnings.warn(message)


# Set the function types for the warning handler.
_WFUNCTYPE = ctypes.CFUNCTYPE(
    ctypes.c_void_p,  # return type of warning handler, void *
    ctypes.c_char_p,  # module
    ctypes.c_char_p,  # fmt
    ctypes.c_void_p,  # va_list
)

_ERROR_HANDLER = _WFUNCTYPE(_handle_error)
_WARNING_HANDLER = _WFUNCTYPE(_handle_warning)


def _set_error_warning_handlers():
    """Setup default python error and warning handlers."""
    old_warning_handler = setWarningHandler()
    old_error_handler = setErrorHandler()

    return old_error_handler, old_warning_handler


def _reset_error_warning_handlers(old_error_handler, old_warning_handler):
    """Restore previous error and warning handlers."""
    setWarningHandler(old_warning_handler)
    setErrorHandler(old_error_handler)


def close(fp):
    """Corresponds to TIFFClose"""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFClose.argtypes = ARGTYPES
    _LIBTIFF.TIFFClose.restype = None
    _LIBTIFF.TIFFClose(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)


def computeStrip(fp, row, sample):
    """Corresponds to TIFFComputeStrip"""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint16]
    _LIBTIFF.TIFFComputeStrip.argtypes = ARGTYPES
    _LIBTIFF.TIFFComputeStrip.restype = ctypes.c_uint32
    stripnum = _LIBTIFF.TIFFComputeStrip(fp, row, sample)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return stripnum


def computeTile(fp, x, y, z, sample):
    """Corresponds to TIFFComputeTile"""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint16,
    ]
    _LIBTIFF.TIFFComputeTile.argtypes = ARGTYPES
    _LIBTIFF.TIFFComputeTile.restype = ctypes.c_uint32
    tilenum = _LIBTIFF.TIFFComputeTile(fp, x, y, z, sample)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return tilenum


def isTiled(fp):
    """Corresponds to TIFFIsTiled"""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]

    _LIBTIFF.TIFFIsTiled.argtypes = ARGTYPES
    _LIBTIFF.TIFFIsTiled.restype = ctypes.c_int

    status = _LIBTIFF.TIFFIsTiled(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return status


def numberOfStrips(fp):
    """Corresponds to TIFFNumberOfStrips."""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFNumberOfStrips.argtypes = ARGTYPES
    _LIBTIFF.TIFFNumberOfStrips.restype = ctypes.c_uint32

    numstrips = _LIBTIFF.TIFFNumberOfStrips(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return numstrips


def numberOfTiles(fp):
    """Corresponds to TIFFNumberOfTiles."""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p]
    _LIBTIFF.TIFFNumberOfTiles.argtypes = ARGTYPES
    _LIBTIFF.TIFFNumberOfTiles.restype = ctypes.c_uint32

    numtiles = _LIBTIFF.TIFFNumberOfTiles(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return numtiles


def readEncodedStrip(fp, stripnum, strip, size=-1):
    """Corresponds to TIFFReadEncodedStrip"""
    err_handler, warn_handler = _set_error_warning_handlers()

    if size == -1:
        size = strip.nbytes

    ARGTYPES = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_int32
    ]
    _LIBTIFF.TIFFReadEncodedStrip.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadEncodedStrip.restype = check_error
    _LIBTIFF.TIFFReadEncodedStrip(
        fp, stripnum, strip.ctypes.data_as(ctypes.c_void_p), size
    )

    _reset_error_warning_handlers(err_handler, warn_handler)

    return strip


def readEncodedTile(fp, tilenum, tile, size=-1):
    """Corresponds to TIFFComputeTile"""
    err_handler, warn_handler = _set_error_warning_handlers()

    if size == -1:
        size = tile.nbytes

    ARGTYPES = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_int32
    ]
    _LIBTIFF.TIFFReadEncodedTile.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadEncodedTile.restype = check_error
    _LIBTIFF.TIFFReadEncodedTile(
        fp,
        tilenum,
        tile.ctypes.data_as(ctypes.c_void_p),
        -1
    )

    _reset_error_warning_handlers(err_handler, warn_handler)

    return tile


def readRGBAStrip(fp, row, strip):
    """Corresponds to TIFFReadRGBAStrip"""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    _LIBTIFF.TIFFReadRGBAStrip.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadRGBAStrip.restype = check_error
    _LIBTIFF.TIFFReadRGBAStrip(fp, row, strip.ctypes.data_as(ctypes.c_void_p))

    _reset_error_warning_handlers(err_handler, warn_handler)

    return strip


def readRGBATile(fp, x, y, tile):
    """Corresponds to TIFFReadRGBATile"""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p
    ]
    _LIBTIFF.TIFFReadRGBATile.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadRGBATile.restype = check_error
    _LIBTIFF.TIFFReadRGBATile(fp, x, y, tile.ctypes.data_as(ctypes.c_void_p))

    _reset_error_warning_handlers(err_handler, warn_handler)

    return tile


def readRGBAImageOriented(
    fp,
    width=None,
    height=None,
    orientation=Orientation.TOPLEFT
):
    """Read an image as if it were RGBA.

    This function corresponds to the TIFFReadRGBAImageOriented function in the
    libtiff library.

    Parameters
    ----------
    fp : ctypes void pointer
        File pointer returned by libtiff.
    width, height : int
        Width and height of the returned image.
    orientation : int
        The raster origin position.

    See Also
    --------
    Orientation
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int32,
        ctypes.c_int32,
    ]

    _LIBTIFF.TIFFReadRGBAImageOriented.argtypes = ARGTYPES
    _LIBTIFF.TIFFReadRGBAImageOriented.restype = check_error

    if width is None:
        width = getFieldDefaulted(fp, "ImageWidth")
    if height is None:
        height = getFieldDefaulted(fp, "ImageLength")

    img = np.zeros((height, width, 4), dtype=np.uint8)
    raster = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    _LIBTIFF.TIFFReadRGBAImageOriented(
        fp,
        width,
        height,
        raster,
        orientation,
        0
    )

    _reset_error_warning_handlers(err_handler, warn_handler)

    return img


def writeEncodedStrip(fp, stripnum, stripdata, size=-1):
    """Corresponds to TIFFWriteEncodedStrip."""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32
    ]
    _LIBTIFF.TIFFWriteEncodedStrip.argtypes = ARGTYPES
    _LIBTIFF.TIFFWriteEncodedStrip.restype = check_error
    raster = stripdata.ctypes.data_as(ctypes.c_void_p)

    if size == -1:
        size = stripdata.nbytes

    _LIBTIFF.TIFFWriteEncodedStrip(fp, stripnum, raster, size)

    _reset_error_warning_handlers(err_handler, warn_handler)


def writeEncodedTile(fp, tilenum, tiledata, size=-1):
    """Corresponds to TIFFWriteEncodedTile."""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32
    ]
    _LIBTIFF.TIFFWriteEncodedTile.argtypes = ARGTYPES
    _LIBTIFF.TIFFWriteEncodedTile.restype = check_error
    raster = tiledata.ctypes.data_as(ctypes.c_void_p)

    if size == -1:
        size = tiledata.nbytes

    _LIBTIFF.TIFFWriteEncodedTile(fp, tilenum, raster, size)

    _reset_error_warning_handlers(err_handler, warn_handler)


def RGBAImageOK(fp):
    """Corresponds to TIFFRGBAImageOK."""
    err_handler, warn_handler = _set_error_warning_handlers()

    emsg = ctypes.create_string_buffer(1024)
    ARGTYPES = [ctypes.c_void_p, ctypes.c_char_p]
    _LIBTIFF.TIFFRGBAImageOK.argtypes = ARGTYPES
    _LIBTIFF.TIFFRGBAImageOK.restype = ctypes.c_int
    ok = _LIBTIFF.TIFFRGBAImageOK(fp, emsg)

    _reset_error_warning_handlers(err_handler, warn_handler)

    if ok:
        return True
    else:
        return False


def getFieldDefaulted(fp, tag):
    """Corresponds to the TIFFGetFieldDefaulted library routine."""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    tag_num = TAGS[tag]["number"]

    # Append the proper return type for the tag.
    tag_type = TAGS[tag]["type"]
    ARGTYPES.append(ctypes.POINTER(TAGS[tag]["type"]))
    _LIBTIFF.TIFFGetFieldDefaulted.argtypes = ARGTYPES

    _LIBTIFF.TIFFGetFieldDefaulted.restype = check_error

    # instantiate the tag value
    item = tag_type()
    _LIBTIFF.TIFFGetFieldDefaulted(fp, tag_num, ctypes.byref(item))

    _reset_error_warning_handlers(err_handler, warn_handler)

    return item.value


def getVersion():
    """Corresponds to the TIFFGetVersion library routine."""
    try:
        _LIBTIFF.TIFFGetVersion.restype = ctypes.c_char_p
    except AttributeError:
        # libtiff not installed
        return "0.0.0"

    v = _LIBTIFF.TIFFGetVersion().decode("utf-8")

    # v would be something like
    #
    # LIBTIFF, Version 4.3.0
    # Copyright (c) 1988-1996 Sam Leffler
    # Copyright (c) 1991-1996 Silicon Graphics, Inc.
    #
    # All we want is the '4.3.0'
    m = re.search(r"(?P<version>\d+\.\d+\.\d+)", v)
    return m.group("version")


def open(filename, mode="r"):
    """Corresponds to TIFFOpen

    Parameters
    ----------
    filename : path or str
        Path to TIFF
    """
    err_handler, warn_handler = _set_error_warning_handlers()

    filename = str(filename)

    ARGTYPES = [ctypes.c_char_p, ctypes.c_char_p]
    _LIBTIFF.TIFFOpen.argtypes = ARGTYPES
    _LIBTIFF.TIFFOpen.restype = ctypes.c_void_p
    file_argument = ctypes.c_char_p(filename.encode())
    mode_argument = ctypes.c_char_p(mode.encode())
    fp = _LIBTIFF.TIFFOpen(file_argument, mode_argument)
    check_error(fp)

    _reset_error_warning_handlers(err_handler, warn_handler)

    return fp


def setErrorHandler(func=_ERROR_HANDLER):
    # The signature of the error handler is
    #     const char *module, const char *fmt, va_list ap
    #
    # The return type is void *
    _LIBTIFF.TIFFSetErrorHandler.argtypes = [_WFUNCTYPE]
    _LIBTIFF.TIFFSetErrorHandler.restype = _WFUNCTYPE
    old_error_handler = _LIBTIFF.TIFFSetErrorHandler(func)
    return old_error_handler


def setField(fp, tag, *value):
    """Corresponds to TIFFSetField"""
    err_handler, warn_handler = _set_error_warning_handlers()

    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]

    # Append the proper return type for the tag.
    tag_num = TAGS[tag]["number"]
    tag_type = TAGS[tag]["type"]

    try:
        for ttype in tag_type:
            ARGTYPES.append(ttype)
    except TypeError:
        ARGTYPES.append(tag_type)

    _LIBTIFF.TIFFSetField.argtypes = ARGTYPES
    _LIBTIFF.TIFFSetField.restype = check_error

    if len(value) == 1 and tag_type == ctypes.c_char_p:
        _LIBTIFF.TIFFSetField(fp, tag_num, ctypes.c_char_p(value[0].encode()))
    elif len(value) == 1:
        _LIBTIFF.TIFFSetField(fp, tag_num, value[0])
    else:
        _LIBTIFF.TIFFSetField(fp, tag_num, *value)

    _reset_error_warning_handlers(err_handler, warn_handler)


def setWarningHandler(func=_WARNING_HANDLER):
    # The signature of the warning handler is
    #     const char *module, const char *fmt, va_list ap
    #
    # The return type is void *
    _LIBTIFF.TIFFSetWarningHandler.argtypes = [_WFUNCTYPE]
    _LIBTIFF.TIFFSetWarningHandler.restype = _WFUNCTYPE
    old_warning_handler = _LIBTIFF.TIFFSetWarningHandler(func)
    return old_warning_handler


def check_error(status):
    """Set a generic function as the restype attribute of all TIFF
    functions that return a int value.  This way we do not have to check
    for error status in each wrapping function and an exception will always be
    appropriately raised.
    """
    msg = ""
    while not EQ.empty():
        msg = EQ.get()
        raise LibTIFFError(msg)

    if status == 0:
        raise RuntimeError("failed")


TAGS = {
    "GPSVersionID": {
        "number": 0,
        "type": ctypes.c_char_p,
    },
    "GPSLatitudeRef": {
        "number": 1,
        "type": ctypes.c_char_p,
    },
    "GPSLatitude": {
        "number": 2,
        "type": ctypes.c_char_p,
    },
    "GPSLongitudeRef": {
        "number": 3,
        "type": ctypes.c_char_p,
    },
    "GPSLongitude": {
        "number": 4,
        "type": ctypes.c_char_p,
    },
    "GPSAltitudeRef": {
        "number": 5,
        "type": ctypes.c_char_p,
    },
    "GPSAltitude": {
        "number": 6,
        "type": ctypes.c_char_p,
    },
    "GPSTimestamp": {
        "number": 7,
        "type": ctypes.c_char_p,
    },
    "GPSSatellites": {
        "number": 8,
        "type": ctypes.c_char_p,
    },
    "GPSStatus": {
        "number": 9,
        "type": ctypes.c_char_p,
    },
    "GPSMeasureMode": {
        "number": 10,
        "type": ctypes.c_char_p,
    },
    "GPSDOP": {
        "number": 11,
        "type": ctypes.c_char_p,
    },
    "GPSSpeedRef": {
        "number": 12,
        "type": ctypes.c_char_p,
    },
    "GPSSpeed": {
        "number": 13,
        "type": ctypes.c_char_p,
    },
    "GPSTrackRef": {
        "number": 14,
        "type": ctypes.c_char_p,
    },
    "GPSTrack": {
        "number": 15,
        "type": ctypes.c_char_p,
    },
    "GPSImgDirectionRef": {
        "number": 16,
        "type": ctypes.c_char_p,
    },
    "GPSImgDirection": {
        "number": 17,
        "type": ctypes.c_char_p,
    },
    "GPSMapDatum": {
        "number": 18,
        "type": ctypes.c_char_p,
    },
    "GPSDestLatitudeRef": {
        "number": 19,
        "type": ctypes.c_char_p,
    },
    "GPSDestLatitude": {
        "number": 20,
        "type": ctypes.c_char_p,
    },
    "GPSDestLongitudeRef": {
        "number": 21,
        "type": ctypes.c_char_p,
    },
    "GPSDestLongitude": {
        "number": 22,
        "type": ctypes.c_char_p,
    },
    "GPSBearingRef": {
        "number": 23,
        "type": ctypes.c_char_p,
    },
    "GPSBearing": {
        "number": 24,
        "type": ctypes.c_char_p,
    },
    "GPSDestDistanceRef": {
        "number": 25,
        "type": ctypes.c_char_p,
    },
    "GPSDestDistance": {
        "number": 26,
        "type": ctypes.c_int16,
    },
    "GPSProcessingMethod": {
        "number": 27,
        "type": ctypes.c_char_p,
    },
    "GPSAreaInformation": {
        "number": 28,
        "type": ctypes.c_char_p,
    },
    "GPSDateStamp": {
        "number": 29,
        "type": ctypes.c_char_p,
    },
    "GPSDifferential": {
        "number": 30,
        "type": ctypes.c_char_p,
    },
    "GPSHPositioningError": {
        "number": 31,
        "type": ctypes.c_int16,
    },
    "SubFileType": {
        "number": 254,
        "type": ctypes.c_int16,
    },
    "OSubFileType": {
        "number": 255,
        "type": ctypes.c_int16,
    },
    "ImageWidth": {
        "number": 256,
        "type": ctypes.c_int32,
    },
    "ImageLength": {
        "number": 257,
        "type": ctypes.c_int32,
    },
    "BitsPerSample": {
        "number": 258,
        "type": ctypes.c_int16,
    },
    "Compression": {
        "number": 259,
        "type": ctypes.c_int16,
    },
    "Photometric": {
        "number": 262,
        "type": ctypes.c_int16,
    },
    "Threshholding": {
        "number": 263,
        "type": ctypes.c_int16,
    },
    "CellWidth": {
        "number": 264,
        "type": ctypes.c_int16,
    },
    "CellLength": {
        "number": 265,
        "type": ctypes.c_int16,
    },
    "FillOrder": {
        "number": 266,
        "type": ctypes.c_int16,
    },
    "DocumentName": {
        "number": 269,
        "type": ctypes.c_char_p,
    },
    "ImageDescription": {
        "number": 270,
        "type": ctypes.c_char_p,
    },
    "Make": {
        "number": 271,
        "type": ctypes.c_char_p,
    },
    "Model": {
        "number": 272,
        "type": ctypes.c_char_p,
    },
    "StripOffsets": {
        "number": 273,
        "type": (ctypes.c_int32, ctypes.c_int64),
    },
    "Orientation": {
        "number": 274,
        "type": ctypes.c_int16,
    },
    "SamplesPerPixel": {
        "number": 277,
        "type": ctypes.c_int16,
    },
    "RowsPerStrip": {
        "number": 278,
        "type": ctypes.c_int16,
    },
    "StripByteCounts": {
        "number": 279,
        "type": None,
    },
    "MinSampleValue": {
        "number": 280,
        "type": ctypes.c_int16,
    },
    "MaxSampleValue": {
        "number": 281,
        "type": ctypes.c_int16,
    },
    "XResolution": {
        "number": 282,
        "type": ctypes.c_double,
    },
    "YResolution": {
        "number": 283,
        "type": ctypes.c_double,
    },
    "PlanarConfig": {
        "number": 284,
        "type": ctypes.c_int16,
    },
    "PageName": {
        "number": 285,
        "type": ctypes.c_char_p,
    },
    "XPosition": {
        "number": 286,
        "type": ctypes.c_double,
    },
    "YPosition": {
        "number": 287,
        "type": ctypes.c_double,
    },
    "FreeOffsets": {
        "number": 288,
        "type": ctypes.c_int32,
    },
    "FreeByteCounts": {
        "number": 289,
        "type": ctypes.c_int32,
    },
    "GrayResponseUnit": {
        "number": 290,
        "type": ctypes.c_int16,
    },
    "GrayResponseCurve": {
        "number": 291,
        "type": None,
    },
    "T4Options": {
        "number": 292,
        "type": None,
    },
    "T6Options": {
        "number": 293,
        "type": None,
    },
    "ResolutionUnit": {
        "number": 296,
        "type": ctypes.c_int16,
    },
    "PageNumber": {
        "number": 297,
        "type": (ctypes.c_int16, ctypes.c_uint16),
    },
    "TransferFunction": {
        "number": 301,
        "type": None,
    },
    "Software": {
        "number": 305,
        "type": ctypes.c_char_p,
    },
    "Datetime": {
        "number": 306,
        "type": ctypes.c_char_p,
    },
    "Artist": {
        "number": 315,
        "type": ctypes.c_char_p,
    },
    "HostComputer": {
        "number": 316,
        "type": ctypes.c_char_p,
    },
    "Predictor": {
        "number": 317,
        "type": ctypes.c_int16,
    },
    "WhitePoint": {
        "number": 318,
        "type": ctypes.c_double,
    },
    "PrimaryChromaticities": {
        "number": 319,
        "type": None,
    },
    "ColorMap": {
        "number": 320,
        "type": (ctypes.c_int16, ctypes.c_uint16),
    },
    "HalfToneHints": {
        "number": 321,
        "type": ctypes.c_int16,
    },
    "TileWidth": {
        "number": 322,
        "type": ctypes.c_int32,
    },
    "TileLength": {
        "number": 323,
        "type": ctypes.c_int32,
    },
    "TileOffsets": {
        "number": 324,
        "type": None,
    },
    "TileByteCounts": {
        "number": 325,
        "type": None,
    },
    "BadFaxLines": {
        "number": 326,
        "type": None,
    },
    "CleanFaxData": {
        "number": 327,
        "type": None,
    },
    "ConsecutiveBadFaxLines": {
        "number": 328,
        "type": None,
    },
    "SubIFDs": {
        "number": 330,
        "type": None,
    },
    "InkSet": {
        "number": 332,
        "type": ctypes.c_uint16,
    },
    "InkNames": {
        "number": 333,
        "type": ctypes.c_char_p,
    },
    "NumberOfInks": {
        "number": 334,
        "type": ctypes.c_uint16,
    },
    "DotRange": {
        "number": 336,
        "type": None,
    },
    "TargetPrinter": {
        "number": 337,
        "type": ctypes.c_uint16,
    },
    "ExtraSamples": {
        "number": 338,
        "type": ctypes.c_uint16,
    },
    "SampleFormat": {
        "number": 339,
        "type": ctypes.c_uint16,
    },
    "SMinSampleValue": {
        "number": 340,
        "type": ctypes.c_double,
    },
    "SMaxSampleValue": {
        "number": 341,
        "type": ctypes.c_double,
    },
    "TransferRange": {
        "number": 342,
        "type": None,
    },
    "ClipPath": {
        "number": 343,
        "type": None,
    },
    "XClipPathUnits": {
        "number": 344,
        "type": None,
    },
    "YClipPathUnits": {
        "number": 345,
        "type": None,
    },
    "Indexed": {
        "number": 346,
        "type": None,
    },
    "JPEGTables": {
        "number": 347,
        "type": None,
    },
    "OPIProxy": {
        "number": 351,
        "type": None,
    },
    "GlobalParametersIFD": {
        "number": 400,
        "type": None,
    },
    "ProfileType": {
        "number": 401,
        "type": None,
    },
    "FaxProfile": {
        "number": 402,
        "type": ctypes.c_uint8,
    },
    "CodingMethods": {
        "number": 403,
        "type": None,
    },
    "VersionYear": {
        "number": 404,
        "type": None,
    },
    "ModeNumber": {
        "number": 405,
        "type": None,
    },
    "Decode": {
        "number": 433,
        "type": None,
    },
    "DefaultImageColor": {
        "number": 434,
        "type": None,
    },
    "JPEGProc": {
        "number": 512,
        "type": None,
    },
    "JPEGInterchangeFormat": {
        "number": 513,
        "type": None,
    },
    "JPEGInterchangeFormatLength": {
        "number": 514,
        "type": None,
    },
    "JPEGRestartInterval": {
        "number": 515,
        "type": None,
    },
    "JPEGLosslessPredictors": {
        "number": 517,
        "type": None,
    },
    "JPEGPointTransforms": {
        "number": 518,
        "type": None,
    },
    "JPEGQTables": {
        "number": 519,
        "type": None,
    },
    "JPEGDCTables": {
        "number": 520,
        "type": None,
    },
    "JPEGACTables": {
        "number": 521,
        "type": None,
    },
    "YCbCrCoefficients": {
        "number": 529,
        "type": (ctypes.c_float, ctypes.c_float, ctypes.c_float),
    },
    "YCbCrSubsampling": {
        "number": 530,
        "type": (ctypes.c_uint16, ctypes.c_uint16),
    },
    "YCbCrPositioning": {
        "number": 531,
        "type": ctypes.c_uint16,
    },
    "ReferenceBlackWhite": {
        "number": 532,
        "type": (
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ),
    },
    "StripRowCounts": {
        "number": 559,
        "type": None,
    },
    "XMLPacket": {
        "number": 700,
        "type": ctypes.c_uint8,
    },
    "ImageID": {
        "number": 32781,
        "type": None,
    },
    "Rating": {
        "number": 18246,
        "type": ctypes.c_uint16,
    },
    "RatingPercent": {
        "number": 18249,
        "type": ctypes.c_uint16,
    },
    "Datatype": {
        "number": 32996,
        "type": None,
    },
    "WANGAnnotation": {
        "number": 32932,
        "type": None,
    },
    "ImageDepth": {
        "number": 32997,
        "type": None,
    },
    "TileDepth": {
        "number": 32998,
        "type": None,
    },
    "CFARepeatPatternDim": {
        "number": 33421,
        "type": None,
    },
    "CFAPattern": {
        "number": 33422,
        "type": None,
    },
    "BatteryLevel": {
        "number": 33423,
        "type": None,
    },
    "Copyright": {
        "number": 33432,
        "type": ctypes.c_char_p,
    },
    "ExposureTime": {
        "number": 33434,
        "type": ctypes.c_double,
    },
    "FNumber": {
        "number": 33437,
        "type": ctypes.c_double,
    },
    "MDFile": {
        "number": 33445,
        "type": None,
    },
    "MDScalePixel": {
        "number": 33446,
        "type": None,
    },
    "MDColorTable": {
        "number": 33447,
        "type": None,
    },
    "MDLabName": {
        "number": 33448,
        "type": None,
    },
    "MDSampleInfo": {
        "number": 33449,
        "type": None,
    },
    "MdPrepDate": {
        "number": 33450,
        "type": None,
    },
    "MDPrepTime": {
        "number": 33451,
        "type": None,
    },
    "MDFileUnits": {
        "number": 33452,
        "type": None,
    },
    "ModelPixelScale": {
        "number": 33550,
        "type": None,
    },
    "IPTC": {
        "number": 33723,
        "type": None,
    },
    "INGRPacketData": {
        "number": 33918,
        "type": None,
    },
    "INGRFlagRegisters": {
        "number": 33919,
        "type": None,
    },
    "IRASbTransformationMatrix": {
        "number": 33920,
        "type": None,
    },
    "ModelTiePoint": {
        "number": 33922,
        "type": None,
    },
    "ModelTransformation": {
        "number": 34264,
        "type": None,
    },
    "Photoshop": {
        "number": 34377,
        "type": None,
    },
    "ExifTag": {
        "number": 34665,
        "type": ctypes.c_int32,
    },
    "ICCProfile": {
        "number": 34675,
        "type": None,
    },
    "ImageLayer": {
        "number": 34732,
        "type": None,
    },
    "GeoKeyDirectory": {
        "number": 34735,
        "type": None,
    },
    "GeoDoubleParams": {
        "number": 34736,
        "type": None,
    },
    "GeoAsciiParams": {
        "number": 34737,
        "type": None,
    },
    "ExposureProgram": {
        "number": 34850,
        "type": ctypes.c_uint16,
    },
    "SpectralSensitivity": {
        "number": 34852,
        "type": ctypes.c_char_p,
    },
    "GPSIFD": {
        "number": 34853,
        "type": None,
    },
    "ISOSpeedRatings": {
        "number": 34855,
        "type": ctypes.c_uint16,
    },
    "OECF": {
        "number": 34856,
        "type": None,
    },
    "Interlace": {
        "number": 34857,
        "type": None,
    },
    "TimeZoneOffset": {
        "number": 34858,
        "type": None,
    },
    "SelfTimerMode": {
        "number": 34859,
        "type": None,
    },
    "SensitivityType": {
        "number": 34864,
        "type": None,
    },
    "StandardOutputSensitivity": {
        "number": 34865,
        "type": None,
    },
    "RecommendedExposureIndex": {
        "number": 34866,
        "type": None,
    },
    "ISOSpeed": {
        "number": 34867,
        "type": None,
    },
    "ISOSpeedLatitudeYYY": {
        "number": 34868,
        "type": None,
    },
    "ISOSpeedLatitudeZZZ": {
        "number": 34869,
        "type": None,
    },
    "HYLAFAXRecvParams": {
        "number": 34908,
        "type": None,
    },
    "HYLAFAXSubAddress": {
        "number": 34909,
        "type": None,
    },
    "HYLAFAXRecvTime": {
        "number": 34910,
        "type": None,
    },
    "ExifVersion": {
        "number": 36864,
        "type": ctypes.c_uint8,
    },
    "DateTimeOriginal": {
        "number": 36867,
        "type": None,
    },
    "DateTimeDigitized": {
        "number": 36868,
        "type": None,
    },
    "OffsetTime": {
        "number": 36880,
        "type": None,
    },
    "OffsetTimeOriginal": {
        "number": 36881,
        "type": None,
    },
    "OffsetTimeDigitized": {
        "number": 36882,
        "type": None,
    },
    "ComponentsConfiguration": {"number": 37121, "type": None},
    "CompressedBitsPerPixel": {
        "number": 37122,
        "type": ctypes.c_uint8,
    },
    "ShutterSpeedValue": {
        "number": 37377,
        "type": ctypes.c_double,
    },
    "ApertureValue": {
        "number": 37378,
        "type": ctypes.c_double,
    },
    "BrightnessValue": {
        "number": 37379,
        "type": ctypes.c_double,
    },
    "ExposureBiasValue": {
        "number": 37380,
        "type": ctypes.c_double,
    },
    "MaxApertureValue": {
        "number": 37381,
        "type": ctypes.c_double,
    },
    "SubjectDistance": {
        "number": 37382,
        "type": ctypes.c_double,
    },
    "MeteringMode": {
        "number": 37383,
        "type": ctypes.c_uint16,
    },
    "LightSource": {
        "number": 37384,
        "type": ctypes.c_uint16,
    },
    "Flash": {
        "number": 37385,
        "type": ctypes.c_uint16,
    },
    "FocalLength": {
        "number": 37386,
        "type": ctypes.c_double,
    },
    "Noise": {
        "number": 37389,
        "type": None,
    },
    "ImageNumber": {
        "number": 37393,
        "type": None,
    },
    "SecurityClassification": {
        "number": 37394,
        "type": None,
    },
    "ImageHistory": {
        "number": 37395,
        "type": None,
    },
    "TIFFEPStandardID": {
        "number": 37398,
        "type": None,
    },
    "MakerNote": {
        "number": 37500,
        "type": ctypes.c_char_p,
    },
    "UserComment": {
        "number": 37510,
        "type": ctypes.c_char_p,
    },
    "SubSecTime": {
        "number": 37520,
        "type": ctypes.c_char_p,
    },
    "SubSecTimeOriginal": {
        "number": 37521,
        "type": ctypes.c_char_p,
    },
    "SubSecTimeDigitized": {
        "number": 37522,
        "type": ctypes.c_char_p,
    },
    "ImageSourceData": {
        "number": 37724,
        "type": None,
    },
    "Temperature": {
        "number": 37888,
        "type": None,
    },
    "Humidity": {
        "number": 37889,
        "type": None,
    },
    "Pressure": {
        "number": 37890,
        "type": None,
    },
    "WaterDepth": {
        "number": 37891,
        "type": None,
    },
    "Acceleration": {
        "number": 37892,
        "type": None,
    },
    "CameraElevationAngle": {
        "number": 37893,
        "type": None,
    },
    "XPTitle": {
        "number": 40091,
        "type": None,
    },
    "XPComment": {
        "number": 40092,
        "type": None,
    },
    "XPAuthor": {
        "number": 40093,
        "type": None,
    },
    "XPKeywords": {
        "number": 40094,
        "type": None,
    },
    "XPSubject": {
        "number": 40095,
        "type": None,
    },
    "FlashPixVersion": {
        "number": 40960,
        "type": None,
    },
    "ColorSpace": {
        "number": 40961,
        "type": ctypes.c_uint16,
    },
    "PixelXDimension": {
        "number": 40962,
        "type": ctypes.c_uint64,
    },
    "PixelYDimension": {
        "number": 40963,
        "type": ctypes.c_uint64,
    },
    "InteroperabilityIFD": {
        "number": 40965,
        "type": None,
    },
    "FocalPlaneXResolution": {
        "number": 41486,
        "type": ctypes.c_double,
    },
    "FocalPlaneYResolution": {
        "number": 41487,
        "type": ctypes.c_double,
    },
    "FocalPlaneResolutionUnit": {
        "number": 41488,
        "type": ctypes.c_uint16,
    },
    "ExposureIndex": {
        "number": 41493,
        "type": ctypes.c_double,
    },
    "SensingMethod": {
        "number": 41495,
        "type": ctypes.c_uint16,
    },
    "SceneType": {
        "number": 41729,
        "type": ctypes.c_uint8,
    },
    "FlashEnergy": {"number": 41483, "type": None},
    "SpatialFrequencyResponse": {"number": 41484, "type": None},
    "SubjectLocation": {"number": 41492, "type": None},
    "FileSource": {"number": 41728, "type": None},
    "ExifCFAPattern": {"number": 41730, "type": None},
    "CustomRendered": {"number": 41985, "type": None},
    "ExposureMode": {
        "number": 41986,
        "type": ctypes.c_uint16,
    },
    "WhiteBalance": {
        "number": 41987,
        "type": ctypes.c_uint16,
    },
    "DigitalZoomRatio": {
        "number": 41988,
        "type": ctypes.c_double,
    },
    "FocalLengthIn35mmFilm": {
        "number": 41989,
        "type": ctypes.c_uint16,
    },
    "SceneCaptureType": {
        "number": 41990,
        "type": ctypes.c_uint16,
    },
    "GainControl": {
        "number": 41991,
        "type": ctypes.c_uint16,
    },
    "Contrast": {
        "number": 41992,
        "type": ctypes.c_uint16,
    },
    "Saturation": {
        "number": 41993,
        "type": ctypes.c_uint16,
    },
    "Sharpness": {
        "number": 41994,
        "type": ctypes.c_uint16,
    },
    "DeviceSettingDescription": {
        "number": 41995,
        "type": ctypes.c_char_p,
    },
    "SubjectDistanceRange": {
        "number": 41996,
        "type": ctypes.c_uint16,
    },
    "ImageUniqueID": {"number": 42016, "type": None},
    "CameraOwnerName": {"number": 42032, "type": None},
    "BodySerialNumber": {"number": 42033, "type": None},
    "LensSpecification": {"number": 42034, "type": None},
    "LensMake": {"number": 42035, "type": None},
    "LensModel": {"number": 42036, "type": None},
    "LensSerialNumber": {"number": 42037, "type": None},
    "CompositeImage": {"number": 42080, "type": None},
    "SourceImageNumberOfCompositeImage": {"number": 42081, "type": None},
    "SourceExposureTimeOfCompositeImage": {"number": 42082, "type": None},
    "GDAL_Metadata": {
        "number": 42112,
        "type": None,
    },
    "GDAL_NoData": {
        "number": 42113,
        "type": None,
    },
    "Gamma": {
        "number": 42240,
        "type": None,
    },
    "OCEScanJobDescription": {
        "number": 50215,
        "type": None,
    },
    "OCEApplicationSelector": {
        "number": 50216,
        "type": None,
    },
    "OCEIdentificationNumber": {
        "number": 50217,
        "type": None,
    },
    "OCEImageLogicCharacteristics": {
        "number": 50218,
        "type": None,
    },
    "PrintImageMatching": {
        "number": 50341,
        "type": None,
    },
    "DNGVersion": {
        "number": 50706,
        "type": None,
    },
    "DNGBackwardVersion": {
        "number": 50707,
        "type": None,
    },
    "UniqueCameraModel": {
        "number": 50708,
        "type": None,
    },
    "LocalizedCameraModel": {
        "number": 50709,
        "type": None,
    },
    "CFAPlaneColor": {
        "number": 50710,
        "type": None,
    },
    "CFALayout": {
        "number": 50711,
        "type": None,
    },
    "LinearizationTable": {
        "number": 50712,
        "type": None,
    },
    "BlackLevelRepeatDim": {
        "number": 50713,
        "type": None,
    },
    "BlackLevel": {
        "number": 50714,
        "type": None,
    },
    "BlackLevelDeltaH": {
        "number": 50715,
        "type": None,
    },
    "BlackLevelDeltaV": {
        "number": 50716,
        "type": None,
    },
    "WhiteLevel": {
        "number": 50717,
        "type": None,
    },
    "DefaultScale": {
        "number": 50718,
        "type": None,
    },
    "DefaultCropOrigin": {
        "number": 50719,
        "type": None,
    },
    "DefaultCropSize": {
        "number": 50720,
        "type": None,
    },
    "ColorMatrix1": {
        "number": 50721,
        "type": None,
    },
    "ColorMatrix2": {
        "number": 50722,
        "type": None,
    },
    "CameraCalibration1": {
        "number": 50723,
        "type": None,
    },
    "CameraCalibration2": {
        "number": 50724,
        "type": None,
    },
    "ReductionMatrix1": {
        "number": 50725,
        "type": None,
    },
    "ReductionMatrix2": {
        "number": 50726,
        "type": None,
    },
    "AnalogBalance": {
        "number": 50727,
        "type": None,
    },
    "AsShotNeutral": {
        "number": 50728,
        "type": None,
    },
    "AsShotWhiteXY": {
        "number": 50729,
        "type": None,
    },
    "BaselineExposure": {
        "number": 50730,
        "type": None,
    },
    "BaselineNoise": {
        "number": 50731,
        "type": None,
    },
    "BaselineSharpness": {
        "number": 50732,
        "type": None,
    },
    "BayerGreenSplit": {
        "number": 50733,
        "type": None,
    },
    "LinearResponseLimit": {
        "number": 50734,
        "type": None,
    },
    "CameraSerialNumber": {
        "number": 50735,
        "type": None,
    },
    "LensInfo": {
        "number": 50736,
        "type": None,
    },
    "ChromaBlurRadius": {
        "number": 50737,
        "type": None,
    },
    "AntiAliasStrength": {
        "number": 50738,
        "type": None,
    },
    "ShadowScale": {
        "number": 50739,
        "type": None,
    },
    "DNGPrivateData": {
        "number": 50740,
        "type": None,
    },
    "MakerNoteSafety": {
        "number": 50741,
        "type": None,
    },
    "CalibrationIllumintant1": {
        "number": 50778,
        "type": None,
    },
    "CalibrationIllumintant2": {
        "number": 50779,
        "type": None,
    },
    "BestQualityScale": {
        "number": 50780,
        "type": None,
    },
    "RawDataUniqueID": {
        "number": 50781,
        "type": None,
    },
    "AliasLayerMetadata": {
        "number": 50784,
        "type": None,
    },
    "OriginalRawFileName": {
        "number": 50827,
        "type": None,
    },
    "OriginalRawFileData": {
        "number": 50828,
        "type": None,
    },
    "ActiveArea": {
        "number": 50829,
        "type": None,
    },
    "MaskedAreas": {
        "number": 50830,
        "type": None,
    },
    "AsShotICCProfile": {
        "number": 50831,
        "type": None,
    },
    "AsShotPreProfileMatrix": {
        "number": 50832,
        "type": None,
    },
    "CurrentICCProfile": {
        "number": 50833,
        "type": None,
    },
    "CurrentPreProfileMatrix": {
        "number": 50834,
        "type": None,
    },
    "ColorimetricReference": {
        "number": 50839,
        "type": None,
    },
    "TIFF_RSID": {
        "number": 50908,
        "type": None,
    },
    "GEO_Metadata": {
        "number": 50909,
        "type": None,
    },
    "CameraCalibrationSignature": {
        "number": 50931,
        "type": None,
    },
    "ProfileCalibrationSignature": {
        "number": 50932,
        "type": None,
    },
    "AsShotProfileName": {
        "number": 50934,
        "type": None,
    },
    "NoiseReductionApplied": {
        "number": 50935,
        "type": None,
    },
    "ProfileName": {
        "number": 50936,
        "type": None,
    },
    "ProfileHueSatMapDims": {
        "number": 50937,
        "type": None,
    },
    "ProfileHueSatMapData1": {
        "number": 50938,
        "type": None,
    },
    "ProfileHueSatMapData2": {
        "number": 50939,
        "type": None,
    },
    "ProfileToneCurve": {
        "number": 50940,
        "type": None,
    },
    "ProfileEmbedPolicy": {
        "number": 50941,
        "type": None,
    },
    "ProfileCopyright": {
        "number": 50942,
        "type": None,
    },
    "ForwardMatrix1": {
        "number": 50964,
        "type": None,
    },
    "ForwardMatrix2": {
        "number": 50965,
        "type": None,
    },
    "PreviewApplicationName": {
        "number": 50966,
        "type": None,
    },
    "PreviewApplicationVersion": {
        "number": 50967,
        "type": None,
    },
    "PreviewSettingsName": {
        "number": 50968,
        "type": None,
    },
    "PreviewSettingsDigest": {
        "number": 50969,
        "type": None,
    },
    "PreviewColorSpace": {
        "number": 50970,
        "type": None,
    },
    "PreviewDateTime": {
        "number": 50971,
        "type": None,
    },
    "RawImageDigest": {
        "number": 50972,
        "type": None,
    },
    "OriginalRawFileDigest": {
        "number": 50973,
        "type": None,
    },
    "SubTileBlockSize": {
        "number": 50974,
        "type": None,
    },
    "RowInterleaveFactor": {
        "number": 50975,
        "type": None,
    },
    "ProfileLookTableDims": {
        "number": 50981,
        "type": None,
    },
    "ProfileLookTableData": {
        "number": 50982,
        "type": None,
    },
    "OpcodeList1": {
        "number": 51008,
        "type": None,
    },
    "OpcodeList2": {
        "number": 51009,
        "type": None,
    },
    "OpcodeList3": {
        "number": 51022,
        "type": None,
    },
    "NoiseProfile": {
        "number": 51041,
        "type": None,
    },
    "JPEGQuality": {
        "number": 65537,
        "type": ctypes.c_int32,
    },
    "JPEGColorMode": {
        "number": 65538,
        "type": ctypes.c_int32,
    },
}

# We need the reverse mapping as well.
TAGNUM2NAME = {value["number"]: key for key, value in TAGS.items()}


def tiff_header(read_buffer):
    """Interpret the uuid raw data as a tiff header."""
    # First 8 should be (73, 73, 42, 8) or (77, 77, 42, 8)
    data = struct.unpack("BB", read_buffer[0:2])
    if data[0] == 73 and data[1] == 73:
        # little endian
        endian = "<"
    elif data[0] == 77 and data[1] == 77:
        # big endian
        endian = ">"
    else:
        msg = (
            f"The byte order indication in the TIFF header "
            f"({read_buffer[0:2]}) is invalid.  It should be either "
            f"{bytes([73, 73])} or {bytes([77, 77])}."
        )
        raise RuntimeError(msg)

    _, offset = struct.unpack(endian + "HI", read_buffer[2:8])

    # This is the 'Exif Image' portion.
    return _Ifd(endian, read_buffer, offset).processed_ifd


class BadTiffTagDatatype(RuntimeError):
    """This exception exists soley to better communicate up the stack that the
    problem exists.
    """

    pass


class _Ifd(object):
    """Corresponds to TIFF IFD data structure on file.

    Attributes
    ----------
    read_buffer : bytes
        Raw byte stream consisting of the UUID data.
    endian : str
        Either '<' for big-endian, or '>' for little-endian.
    num_tags : int
        Number of tags in the IFD.
    raw_ifd : dictionary
        Maps tag number to "mildly-interpreted" tag value.
    processed_ifd : dictionary
        Maps tag name to "mildly-interpreted" tag value.
    """

    def __init__(self, endian, read_buffer, offset):
        self.endian = endian
        self.read_buffer = read_buffer
        self.processed_ifd = OrderedDict()

        (self.num_tags,) = struct.unpack(
            endian + "H",
            read_buffer[offset:offset + 2]
        )

        fmt = self.endian + "HHII" * self.num_tags
        ifd_buffer = read_buffer[offset + 2:offset + 2 + self.num_tags * 12]
        data = struct.unpack(fmt, ifd_buffer)
        self.raw_ifd = OrderedDict()
        for j, tag in enumerate(data[0::4]):
            # The offset to the tag offset/payload is the offset to the IFD
            # plus 2 bytes for the number of tags plus 12 bytes for each
            # tag entry plus 8 bytes to the offset/payload itself.
            toffp = read_buffer[offset + 10 + j * 12:offset + 10 + j * 12 + 4]
            self.raw_ifd[tag] = self.parse_tag(
                tag, data[j * 4 + 1], data[j * 4 + 2], toffp
            )

        self.post_process()

    def parse_tag(self, tag, dtype, count, offset_buf):
        """Interpret an Exif image tag data payload."""

        try:
            fmt = DATATYPE2FMT[dtype]["format"] * count
            payload_size = DATATYPE2FMT[dtype]["nbytes"] * count
        except KeyError:
            msg = f"Invalid TIFF tag datatype ({dtype})."
            raise BadTiffTagDatatype(msg)

        if payload_size <= 4:
            # Interpret the payload from the 4 bytes in the tag entry.
            target_buffer = offset_buf[:payload_size]
        else:
            # Interpret the payload at the offset specified by the 4 bytes in
            # the tag entry.
            (offset,) = struct.unpack(self.endian + "I", offset_buf)
            target_buffer = self.read_buffer[offset:offset + payload_size]

        if dtype == 2:
            # ASCII
            payload = target_buffer.decode("utf-8").rstrip("\x00")

        else:
            payload = struct.unpack(self.endian + fmt, target_buffer)
            if dtype == 5 or dtype == 10:
                # Rational or Signed Rational.  Construct the list of values.
                rational_payload = []
                for j in range(count):
                    try:
                        value = float(payload[j * 2]) / float(payload[j * 2 + 1])  # noqa : E501
                    except ZeroDivisionError:
                        value = np.nan
                    rational_payload.append(value)
                payload = np.array(rational_payload)
            if count == 1:
                # If just a single value, then return a scalar instead of a
                # tuple.
                payload = payload[0]
            else:
                payload = np.array(
                    payload, dtype=DATATYPE2FMT[dtype]["nptype"]
                )

        return payload

    def post_process(self):
        """Map the tag name instead of tag number to the tag value."""
        for tag, value in self.raw_ifd.items():
            try:
                tag_name = TAGNUM2NAME[tag]
            except KeyError:
                # Ok, we don't recognize this tag.  Just use the numeric id.
                msg = f"Unrecognized UUID box TIFF tag ({tag})."
                warnings.warn(msg, UserWarning)
                tag_name = tag

            if tag_name in ("ExifTag", 'GPSIFD'):
                # There's an IFD at the offset specified here.
                ifd = _Ifd(self.endian, self.read_buffer, value)
                self.processed_ifd[tag_name] = ifd.processed_ifd
            else:
                # just a regular tag, treat it as a simple value
                self.processed_ifd[tag_name] = value


# maps the TIFF enumerated datatype to the corresponding structs datatype code,
# the data width, and the corresponding numpy datatype
DATATYPE2FMT = {
    1: {"format": "B", "nbytes": 1, "nptype": np.ubyte},
    2: {"format": "B", "nbytes": 1, "nptype": str},
    3: {"format": "H", "nbytes": 2, "nptype": np.uint16},
    4: {"format": "I", "nbytes": 4, "nptype": np.uint32},
    5: {"format": "II", "nbytes": 8, "nptype": np.double},
    6: {"format": "b", "nbytes": 1, "nptype": np.int8},
    7: {"format": "B", "nbytes": 1, "nptype": np.uint8},
    8: {"format": "h", "nbytes": 2, "nptype": np.int16},
    9: {"format": "i", "nbytes": 4, "nptype": np.int32},
    10: {"format": "ii", "nbytes": 8, "nptype": np.double},
    11: {"format": "f", "nbytes": 4, "nptype": np.double},
    12: {"format": "d", "nbytes": 8, "nptype": np.double},
    13: {"format": "I", "nbytes": 4, "nptype": np.uint32},
    16: {"format": "Q", "nbytes": 8, "nptype": np.uint64},
    17: {"format": "q", "nbytes": 8, "nptype": np.int64},
    18: {"format": "Q", "nbytes": 8, "nptype": np.uint64},
}
