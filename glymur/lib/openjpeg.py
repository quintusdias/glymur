"""Wraps library calls to openjpeg.
"""

# pylint: disable=R0903

import ctypes
import sys

from .config import glymur_config
_, OPENJPEG = glymur_config()

PATH_LEN = 4096  # maximum allowed size for filenames


def version():
    """Wrapper for opj_version library routine."""
    OPENJPEG.opj_version.restype = ctypes.c_char_p
    library_version = OPENJPEG.opj_version()
    if sys.hexversion >= 0x03000000:
        return library_version.decode('utf-8')
    else:
        return library_version

# Need to get the minor version, make sure we are at least at 1.4.x
if OPENJPEG is not None:
    _MINOR = version().split('.')[1]
else:
    # Does not really matter.  But version should not be called if there is no
    # OpenJPEG library found.
    _MINOR = 0
    # Redefine version so that we can use it.
    def version():
        return '0.0.0'


class EventMgrType(ctypes.Structure):
    """Message handler object.

    Corresponds to event_mgr_t type in openjpeg headers.
    """
    _fields_ = [("error_handler", ctypes.c_void_p),
                ("warning_handler", ctypes.c_void_p),
                ("info_handler", ctypes.c_void_p)]


class CommonStructType(ctypes.Structure):
    """Common fields between JPEG 2000 compression and decompression contextx.
    """
    _fields_ = [("event_mgr", ctypes.POINTER(EventMgrType)),
                ("client_data", ctypes.c_void_p),
                ("is_decompressor", ctypes.c_bool),
                ("codec_format", ctypes.c_int),
                ("j2k_handle", ctypes.c_void_p),
                ("jp2_handle", ctypes.c_void_p),
                ("mj2_handle", ctypes.c_void_p)]


class DecompressionInfoType(ctypes.Structure):
    """This is for decompression contexts.

    Corresponds to dinfo_t type in openjpeg headers.
    """
    pass


class CioType(ctypes.Structure):
    """Byte input-output stream (CIO)

    Corresponds to cio_t in openjpeg headers.
    """
    _fields_ = [("cinfo", ctypes.POINTER(CommonStructType)),  # codec context
                # STREAM_READ or STREAM_WRITE
                ("openmode", ctypes.c_int),
                # pointer to start of buffer
                ("buffer", ctypes.POINTER(ctypes.c_char)),
                # buffer size in bytes
                ("length", ctypes.c_int),
                # pointer to start of stream
                ("start", ctypes.c_char_p),
                # pointer to end of stream
                ("end", ctypes.c_char_p),
                # pointer to current position
                ("bp", ctypes.c_char_p)]


class DecompressionParametersType(ctypes.Structure):
    """Decompression parameters.

    Corresponds to dparameters_t type in openjpeg headers.
    """
    # cp_reduce:  the number of highest resolution levels to be discarded
    _fields_ = [("cp_reduce",         ctypes.c_int),
                # cp_layer:  the maximum number of quality layers to decode
                ("cp_layer",          ctypes.c_int),
                # infile:  input file name
                ("infile",            ctypes.c_char * PATH_LEN),
                # outfile:  output file name
                ("outfile",           ctypes.c_char * PATH_LEN),
                # decod_format:  input file format 0: J2K, 1: JP2, 2: JPT
                ("decod_format",      ctypes.c_int),
                # cod_format:  output file format 0: PGX, 1: PxM, 2: BMP
                ("cod_format",        ctypes.c_int),
                # jpwl_correct:  activates the JPWL correction capabilities
                ("jpwl_correct",      ctypes.c_bool),
                # jpwl_exp_comps:  expected number of components
                ("jpwl_exp_comps",    ctypes.c_int),
                # jpwl_max_tiles:  maximum number of tiles
                ("jpwl_max_tiles",    ctypes.c_int),
                # cp_limit_decoding:  whether decoding should be done on the
                # entire codestream or be limited to the main header
                ("cp_limit_decoding", ctypes.c_int)]

    if _MINOR == '5':
        _fields_.append(("flags",             ctypes.c_uint))


class ImageCompType(ctypes.Structure):
    """Defines a single image component.

    Corresponds to image_comp_t type in openjpeg.
    """
    _fields_ = [("dx", ctypes.c_int),
                ("dy", ctypes.c_int),
                ("w", ctypes.c_int),
                ("h", ctypes.c_int),
                ("x0", ctypes.c_int),
                ("y0", ctypes.c_int),
                ("prec", ctypes.c_int),
                ("bpp", ctypes.c_int),
                ("sgnd", ctypes.c_int),
                ("resno_decoded", ctypes.c_int),
                ("factor", ctypes.c_int),
                ("data", ctypes.POINTER(ctypes.c_int))]


class ImageType(ctypes.Structure):
    """Defines image data and characteristics.

    Corresponds to image_t type in openjpeg headers.
    """
    _fields_ = [("x0", ctypes.c_int),
                ("y0", ctypes.c_int),
                ("x1", ctypes.c_int),
                ("y1", ctypes.c_int),
                ("numcomps", ctypes.c_int),
                ("color_space", ctypes.c_int),
                ("comps", ctypes.POINTER(ImageCompType)),
                ("icc_profile_buf", ctypes.c_char_p),
                ("icc_profile_len", ctypes.c_int)]


def cio_open(cinfo, src):
    """Wrapper for openjpeg library function opj_cio_open."""
    argtypes = [ctypes.POINTER(CommonStructType), ctypes.c_char_p,
                ctypes.c_int]
    OPENJPEG.opj_cio_open.argtypes = argtypes
    OPENJPEG.opj_cio_open.restype = ctypes.POINTER(CioType)

    cio = OPENJPEG.opj_cio_open(ctypes.cast(cinfo,
                                            ctypes.POINTER(CommonStructType)),
                                src, len(src))
    return cio


def cio_close(cio):
    """Wraps openjpeg library function cio_close.
    """
    OPENJPEG.opj_cio_close.argtypes = [ctypes.POINTER(CioType)]
    OPENJPEG.opj_cio_close(cio)


def create_decompress(fmt):
    """Wraps openjpeg library function opj_create_decompress.
    """
    OPENJPEG.opj_create_decompress.argtypes = [ctypes.c_int]
    restype = ctypes.POINTER(DecompressionInfoType)
    OPENJPEG.opj_create_decompress.restype = restype
    dinfo = OPENJPEG.opj_create_decompress(fmt)
    return dinfo


def decode(dinfo, cio):
    """Wrapper for opj_decode.
    """
    argtypes = [ctypes.POINTER(DecompressionInfoType), ctypes.POINTER(CioType)]
    OPENJPEG.opj_decode.argtypes = argtypes
    OPENJPEG.opj_decode.restype = ctypes.POINTER(ImageType)
    image = OPENJPEG.opj_decode(dinfo, cio)
    return image


def destroy_decompress(dinfo):
    """Wraps openjpeg library function opj_destroy_decompress."""
    argtypes = [ctypes.POINTER(DecompressionInfoType)]
    OPENJPEG.opj_destroy_decompress.argtypes = argtypes
    OPENJPEG.opj_destroy_decompress(dinfo)


def image_destroy(image):
    """Wraps openjpeg library function opj_image_destroy."""
    OPENJPEG.opj_image_destroy.argtypes = [ctypes.POINTER(ImageType)]
    OPENJPEG.opj_image_destroy(image)


def set_default_decoder_parameters(dparams_p):
    """Wrapper for opj_set_default_decoder_parameters.
    """
    argtypes = [ctypes.POINTER(DecompressionParametersType)]
    OPENJPEG.opj_set_default_decoder_parameters.argtypes = argtypes
    OPENJPEG.opj_set_default_decoder_parameters(dparams_p)


def set_event_mgr(dinfo, event_mgr, context=None):
    """Wrapper for openjpeg library function opj_set_event_mgr.
    """
    argtypes = [ctypes.POINTER(CommonStructType),
                ctypes.POINTER(EventMgrType),
                ctypes.c_void_p]
    OPENJPEG.opj_set_event_mgr.argtypes = argtypes
    OPENJPEG.opj_set_event_mgr(ctypes.cast(dinfo,
                                           ctypes.POINTER(CommonStructType)),
                               event_mgr, context)


def setup_decoder(dinfo, dparams):
    """Wrapper for openjpeg library function opj_setup_decoder."""
    argtypes = [ctypes.POINTER(DecompressionInfoType),
                ctypes.POINTER(DecompressionParametersType)]
    OPENJPEG.opj_setup_decoder.argtypes = argtypes
    OPENJPEG.opj_setup_decoder(dinfo, dparams)
