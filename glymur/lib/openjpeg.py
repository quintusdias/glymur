"""Wraps library calls to openjpeg.
"""

import ctypes
from ctypes.util import find_library
import platform
import os

if os.name == "nt":
    path = os.path.join('C:\\', 'Program files', 'OpenJPEG 1.5',
                        'bin', 'openjpeg.dll')
    OPENJPEG = ctypes.windll.LoadLibrary(path)
else:
    if platform.system() == 'Darwin':
        OPENJPEG = ctypes.CDLL('/opt/local/lib/libopenjpeg.dylib')
    elif platform.system() == 'Linux':
        OPENJPEG = ctypes.CDLL(find_library('openjpeg'))

PATH_LEN = 4096  # maximum allowed size for filenames


class event_mgr_t(ctypes.Structure):
    """Message handler object.
    """
    _fields_ = [("error_handler", ctypes.c_void_p),
                ("warning_handler", ctypes.c_void_p),
                ("info_handler", ctypes.c_void_p)]

class common_struct_t(ctypes.Structure):
    """Common fields between JPEG 2000 compression and decompression contextx.
    """
    _fields_ = [("event_mgr", ctypes.POINTER(event_mgr_t)),
                ("client_data", ctypes.c_void_p),
                ("is_decompressor", ctypes.c_bool),
                ("codec_format", ctypes.c_int),
                ("j2k_handle", ctypes.c_void_p),
                ("jp2_handle", ctypes.c_void_p),
                ("mj2_handle", ctypes.c_void_p)]


class dinfo_t(ctypes.Structure):
    """Common fields between JPEG 2000 compression and decompression contextx.
    This is for decompression contexts.
    """
    pass


class cio_t(ctypes.Structure):
    _fields_ = [# codec context
                ("cinfo",         ctypes.POINTER(common_struct_t)),
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


class dparameters_t(ctypes.Structure):
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
                ("cp_limit_decoding", ctypes.c_int),
                ("flags",             ctypes.c_uint)]


class image_comp_t(ctypes.Structure):
    """Defines a single image component. """
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


class image_t(ctypes.Structure):
    """Defines image data and characteristics."""
    _fields_ = [("x0", ctypes.c_int),
                ("y0", ctypes.c_int),
                ("x1", ctypes.c_int),
                ("y1", ctypes.c_int),
                ("numcomps", ctypes.c_int),
                ("color_space", ctypes.c_int),
                ("comps", ctypes.POINTER(image_comp_t)),
                ("icc_profile_buf", ctypes.c_char_p),
                ("icc_profile_len", ctypes.c_int)]

def cio_open(cinfo, src):
    """Wrapper for openjpeg library function opj_cio_open."""
    argtypes = [ctypes.POINTER(common_struct_t), ctypes.c_char_p, ctypes.c_int]
    OPENJPEG.opj_cio_open.argtypes = argtypes
    OPENJPEG.opj_cio_open.restype = ctypes.POINTER(cio_t)

    cio = OPENJPEG.opj_cio_open(ctypes.cast(cinfo, ctypes.POINTER(common_struct_t)),
                                src, len(src))
    return cio

def cio_close(cio):
    """Wraps openjpeg library function cio_close.
    """
    OPENJPEG.opj_cio_close.argtypes = [ctypes.POINTER(cio_t)]
    OPENJPEG.opj_cio_close(cio)

def create_decompress(fmt):
    """Wraps openjpeg library function opj_create_decompress.
    """
    OPENJPEG.opj_create_decompress.argtypes = [ctypes.c_int]
    OPENJPEG.opj_create_decompress.restype = ctypes.POINTER(dinfo_t)
    dinfo = OPENJPEG.opj_create_decompress(fmt)
    return dinfo

def decode(dinfo, cio):
    """Wrapper for opj_decode.
    """
    argtypes = [ctypes.POINTER(dinfo_t), ctypes.POINTER(cio_t)]
    OPENJPEG.opj_decode.argtypes = argtypes
    OPENJPEG.opj_decode.restype = ctypes.POINTER(image_t)
    image = OPENJPEG.opj_decode(dinfo, cio)
    return image

def destroy_decompress(dinfo):
    """Wraps openjpeg library function opj_destroy_decompress."""
    OPENJPEG.opj_destroy_decompress.argtypes = [ctypes.POINTER(dinfo_t)]
    OPENJPEG.opj_destroy_decompress(dinfo)

def image_destroy(image):
    """Wraps openjpeg library function opj_image_destroy."""
    OPENJPEG.opj_image_destroy.argtypes = [ctypes.POINTER(image_t)]
    OPENJPEG.opj_image_destroy(image)

def set_default_decoder_parameters(dparams_p):
    """Wrapper for opj_set_default_decoder_parameters.
    """
    argtypes = [ctypes.POINTER(dparameters_t)]
    OPENJPEG.opj_set_default_decoder_parameters.argtypes = argtypes
    OPENJPEG.opj_set_default_decoder_parameters(dparams_p)


def set_event_mgr(dinfo, event_mgr, context=None):
    """Wrapper for openjpeg library function opj_set_event_mgr.
    """
    argtypes = [ctypes.POINTER(common_struct_t),
                ctypes.POINTER(event_mgr_t),
                ctypes.c_void_p]
    OPENJPEG.opj_set_event_mgr(ctypes.cast(dinfo,
                                           ctypes.POINTER(common_struct_t)),
                               event_mgr, context)

def setup_decoder(dinfo, dparams):
    """Wrapper for openjpeg library function opj_setup_decoder."""
    argtypes = [ctypes.POINTER(dinfo_t), ctypes.POINTER(dparameters_t)]
    OPENJPEG.opj_setup_decoder.argtypes = argtypes
    OPENJPEG.opj_setup_decoder(dinfo, dparams)

def version():
    """Wrapper for opj_version library routine."""
    OPENJPEG.opj_version.restype = ctypes.c_char_p
    v = OPENJPEG.opj_version()
    return v.decode('utf-8')
