"""Wraps library calls to openjpeg.
"""

import ctypes
from ctypes.util import find_library
import platform
import os

if os.name == "nt":
    _OPENJPEG = ctypes.windll.LoadLibrary('openjpeg')
else:
    if platform.system() == 'Darwin':
        _OPENJPEG = ctypes.CDLL('/opt/local/lib/libopenjpeg.dylib')
    elif platform.system() == 'Linux':
        _OPENJPEG = ctypes.CDLL(find_library('openjpeg'))

OPJ_PATH_LEN = 4096  # maximum allowed size for filenames


class dparameters_t(ctypes.Structure):
    # cp_reduce:  the number of highest resolution levels to be discarded
    _fields_ = [("cp_reduce",         ctypes.c_int),
                # cp_layer:  the maximum number of quality layers to decode
                ("cp_layer",          ctypes.c_int),
                # infile:  input file name
                ("infile",            ctypes.c_char * OPJ_PATH_LEN),
                # outfile:  output file name
                ("outfile",           ctypes.c_char * OPJ_PATH_LEN),
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


def _set_default_decoder_parameters(dparams_p):
    """Wrapper for opj_set_default_decoder_parameters.
    """
    argtypes = [ctypes.POINTER(dparameters_t)]
    _OPENJPEG.opj_set_default_decoder_parameters.argtypes = argtypes
    _OPENJPEG.opj_set_default_decoder_parameters(dparams_p)


def _version():
    """Wrapper for opj_version library routine."""
    _OPENJPEG.opj_version.restype = ctypes.c_char_p
    v = _OPENJPEG.opj_version()
    return v.decode('utf-8')
