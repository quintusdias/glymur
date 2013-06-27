"""
Wraps individual functions in openjp2 library.
"""


def _glymurrc_fname():
    """Return the path to the configuration file.

    Search order:
        1) current working directory
        2) environ var XDG_CONFIG_HOME
        3) $HOME/.config/glymur/glymurrc
    """

    # Current directory.
    fname = os.path.join(os.getcwd(), 'glymurrc')
    if os.path.exists(fname):
        return fname

    # Either GLYMURCONFIGDIR/glymurrc or $HOME/.glymur/glymurrc
    confdir = _get_configdir()
    if confdir is not None:
        fname = os.path.join(confdir, 'glymurrc')
        if os.path.exists(fname):
            return fname
        else:
            msg = "Configuration file '{0}' does not exist.".format(confdir)
            warnings.warn(msg, UserWarning)

    # didn't find a configuration file.
    return None


def _config():
    """Read configuration file.

    Based on matplotlib.
    """
    filename = _glymurrc_fname()
    if filename is not None:
        # Read the configuration file for the library location.
        parser = ConfigParser()
        parser.read(filename)
        libopenjp2_path = parser.get('library', 'openjp2')
    else:
        # No help from the config file, try to find it ourselves.
        from ctypes.util import find_library
        libopenjp2_path = find_library('openjp2')

    if libopenjp2_path is None:
        return None

    try:
        _OPENJP2 = ctypes.CDLL(libopenjp2_path)
    except OSError:
        msg = '"Library {0}" could not be loaded.  Operating in degraded mode.'
        msg = msg.format(libopenjp2_path)
        warnings.warn(msg, UserWarning)
        _OPENJP2 = None
    return _OPENJP2


def _get_configdir():
    """Return string representing the configuration directory.

    Default is $HOME/.config/glymur.  You can override this with the
    XDG_CONFIG_HOME environment variable.
    """

    if 'XDG_CONFIG_HOME' in os.environ:
        return os.path.join(os.environ['XDG_CONFIG_HOME'], 'glymur')

    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.config', 'glymur')

import ctypes
import os
import warnings

import sys
if sys.hexversion <= 0x03000000:
    from ConfigParser import SafeConfigParser as ConfigParser
    from ConfigParser import NoOptionError
else:
    from configparser import ConfigParser
    from configparser import NoOptionError

_OPENJP2 = _config()

import numpy as np

# Progression order
LRCP = 0
RLCP = 1
RPCL = 2
PCRL = 3
CPRL = 4


_ERROR_MSG_LST = []

# Map certain atomic OpenJPEG datatypes to the ctypes equivalents.
_bool_t = ctypes.c_int32
_codec_t_p = ctypes.c_void_p
_prog_order_t_p = ctypes.c_int32
_cinema_mode_t = ctypes.c_int32
_rsiz_capabilities_t = ctypes.c_int32
_stream_t_p = ctypes.c_void_p

_PATH_LEN = 4096
_J2K_MAXRLVLS = 33
_J2K_MAXBANDS = (3 * _J2K_MAXRLVLS - 2)

_JPWL_MAX_NO_TILESPECS = 16

_TRUE = 1
_FALSE = 0

# supported color spaces
_CLRSPC_UNKNOWN = -1
_CLRSPC_UNSPECIFIED = 0
_CLRSPC_SRGB = 1
_CLRSPC_GRAY = 2
_CLRSPC_YCC = 3
color_space_t = ctypes.c_int

# supported codec
_codec_format_t = ctypes.c_int
_CODEC_UNKNOWN = -1
_CODEC_J2K = 0
_CODEC_JPT = 1
_CODEC_JP2 = 2


class _poc_t(ctypes.Structure):
    """Progression order changes."""
    # Resolution num start, Component num start, given by POC
    _fields_ = [
        ("resno0",     ctypes.c_uint32),
        ("compno0",    ctypes.c_uint32),

        # Layer num end,Resolution num end, Component num end, given by POC
        ("layno1",     ctypes.c_uint32),
        ("resno1",     ctypes.c_uint32),
        ("compno1",    ctypes.c_uint32),

        # Layer num start,Precinct num start, Precinct num end
        ("layno0",     ctypes.c_uint32),
        ("precno0",    ctypes.c_uint32),
        ("precno1",    ctypes.c_uint32),

        # Progression order enum
        ("prg1",       _prog_order_t_p),
        ("prg",        _prog_order_t_p),

        # Progression order string
        ("progorder",  ctypes.c_char * 5),

        # Tile number
        ("tile",       ctypes.c_uint32),

        # Start and end values for Tile width and height*
        ("tx0",       ctypes.c_int32),
        ("tx1",       ctypes.c_int32),
        ("ty0",       ctypes.c_int32),
        ("ty1",       ctypes.c_int32),

        # Start value, initialised in pi_initialise_encode
        ("layS",       ctypes.c_uint32),
        ("resS",       ctypes.c_uint32),
        ("compS",      ctypes.c_uint32),
        ("prcS",       ctypes.c_uint32),

        # End value, initialised in pi_initialise_encode
        ("layE",       ctypes.c_uint32),
        ("resE",       ctypes.c_uint32),
        ("compE",      ctypes.c_uint32),
        ("prcE",       ctypes.c_uint32),

        # Start and end values of Tile width and height, initialised in
        # pi_initialise_encode
        ("txS",        ctypes.c_uint32),
        ("txE",        ctypes.c_uint32),
        ("tyS",        ctypes.c_uint32),
        ("tyE",        ctypes.c_uint32),
        ("dx",         ctypes.c_uint32),
        ("dy",         ctypes.c_uint32),

        # Temporary values for Tile parts, initialised in pi_create_encode
        ("lay_t",      ctypes.c_uint32),
        ("res_t",      ctypes.c_uint32),
        ("comp_t",     ctypes.c_uint32),
        ("prec_t",     ctypes.c_uint32),
        ("tx0_t",      ctypes.c_uint32),
        ("ty0_t",      ctypes.c_uint32)]


class _dparameters_t(ctypes.Structure):
    """Decompression parameters"""
    _fields_ = [
        # Set the number of highest resolutio levels to be discarded.  The
        # image resolution is effectively divided by 2 to the power of
        # discarded levels.  The reduce factor is limited by the smallest
        # total number of decomposition levels among tiles.  If not equal to
        # zero, then the original dimension is divided by 2^(reduce).  If
        # equal to zero or not used, the image is decoded to the full
        # resolution.
        ("cp_reduce",         ctypes.c_uint32),

        # Set the maximum number of quality layers to decode.  If there are
        # fewer quality layers than the specified number, all the quality
        # layers are decoded.
        #
        # If != 0, then only the first cp_layer layers are decoded.
        # If == 0 or not used, all the quality layers are decoded.
        ("cp_layer",          ctypes.c_uint32),

        # input file name
        ("infile",            ctypes.c_char * _PATH_LEN),

        # output file name
        ("outfile",           ctypes.c_char * _PATH_LEN),

        # input file format 0: PGX, 1: PxM, 2: BMP 3:TIF
        # output file format 0: J2K, 1: JP2, 2: JPT
        ("decod_format",      ctypes.c_int),
        ("cod_format",        ctypes.c_int),

        # Decoding area left and right boundary.
        # Decoding area upper and lower boundary.
        ("DA_x0",             ctypes.c_uint32),
        ("DA_x1",             ctypes.c_uint32),
        ("DA_y0",             ctypes.c_uint32),
        ("DA_y1",             ctypes.c_uint32),

        # verbose mode
        ("m_verbose",         _bool_t),

        # tile number of the decoded tile
        ("tile_index",        ctypes.c_uint32),

        # number of tiles to decode
        ("nb_tile_to_decode", ctypes.c_uint32),

        # activates the JPWL correction capabilities
        ("jpwl_correct",      _bool_t),

        # activates the JPWL correction capabilities
        ("jpwl_exp_comps",    ctypes.c_int32),

        # maximum number of tiles
        ("jpwl_max_tiles",    ctypes.c_int32),

        # maximum number of tiles
        ("flags",             ctypes.c_uint32)]


class _cparameters_t(ctypes.Structure):
    """Compression parameters"""
    _fields_ = [
        # size of tile:
        #     tile_size_on = false (not in argument) or
        #                  = true (in argument)
        ("tile_size_on",     _bool_t),

        # XTOsiz, YTOsiz
        ("cp_tx0",           ctypes.c_int),
        ("cp_ty0",           ctypes.c_int),

        # XTsiz, YTsiz
        ("cp_tdx",           ctypes.c_int),
        ("cp_tdy",           ctypes.c_int),

        # allocation by rate/distortion
        ("cp_disto_alloc",   ctypes.c_int),

        # allocation by fixed layer
        ("cp_fixed_alloc",   ctypes.c_int),

        # add fixed_quality
        ("cp_fixed_quality", ctypes.c_int),

        # fixed layer
        ("cp_matrice",       ctypes.c_void_p),

        # comment for coding
        ("cp_comment",       ctypes.c_char_p),

        # csty : coding style
        ("csty",             ctypes.c_int),

        # progression order (default OPJ_LRCP)
        ("prog_order",       ctypes.c_int),

        # progression order changes
        ("poc",              _poc_t * 32),

        # number of progression order changes (POC), default to 0
        ("numpocs",          ctypes.c_uint),

        # number of layers
        ("tcp_numlayers",    ctypes.c_int),

        # rates of layers
        ("tcp_rates",        ctypes.c_float * 100),

        # different psnr for successive layers
        ("tcp_distoratio",   ctypes.c_float * 100),

        # number of resolutions
        ("numresolution",    ctypes.c_int),

        # initial code block width, default to 64
        ("cblockw_init",     ctypes.c_int),

        # initial code block height, default to 64
        ("cblockh_init",     ctypes.c_int),

        # mode switch (cblk_style)
        ("mode",             ctypes.c_int),

        # 1 : use the irreversible DWT 9-7
        # 0 : use lossless compression (default)
        ("irreversible",     ctypes.c_int),

        # region of interest: affected component in [0..3], -1 means no ROI
        ("roi_compno",       ctypes.c_int),

        # region of interest: upshift value
        ("roi_shift",        ctypes.c_int),

        # number of precinct size specifications
        ("res_spec",         ctypes.c_int),

        # initial precinct width
        ("prcw_init",        ctypes.c_int * _J2K_MAXRLVLS),

        # initial precinct height
        ("prch_init",        ctypes.c_int * _J2K_MAXRLVLS),

        # input file name
        ("infile",           ctypes.c_char * _PATH_LEN),

        # output file name
        ("outfile",          ctypes.c_char * _PATH_LEN),

        # DEPRECATED.
        ("index_on",         ctypes.c_int),

        # DEPRECATED.
        ("index",            ctypes.c_char * _PATH_LEN),

        # subimage encoding: origin image offset in x direction
        # subimage encoding: origin image offset in y direction
        ("image_offset_x0",  ctypes.c_int),
        ("image_offset_y0",  ctypes.c_int),

        # subsampling value for dx
        # subsampling value for dy
        ("subsampling_dx",  ctypes.c_int),
        ("subsampling_dy",  ctypes.c_int),

        # input file format 0: PGX, 1: PxM, 2: BMP 3:TIF
        # output file format 0: J2K, 1: JP2, 2: JPT
        ("decod_format",    ctypes.c_int),
        ("cod_format",      ctypes.c_int),

        # JPWL encoding parameters
        # enables writing of EPC in MH, thus activating JPWL
        ("jpwl_epc_on",     _bool_t),

        # error protection method for MH (0,1,16,32,37-128)
        ("jpwl_hprot_mh",   ctypes.c_int),

        # tile number of header protection specification (>=0)
        ("jpwl_hprot_tph_tileno", ctypes.c_int * _JPWL_MAX_NO_TILESPECS),

        # error protection methods for TPHs (0,1,16,32,37-128)
        ("jpwl_hprot_tph",        ctypes.c_int * _JPWL_MAX_NO_TILESPECS),

        # tile number of packet protection specification (>=0)
        ("jpwl_pprot_tileno",     ctypes.c_int * _JPWL_MAX_NO_TILESPECS),

        # packet number of packet protection specification (>=0)
        ("jpwl_pprot_packno",     ctypes.c_int * _JPWL_MAX_NO_TILESPECS),

        # error protection methods for packets (0,1,16,32,37-128)
        ("jpwl_pprot",            ctypes.c_int * _JPWL_MAX_NO_TILESPECS),

        # enables writing of ESD, (0=no/1/2 bytes)
        ("jpwl_sens_size",        ctypes.c_int),

        # sensitivity addressing size (0=auto/2/4 bytes)
        ("jpwl_sens_addr",        ctypes.c_int),

        # sensitivity range (0-3)
        ("jpwl_sens_range",       ctypes.c_int),

        # sensitivity method for MH (-1=no,0-7)
        ("jpwl_sens_mh",          ctypes.c_int),

        # tile number of sensitivity specification (>=0)
        ("jpwl_sens_tph_tileno",  ctypes.c_int * _JPWL_MAX_NO_TILESPECS),

        # sensitivity methods for TPHs (-1=no,0-7)
        ("jpwl_sens_tph",         ctypes.c_int * _JPWL_MAX_NO_TILESPECS),

        # Digital Cinema compliance 0-not compliant, 1-compliant
        ("cp_cinema",             _cinema_mode_t),

        # Maximum rate for each component.
        # If == 0, component size limitation is not considered
        ("max_comp_size",         ctypes.c_int),

        # Profile name
        ("cp_rsiz",               _rsiz_capabilities_t),

        # Tile part generation
        ("tp_on",                 ctypes.c_uint8),

        # Flag for Tile part generation
        ("tp_flag",               ctypes.c_uint8),

        # MCT (multiple component transform)
        ("tcp_mct",               ctypes.c_uint8),

        # Enable JPIP indexing
        ("jpip_on",               _bool_t),

        # Naive implementation of MCT restricted to a single reversible array
        # based encoding without offset concerning all the components.
        ("mct_data",              ctypes.c_void_p)]


class _image_comp_t(ctypes.Structure):
    """defines a single image component"""
    _fields_ = [
        # XRsiz, YRsiz:  horizontal, vertical separation of ith component with
        # respect to the reference grid
        ("dx",                  ctypes.c_uint32),
        ("dy",                  ctypes.c_uint32),

        # data width and height
        ("w",                   ctypes.c_uint32),
        ("h",                   ctypes.c_uint32),

        # x, y component offset compared to the whole image
        ("x0",                  ctypes.c_uint32),
        ("y0",                  ctypes.c_uint32),

        # component depth in bits
        ("prec",                ctypes.c_uint32),

        # component depth in bits
        ("bpp",                 ctypes.c_uint32),

        # signed (1) or unsigned (0)
        ("sgnd",                ctypes.c_uint32),

        # number of decoded resolution
        ("resno_decoded",       ctypes.c_uint32),

        # number of division by 2 of the out image component as compared to the
        # original size of the image
        ("factor",              ctypes.c_uint32),

        # image component data
        ("data",                ctypes.POINTER(ctypes.c_int32))]


class _image_t(ctypes.Structure):
    """defines image data and characteristics"""
    _fields_ = [
        # XOsiz, YOsiz:  horizontal and vertical offset from the origin of the
        # reference grid to the left side of the image area
        ("x0",                  ctypes.c_uint32),
        ("y0",                  ctypes.c_uint32),

        # Xsiz, Ysiz:  width and height of the reference grid.
        ("x1",                  ctypes.c_uint32),
        ("y1",                  ctypes.c_uint32),

        # number of components in the image
        ("numcomps",            ctypes.c_uint32),

        # color space:  should be sRGB, greyscale, or YUV
        ("color_space",         color_space_t),

        # image components
        ("comps",               ctypes.POINTER(_image_comp_t)),

        # restricted ICC profile buffer
        ("icc_profile_buf",     ctypes.POINTER(ctypes.c_uint8)),

        # restricted ICC profile buffer length
        ("icc_profile_len",     ctypes.c_uint32)]


class _image_comptparm_t(ctypes.Structure):
    """component parameters structure used by image_create function"""
    _fields_ = [
        # XRsiz, YRsiz: horizontal, vertical separation of a sample of ith
        # component with respect to the reference grid
        ("dx",              ctypes.c_uint32),
        ("dy",              ctypes.c_uint32),

        # data width, height
        ("w",              ctypes.c_uint32),
        ("h",              ctypes.c_uint32),

        # x, y component offset compared to the whole image
        ("x0",              ctypes.c_uint32),
        ("y0",              ctypes.c_uint32),

        # precision
        ("prec",            ctypes.c_uint32),

        # image depth in bits
        ("bpp",             ctypes.c_uint32),

        # signed (1) / unsigned (0)
        ("sgnd",            ctypes.c_uint32)]


class _tccp_info_t(ctypes.Structure):
    """tile-component coding parameters information"""
    _fields_ = [
        # component index
        ("compno",          ctypes.c_uint32),

        # coding style
        ("csty",            ctypes.c_uint32),

        # number of resolutions
        ("numresolutions",  ctypes.c_uint32),

        # code-blocks width
        ("cblkw",           ctypes.c_uint32),

        # code-blocks height
        ("cblkh",           ctypes.c_uint32),

        # code-block coding style
        ("cblksty",         ctypes.c_uint32),

        # discrete wavelet transform identifier
        ("qmfbid",          ctypes.c_uint32),

        # quantization style
        ("qntsty",          ctypes.c_uint32),

        # stepsizes used for quantization
        ("stepsizes_mant",  ctypes.c_uint32 * _J2K_MAXBANDS),
        ("stepsizes_expn",  ctypes.c_uint32 * _J2K_MAXBANDS),

        # stepsizes used for quantization
        ("numgbits",        ctypes.c_uint32),

        # region of interest shift
        ("roishift",        ctypes.c_int32),

        # precinct width
        ("prcw",            ctypes.c_uint32 * _J2K_MAXRLVLS),

        # precinct width
        ("prch",            ctypes.c_uint32 * _J2K_MAXRLVLS)]


class _tile_info_v2_t(ctypes.Structure):
    """tile coding parameters information"""
    _fields_ = [
        # number (index) of tile
        ("tileno",          ctypes.c_int32),

        # coding style
        ("csty",            ctypes.c_uint32),

        # progression order
        ("prg",             _prog_order_t_p),

        # number of layers
        ("numlayers",       ctypes.c_uint32),

        # multi-component transform identifier
        ("mct",             ctypes.c_uint32),

        # information concerning tile component parameters
        ("tccp_info",       ctypes.POINTER(_tccp_info_t))]


class _codestream_info_v2_t(ctypes.Structure):
    """information about the codestream"""
    _fields_ = [
        # tile info
        # tile origin in x, y (XTOsiz, YTOsiz)
        ("tx0",       ctypes.c_uint32),
        ("ty0",       ctypes.c_uint32),

        # tile size in x, y = XTsiz, YTsiz
        ("tdx",       ctypes.c_uint32),
        ("tdy",       ctypes.c_uint32),

        # number of tiles in X, Y
        ("tw",        ctypes.c_uint32),
        ("th",        ctypes.c_uint32),

        # number of components
        ("nbcomps",   ctypes.c_uint32),

        # default information regarding tiles inside of image
        ("m_default_tile_info",   _tile_info_v2_t),

        # information regarding tiles inside of image
        ("tile_info",             ctypes.POINTER(_tile_info_v2_t))]

# Restrict the input and output argument types for each function used in the
# API.
if _OPENJP2 is not None:
    _OPENJP2.opj_create_compress.argtypes = [_codec_format_t]
    _OPENJP2.opj_create_compress.restype = _codec_t_p

    _OPENJP2.opj_create_decompress.argtypes = [_codec_format_t]
    _OPENJP2.opj_create_decompress.restype = _codec_t_p

    _argtypes = [_codec_t_p, _stream_t_p, ctypes.POINTER(_image_t)]
    _OPENJP2.opj_decode.argtypes = _argtypes

    _argtypes = [_codec_t_p, ctypes.c_uint32,
                 ctypes.POINTER(ctypes.c_uint8),
                 ctypes.c_uint32,
                 _stream_t_p]
    _OPENJP2.opj_decode_tile_data.argtypes = _argtypes

    _argtypes = [ctypes.POINTER(ctypes.POINTER(_codestream_info_v2_t))]
    _OPENJP2.opj_destroy_cstr_info.argtypes = _argtypes
    _OPENJP2.opj_destroy_cstr_info.restype = ctypes.c_void_p

    _argtypes = [_codec_t_p, _stream_t_p]
    _OPENJP2.opj_encode.argtypes = _argtypes

    _OPENJP2.opj_get_cstr_info.argtypes = [_codec_t_p]
    _OPENJP2.opj_get_cstr_info.restype = ctypes.POINTER(_codestream_info_v2_t)

    _argtypes = [_codec_t_p,
                 _stream_t_p,
                 ctypes.POINTER(_image_t),
                 ctypes.c_uint32]
    _OPENJP2.opj_get_decoded_tile.argtypes = _argtypes

    _argtypes = [ctypes.c_uint32,
                 ctypes.POINTER(_image_comptparm_t),
                 color_space_t]
    _OPENJP2.opj_image_create.argtypes = _argtypes
    _OPENJP2.opj_image_create.restype = ctypes.POINTER(_image_t)

    _argtypes = [ctypes.c_uint32,
                 ctypes.POINTER(_image_comptparm_t),
                 color_space_t]
    _OPENJP2.opj_image_tile_create.argtypes = _argtypes
    _OPENJP2.opj_image_tile_create.restype = ctypes.POINTER(_image_t)

    _OPENJP2.opj_image_destroy.argtypes = [ctypes.POINTER(_image_t)]

    _argtypes = [_stream_t_p, _codec_t_p,
                 ctypes.POINTER(ctypes.POINTER(_image_t))]
    _OPENJP2.opj_read_header.argtypes = _argtypes

    _argtypes = [_codec_t_p,
                 _stream_t_p,
                 ctypes.POINTER(ctypes.c_uint32),
                 ctypes.POINTER(ctypes.c_uint32),
                 ctypes.POINTER(ctypes.c_int32),
                 ctypes.POINTER(ctypes.c_int32),
                 ctypes.POINTER(ctypes.c_int32),
                 ctypes.POINTER(ctypes.c_int32),
                 ctypes.POINTER(ctypes.c_uint32),
                 ctypes.POINTER(_bool_t)]
    _OPENJP2.opj_read_tile_header.argtypes = _argtypes

    _argtypes = [_codec_t_p, ctypes.POINTER(_image_t), ctypes.c_int32,
                 ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    _OPENJP2.opj_set_decode_area.argtypes = _argtypes

    _argtypes = [ctypes.POINTER(_cparameters_t)]
    _OPENJP2.opj_set_default_encoder_parameters.argtypes = _argtypes

    _argtypes = [ctypes.POINTER(_dparameters_t)]
    _OPENJP2.opj_set_default_decoder_parameters.argtypes = _argtypes

    _argtypes = [_codec_t_p, ctypes.c_void_p, ctypes.c_void_p]
    _OPENJP2.opj_set_error_handler.argtypes = _argtypes
    _OPENJP2.opj_set_info_handler.argtypes = _argtypes
    _OPENJP2.opj_set_warning_handler.argtypes = _argtypes

    _argtypes = [_codec_t_p, ctypes.POINTER(_dparameters_t)]
    _OPENJP2.opj_setup_decoder.argtypes = _argtypes

    _argtypes = [_codec_t_p,
                 ctypes.POINTER(_cparameters_t),
                 ctypes.POINTER(_image_t)]
    _OPENJP2.opj_setup_encoder.argtypes = _argtypes

    _argtypes = [ctypes.c_char_p, ctypes.c_int32]
    _OPENJP2.opj_stream_create_default_file_stream_v3.argtypes = _argtypes
    _OPENJP2.opj_stream_create_default_file_stream_v3.restype = _stream_t_p

    _argtypes = [_codec_t_p, ctypes.POINTER(_image_t), _stream_t_p]
    _OPENJP2.opj_start_compress.argtypes = _argtypes

    _OPENJP2.opj_end_compress.argtypes = [_codec_t_p, _stream_t_p]
    _OPENJP2.opj_end_decompress.argtypes = [_codec_t_p, _stream_t_p]

    _OPENJP2.opj_stream_destroy_v3.argtypes = [_stream_t_p]
    _OPENJP2.opj_destroy_codec.argtypes = [_codec_t_p]

    _argtypes = [_codec_t_p,
                 ctypes.c_uint32,
                 ctypes.POINTER(ctypes.c_uint8),
                 ctypes.c_uint32,
                 _stream_t_p]
    _OPENJP2.opj_write_tile.argtypes = _argtypes


def _check_error(status):
    """Set a generic function as the restype attribute of all OpenJPEG
    functions that return a _bool_t value.  This way we do not have to check
    for error status in each wrapping function and an exception will always be
    appropriately raised.
    """
    global _ERROR_MSG_LST
    if status != 1:
        if len(_ERROR_MSG_LST) > 0:
            # clear out the existing error message so that we don't pick up
            # a bad one next time around.
            msg = '\n'.join(_ERROR_MSG_LST)
            _ERROR_MSG_LST = []
            raise IOError(msg)
        else:
            raise IOError("OpenJPEG function failure.")

# These library functions all return an error status.  Circumvent that and
# force # them to raise an exception.
_fcns = ['opj_decode', 'opj_decode_tile_data', 'opj_end_compress',
         'opj_encode', 'opj_end_decompress', 'opj_get_decoded_tile',
         'opj_read_header', 'opj_read_tile_header', 'opj_set_decode_area',
         'opj_set_error_handler', 'opj_set_info_handler',
         'opj_set_warning_handler',
         'opj_setup_decoder', 'opj_setup_encoder', 'opj_start_compress',
         'opj_write_tile']
if _OPENJP2 is not None:
    for _fcn in _fcns:
        _attr = getattr(_OPENJP2, _fcn)
        setattr(_attr, 'restype', _check_error)


def _create_compress(codec_format):
    """Creates a J2K/JP2 compress structure.

    Wraps the openjp2 library function opj_create_compress.

    Parameters
    ----------
    codec_format : int
        Specifies codec to select.  Should be one of _CODEC_J2K or _CODEC_JP2.

    Returns
    -------
    codec :  Reference to _codec_t_p instance.
    """
    codec = _OPENJP2.opj_create_compress(codec_format)
    return codec


def _decode(codec, stream, image):
    """Reads an entire image.

    Wraps the openjp2 library function opj_decode.

    Parameters
    ----------
    codec : _codec_t_p
        The JPEG2000 codec
    stream : _stream_t_p
        The stream to decode.
    image : _image_t
        Output image structure.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_decode fails.
    """
    _OPENJP2.opj_decode(codec, stream, image)


def _decode_tile_data(codec, tidx, data, data_size, stream):
    """Reads tile data.

    Wraps the openjp2 library function opj_decode_tile_data.

    Parameters
    ----------
    codec : _codec_t_p
        The JPEG2000 codec
    tile_index : int
        The index of the tile being decoded
    data : array
        Holds a memory block into which data will be decoded.
    data_size : int
        The size of data in bytes
    stream : _stream_t_p
        The stream to decode.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_decode fails.
    """
    datap = data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    _OPENJP2.opj_decode_tile_data(codec,
                                  ctypes.c_uint32(tidx),
                                  datap,
                                  ctypes.c_uint32(data_size),
                                  stream)
    return codec


def _create_decompress(codec_format):
    """Creates a J2K/JP2 decompress structure.

    Wraps the openjp2 library function opj_create_decompress.

    Parameters
    ----------
    codec_format : int
        Specifies codec to select.  Should be one of _CODEC_J2K or _CODEC_JP2.

    Returns
    -------
    codec : Reference to _codec_t_p instance.
    """
    codec = _OPENJP2.opj_create_decompress(codec_format)
    return codec


def _destroy_codec(codec):
    """Destroy a decompressor handle.

    Wraps the openjp2 library function opj_destroy_codec.

    Parameters
    ----------
    codec : _codec_t_p
        Decompressor handle to destroy.
    """
    _OPENJP2.opj_destroy_codec(codec)


def _encode(codec, stream):
    """Wraps openjp2 library function opj_encode.

    Encode an image into a JPEG 2000 codestream.

    Parameters
    ----------
    codec : _codec_t_p
        The jpeg2000 codec.
    stream : _stream_t_p
        The stream to which data is written.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_encode fails.
    """
    _OPENJP2.opj_encode(codec, stream)


def _get_cstr_info(codec):
    """get the codestream information from the codec

    Wraps the openjp2 library function opj_get_cstr_info.

    Parameters
    ----------
    codec : _codec_t_p
        The jpeg2000 codec.

    Returns
    -------
    cstr_info_p : _codestream_info_v2_t
        Reference to codestream information.
    """
    cstr_info_p = _OPENJP2.opj_get_cstr_info(codec)
    return cstr_info_p


def _get_decoded_tile(codec, stream, imagep, tile_index):
    """get the decoded tile from the codec

    Wraps the openjp2 library function opj_get_decoded_tile.

    Parameters
    ----------
    codec : _codec_t_p
        The jpeg2000 codec.
    stream : _stream_t_p
        The input stream.
    image : _image_t
        Output image structure.
    tiler_index : int
        Index of the tile which will be decoded.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_get_decoded_tile fails.
    """
    _OPENJP2.opj_get_decoded_tile(codec, stream, imagep, tile_index)


def _destroy_cstr_info(cstr_info_p):
    """destroy codestream information after compression or decompression

    Wraps the openjp2 library function opj_destroy_cstr_info.

    Parameters
    ----------
    cstr_info_p : _codestream_info_v2_t pointer
        Pointer to codestream info structure.
    """
    _OPENJP2.opj_destroy_cstr_info(ctypes.byref(cstr_info_p))


def _end_compress(codec, stream):
    """End of compressing the current image.

    Wraps the openjp2 library function opj_end_compress.

    Parameters
    ----------
    codec : _codec_t_p
        Compressor handle.
    stream : _stream_t_p
        Output stream buffer.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_end_compress fails.
    """
    _OPENJP2.opj_end_compress(codec, stream)


def _end_decompress(codec, stream):
    """End of decompressing the current image.

    Wraps the openjp2 library function opj_end_decompress.

    Parameters
    ----------
    codec : _codec_t_p
        Compressor handle.
    stream : _stream_t_p
        Output stream buffer.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_end_decompress fails.
    """
    _OPENJP2.opj_end_decompress(codec, stream)


def _image_destroy(image):
    """Deallocate any resources associated with an image.

    Wraps the openjp2 library function opj_image_destroy.

    Parameters
    ----------
    image : _image_t pointer
        Image resource to be disposed.
    """
    _OPENJP2.opj_image_destroy(image)


def _image_create(comptparms, clrspc):
    """Creates a new image structure.

    Wraps the openjp2 library function opj_image_create.

    Parameters
    ----------
    cmptparms : comptparms_t
        The component parameters.
    clrspc : int
        Specifies the color space.

    Returns
    -------
    image : _image_t
        Reference to _image_t instance.
    """
    image = _OPENJP2.opj_image_create(len(comptparms),
                                      comptparms,
                                      clrspc)
    return image


def _image_tile_create(comptparms, clrspc):
    """Creates a new image structure.

    Wraps the openjp2 library function opj_image_tile_create.

    Parameters
    ----------
    cmptparms : comptparms_t
        The component parameters.
    clrspc : int
        Specifies the color space.

    Returns
    -------
    image : _image_t
        Reference to _image_t instance.
    """
    image = _OPENJP2.opj_image_tile_create(len(comptparms),
                                           comptparms,
                                           clrspc)
    return image


def _read_header(stream, codec):
    """Decodes an image header.

    Wraps the openjp2 library function opj_read_header.

    Parameters
    ----------
    stream: _stream_t_p
        The JPEG2000 stream.
    codec:  codec_t
        The JPEG2000 codec to read.

    Returns
    -------
    imagep : reference to _image_t instance
        The image structure initialized with image characteristics.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_read_header fails.
    """
    imagep = ctypes.POINTER(_image_t)()
    _OPENJP2.opj_read_header(stream, codec, ctypes.byref(imagep))
    return imagep


def _read_tile_header(codec, stream):
    """Reads a tile header.

    Wraps the openjp2 library function opj_read_tile_header.

    Parameters
    ----------
    codec : codec_t
        The JPEG2000 codec to read.
    stream : _stream_t_p
        The JPEG2000 stream.

    Returns
    -------
    tile_index : int
        index of the tile being decoded
    data_size : int
        number of bytes for the decoded area
    x0, y0 : int
        upper left-most coordinate of tile
    x1, y1 : int
        lower right-most coordinate of tile
    ncomps : int
        number of components in the tile
    go_on : bool
        indicates that decoding should continue

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_read_tile_header fails.
    """
    tile_index = ctypes.c_uint32()
    data_size = ctypes.c_uint32()
    x0 = ctypes.c_int32()
    y0 = ctypes.c_int32()
    x1 = ctypes.c_int32()
    y1 = ctypes.c_int32()
    ncomps = ctypes.c_uint32()
    go_on = _bool_t()
    _OPENJP2.opj_read_tile_header(codec,
                                  stream,
                                  ctypes.byref(tile_index),
                                  ctypes.byref(data_size),
                                  ctypes.byref(x0),
                                  ctypes.byref(y0),
                                  ctypes.byref(x1),
                                  ctypes.byref(y1),
                                  ctypes.byref(ncomps),
                                  ctypes.byref(go_on))
    go_on = bool(go_on.value)
    return (tile_index.value,
            data_size.value,
            x0.value,
            y0.value,
            x1.value,
            y1.value,
            ncomps.value,
            go_on)


def _set_decode_area(codec, image, start_x=0, start_y=0, end_x=0, end_y=0):
    """Wraps openjp2 library function opj_set_decode area.

    Sets the given area to be decoded.  This function should be called right
    after read_header and before any tile header reading.

    Parameters
    ----------
    codec : _codec_t_p
        Codec initialized by create_decompress function.
    image : _image_t pointer
        The decoded image previously set by read_header.
    start_x, start_y : optional, int
        The left and upper position of the rectangle to decode.
    end_x, end_y : optional, int
        The right and lower position of the rectangle to decode.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_set_decode_area fails.
    """
    _OPENJP2.opj_set_decode_area(codec, image,
                                 ctypes.c_int32(start_x),
                                 ctypes.c_int32(start_y),
                                 ctypes.c_int32(end_x),
                                 ctypes.c_int32(end_y))


def _set_default_decoder_parameters():
    """Wraps openjp2 library function opj_set_default_decoder_parameters.

    Sets decoding parameters to default values.

    Returns
    -------
    dparam : _dparameters_t
        Decompression parameters.
    """
    dparams = _dparameters_t()
    _OPENJP2.opj_set_default_decoder_parameters(ctypes.byref(dparams))
    return dparams


def _set_default_encoder_parameters():
    """Wraps openjp2 library function opj_set_default_encoder_parameters.

    Sets encoding parameters to default values.  That means

        lossless
        1 tile
        size of precinct : 2^15 x 2^15 (means 1 precinct)
        size of code-block : 64 x 64
        number of resolutions: 6
        no SOP marker in the codestream
        no EPH marker in the codestream
        no sub-sampling in x or y direction
        no mode switch activated
        progression order: LRCP
        no index file
        no ROI upshifted
        no offset of the origin of the image
        no offset of the origin of the tiles
        reversible DWT 5-3

    The signature for this function differs from its C library counterpart, as
    the the C function pass-by-reference parameter becomes the Python return
    value.

    Returns
    -------
    cparameters : _cparameters_t
        Compression parameters.
    """
    cparams = _cparameters_t()
    _OPENJP2.opj_set_default_encoder_parameters(ctypes.byref(cparams))
    return cparams


def _set_error_handler(codec, handler, data=None):
    """Wraps openjp2 library function opj_set_error_handler.

    Set the error handler use by openjpeg.

    Parameters
    ----------
    codec : _codec_t_p
        Codec initialized by create_compress function.
    handler : python function
        The callback function to be used.
    user_data : anything
        User/client data.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_set_error_handler fails.
    """
    _OPENJP2.opj_set_error_handler(codec, handler, data)


def _set_info_handler(codec, handler, data=None):
    """Wraps openjp2 library function opj_set_info_handler.

    Set the info handler use by openjpeg.

    Parameters
    ----------
    codec : _codec_t_p
        Codec initialized by create_compress function.
    handler : python function
        The callback function to be used.
    user_data : anything
        User/client data.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_set_info_handler fails.
    """
    _OPENJP2.opj_set_info_handler(codec, handler, data)


def _set_warning_handler(codec, handler, data=None):
    """Wraps openjp2 library function opj_set_warning_handler.

    Set the warning handler use by openjpeg.

    Parameters
    ----------
    codec : _codec_t_p
        Codec initialized by create_compress function.
    handler : python function
        The callback function to be used.
    user_data : anything
        User/client data.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_set_warning_handler fails.
    """
    _OPENJP2.opj_set_warning_handler(codec, handler, data)


def _setup_decoder(codec, dparams):
    """Wraps openjp2 library function opj_setup_decoder.

    Setup the decoder with decompression parameters.

    Parameters
    ----------
    codec:  _codec_t_p
        Codec initialized by create_compress function.
    dparams:  _dparameters_t
        Decompression parameters.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_setup_decoder fails.
    """
    _OPENJP2.opj_setup_decoder(codec, ctypes.byref(dparams))


def _setup_encoder(codec, cparams, image):
    """Wraps openjp2 library function opj_setup_encoder.

    Setup the encoder parameters using the current image and using user
    parameters.

    Parameters
    ----------
    codec : _codec_t_p
        codec initialized by create_compress function
    cparams : _cparameters_t
        compression parameters
    image : _image_t
        input-filled image

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_setup_encoder fails.
    """
    _OPENJP2.opj_setup_encoder(codec, ctypes.byref(cparams), image)


def _start_compress(codec, image, stream):
    """Wraps openjp2 library function opj_start_compress.

    Start to compress the current image.

    Parameters
    ----------
    codec : _codec_t_p
        Compressor handle.
    image : pointer to _image_t
        Input filled image.
    stream : _stream_t_p
        Input stream.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_start_compress fails.
    """
    _OPENJP2.opj_start_compress(codec, image, stream)


def _stream_create_default_file_stream_v3(fname, a_read_stream):
    """Wraps openjp2 library function opj_stream_create_default_vile_stream_v3.

    Sets the stream to be a file stream.

    Parameters
    ----------
    fname : str
        Specifies a file.
    a_read_stream:  bool
        True (read) or False (write)

    Returns
    -------
    stream : stream_t
        An OpenJPEG file stream.
    """
    tf = 1 if a_read_stream else 0
    fn = ctypes.c_char_p(fname.encode())
    stream = _OPENJP2.opj_stream_create_default_file_stream_v3(fn, tf)
    return stream


def _stream_destroy_v3(stream):
    """Wraps openjp2 library function opj_stream_destroy.

    Destroys the stream created by create_stream_v3.

    Parameters
    ----------
    stream : _stream_t_p
        The file stream.
    """
    _OPENJP2.opj_stream_destroy_v3(stream)


def _write_tile(codec, tile_index, data, data_size, stream):
    """Wraps openjp2 library function opj_write_tile.

    Write a tile into an image.

    Parameters
    ----------
    codec : _codec_t_p
        The jpeg2000 codec
    tile_index : int
        The index of the tile to write, zero-indexing assumed
    data : array
        Image data arranged in usual C-order
    data_size : int
        Size of a tile in bytes
    stream : _stream_t_p
        The stream to write data to

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_write_tile fails.
    """
    datap = data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    _OPENJP2.opj_write_tile(codec,
                            ctypes.c_uint32(int(tile_index)),
                            datap,
                            ctypes.c_uint32(int(data_size)),
                            stream)


def _set_error_message(msg):
    """The openjpeg error handler has recorded an error message."""
    global _ERROR_MSG_LST
    _ERROR_MSG_LST.append(msg)
