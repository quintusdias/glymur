"""
Wraps individual functions in openjp2 library.
"""

import ctypes
import re
import sys
import textwrap

from .config import glymur_config

OPENJP2, OPENJPEG = glymur_config()

class OpenJPEGLibraryError(IOError):
    """
    Issue when the OpenJPEG library signals an error.
    """
    pass


def version():
    """Wrapper for opj_version library routine."""
    try:
        OPENJP2.opj_version.restype = ctypes.c_char_p
    except:
        return "0.0.0"

    library_version = OPENJP2.opj_version()
    if sys.hexversion >= 0x03000000:
        return library_version.decode('utf-8')
    else:
        return library_version

if OPENJP2 is not None:
    _MAJOR, _MINOR, _PATCH = version().split('.')
else:
    _MINOR = 0

ERROR_MSG_LST = []

# Map certain atomic OpenJPEG datatypes to the ctypes equivalents.
BOOL_TYPE = ctypes.c_int32
CODEC_TYPE = ctypes.c_void_p
PROG_ORDER_TYPE = ctypes.c_int32
CINEMA_MODE_TYPE = ctypes.c_int32
RSIZ_CAPABILITIES_TYPE = ctypes.c_int32
STREAM_TYPE_P = ctypes.c_void_p

PATH_LEN = 4096
J2K_MAXRLVLS = 33
J2K_MAXBANDS = (3 * J2K_MAXRLVLS - 2)

JPWL_MAX_NO_TILESPECS = 16

TRUE = 1
FALSE = 0

# supported color spaces
CLRSPC_UNKNOWN = -1
CLRSPC_UNSPECIFIED = 0
CLRSPC_SRGB = 1
CLRSPC_GRAY = 2
CLRSPC_YCC = 3
CLRSPC_EYCC = 4
COLOR_SPACE_TYPE = ctypes.c_int

# supported codec
CODEC_FORMAT_TYPE = ctypes.c_int
CODEC_UNKNOWN = -1
CODEC_J2K = 0
CODEC_JPT = 1
CODEC_JP2 = 2


class PocType(ctypes.Structure):
    """Progression order changes.

    Corresponds to poc_t type in openjp2 headers.
    """
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
        ("prg1",       PROG_ORDER_TYPE),
        ("prg",        PROG_ORDER_TYPE),

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

    def __str__(self):
        msg = "{0}:\n".format(self.__class__)
        for field_name, _ in self._fields_:
            msg += "    {0}: {1}\n".format(
                field_name, getattr(self, field_name))
        return msg


class DecompressionParametersType(ctypes.Structure):
    """Decompression parameters.

    Corresponds to dparameters_t type in openjp2 headers.
    """
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
        ("infile",            ctypes.c_char * PATH_LEN),

        # output file name
        ("outfile",           ctypes.c_char * PATH_LEN),

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
        ("m_verbose",         BOOL_TYPE),

        # tile number of the decoded tile
        ("tile_index",        ctypes.c_uint32),

        # number of tiles to decode
        ("nb_tile_to_decode", ctypes.c_uint32),

        # activates the JPWL correction capabilities
        ("jpwl_correct",      BOOL_TYPE),

        # activates the JPWL correction capabilities
        ("jpwl_exp_comps",    ctypes.c_int32),

        # maximum number of tiles
        ("jpwl_max_tiles",    ctypes.c_int32),

        # maximum number of tiles
        ("flags",             ctypes.c_uint32)]

    def __str__(self):
        msg = "{0}:\n".format(self.__class__)
        for field_name, _ in self._fields_:
            msg += "    {0}: {1}\n".format(
                field_name, getattr(self, field_name))
        return msg


class CompressionParametersType(ctypes.Structure):
    """Compression parameters.

    Corresponds to cparameters_t type in openjp2 headers.
    """
    _fields_ = [
        # size of tile:
        #     tile_size_on = false (not in argument) or
        #                  = true (in argument)
        ("tile_size_on",     BOOL_TYPE),

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
        ("poc",              PocType * 32),

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
        ("prcw_init",        ctypes.c_int * J2K_MAXRLVLS),

        # initial precinct height
        ("prch_init",        ctypes.c_int * J2K_MAXRLVLS),

        # input file name
        ("infile",           ctypes.c_char * PATH_LEN),

        # output file name
        ("outfile",          ctypes.c_char * PATH_LEN),

        # DEPRECATED.
        ("index_on",         ctypes.c_int),

        # DEPRECATED.
        ("index",            ctypes.c_char * PATH_LEN),

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
        ("jpwl_epc_on",     BOOL_TYPE),

        # error protection method for MH (0,1,16,32,37-128)
        ("jpwl_hprot_mh",   ctypes.c_int),

        # tile number of header protection specification (>=0)
        ("jpwl_hprot_tph_tileno", ctypes.c_int * JPWL_MAX_NO_TILESPECS),

        # error protection methods for TPHs (0,1,16,32,37-128)
        ("jpwl_hprot_tph",        ctypes.c_int * JPWL_MAX_NO_TILESPECS),

        # tile number of packet protection specification (>=0)
        ("jpwl_pprot_tileno",     ctypes.c_int * JPWL_MAX_NO_TILESPECS),

        # packet number of packet protection specification (>=0)
        ("jpwl_pprot_packno",     ctypes.c_int * JPWL_MAX_NO_TILESPECS),

        # error protection methods for packets (0,1,16,32,37-128)
        ("jpwl_pprot",            ctypes.c_int * JPWL_MAX_NO_TILESPECS),

        # enables writing of ESD, (0=no/1/2 bytes)
        ("jpwl_sens_size",        ctypes.c_int),

        # sensitivity addressing size (0=auto/2/4 bytes)
        ("jpwl_sens_addr",        ctypes.c_int),

        # sensitivity range (0-3)
        ("jpwl_sens_range",       ctypes.c_int),

        # sensitivity method for MH (-1=no,0-7)
        ("jpwl_sens_mh",          ctypes.c_int),

        # tile number of sensitivity specification (>=0)
        ("jpwl_sens_tph_tileno",  ctypes.c_int * JPWL_MAX_NO_TILESPECS),

        # sensitivity methods for TPHs (-1=no,0-7)
        ("jpwl_sens_tph",         ctypes.c_int * JPWL_MAX_NO_TILESPECS),

        # Digital Cinema compliance 0-not compliant, 1-compliant
        ("cp_cinema",             CINEMA_MODE_TYPE),

        # Maximum rate for each component.
        # If == 0, component size limitation is not considered
        ("max_comp_size",         ctypes.c_int),

        # Profile name
        ("cp_rsiz",               RSIZ_CAPABILITIES_TYPE),

        # Tile part generation
        ("tp_on",                 ctypes.c_uint8),

        # Flag for Tile part generation
        ("tp_flag",               ctypes.c_uint8),

        # MCT (multiple component transform)
        ("tcp_mct",               ctypes.c_uint8),

        # Enable JPIP indexing
        ("jpip_on",               BOOL_TYPE),

        # Naive implementation of MCT restricted to a single reversible array
        # based encoding without offset concerning all the components.
        ("mct_data",              ctypes.c_void_p)]

    if _MINOR == '1':
        # Maximum size (in bytes) for the whole codestream.
        # If == 0, codestream size limitation is not considered.
        # If it does not comply with tcp_rates, max_cs_size prevails and a
        # warning is issued.
        _fields_.append(("max_cs_size",               ctypes.c_int32))

        # To be used to combine OPJ_PROFILE_*, OPJ_EXTENSION_* and (sub)levels
        # values.
        _fields_.append(("rsiz",                      ctypes.c_uint16))

    def __str__(self):
        msg = "{0}:\n".format(self.__class__)
        for field_name, _ in self._fields_:

            if field_name == 'poc':
                msg += "    numpocs: {0}\n".format(self.numpocs)
                for j in range(self.numpocs):
                    msg += "        [#{0}]:".format(j)
                    msg += "            {0}".format(str(self.poc[j]))

            elif field_name in ['tcp_rates', 'tcp_distoratio']:
                lst = []
                arr = getattr(self, field_name)
                lst = [arr[j] for j in range(self.tcp_numlayers)]
                msg += "    {0}: {1}\n".format(field_name, lst)

            elif field_name in ['prcw_init', 'prch_init']:
                pass

            elif field_name == 'res_spec':
                prcw_init = [self.prcw_init[j] for j in range(self.res_spec)]
                prch_init = [self.prch_init[j] for j in range(self.res_spec)]
                msg += "    res_spec: {0}\n".format(self.res_spec)
                msg += "    prch_init: {0}\n".format(prch_init)
                msg += "    prcw_init: {0}\n".format(prcw_init)

            elif field_name in [
                    'jpwl_hprot_tph_tileno', 'jpwl_hprot_tph',
                    'jpwl_pprot_tileno', 'jpwl_pprot_packno', 'jpwl_pprot',
                    'jpwl_sens_tph_tileno', 'jpwl_sens_tph']:
                arr = getattr(self, field_name)
                lst = [arr[j] for j in range(JPWL_MAX_NO_TILESPECS)]
                msg += "    {0}: {1}\n".format(field_name, lst)

            else:
                msg += "    {0}: {1}\n".format(
                    field_name, getattr(self, field_name))
        return msg


class ImageCompType(ctypes.Structure):
    """Defines a single image component.

    Corresponds to image_comp_t type in openjp2 headers.
    """
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

    if _MINOR == '1':
        _fields_.append(("alpha",               ctypes.c_uint16))

    def __str__(self):
        msg = "{0}:\n".format(self.__class__)
        for field_name, _ in self._fields_:
            msg += "    {0}: {1}\n".format(
                field_name, getattr(self, field_name))
        return msg


class ImageType(ctypes.Structure):
    """Defines image data and characteristics.

    Corresponds to image_t type in openjp2 headers.
    """
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
        ("color_space",         COLOR_SPACE_TYPE),

        # image components
        ("comps",               ctypes.POINTER(ImageCompType)),

        # restricted ICC profile buffer
        ("icc_profile_buf",     ctypes.POINTER(ctypes.c_uint8)),

        # restricted ICC profile buffer length
        ("icc_profile_len",     ctypes.c_uint32)]

    def __str__(self):
        msg = "{0}:\n".format(self.__class__)
        for field_name, _ in self._fields_:

            if field_name == "numcomps":
                msg += "    numcomps: {0}\n".format(self.numcomps)
                for j in range(self.numcomps):
                    msg += "        comps[#{0}]:\n".format(j)
                    msg += textwrap.indent(str(self.comps[j]), ' ' * 12)

            elif field_name == "comps":
                # handled above
                pass

            else:
                msg += "    {0}: {1}\n".format(
                    field_name, getattr(self, field_name))

        return msg


class ImageComptParmType(ctypes.Structure):
    """Component parameters structure used by image_create function.

    Corresponds to image_comptparm_t type in openjp2 headers.
    """
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

    def __str__(self):
        msg = "{0}:\n".format(self.__class__)
        for field_name, _ in self._fields_:
            msg += "    {0}: {1}\n".format(
                field_name, getattr(self, field_name))
        return msg


def check_error(status):
    """Set a generic function as the restype attribute of all OpenJPEG
    functions that return a BOOL_TYPE value.  This way we do not have to check
    for error status in each wrapping function and an exception will always be
    appropriately raised.
    """
    global ERROR_MSG_LST
    if status != 1:
        if len(ERROR_MSG_LST) > 0:
            # clear out the existing error message so that we don't pick up
            # a bad one next time around.
            msg = '\n'.join(ERROR_MSG_LST)
            ERROR_MSG_LST = []
            raise OpenJPEGLibraryError(msg)
        else:
            raise OpenJPEGLibraryError("OpenJPEG function failure.")


def create_compress(codec_format):
    """Creates a J2K/JP2 compress structure.

    Wraps the openjp2 library function opj_create_compress.

    Parameters
    ----------
    codec_format : int
        Specifies codec to select.  Should be one of CODEC_J2K or CODEC_JP2.

    Returns
    -------
    codec :  Reference to CODEC_TYPE instance.
    """
    OPENJP2.opj_create_compress.restype = CODEC_TYPE
    OPENJP2.opj_create_compress.argtypes = [CODEC_FORMAT_TYPE]

    codec = OPENJP2.opj_create_compress(codec_format)
    return codec


def decode(codec, stream, image):
    """Reads an entire image.

    Wraps the openjp2 library function opj_decode.

    Parameters
    ----------
    codec : CODEC_TYPE
        The JPEG2000 codec
    stream : STREAM_TYPE_P
        The stream to decode.
    image : ImageType
        Output image structure.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_decode fails.
    """
    OPENJP2.opj_decode.argtypes = [CODEC_TYPE, STREAM_TYPE_P,
                                   ctypes.POINTER(ImageType)]
    OPENJP2.opj_decode.restype = check_error

    OPENJP2.opj_decode(codec, stream, image)


def decode_tile_data(codec, tidx, data, data_size, stream):
    """Reads tile data.

    Wraps the openjp2 library function opj_decode_tile_data.

    Parameters
    ----------
    codec : CODEC_TYPE
        The JPEG2000 codec
    tile_index : int
        The index of the tile being decoded
    data : array
        Holds a memory block into which data will be decoded.
    data_size : int
        The size of data in bytes
    stream : STREAM_TYPE_P
        The stream to decode.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_decode fails.
    """
    OPENJP2.opj_decode_tile_data.argtypes = [CODEC_TYPE,
                                             ctypes.c_uint32,
                                             ctypes.POINTER(ctypes.c_uint8),
                                             ctypes.c_uint32,
                                             STREAM_TYPE_P]
    OPENJP2.opj_decode_tile_data.restype = check_error

    datap = data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    OPENJP2.opj_decode_tile_data(codec,
                                 ctypes.c_uint32(tidx),
                                 datap,
                                 ctypes.c_uint32(data_size),
                                 stream)


def create_decompress(codec_format):
    """Creates a J2K/JP2 decompress structure.

    Wraps the openjp2 library function opj_create_decompress.

    Parameters
    ----------
    codec_format : int
        Specifies codec to select.  Should be one of CODEC_J2K or CODEC_JP2.

    Returns
    -------
    codec : Reference to CODEC_TYPE instance.
    """
    OPENJP2.opj_create_decompress.argtypes = [CODEC_FORMAT_TYPE]
    OPENJP2.opj_create_decompress.restype = CODEC_TYPE

    codec = OPENJP2.opj_create_decompress(codec_format)
    return codec


def destroy_codec(codec):
    """Destroy a decompressor handle.

    Wraps the openjp2 library function opj_destroy_codec.

    Parameters
    ----------
    codec : CODEC_TYPE
        Decompressor handle to destroy.
    """
    OPENJP2.opj_destroy_codec.argtypes = [CODEC_TYPE]
    OPENJP2.opj_destroy_codec.restype = ctypes.c_void_p
    OPENJP2.opj_destroy_codec(codec)


def encode(codec, stream):
    """Wraps openjp2 library function opj_encode.

    Encode an image into a JPEG 2000 codestream.

    Parameters
    ----------
    codec : CODEC_TYPE
        The jpeg2000 codec.
    stream : STREAM_TYPE_P
        The stream to which data is written.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_encode fails.
    """
    OPENJP2.opj_encode.argtypes = [CODEC_TYPE, STREAM_TYPE_P]
    OPENJP2.opj_encode.restype = check_error

    OPENJP2.opj_encode(codec, stream)


def get_decoded_tile(codec, stream, imagep, tile_index):
    """get the decoded tile from the codec

    Wraps the openjp2 library function opj_get_decoded_tile.

    Parameters
    ----------
    codec : CODEC_TYPE
        The jpeg2000 codec.
    stream : STREAM_TYPE_P
        The input stream.
    image : ImageType
        Output image structure.
    tiler_index : int
        Index of the tile which will be decoded.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_get_decoded_tile fails.
    """
    OPENJP2.opj_get_decoded_tile.argtypes = [CODEC_TYPE,
                                             STREAM_TYPE_P,
                                             ctypes.POINTER(ImageType),
                                             ctypes.c_uint32]
    OPENJP2.opj_get_decoded_tile.restype = check_error

    OPENJP2.opj_get_decoded_tile(codec, stream, imagep, tile_index)


def end_compress(codec, stream):
    """End of compressing the current image.

    Wraps the openjp2 library function opj_end_compress.

    Parameters
    ----------
    codec : CODEC_TYPE
        Compressor handle.
    stream : STREAM_TYPE_P
        Output stream buffer.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_end_compress fails.
    """
    OPENJP2.opj_end_compress.argtypes = [CODEC_TYPE, STREAM_TYPE_P]
    OPENJP2.opj_end_compress.restype = check_error
    OPENJP2.opj_end_compress(codec, stream)


def end_decompress(codec, stream):
    """End of decompressing the current image.

    Wraps the openjp2 library function opj_end_decompress.

    Parameters
    ----------
    codec : CODEC_TYPE
        Compressor handle.
    stream : STREAM_TYPE_P
        Output stream buffer.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_end_decompress fails.
    """
    OPENJP2.opj_end_decompress.argtypes = [CODEC_TYPE, STREAM_TYPE_P]
    OPENJP2.opj_end_decompress.restype = check_error
    OPENJP2.opj_end_decompress(codec, stream)


def image_destroy(image):
    """Deallocate any resources associated with an image.

    Wraps the openjp2 library function opj_image_destroy.

    Parameters
    ----------
    image : ImageType pointer
        Image resource to be disposed.
    """
    OPENJP2.opj_image_destroy.argtypes = [ctypes.POINTER(ImageType)]
    OPENJP2.opj_image_destroy.restype = ctypes.c_void_p

    OPENJP2.opj_image_destroy(image)


def image_create(comptparms, clrspc):
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
    image : ImageType
        Reference to ImageType instance.
    """
    OPENJP2.opj_image_create.argtypes = [ctypes.c_uint32,
                                         ctypes.POINTER(ImageComptParmType),
                                         COLOR_SPACE_TYPE]
    OPENJP2.opj_image_create.restype = ctypes.POINTER(ImageType)

    image = OPENJP2.opj_image_create(len(comptparms),
                                     comptparms,
                                     clrspc)
    return image


def image_tile_create(comptparms, clrspc):
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
    image : ImageType
        Reference to ImageType instance.
    """
    ARGTYPES = [ctypes.c_uint32,
                ctypes.POINTER(ImageComptParmType),
                COLOR_SPACE_TYPE]
    OPENJP2.opj_image_tile_create.argtypes = ARGTYPES
    OPENJP2.opj_image_tile_create.restype = ctypes.POINTER(ImageType)

    image = OPENJP2.opj_image_tile_create(len(comptparms),
                                          comptparms,
                                          clrspc)
    return image


def read_header(stream, codec):
    """Decodes an image header.

    Wraps the openjp2 library function opj_read_header.

    Parameters
    ----------
    stream: STREAM_TYPE_P
        The JPEG2000 stream.
    codec:  codec_t
        The JPEG2000 codec to read.

    Returns
    -------
    imagep : reference to ImageType instance
        The image structure initialized with image characteristics.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_read_header fails.
    """
    ARGTYPES = [STREAM_TYPE_P, CODEC_TYPE,
                ctypes.POINTER(ctypes.POINTER(ImageType))]
    OPENJP2.opj_read_header.argtypes = ARGTYPES
    OPENJP2.opj_read_header.restype = check_error

    imagep = ctypes.POINTER(ImageType)()
    OPENJP2.opj_read_header(stream, codec, ctypes.byref(imagep))
    return imagep


def read_tile_header(codec, stream):
    """Reads a tile header.

    Wraps the openjp2 library function opj_read_tile_header.

    Parameters
    ----------
    codec : codec_t
        The JPEG2000 codec to read.
    stream : STREAM_TYPE_P
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
    ARGTYPES = [CODEC_TYPE,
                STREAM_TYPE_P,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(BOOL_TYPE)]
    OPENJP2.opj_read_tile_header.argtypes = ARGTYPES
    OPENJP2.opj_read_tile_header.restype = check_error

    tile_index = ctypes.c_uint32()
    data_size = ctypes.c_uint32()
    col0 = ctypes.c_int32()
    row0 = ctypes.c_int32()
    col1 = ctypes.c_int32()
    row1 = ctypes.c_int32()
    ncomps = ctypes.c_uint32()
    go_on = BOOL_TYPE()
    OPENJP2.opj_read_tile_header(codec,
                                 stream,
                                 ctypes.byref(tile_index),
                                 ctypes.byref(data_size),
                                 ctypes.byref(col0),
                                 ctypes.byref(row0),
                                 ctypes.byref(col1),
                                 ctypes.byref(row1),
                                 ctypes.byref(ncomps),
                                 ctypes.byref(go_on))
    go_on = bool(go_on.value)
    return (tile_index.value,
            data_size.value,
            col0.value,
            row0.value,
            col1.value,
            row1.value,
            ncomps.value,
            go_on)


def set_decode_area(codec, image, start_x=0, start_y=0, end_x=0, end_y=0):
    """Wraps openjp2 library function opj_set_decode area.

    Sets the given area to be decoded.  This function should be called right
    after read_header and before any tile header reading.

    Parameters
    ----------
    codec : CODEC_TYPE
        Codec initialized by create_decompress function.
    image : ImageType pointer
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
    OPENJP2.opj_set_decode_area.argtypes = [CODEC_TYPE,
                                            ctypes.POINTER(ImageType),
                                            ctypes.c_int32,
                                            ctypes.c_int32,
                                            ctypes.c_int32,
                                            ctypes.c_int32]
    OPENJP2.opj_set_decode_area.restype = check_error

    OPENJP2.opj_set_decode_area(codec, image,
                                ctypes.c_int32(start_x),
                                ctypes.c_int32(start_y),
                                ctypes.c_int32(end_x),
                                ctypes.c_int32(end_y))


def set_default_decoder_parameters():
    """Wraps openjp2 library function opj_set_default_decoder_parameters.

    Sets decoding parameters to default values.

    Returns
    -------
    dparam : DecompressionParametersType
        Decompression parameters.
    """
    ARGTYPES = [ctypes.POINTER(DecompressionParametersType)]
    OPENJP2.opj_set_default_decoder_parameters.argtypes = ARGTYPES
    OPENJP2.opj_set_default_decoder_parameters.restype = ctypes.c_void_p

    dparams = DecompressionParametersType()
    OPENJP2.opj_set_default_decoder_parameters(ctypes.byref(dparams))
    return dparams


def set_default_encoder_parameters():
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
    cparameters : CompressionParametersType
        Compression parameters.
    """
    ARGTYPES = [ctypes.POINTER(CompressionParametersType)]
    OPENJP2.opj_set_default_encoder_parameters.argtypes = ARGTYPES
    OPENJP2.opj_set_default_encoder_parameters.restype = ctypes.c_void_p

    cparams = CompressionParametersType()
    OPENJP2.opj_set_default_encoder_parameters(ctypes.byref(cparams))
    return cparams


def set_error_handler(codec, handler, data=None):
    """Wraps openjp2 library function opj_set_error_handler.

    Set the error handler use by openjpeg.

    Parameters
    ----------
    codec : CODEC_TYPE
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
    OPENJP2.opj_set_error_handler.argtypes = [CODEC_TYPE,
                                              ctypes.c_void_p,
                                              ctypes.c_void_p]
    OPENJP2.opj_set_error_handler.restype = check_error
    OPENJP2.opj_set_error_handler(codec, handler, data)


def set_info_handler(codec, handler, data=None):
    """Wraps openjp2 library function opj_set_info_handler.

    Set the info handler use by openjpeg.

    Parameters
    ----------
    codec : CODEC_TYPE
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
    OPENJP2.opj_set_info_handler.argtypes = [CODEC_TYPE,
                                             ctypes.c_void_p,
                                             ctypes.c_void_p]
    OPENJP2.opj_set_info_handler.restype = check_error
    OPENJP2.opj_set_info_handler(codec, handler, data)


def set_warning_handler(codec, handler, data=None):
    """Wraps openjp2 library function opj_set_warning_handler.

    Set the warning handler use by openjpeg.

    Parameters
    ----------
    codec : CODEC_TYPE
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
    OPENJP2.opj_set_warning_handler.argtypes = [CODEC_TYPE,
                                                ctypes.c_void_p,
                                                ctypes.c_void_p]
    OPENJP2.opj_set_warning_handler.restype = check_error

    OPENJP2.opj_set_warning_handler(codec, handler, data)


def setup_decoder(codec, dparams):
    """Wraps openjp2 library function opj_setup_decoder.

    Setup the decoder with decompression parameters.

    Parameters
    ----------
    codec:  CODEC_TYPE
        Codec initialized by create_compress function.
    dparams:  DecompressionParametersType
        Decompression parameters.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_setup_decoder fails.
    """
    ARGTYPES = [CODEC_TYPE, ctypes.POINTER(DecompressionParametersType)]
    OPENJP2.opj_setup_decoder.argtypes = ARGTYPES
    OPENJP2.opj_setup_decoder.restype = check_error

    OPENJP2.opj_setup_decoder(codec, ctypes.byref(dparams))


def setup_encoder(codec, cparams, image):
    """Wraps openjp2 library function opj_setup_encoder.

    Setup the encoder parameters using the current image and using user
    parameters.

    Parameters
    ----------
    codec : CODEC_TYPE
        codec initialized by create_compress function
    cparams : CompressionParametersType
        compression parameters
    image : ImageType
        input-filled image

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_setup_encoder fails.
    """
    ARGTYPES = [CODEC_TYPE,
                ctypes.POINTER(CompressionParametersType),
                ctypes.POINTER(ImageType)]
    OPENJP2.opj_setup_encoder.argtypes = ARGTYPES
    OPENJP2.opj_setup_encoder.restype = check_error
    OPENJP2.opj_setup_encoder(codec, ctypes.byref(cparams), image)


def start_compress(codec, image, stream):
    """Wraps openjp2 library function opj_start_compress.

    Start to compress the current image.

    Parameters
    ----------
    codec : CODEC_TYPE
        Compressor handle.
    image : pointer to ImageType
        Input filled image.
    stream : STREAM_TYPE_P
        Input stream.

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_start_compress fails.
    """
    OPENJP2.opj_start_compress.argtypes = [CODEC_TYPE,
                                           ctypes.POINTER(ImageType),
                                           STREAM_TYPE_P]
    OPENJP2.opj_start_compress.restype = check_error

    OPENJP2.opj_start_compress(codec, image, stream)


def _stream_create_default_file_stream_2p0(fptr, isa_read_stream):
    """Wraps openjp2 library function opj_stream_create_default_vile_stream.

    Sets the stream to be a file stream.  This is valid only for version 2.0.0
    of OpenJPEG.

    Parameters
    ----------
    fptr : ctypes.c_void_p
        Corresponds to C file pointer.  Must be obtained from libc.fopen.
    isa_read_stream:  bool
        True (read) or False (write)

    Returns
    -------
    stream : stream_t
        An OpenJPEG file stream.
    """
    ARGTYPES = [ctypes.c_void_p, ctypes.c_int32]
    OPENJP2.opj_stream_create_default_file_stream.argtypes = ARGTYPES
    OPENJP2.opj_stream_create_default_file_stream.restype = STREAM_TYPE_P
    read_stream = 1 if isa_read_stream else 0
    stream = OPENJP2.opj_stream_create_default_file_stream(fptr, read_stream)
    return stream


def _stream_create_default_file_stream_2p1(fname, isa_read_stream):
    """Wraps openjp2 library function opj_stream_create_default_vile_stream.

    Sets the stream to be a file stream.  This function is only valid for the
    2.1 version of the openjp2 library.

    Parameters
    ----------
    fname : str
        Specifies a file.
    isa_read_stream:  bool
        True (read) or False (write)

    Returns
    -------
    stream : stream_t
        An OpenJPEG file stream.
    """
    ARGTYPES = [ctypes.c_char_p, ctypes.c_int32]
    OPENJP2.opj_stream_create_default_file_stream.argtypes = ARGTYPES
    OPENJP2.opj_stream_create_default_file_stream.restype = STREAM_TYPE_P
    read_stream = 1 if isa_read_stream else 0
    file_argument = ctypes.c_char_p(fname.encode())
    stream = OPENJP2.opj_stream_create_default_file_stream(file_argument,
                                                           read_stream)
    return stream

if re.match(r'''2.0''', version()):
    stream_create_default_file_stream = _stream_create_default_file_stream_2p0
else:
    stream_create_default_file_stream = _stream_create_default_file_stream_2p1


def stream_destroy(stream):
    """Wraps openjp2 library function opj_stream_destroy.

    Destroys the stream created by create_stream.

    Parameters
    ----------
    stream : STREAM_TYPE_P
        The file stream.
    """
    OPENJP2.opj_stream_destroy.argtypes = [STREAM_TYPE_P]
    OPENJP2.opj_stream_destroy.restype = ctypes.c_void_p
    OPENJP2.opj_stream_destroy(stream)


def write_tile(codec, tile_index, data, data_size, stream):
    """Wraps openjp2 library function opj_write_tile.

    Write a tile into an image.

    Parameters
    ----------
    codec : CODEC_TYPE
        The jpeg2000 codec
    tile_index : int
        The index of the tile to write, zero-indexing assumed
    data : array
        Image data arranged in usual C-order
    data_size : int
        Size of a tile in bytes
    stream : STREAM_TYPE_P
        The stream to write data to

    Raises
    ------
    RuntimeError
        If the OpenJPEG library routine opj_write_tile fails.
    """
    OPENJP2.opj_write_tile.argtypes = [CODEC_TYPE,
                                       ctypes.c_uint32,
                                       ctypes.POINTER(ctypes.c_uint8),
                                       ctypes.c_uint32,
                                       STREAM_TYPE_P]
    OPENJP2.opj_write_tile.restype = check_error

    datap = data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    OPENJP2.opj_write_tile(codec,
                           ctypes.c_uint32(int(tile_index)),
                           datap,
                           ctypes.c_uint32(int(data_size)),
                           stream)


def set_error_message(msg):
    """The openjpeg error handler has recorded an error message."""
    ERROR_MSG_LST.append(msg)
