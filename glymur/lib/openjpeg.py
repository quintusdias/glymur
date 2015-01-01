"""Wraps library calls to openjpeg.
"""

import ctypes
import sys

from .config import glymur_config

_, OPENJPEG = glymur_config()

# Maximum number of tile parts expected by JPWL: increase at your will
JPWL_MAX_NO_TILESPECS = 16

J2K_MAXRLVLS = 33  # Number of maximum resolution level authorized
PATH_LEN = 4096    # maximum allowed size for filenames


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


STREAM_READ = 0x0001  # The stream was opened for reading.
STREAM_WRITE = 0x0002  # The stream was opened for writing.


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


class CompressionInfoType(CommonStructType):
    """Common fields between JPEG-2000 compression and decompression contexts.
    This is for compression contexts.  Corresponds to common_struct_t.
    """
    pass


class PocType(ctypes.Structure):
    """Progression order changes."""
    _fields_ = [("resno", ctypes.c_int),
                # Resolution num start, Component num start, given by POC
                ("compno0", ctypes.c_int),

                # Layer num end,Resolution num end, Component num end, given
                # by POC
                ("layno1", ctypes.c_int),
                ("resno1", ctypes.c_int),
                ("compno1", ctypes.c_int),

                # Layer num start,Precinct num start, Precinct num end
                ("layno0", ctypes.c_int),
                ("precno0", ctypes.c_int),
                ("precno1", ctypes.c_int),

                # Progression order enum
                # OPJ_PROG_ORDER prg1,prg;
                ("prg1", ctypes.c_int),
                ("prg", ctypes.c_int),

                # Progression order string
                # char progorder[5];
                ("progorder",            ctypes.c_char * 5),

                # Tile number
                # int tile;
                ("tile", ctypes.c_int),

                ("tx0", ctypes.c_int),
                ("tx1", ctypes.c_int),
                ("ty0", ctypes.c_int),
                ("ty1", ctypes.c_int),
                ("layS", ctypes.c_int),
                ("resS", ctypes.c_int),
                ("compS", ctypes.c_int),
                ("prcS", ctypes.c_int),
                ("layE", ctypes.c_int),
                ("resE", ctypes.c_int),
                ("compE", ctypes.c_int),
                ("prcE", ctypes.c_int),
                ("txS", ctypes.c_int),
                ("txE", ctypes.c_int),
                ("tyS", ctypes.c_int),
                ("tyE", ctypes.c_int),
                ("dx", ctypes.c_int),
                ("dy", ctypes.c_int),
                ("lay_t", ctypes.c_int),
                ("res_t", ctypes.c_int),
                ("comp_t", ctypes.c_int),
                ("prc_t", ctypes.c_int),
                ("tx0_t", ctypes.c_int),
                ("ty0_t", ctypes.c_int)]


class CompressionParametersType(ctypes.Structure):
    """Compression parameters.

    Corresponds to cparameters_t type in openjp2 headers.
    """
    _fields_ = [
        # size of tile:
        #     tile_size_on = false (not in argument) or
        #                  = true (in argument)
        ("tile_size_on",     ctypes.c_int),

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
        ("jpwl_epc_on",     ctypes.c_int),

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
        ("cp_cinema",             ctypes.c_int),

        # Maximum rate for each component.
        # If == 0, component size limitation is not considered
        ("max_comp_size",         ctypes.c_int),

        # Profile name
        ("cp_rsiz",               ctypes.c_int),

        # Tile part generation
        ("tp_on",                 ctypes.c_uint8),

        # Flag for Tile part generation
        ("tp_flag",               ctypes.c_uint8),

        # MCT (multiple component transform)
        ("tcp_mct",               ctypes.c_uint8),

        # Enable JPIP indexing
        ("jpip_on",               ctypes.c_int)]


class DecompressionInfoType(ctypes.Structure):
    """This is for decompression contexts.

    Corresponds to dinfo_t type in openjpeg headers.
    """
    pass


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


class ImageComptParmType(ctypes.Structure):
    """Component parameters structure used by the opj_image_create function.
    """
    _fields_ = [("dx", ctypes.c_int),
                # XRsiz: horizontal separation of a sample of ith component
                # with respect to the reference grid

                # YRsiz: vertical separation of a sample of ith component with
                # respect to the reference grid */
                ("dy", ctypes.c_int),

                # data width, height
                ("w", ctypes.c_int),
                ("h", ctypes.c_int),

                # x component offset compared to the whole image
                # y component offset compared to the whole image
                ("x0", ctypes.c_int),
                ("y0", ctypes.c_int),

                # precision
                ('prec', ctypes.c_int),

                # image depth in bits
                ('bpp', ctypes.c_int),

                # signed (1) / unsigned (0)
                ('sgnd', ctypes.c_int)]


class ImageCompType(ctypes.Structure):
    """Defines a single image component. """
    _fields_ = [("dx",        ctypes.c_int),
                ("dy",            ctypes.c_int),
                ("w",             ctypes.c_int),
                ("h",             ctypes.c_int),
                ("x0",            ctypes.c_int),
                ("y0",            ctypes.c_int),
                ("prec",          ctypes.c_int),
                ("bpp",           ctypes.c_int),
                ("sgnd",          ctypes.c_int),
                ("resno_decoded", ctypes.c_int),
                ("factor",        ctypes.c_int),
                ("data",          ctypes.POINTER(ctypes.c_int))]


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


def cio_open(cinfo, src=None):
    """Wrapper for openjpeg library function opj_cio_open."""
    argtypes = [ctypes.POINTER(CommonStructType), ctypes.c_char_p,
                ctypes.c_int]
    OPENJPEG.opj_cio_open.argtypes = argtypes
    OPENJPEG.opj_cio_open.restype = ctypes.POINTER(CioType)

    if src is None:
        length = 0
    else:
        length = len(src)

    cio = OPENJPEG.opj_cio_open(ctypes.cast(cinfo,
                                            ctypes.POINTER(CommonStructType)),
                                src,
                                length)
    return cio


def cio_close(cio):
    """Wraps openjpeg library function cio_close.
    """
    OPENJPEG.opj_cio_close.argtypes = [ctypes.POINTER(CioType)]
    OPENJPEG.opj_cio_close(cio)


def cio_tell(cio):
    """Get position in byte stream."""
    OPENJPEG.cio_tell.argtypes = [ctypes.POINTER(CioType)]
    OPENJPEG.cio_tell.restype = ctypes.c_int
    pos = OPENJPEG.cio_tell(cio)
    return pos


def create_compress(fmt):
    """Wrapper for openjpeg library function opj_create_compress.

    Creates a J2K/JPT/JP2 compression structure.
    """
    OPENJPEG.opj_create_compress.argtypes = [ctypes.c_int]
    OPENJPEG.opj_create_compress.restype = ctypes.POINTER(CompressionInfoType)
    cinfo = OPENJPEG.opj_create_compress(fmt)
    return cinfo


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


def destroy_compress(cinfo):
    """Wrapper for openjpeg library function opj_destroy_compress.

    Release resources for a compressor handle.
    """
    argtypes = [ctypes.POINTER(CompressionInfoType)]
    OPENJPEG.opj_destroy_compress.argtypes = argtypes
    OPENJPEG.opj_destroy_compress(cinfo)


def encode(cinfo, cio, image):
    """Wrapper for openjpeg library function opj_encode.

    Encodes an image into a JPEG-2000 codestream.

    Parameters
    ----------
    cinfo : compression handle

    cio : output buffer stream

    image : image to encode
    """
    argtypes = [ctypes.POINTER(CompressionInfoType),
                ctypes.POINTER(CioType),
                ctypes.POINTER(ImageType)]
    OPENJPEG.opj_encode.argtypes = argtypes
    OPENJPEG.opj_encode.restype = ctypes.c_int
    status = OPENJPEG.opj_encode(cinfo, cio, image)
    return status


def destroy_decompress(dinfo):
    """Wraps openjpeg library function opj_destroy_decompress."""
    argtypes = [ctypes.POINTER(DecompressionInfoType)]
    OPENJPEG.opj_destroy_decompress.argtypes = argtypes
    OPENJPEG.opj_destroy_decompress(dinfo)


def image_create(cmptparms, cspace):
    """Wrapper for openjpeg library function opj_image_create.
    """
    lst = [ctypes.c_int, ctypes.POINTER(ImageComptParmType), ctypes.c_int]
    OPENJPEG.opj_image_create.argtypes = lst
    OPENJPEG.opj_image_create.restype = ctypes.POINTER(ImageType)

    image = OPENJPEG.opj_image_create(len(cmptparms), cmptparms, cspace)
    return(image)


def image_destroy(image):
    """Wraps openjpeg library function opj_image_destroy."""
    OPENJPEG.opj_image_destroy.argtypes = [ctypes.POINTER(ImageType)]
    OPENJPEG.opj_image_destroy(image)


def set_default_encoder_parameters():
    """Wrapper for openjpeg library function opj_set_default_encoder_parameters.
    """
    cparams = CompressionParametersType()
    argtypes = [ctypes.POINTER(CompressionParametersType)]
    OPENJPEG.opj_set_default_encoder_parameters.argtypes = argtypes
    OPENJPEG.opj_set_default_encoder_parameters(ctypes.byref(cparams))
    return cparams


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


def setup_encoder(cinfo, cparameters, image):
    """Wrapper for openjpeg library function opj_setup_decoder."""
    argtypes = [ctypes.POINTER(CompressionInfoType),
                ctypes.POINTER(CompressionParametersType),
                ctypes.POINTER(ImageType)]
    OPENJPEG.opj_setup_encoder.argtypes = argtypes
    OPENJPEG.opj_setup_encoder(cinfo, cparameters, image)


def setup_decoder(dinfo, dparams):
    """Wrapper for openjpeg library function opj_setup_decoder."""
    argtypes = [ctypes.POINTER(DecompressionInfoType),
                ctypes.POINTER(DecompressionParametersType)]
    OPENJPEG.opj_setup_decoder.argtypes = argtypes
    OPENJPEG.opj_setup_decoder(dinfo, dparams)
