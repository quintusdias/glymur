"""Core definitions to be shared amongst the modules."""

# standard library imports
from contextlib import ExitStack
import ctypes
import warnings

# 3rd party library imports
import numpy as np

# local imports
from . import get_option, version
from .lib import openjp2 as opj2


# Progression order
LRCP = 0
RLCP = 1
RPCL = 2
PCRL = 3
CPRL = 4

STD = 0
CINEMA2K = 3
CINEMA4K = 4

RSIZ = {"STD": STD, "CINEMA2K": CINEMA2K, "CINEMA4K": CINEMA4K}

OFF = 0
CINEMA2K_24 = 1
CINEMA2K_48 = 2
CINEMA4K_24 = 3

OPJ_OFF = 0  # Not Digital Cinema
OPJ_CINEMA2K_24 = 1  # 2K Digital Cinema at 24 fps
OPJ_CINEMA2K_48 = 2  # 2K Digital Cinema at 48 fps
OPJ_CINEMA4K_24 = 3  # 4K Digital Cinema at 24 fps

# no profile, conform to 15444-1
OPJ_PROFILE_NONE = 0x0000
# Profile 0 as described in 15444-1,Table A.45
OPJ_PROFILE_0 = 0x0001
# Profile 1 as described in 15444-1,Table A.45
OPJ_PROFILE_1 = 0x0002
# At least 1 extension defined in 15444-2 (Part-2)
OPJ_PROFILE_PART2 = 0x8000
# 2K cinema profile defined in 15444-1 AMD1
OPJ_PROFILE_CINEMA_2K = 0x0003
# 4K cinema profile defined in 15444-1 AMD1
OPJ_PROFILE_CINEMA_4K = 0x0004
# Scalable 2K cinema profile defined in 15444-1 AMD2
OPJ_PROFILE_CINEMA_S2K = 0x0005
# Scalable 4K cinema profile defined in 15444-1 AMD2
OPJ_PROFILE_CINEMA_S4K = 0x0006
# Long term storage cinema profile defined in 15444-1 AMD2
OPJ_PROFILE_CINEMA_LTS = 0x0007
# Single Tile Broadcast profile defined in 15444-1 AMD3
OPJ_PROFILE_BC_SINGLE = 0x0100
# Multi Tile Broadcast profile defined in 15444-1 AMD3
OPJ_PROFILE_BC_MULTI = 0x0200
# Multi Tile Reversible Broadcast profile defined in 15444-1 AMD3
OPJ_PROFILE_BC_MULTI_R = 0x0300
# 2K Single Tile Lossy IMF profile defined in 15444-1 AMD 8
OPJ_PROFILE_IMF_2K = 0x0400
# 4K Single Tile Lossy IMF profile defined in 15444-1 AMD 8
OPJ_PROFILE_IMF_4K = 0x0401
# 8K Single Tile Lossy IMF profile defined in 15444-1 AMD 8
OPJ_PROFILE_IMF_8K = 0x0402
# 2K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8
OPJ_PROFILE_IMF_2K_R = 0x0403
# 4K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8
OPJ_PROFILE_IMF_4K_R = 0x0800
# 8K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8
OPJ_PROFILE_IMF_8K_R = 0x0801

# JPEG 2000 codestream and component size limits in cinema profiles
#
# Maximum codestream length for 24fps
OPJ_CINEMA_24_CS = 1302083
# Maximum codestream length for 48fps
OPJ_CINEMA_48_CS = 651041
# Maximum size per color component for 2K & 4K @ 24fps
OPJ_CINEMA_24_COMP = 1041666
# Maximum size per color component for 2K @ 48fps
OPJ_CINEMA_48_COMP = 520833


PROGRESSION_ORDER = {
    "LRCP": LRCP,
    "RLCP": RLCP,
    "RPCL": RPCL,
    "PCRL": PCRL,
    "CPRL": CPRL,
}

WAVELET_XFORM_9X7_IRREVERSIBLE = 0
WAVELET_XFORM_5X3_REVERSIBLE = 1

ENUMERATED_COLORSPACE = 1
RESTRICTED_ICC_PROFILE = 2
ANY_ICC_PROFILE = 3
VENDOR_COLOR_METHOD = 4

# Registration values for comment markers.
RCME_BINARY = 0  # binary value comments
RCME_ISO_8859_1 = 1  # comments in latin-1 codec

# enumerated colorspaces
CMYK = 12
SRGB = 16
GREYSCALE = 17
YCC = 18
E_SRGB = 20
ROMM_RGB = 21


_COLORSPACE_MAP_DISPLAY = {
    CMYK: "CMYK",
    SRGB: "sRGB",
    GREYSCALE: "greyscale",
    YCC: "YCC",
    E_SRGB: "e-sRGB",
    ROMM_RGB: "ROMM-RGB",
}

# enumerated color channel types
COLOR = 0
OPACITY = 1
PRE_MULTIPLIED_OPACITY = 2
_UNSPECIFIED = 65535


_COLOR_TYPE_MAP_DISPLAY = {
    COLOR: "color",
    OPACITY: "opacity",
    PRE_MULTIPLIED_OPACITY: "pre-multiplied opacity",
    _UNSPECIFIED: "unspecified",
}

# color channel definitions.
RED = 1
GREEN = 2
BLUE = 3
GREY = 1
WHOLE_IMAGE = 0

# enumerated color channel associations
_COLORSPACE = {
    SRGB: {"R": 1, "G": 2, "B": 3},
    GREYSCALE: {"Y": 1},
    YCC: {"Y": 1, "Cb": 2, "Cr": 3},
    E_SRGB: {"R": 1, "G": 2, "B": 3},
    ROMM_RGB: {"R": 1, "G": 2, "B": 3},
}


class InvalidJp2kWarning(UserWarning):
    """Issue this warning in case the file is technically invalid but we can
    still read the image.
    """

    pass


class InvalidJp2kError(RuntimeError):
    """Raise this exception in case we cannot parse a valid JP2 file."""

    pass


class _OpenJPEG(object):

    def __init__(self, filename, verbose):

        self.filename = filename

        self._codec_format = None
        self._decoded_components = None
        self._ignore_pclr_cmap_cdef = False
        self._layer = 0
        self._verbose = verbose

        self._tilesize = None

    @property
    def cbsize(self):
        return self._cbsize

    @cbsize.setter
    def cbsize(self, cbsize):
        self._cbsize = cbsize

    @property
    def cinema2k(self):
        return self._cinema2k

    @cinema2k.setter
    def cinema2k(self, cinema2k):
        self._cinema2k = cinema2k

    @property
    def cinema4k(self):
        return self._cinema4k

    @cinema4k.setter
    def cinema4k(self, cinema4k):
        self._cinema4k = cinema4k

    @property
    def colorspace(self):
        return self._colorspace

    @colorspace.setter
    def colorspace(self, colorspace):
        self._colorspace = colorspace

    @property
    def cratios(self):
        return self._cratios

    @cratios.setter
    def cratios(self, cratios):
        self._cratios = cratios

    @property
    def codec_format(self):
        return self._codec_format

    @codec_format.setter
    def codec_format(self, codec_format):
        self._codec_format = codec_format

    @property
    def decoded_components(self):
        return self._decoded_components

    @decoded_components.setter
    def decoded_components(self, components):
        self._decoded_components = components

    @property
    def eph(self):
        return self._eph

    @eph.setter
    def eph(self, eph):
        self._eph = eph

    @property
    def grid_offset(self):
        return self._grid_offset

    @grid_offset.setter
    def grid_offset(self, grid_offset):
        self._grid_offset = grid_offset

    @property
    def ignore_pclr_cmap_cdef(self):
        return self._ignore_pclr_cmap_cdef

    @ignore_pclr_cmap_cdef.setter
    def ignore_pclr_cmap_cdef(self, ignore_pclr_cmap_cdef):
        self._ignore_pclr_cmap_cdef = ignore_pclr_cmap_cdef

    @property
    def irreversible(self):
        return self._irreversible

    @irreversible.setter
    def irreversible(self, irreversible):
        self._irreversible = irreversible

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer):
        self._layer = layer

    @property
    def mct(self):
        return self._mct

    @mct.setter
    def mct(self, mct):
        self._mct = mct

    @property
    def modesw(self):
        return self._modesw

    @modesw.setter
    def modesw(self, modesw):
        self._modesw = modesw

    @property
    def numres(self):
        return self._numres

    @numres.setter
    def numres(self, numres):
        self._numres = numres

    @property
    def plt(self):
        return self._plt

    @plt.setter
    def plt(self, plt):
        self._plt = plt

    @property
    def prog(self):
        return self._prog

    @prog.setter
    def prog(self, prog):
        self._prog = prog

    @property
    def psizes(self):
        return self._psizes

    @psizes.setter
    def psizes(self, psizes):
        self._psizes = psizes

    @property
    def psnr(self):
        return self._psnr

    @psnr.setter
    def psnr(self, psnr):
        self._psnr = psnr

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def sop(self):
        return self._sop

    @sop.setter
    def sop(self, sop):
        self._sop = sop

    @property
    def subsam(self):
        return self._subsam

    @subsam.setter
    def subsam(self, subsam):
        self._subsam = subsam

    @property
    def tilesize(self):
        return self._tilesize

    @tilesize.setter
    def tilesize(self, tilesize):
        self._tilesize = tilesize

    @property
    def tlm(self):
        return self._tlm

    @tlm.setter
    def tlm(self, tlm):
        self._tlm = tlm

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    def read(self):
        """Read a JPEG 2000 image using libopenjp2.

        Returns
        -------
        ndarray or lst
            Either the image as an ndarray or a list of ndarrays, each item
            corresponding to one band.
        """
        with ExitStack() as stack:
            stream = opj2.stream_create_default_file_stream(self.filename, True)  # noqa : E501
            stack.callback(opj2.stream_destroy, stream)
            codec = opj2.create_decompress(self.codec_format)
            stack.callback(opj2.destroy_codec, codec)

            opj2.set_error_handler(codec, opj2._ERROR_CALLBACK)
            opj2.set_warning_handler(codec, opj2._WARNING_CALLBACK)

            if self._verbose:
                opj2.set_info_handler(codec, opj2._INFO_CALLBACK)
            else:
                opj2.set_info_handler(codec, None)

            opj2.setup_decoder(codec, self._dparams)
            # if version.openjpeg_version >= "2.2.0":
            if version.openjpeg_version >= "2.2.0":
                opj2.codec_set_threads(codec, get_option("lib.num_threads"))

            raw_image = opj2.read_header(stream, codec)
            stack.callback(opj2.image_destroy, raw_image)

            if self._decoded_components is not None:
                opj2.set_decoded_components(codec, self._decoded_components)

            if self._dparams.nb_tile_to_decode:
                opj2.get_decoded_tile(
                    codec, stream, raw_image, self._dparams.tile_index
                )
            else:
                opj2.set_decode_area(
                    codec,
                    raw_image,
                    self._dparams.DA_x0,
                    self._dparams.DA_y0,
                    self._dparams.DA_x1,
                    self._dparams.DA_y1,
                )
                opj2.decode(codec, stream, raw_image)

            opj2.end_decompress(codec, stream)

            image = self.extract_image(raw_image)

        return image

    def populate_dparams(self, rlevel, tile=None, area=None):
        """Populate decompression structure with appropriate input parameters.

        Parameters
        ----------
        rlevel : int
            Factor by which to rlevel output resolution.
        area : tuple
            Specifies decoding image area,
            (first_row, first_col, last_row, last_col)
        tile : int
            Number of tile to decode.
        """
        dparam = opj2.set_default_decoder_parameters()

        infile = self.filename.encode()
        nelts = opj2.PATH_LEN - len(infile)
        infile += b"0" * nelts
        dparam.infile = infile

        # Return raw codestream components instead of "interpolating" the
        # colormap?
        dparam.flags |= 1 if self.ignore_pclr_cmap_cdef else 0

        dparam.decod_format = self.codec_format
        dparam.cp_layer = self.layer

        dparam.cp_reduce = rlevel

        if area is not None:
            if area[0] < 0 or area[1] < 0 or area[2] <= 0 or area[3] <= 0:
                msg = (
                    f"The upper left corner coordinates must be nonnegative "
                    f"and the lower right corner coordinates must be positive."
                    f"  The specified upper left and lower right coordinates "
                    f"are ({area[0]}, {area[1]}) and ({area[2]}, {area[3]})."
                )
                raise ValueError(msg)
            dparam.DA_y0 = area[0]
            dparam.DA_x0 = area[1]
            dparam.DA_y1 = area[2]
            dparam.DA_x1 = area[3]

        if tile is not None:
            dparam.tile_index = tile
            dparam.nb_tile_to_decode = 1

        self._dparams = dparam

    def read_bands(
        self, rlevel, layer, area, tile, verbose, ignore_pclr_cmap_cdef
    ):
        if version.openjpeg_version < "2.4.0":
            msg = (
                "The minimum supported version of OpenJPEG is 2.4.0.  "
                f"Your version is {version.openjpeg_version}."
            )
            raise RuntimeError(msg)

        self.ignore_pclr_cmap_cdef = ignore_pclr_cmap_cdef
        self.layer = layer
        self.populate_dparams(rlevel, tile=tile, area=area)
        lst = self.read()
        return lst

    def extract_image(self, raw_image):
        """Extract unequally-sized image bands.

        Parameters
        ----------
        raw_image : reference to openjpeg ImageType instance
            The image structure initialized with image characteristics.

        Returns
        -------
        list or ndarray
            If the JPEG 2000 image has unequally-sized components, they are
            extracted into a list, otherwise a numpy array.

        """
        ncomps = raw_image.contents.numcomps

        # Make a pass thru the image, see if any of the band datatypes or
        # dimensions differ.
        dtypes, nrows, ncols = [], [], []
        for k in range(raw_image.contents.numcomps):
            component = raw_image.contents.comps[k]
            dtypes.append(self.component2dtype(component))
            nrows.append(component.h)
            ncols.append(component.w)
        is_cube = all(
            r == nrows[0] and c == ncols[0] and d == dtypes[0]
            for r, c, d in zip(nrows, ncols, dtypes)
        )

        if is_cube:
            image = np.zeros((nrows[0], ncols[0], ncomps), dtypes[0])
        else:
            image = []

        for k in range(raw_image.contents.numcomps):
            component = raw_image.contents.comps[k]

            self.validate_nonzero_image_size(nrows[k], ncols[k], k)

            addr = ctypes.addressof(component.data.contents)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                band_i32 = np.ctypeslib.as_array(
                    (ctypes.c_int32 * nrows[k] * ncols[k]).from_address(addr)
                )
                band = np.reshape(
                    band_i32.astype(dtypes[k]), (nrows[k], ncols[k])
                )

                if is_cube:
                    image[:, :, k] = band
                else:
                    image.append(band)

        if is_cube and image.shape[2] == 1:
            # The third dimension has just a single layer.  Make the image
            # data 2D instead of 3D.
            image.shape = image.shape[0:2]

        return image

    def component2dtype(self, component):
        """Determine the appropriate numpy datatype for an OpenJPEG component.

        Parameters
        ----------
        component : ctypes pointer to ImageCompType (image_comp_t)
            single image component structure.

        Returns
        -------
        builtins.type
            numpy datatype to be used to construct an image array
        """
        if component.prec > 16:
            msg = f"Unhandled precision: {component.prec} bits."
            raise ValueError(msg)

        if component.sgnd:
            if component.prec <= 8:
                dtype = np.int8
            else:
                dtype = np.int16
        else:
            if component.prec <= 8:
                dtype = np.uint8
            else:
                dtype = np.uint16

        return dtype

    def validate_nonzero_image_size(self, nrows, ncols, component_index):
        """The image cannot have area of zero."""
        if nrows == 0 or ncols == 0:
            # Letting this situation continue would segfault openjpeg.
            msg = (
                f"Component {component_index} has dimensions "
                f"{nrows} x {ncols}"
            )
            raise InvalidJp2kError(msg)

    def populate_cparams(self, img_array):
        """Directs processing of write method arguments.

        Parameters
        ----------
        img_array : ndarray
            Image data to be written to file.
        kwargs : dictionary
            Non-image keyword inputs provided to write method.
        """
        cparams = opj2.set_default_encoder_parameters()

        outfile = self.filename.encode()
        num_pad_bytes = opj2.PATH_LEN - len(outfile)
        outfile += b"0" * num_pad_bytes
        cparams.outfile = outfile

        cparams.codec_fmt = self.codec_format

        cparams.irreversible = 1 if self.irreversible else 0

        if self.cinema2k:
            # cinema2k is an integer, so this test is "truthy"
            self.cparams = cparams
            self.set_cinema_params("cinema2k", self.cinema2k)

        if self.cinema4k:
            self.cparams = cparams
            self.set_cinema_params("cinema4k", self.cinema4k)

        if self.cbsize is not None:
            cparams.cblockw_init = self.cbsize[1]
            cparams.cblockh_init = self.cbsize[0]

        if self.cratios is not None:
            cparams.tcp_numlayers = len(self.cratios)
            for j, cratio in enumerate(self.cratios):
                cparams.tcp_rates[j] = cratio
            cparams.cp_disto_alloc = 1

        cparams.csty |= 0x02 if self.sop else 0
        cparams.csty |= 0x04 if self.eph else 0

        if self.grid_offset is not None:
            cparams.image_offset_x0 = self.grid_offset[1]
            cparams.image_offset_y0 = self.grid_offset[0]

        if self.modesw is not None:
            # The None check is for backwards compatibility.
            for shift in range(6):
                power_of_two = 1 << shift
                if self.modesw & power_of_two:
                    cparams.mode |= power_of_two

        cparams.numresolution = self.numres

        if self.prog is not None:
            cparams.prog_order = PROGRESSION_ORDER[self.prog.upper()]

        if self.psnr is not None:
            cparams.tcp_numlayers = len(self.psnr)
            for j, snr_layer in enumerate(self.psnr):
                cparams.tcp_distoratio[j] = snr_layer
            cparams.cp_fixed_quality = 1

        if self.psizes is not None:
            for j, (prch, prcw) in enumerate(self.psizes):
                cparams.prcw_init[j] = prcw
                cparams.prch_init[j] = prch
            cparams.csty |= 0x01
            cparams.res_spec = len(self.psizes)

        if self.subsam is not None:
            cparams.subsampling_dy = self.subsam[0]
            cparams.subsampling_dx = self.subsam[1]

        if self.tilesize is not None:
            cparams.cp_tdx = self.tilesize[1]
            cparams.cp_tdy = self.tilesize[0]
            cparams.tile_size_on = opj2.TRUE

        if self.mct is None:

            # If the multi component transform was not specified, we infer
            # that it should be used if the color space is RGB.
            cparams.tcp_mct = 1 if self.colorspace == opj2.CLRSPC_SRGB else 0

        elif self.mct and self.colorspace == opj2.CLRSPC_GRAY:

            # the MCT was requested, but the colorspace is gray
            # i.e. 1 component.  NOT ON MY WATCH!
            msg = (
                "You cannot specify usage of the multi component transform "
                "if the colorspace is gray."
            )
            raise InvalidJp2kError(msg)

        else:

            # The MCT was either not specified
            # or it was specified AND the colorspace is going to be RGB.
            # In either case, we can use the MCT as requested.
            cparams.tcp_mct = 1 if self._mct else 0

        # Set defaults to lossless to begin.
        if cparams.tcp_numlayers == 0:
            cparams.tcp_rates[0] = 0
            cparams.tcp_numlayers += 1
            cparams.cp_disto_alloc = 1

        self._validate_compression_params(img_array, cparams)

        self.cparams = cparams

    def _validate_compression_params(self, img_array, cparams):
        """Check that the compression parameters are valid.

        Parameters
        ----------
        img_array : ndarray
            Image data to be written to file.
        cparams : CompressionParametersType(ctypes.Structure)
            Corresponds to cparameters_t type in openjp2 headers.
        """
        self._validate_codeblock_size(cparams)
        self._validate_precinct_size(cparams)
        self._validate_image_rank(img_array)
        self._validate_image_datatype(img_array)

    def _validate_codeblock_size(self, cparams):
        """Code block dimensions must satisfy certain restrictions.

        They must both be a power of 2 and the total area defined by the width
        and height cannot be either too great or too small for the codec.
        """
        if cparams.cblockw_init != 0 and cparams.cblockh_init != 0:
            # These fields ARE zero if uninitialized.
            width = cparams.cblockw_init
            height = cparams.cblockh_init
            if height * width > 4096 or height < 4 or width < 4:
                msg = (
                    f"The code block area is specified as {height} x {width} "
                    f"= {height * width} square pixels.  Code block area "
                    f"cannot exceed 4096 square pixels.  Code block height "
                    f"and width dimensions must be larger than 4 pixels."
                )
                raise InvalidJp2kError(msg)
            if np.log2(height) != np.floor(np.log2(height)) or np.log2(
                width
            ) != np.floor(np.log2(width)):
                msg = (
                    f"Bad code block size ({height} x {width}).  "
                    f"The dimensions must be powers of 2."
                )
                raise InvalidJp2kError(msg)

    def _validate_precinct_size(self, cparams):
        """Precinct dimensions must satisfy certain restrictions if specified.

        They must both be a power of 2 and must both be at least twice the
        size of their codeblock size counterparts.
        """
        code_block_specified = False
        if cparams.cblockw_init != 0 and cparams.cblockh_init != 0:
            code_block_specified = True

        if cparams.res_spec != 0:
            # precinct size was not specified if this field is zero.
            for j in range(cparams.res_spec):
                prch = cparams.prch_init[j]
                prcw = cparams.prcw_init[j]
                if j == 0 and code_block_specified:
                    height, width = cparams.cblockh_init, cparams.cblockw_init
                    if prch < height * 2 or prcw < width * 2:
                        msg = (
                            f"The highest resolution precinct size "
                            f"({prch} x {prcw}) must be at least twice that "
                            f"of the code block size ({height} x {width})."
                        )
                        raise InvalidJp2kError(msg)
                if (
                    np.log2(prch) != np.floor(np.log2(prch))
                    or np.log2(prcw) != np.floor(np.log2(prcw))
                ):
                    msg = (
                        f"Bad precinct size ({prch} x {prcw}).  Precinct "
                        f"dimensions must be powers of 2."
                    )
                    raise InvalidJp2kError(msg)

    def _validate_image_rank(self, img_array):
        """Images must be either 2D or 3D."""
        if img_array.ndim == 1 or img_array.ndim > 3:
            msg = f"{img_array.ndim}D imagery is not allowed."
            raise InvalidJp2kError(msg)

    def _validate_image_datatype(self, img_array):
        """Only uint8 and uint16 images are currently supported."""
        if img_array.dtype != np.uint8 and img_array.dtype != np.uint16:
            msg = (
                "Only uint8 and uint16 datatypes are currently supported when "
                "writing."
            )
            raise InvalidJp2kError(msg)

    def write(self, img_array):
        """Write image data to a JP2/JPX/J2k file.  Intended usage of the
        various parameters follows that of OpenJPEG's opj_compress utility.

        This method can only be used to create JPEG 2000 images that can fit
        in memory.
        """
        self.determine_colorspace()
        self.populate_cparams(img_array)

        if img_array.ndim == 2:
            # Force the image to be 3D.  This makes it easier to copy the
            # image data later on.
            numrows, numcols = img_array.shape
            # img_array.shape = (numrows, numcols, 1)
            img_array = img_array.reshape(numrows, numcols, 1)

        self.populate_comptparms(img_array)

        with ExitStack() as stack:
            image = opj2.image_create(self.comptparms, self.colorspace)
            stack.callback(opj2.image_destroy, image)

            self.populate_image_struct(image, img_array)

            codec = opj2.create_compress(self.cparams.codec_fmt)
            stack.callback(opj2.destroy_codec, codec)

            if self._verbose:
                info_handler = opj2._INFO_CALLBACK
            else:
                info_handler = None

            opj2.set_info_handler(codec, info_handler)
            opj2.set_warning_handler(codec, opj2._WARNING_CALLBACK)
            opj2.set_error_handler(codec, opj2._ERROR_CALLBACK)

            opj2.setup_encoder(codec, self.cparams, image)

            if self.plt:
                opj2.encoder_set_extra_options(codec, plt=self.plt)

            if self.tlm:
                opj2.encoder_set_extra_options(codec, tlm=self.tlm)

            strm = opj2.stream_create_default_file_stream(self.filename, False)

            num_threads = get_option("lib.num_threads")
            if version.openjpeg_version >= "2.4.0":
                opj2.codec_set_threads(codec, num_threads)
            elif num_threads > 1:
                msg = (
                    f"Threaded encoding is not supported in library versions "
                    f"prior to 2.4.0.  Your version is "
                    f"{version.openjpeg_version}."
                )
                warnings.warn(msg, UserWarning)

            stack.callback(opj2.stream_destroy, strm)

            opj2.start_compress(codec, image, strm)
            opj2.encode(codec, strm)
            opj2.end_compress(codec, strm)

    def populate_image_struct(
        self, image, imgdata, tile_x_factor=1, tile_y_factor=1
    ):
        """Populates image struct needed for compression.

        Parameters
        ----------
        image : ImageType(ctypes.Structure)
            Corresponds to image_t type in openjp2 headers.
        imgdata : ndarray
            Image data to be written to file.
        tile_x_factor, tile_y_factor: int
            Used only when writing tile-by-tile.  In this case, the image data
            that we have is only the size of a single tile.
        """

        if len(self.shape) < 3:
            (numrows, numcols), num_comps = self.shape, 1
        else:
            numrows, numcols, num_comps = self.shape

        for k in range(num_comps):
            self.validate_nonzero_image_size(numrows, numcols, k)

        # set image offset and reference grid
        image.contents.x0 = self.cparams.image_offset_x0
        image.contents.y0 = self.cparams.image_offset_y0
        image.contents.x1 = (
            image.contents.x0
            + (numcols - 1) * self.cparams.subsampling_dx * tile_x_factor
            + 1
        )
        image.contents.y1 = (
            image.contents.y0
            + (numrows - 1) * self.cparams.subsampling_dy * tile_y_factor
            + 1
        )

        if tile_x_factor != 1 or tile_y_factor != 1:
            # don't stage the data if writing tiles
            return image

        # Stage the image data to the openjpeg data structure.
        for k in range(0, num_comps):
            if self.cparams.rsiz in (
                OPJ_PROFILE_CINEMA_2K, OPJ_PROFILE_CINEMA_4K,
            ):
                image.contents.comps[k].prec = 12
                image.contents.comps[k].bpp = 12

            layer = np.ascontiguousarray(imgdata[:, :, k], dtype=np.int32)
            dest = image.contents.comps[k].data
            src = layer.ctypes.data
            ctypes.memmove(dest, src, layer.nbytes)

        return image

    def determine_colorspace(self):
        """Determine the colorspace from the supplied inputs."""

        if self.colorspace is None:
            # Must infer the colorspace from the image dimensions.
            if len(self.shape) < 3:
                # A single channel image is grayscale.
                self.colorspace = opj2.CLRSPC_GRAY
            elif self.shape[2] == 1 or self.shape[2] == 2:
                # A single channel image or an image with two channels is going
                # to be greyscale.
                self.colorspace = opj2.CLRSPC_GRAY
            else:
                # Anything else must be RGB, right?
                self.colorspace = opj2.CLRSPC_SRGB
        else:
            if self.colorspace.lower() not in ("rgb", "grey", "gray"):
                msg = f'Invalid colorspace "{self.colorspace}".'
                raise InvalidJp2kError(msg)
            elif self.colorspace.lower() == "rgb" and self.shape[2] < 3:
                msg = "RGB colorspace requires at least 3 components."
                raise InvalidJp2kError(msg)

            # Turn the colorspace from a string to the enumerated value that
            # the library expects.
            COLORSPACE_MAP = {
                "rgb": opj2.CLRSPC_SRGB,
                "gray": opj2.CLRSPC_GRAY,
                "grey": opj2.CLRSPC_GRAY,
                "ycc": opj2.CLRSPC_YCC,
            }

            self.colorspace = COLORSPACE_MAP[self.colorspace.lower()]

    def populate_comptparms(self, img_array):
        """Instantiate and populate comptparms structure.

        This structure defines the image components.

        Parameters
        ----------
        img_array : ndarray
            Image data to be written to file.
        """
        # Only two precisions are possible.
        if img_array.dtype == np.uint8:
            comp_prec = 8
        else:
            comp_prec = 16

        if len(self.shape) < 3:
            (numrows, numcols), num_comps = self.shape, 1
        else:
            numrows, numcols, num_comps = self.shape

        comptparms = (opj2.ImageComptParmType * num_comps)()
        for j in range(num_comps):
            comptparms[j].dx = self.cparams.subsampling_dx
            comptparms[j].dy = self.cparams.subsampling_dy
            comptparms[j].w = numcols
            comptparms[j].h = numrows
            comptparms[j].x0 = self.cparams.image_offset_x0
            comptparms[j].y0 = self.cparams.image_offset_y0
            comptparms[j].prec = comp_prec
            comptparms[j].bpp = comp_prec
            comptparms[j].sgnd = 0

        self.comptparms = comptparms

    def set_cinema_params(self, cinema_mode, fps):
        """Populate compression parameters structure for cinema2K.

        Parameters
        ----------
        params : ctypes struct
            Corresponds to compression parameters structure used by the
            library.
        cinema_mode : {'cinema2k', 'cinema4k}
            Use either Cinema2K or Cinema4K profile.
        fps : {24, 48}
            Frames per second.
        """
        # Cinema modes imply MCT.
        self.cparams.tcp_mct = 1

        if cinema_mode == "cinema2k":
            if fps not in [24, 48]:
                msg = "Cinema2K frame rate must be either 24 or 48."
                raise ValueError(msg)

            if fps == 24:
                self.cparams.rsiz = OPJ_PROFILE_CINEMA_2K
                self.cparams.max_comp_size = OPJ_CINEMA_24_COMP
                self.cparams.max_cs_size = OPJ_CINEMA_24_CS
            else:
                self.cparams.rsiz = OPJ_PROFILE_CINEMA_2K
                self.cparams.max_comp_size = OPJ_CINEMA_48_COMP
                self.cparams.max_cs_size = OPJ_CINEMA_48_CS

        else:
            # cinema4k
            self.cparams.rsiz = OPJ_PROFILE_CINEMA_4K
