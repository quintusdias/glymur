"""This file is part of glymur, a Python interface for accessing JPEG 2000.

http://glymur.readthedocs.org

Copyright 2013 John Evans

License:  MIT
"""

# Standard library imports...
from __future__ import annotations
from collections import Counter
from contextlib import ExitStack
import ctypes
import pathlib
import shutil
import struct
from typing import List, Tuple
from uuid import UUID
import warnings

# Third party library imports
import numpy as np

# Local imports...
import glymur
from . import core, version, get_option
from .jp2kr import Jp2kr
from .jp2box import (
    ColourSpecificationBox,
    ContiguousCodestreamBox,
    FileTypeBox,
    ImageHeaderBox,
    InvalidJp2kError,
    JP2HeaderBox,
    JPEG2000SignatureBox,
)
from .lib import openjp2 as opj2


class Jp2k(Jp2kr):
    """Write JPEG 2000 files (and optionally read them as well).

    Parameters
    ----------
    filename : str or path
        The path to JPEG 2000 file.
    data : np.ndarray, optional
        Image data to be written to file.
    shape : Tuple[int, int, ...], optional
        Size of image data, only required when image_data is not provided.
    capture_resolution : Tuple[int, int], optional
        Capture solution (VRES, HRES).  This appends a capture resolution
        box onto the end of the JP2 file when it is created.
    cbsize : Tuple[int, int], optional
        Code block size (NROWS, NCOLS)
    cinema2k : int, optional
        Frames per second, either 24 or 48.
    cinema4k : bool, optional
        Set to True to specify Cinema4K mode, defaults to false.
    colorspace : {'rgb', 'gray'}, optional
        The image color space.  If not supplied, it will be inferred.
    cratios : Tuple[int, ...], optional
        Compression ratios for successive layers.
    display_resolution : Tuple[int, int], optional
        Display solution (VRES, HRES).  This appends a display resolution
        box onto the end of the JP2 file when it is created.
    eph : bool, optional
        If true, write EPH marker after each header packet.
    grid_offset : Tuple[int, int], optional
        Offset (DY, DX) of the origin of the image in the reference grid.
    irreversible : bool, optional
        If true, use the irreversible DWT 9-7 transform.
    mct : bool, optional
        Usage of the multi component transform to write an image.  If not
        specified, defaults to True if the color space is RGB, false if the
        color space is grayscale.
    modesw : int, optional
        mode switch
            1 = BYPASS(LAZY)
            2 = RESET
            4 = RESTART(TERMALL)
            8 = VSC
            16 = ERTERM(SEGTERM)
            32 = SEGMARK(SEGSYM)
    numres : int, optional
        Number of resolutions, defaults to 6.  This number will be equal to
        the number of thumbnails plus the original image.
    plt : bool, optional
        Generate PLT markers.
    prog : {'LRCP', 'RLCP', 'RPCL', 'PCRL', 'CPRL'}, optional
        Progression order.  If not specified, the chosen progression order
        will be 'CPRL' if either cinema2k or cinema4k is specified,
        otherwise defaulting to 'LRCP'.
    psizes : List[Tuple[int, int]], optional
        Precinct sizes, each precinct size tuple is defined as
        (height, width).
    psnr : Tuple[int, ...] or None
        Different PSNR for successive layers.  If the last layer is desired
        to be lossless, specify 0 for the last value.
    sop : bool, optional
        If true, write SOP marker before each packet.
    subsam : Tuple[int, int], optional
        Subsampling factors (dy, dx).
    tilesize : Tuple[int, int], optional
        Tile size in terms of (numrows, numcols), not (X, Y).
    tlm : bool, optional
        Generate TLM markers.
    verbose : bool, optional
        Print informational messages produced by the OpenJPEG library.

    """

    def __init__(
        self,
        filename: str | pathlib.Path,
        data: np.ndarray | None = None,
        capture_resolution: Tuple[int, int] | None = None,
        cbsize: Tuple[int, int] | None = None,
        cinema2k: int = 0,
        cinema4k: bool = False,
        colorspace: str | None = None,
        cratios: Tuple[int, ...] | None = None,
        display_resolution: Tuple[int, int] | None = None,
        eph: bool = False,
        grid_offset: Tuple[int, int] | None = None,
        irreversible: bool = False,
        mct: bool | None = None,
        modesw: int = 0,
        numres: int = 6,
        plt: bool = False,
        prog: str | None = None,
        psizes: List[Tuple[int, int]] | None = None,
        psnr: Tuple[int, ...] | None = None,
        shape: Tuple[int, int, ...] | None = None,
        sop: bool = False,
        subsam: Tuple[int, int] | None = None,
        tilesize: Tuple[int, int] | None = None,
        tlm: bool = False,
        verbose: bool = False,
    ):
        try:
            super().__init__(filename, verbose=verbose)
        except FileNotFoundError:
            # must assume we are writing
            pass

        # In case of pathlib.Paths...
        self.filename = str(filename)
        self.path = pathlib.Path(self.filename)

        self._capture_resolution = capture_resolution
        self._cbsize = cbsize
        self._cinema2k = cinema2k
        self._cinema4k = cinema4k
        self._colorspace = colorspace
        self._cratios = cratios
        self._display_resolution = display_resolution
        self._eph = eph
        self._grid_offset = grid_offset
        self._irreversible = irreversible
        self._mct = mct
        self._modesw = modesw
        self._numres = numres if numres is not None else 6
        self._plt = plt
        self._prog = prog
        self._psizes = psizes
        self._psnr = psnr
        self._sop = sop
        self._subsam = subsam
        self._tilesize_w = tilesize
        self._tlm = tlm
        self._verbose = verbose

        if shape is not None:
            self._shape = shape
        elif data is not None:
            self._shape = data.shape
        elif not hasattr(self, "shape"):
            # We must be writing via slicing.
            # Must be determined when writing.
            self._shape = None

        if not hasattr(self, "_codec_format"):
            # Only set codec format if the superclass has not done so, i.e.
            # we are writing instead of reading.
            if self.filename[-4:].endswith((".jp2", ".JP2", ".jpx", "JPX")):
                self._codec_format = opj2.CODEC_JP2
            else:
                self._codec_format = opj2.CODEC_J2K

        self._validate_kwargs()

        if data is None:
            # Expecting to write either by tiles or by setitem.  Do not
            # parse just yet, as there is nothing to parse.  We are done for
            # now.
            return

        else:
            # We are writing a JP2/J2K/JPX file where the image is
            # contained in memory.
            self[:] = data

    def __repr__(self):
        msg = f"glymur.Jp2k('{self.path}')"
        return msg

    def finalize(self, force_parse=False):
        """For now, the only remaining tasks are to possibly parse the file
        and to possibly write out a ResolutionBox.  There could be other
        possibilities in the future.

        Parameters
        ----------
        force : bool
            If true, then run finalize operations
        """
        self._parse(force=force_parse)

        if (
            self._capture_resolution is None
            and self._display_resolution is None
        ):
            # ... and we don't have any extra boxes, so we are done
            return

        # So we DO have extra boxes.
        self._insert_resolution_superbox()

    def _insert_resolution_superbox(self):
        """As a close-out task, insert a resolution superbox into the jp2
        header box if we were so instructed.  This requires a wrapping
        operation.
        """
        jp2h = next(filter(lambda x: x.box_id == "jp2h", self.box), None)

        extra_boxes = []
        if self._capture_resolution is not None:
            resc = glymur.jp2box.CaptureResolutionBox(
                self._capture_resolution[0],
                self._capture_resolution[1],
            )
            extra_boxes.append(resc)

        if self._display_resolution is not None:
            resd = glymur.jp2box.DisplayResolutionBox(
                self._display_resolution[0],
                self._display_resolution[1],
            )
            extra_boxes.append(resd)

        rbox = glymur.jp2box.ResolutionBox(extra_boxes)
        jp2h.box.append(rbox)

        temp_filename = self.filename + ".tmp"
        self.wrap(temp_filename, boxes=self.box)
        shutil.move(temp_filename, self.filename)
        self._parse(force=True)

    def _validate_kwargs(self):
        """Validate keyword parameters passed to the constructor."""
        non_cinema_args = (
            self._mct,
            self._cratios,
            self._psnr,
            self._irreversible,
            self._cbsize,
            self._eph,
            self._grid_offset,
            self._modesw,
            self._prog,
            self._psizes,
            self._sop,
            self._subsam,
        )
        if (self._cinema2k or self._cinema4k) and not all(
            [arg is None or not arg for arg in non_cinema_args]
        ):
            msg = "Do not specify cinema2k/cinema4k along with other options."
            raise InvalidJp2kError(msg)

        if self._psnr is not None:
            if self._cratios is not None:
                msg = "Cannot specify cratios and psnr options together."
                raise InvalidJp2kError(msg)

            if 0 in self._psnr and self._psnr[-1] != 0:
                msg = (
                    "If a zero value is supplied in the PSNR keyword "
                    "argument, it must be in the final position."
                )
                raise InvalidJp2kError(msg)

            if (
                0 in self._psnr and np.any(np.diff(self._psnr[:-1]) < 0)
                or 0 not in self._psnr and np.any(np.diff(self._psnr) < 0)
            ):
                msg = (
                    "PSNR values must be increasing, with one exception - "
                    "zero may be in the final position to indicate a lossless "
                    "layer."
                )
                raise InvalidJp2kError(msg)

        if (
            self._codec_format == opj2.CODEC_J2K
            and self._colorspace is not None
        ):
            msg = "Do not specify a colorspace when writing a raw codestream."
            raise InvalidJp2kError(msg)

        if (
            self._codec_format == opj2.CODEC_J2K
            and self._capture_resolution is not None
            and self._display_resolution is not None
        ):
            msg = (
                "Do not specify capture/display resolution when writing a raw "
                "codestream."
            )
            raise InvalidJp2kError(msg)

        if (
            self._shape is not None
            and self._tilesize_w is not None
            and (
                self.tilesize[0] > self.shape[0]
                or self.tilesize[1] > self.shape[1]
            )
        ):
            msg = (
                f"The tile size {self.tilesize} cannot exceed the image "
                f"size {self.shape[:2]}."
            )
            raise RuntimeError(msg)

    def get_tilewriters(self):
        """Return an object that facilitates writing tile by tile.

        The tiles are written out left-to-right, tile-row-by-tile-row.
        You must have image data ready to feed each tile writer, and you
        cannot skip a tile.

        You can use this method to write extremely large images that cannot
        fit into memory, tile by tile.

        Examples
        --------
        >>> import skimage.data
        >>> img = skimage.data.moon()
        >>> print(img.shape)
        (512, 512)
        >>> shape = img.shape[0] * 2, img.shape[1] * 2
        >>> tilesize = (img.shape[0], img.shape[1])
        >>> j = Jp2k('moon-4.jp2', shape=shape, tilesize=tilesize)
        >>> for tw in j.get_tilewriters():
        ...     tw[:] = img
        >>> j = Jp2kr('moon-4.jp2')
        >>> print(j.shape)
        (1024, 1024)
        """

        if self.shape[:2] == self.tilesize:
            msg = (
                "Do not write an image tile-by-tile "
                "if there is only one tile in the first place.  "
                "See issue #586"
            )
            raise RuntimeError(msg)

        return _TileWriter(self)

    def _set_cinema_params(self, cinema_mode, fps):
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
        self._cparams.tcp_mct = 1

        if cinema_mode == "cinema2k":
            if fps not in [24, 48]:
                msg = "Cinema2K frame rate must be either 24 or 48."
                raise ValueError(msg)

            if fps == 24:
                self._cparams.rsiz = core.OPJ_PROFILE_CINEMA_2K
                self._cparams.max_comp_size = core.OPJ_CINEMA_24_COMP
                self._cparams.max_cs_size = core.OPJ_CINEMA_24_CS
            else:
                self._cparams.rsiz = core.OPJ_PROFILE_CINEMA_2K
                self._cparams.max_comp_size = core.OPJ_CINEMA_48_COMP
                self._cparams.max_cs_size = core.OPJ_CINEMA_48_CS

        else:
            # cinema4k
            self._cparams.rsiz = core.OPJ_PROFILE_CINEMA_4K

    def _populate_cparams(self, img_array):
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

        cparams.codec_fmt = self._codec_format

        cparams.irreversible = 1 if self._irreversible else 0

        if self._cinema2k:
            # cinema2k is an integer, so this test is "truthy"
            self._cparams = cparams
            self._set_cinema_params("cinema2k", self._cinema2k)

        if self._cinema4k:
            self._cparams = cparams
            self._set_cinema_params("cinema4k", self._cinema4k)

        if self._cbsize is not None:
            cparams.cblockw_init = self._cbsize[1]
            cparams.cblockh_init = self._cbsize[0]

        if self._cratios is not None:
            cparams.tcp_numlayers = len(self._cratios)
            for j, cratio in enumerate(self._cratios):
                cparams.tcp_rates[j] = cratio
            cparams.cp_disto_alloc = 1

        cparams.csty |= 0x02 if self._sop else 0
        cparams.csty |= 0x04 if self._eph else 0

        if self._grid_offset is not None:
            cparams.image_offset_x0 = self._grid_offset[1]
            cparams.image_offset_y0 = self._grid_offset[0]

        if self._modesw is not None:
            # The None check is for backwards compatibility.
            for shift in range(6):
                power_of_two = 1 << shift
                if self._modesw & power_of_two:
                    cparams.mode |= power_of_two

        cparams.numresolution = self._numres

        if self._prog is not None:
            cparams.prog_order = core.PROGRESSION_ORDER[self._prog.upper()]

        if self._psnr is not None:
            cparams.tcp_numlayers = len(self._psnr)
            for j, snr_layer in enumerate(self._psnr):
                cparams.tcp_distoratio[j] = snr_layer
            cparams.cp_fixed_quality = 1

        if self._psizes is not None:
            for j, (prch, prcw) in enumerate(self._psizes):
                cparams.prcw_init[j] = prcw
                cparams.prch_init[j] = prch
            cparams.csty |= 0x01
            cparams.res_spec = len(self._psizes)

        if self._subsam is not None:
            cparams.subsampling_dy = self._subsam[0]
            cparams.subsampling_dx = self._subsam[1]

        if self._tilesize_w is not None:
            cparams.cp_tdx = self.tilesize[1]
            cparams.cp_tdy = self.tilesize[0]
            cparams.tile_size_on = opj2.TRUE

        if self._mct is None:

            # If the multi component transform was not specified, we infer
            # that it should be used if the color space is RGB.
            cparams.tcp_mct = 1 if self._colorspace == opj2.CLRSPC_SRGB else 0

        elif self._mct and self._colorspace == opj2.CLRSPC_GRAY:

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

        self._cparams = cparams

    def _write(self, img_array):
        """Write image data to a JP2/JPX/J2k file.  Intended usage of the
        various parameters follows that of OpenJPEG's opj_compress utility.

        This method can only be used to create JPEG 2000 images that can fit
        in memory.
        """
        if version.openjpeg_version < "2.3.0":
            msg = (
                "You must have at least version 2.3.0 of OpenJPEG in order to "
                "write images."
            )
            raise RuntimeError(msg)

        if hasattr(self, "_cparams"):
            msg = (
                "You cannot write image data to a JPEG 2000 file "
                "that already exists."
            )
            raise RuntimeError(msg)

        self._determine_colorspace()
        self._populate_cparams(img_array)

        if img_array.ndim == 2:
            # Force the image to be 3D.  This makes it easier to copy the
            # image data later on.
            numrows, numcols = img_array.shape
            img_array = img_array.reshape(numrows, numcols, 1)

        self._populate_comptparms(img_array)

        self._write_openjp2(img_array)

        # if writing the entire image, we need to parse ourselves in case
        # further operations are needed
        self.finalize(force_parse=True)

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

    def _determine_colorspace(self):
        """Determine the colorspace from the supplied inputs."""
        if self._colorspace is None:
            # Must infer the colorspace from the image dimensions.
            if len(self.shape) < 3:
                # A single channel image is grayscale.
                self._colorspace = opj2.CLRSPC_GRAY
            elif self.shape[2] == 1 or self.shape[2] == 2:
                # A single channel image or an image with two channels is going
                # to be greyscale.
                self._colorspace = opj2.CLRSPC_GRAY
            else:
                # Anything else must be RGB, right?
                self._colorspace = opj2.CLRSPC_SRGB
        else:
            if self._colorspace.lower() not in ("rgb", "grey", "gray"):
                msg = f'Invalid colorspace "{self._colorspace}".'
                raise InvalidJp2kError(msg)
            elif self._colorspace.lower() == "rgb" and self.shape[2] < 3:
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

            self._colorspace = COLORSPACE_MAP[self._colorspace.lower()]

    def _write_openjp2(self, img_array):
        """Write JPEG 2000 file using OpenJPEG 2.x interface."""
        with ExitStack() as stack:
            image = opj2.image_create(self._comptparms, self._colorspace)
            stack.callback(opj2.image_destroy, image)

            self._populate_image_struct(image, img_array)

            codec = opj2.create_compress(self._cparams.codec_fmt)
            stack.callback(opj2.destroy_codec, codec)

            if self._verbose:
                info_handler = opj2._INFO_CALLBACK
            else:
                info_handler = None

            opj2.set_info_handler(codec, info_handler)
            opj2.set_warning_handler(codec, opj2._WARNING_CALLBACK)
            opj2.set_error_handler(codec, opj2._ERROR_CALLBACK)

            opj2.setup_encoder(codec, self._cparams, image)

            if self._plt:
                opj2.encoder_set_extra_options(codec, plt=self._plt)

            if self._tlm:
                opj2.encoder_set_extra_options(codec, tlm=self._tlm)

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

    def append(self, box):
        """
        Append a metadata box to the JP2 file.  This will not result in a
        file-copy operation.  Only XML UUID (XMP), or ASOC boxes can be
        appended at this time.

        Parameters
        ----------
        box : Jp2Box
            Instance of a JP2 box.

        Examples
        --------
        >>> import io, shutil, lxml.etree as ET
        >>> _ = shutil.copyfile(glymur.data.nemo(), 'new-nemo.jp2')
        >>> j = glymur.Jp2k('new-nemo.jp2')
        >>> b = io.BytesIO(b'''
        ... <info>
        ...     <city>Nashville</city>
        ...     <city>Knoxville</city>
        ...     <city>Whoville</city>
        ... </info>
        ... ''')
        >>> doc = ET.parse(b)
        >>> xmlbox = glymur.jp2box.XMLBox(xml=doc)
        >>> j.append(xmlbox)
        >>> glymur.set_option('print.codestream', False)
        >>> print(j)
        File:  new-nemo.jp2
        JPEG 2000 Signature Box (jP  ) @ (0, 12)
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
        Contiguous Codestream Box (jp2c) @ (77, 1132296)
        XML Box (xml ) @ (1132373, 102)
            <info>
                <city>Nashville</city>
                <city>Knoxville</city>
                <city>Whoville</city>
            </info>
        """
        if self._codec_format == opj2.CODEC_J2K:
            msg = "You cannot append to a J2K file (raw codestream)."
            raise RuntimeError(msg)

        box_is_asoc = box.box_id == "asoc"
        box_is_xml = box.box_id == "xml "
        box_is_xmp = box.box_id == "uuid" and (
            box.uuid == UUID("be7acfcb-97a9-42e8-9c71-999491e3afac")
            or box.uuid == UUID("b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03")
        )
        if not (box_is_asoc or box_is_xml or box_is_xmp):
            msg = (
                "Only ASOC, XML, or UUID (XMP or GeoTIFF) boxes can currently "
                "be appended."
            )
            raise RuntimeError(msg)

        # Check the last box.  If the length field is zero, then rewrite
        # the length field to reflect the true length of the box.
        with self.path.open("rb") as ifile:
            offset = self.box[-1].offset
            ifile.seek(offset)
            read_buffer = ifile.read(4)
            (box_length,) = struct.unpack(">I", read_buffer)
            if box_length == 0:
                # Reopen the file in write mode and rewrite the length field.
                true_box_length = self.path.stat().st_size - offset
                with self.path.open("r+b") as ofile:
                    ofile.seek(offset)
                    write_buffer = struct.pack(">I", true_box_length)
                    ofile.write(write_buffer)

        # Can now safely append the box.
        with self.path.open("ab") as ofile:
            box.write(ofile)

        self._parse(force=True)

    def wrap(self, filename, boxes=None):
        """
        Create a new JP2/JPX file wrapped in a new set of JP2 boxes.

        This method is primarily aimed at wrapping a raw codestream in a set of
        of JP2 boxes (turning it into a JP2 file instead of just a raw
        codestream), or rewrapping a codestream in a JP2 file in a new "jacket"
        of JP2 boxes.  Wrapping a raw codestream preserves the internal
        structure of the codestream, whereas simply writing it back out by
        invoking the Jp2k constructor might rewrite the internal structure.

        Parameters
        ----------
        filename : str
            JP2 file to be created from a raw codestream.
        boxes : list
            JP2 box definitions to define the JP2 file format.  If not
            provided, a default ""jacket" is assumed, consisting of JP2
            signature, file type, JP2 header, and contiguous codestream boxes.
            A JPX file rewrapped without the boxes argument results in a JP2
            file encompassing the first codestream.

        Returns
        -------
        Jp2k
            Newly wrapped Jp2k object.

        Examples
        --------

        >>> j2c = glymur.Jp2k(glymur.data.goodstuff())
        >>> jp2 = j2c.wrap('new-goodstuff.jp2')
        >>> glymur.set_option('print.short', True)
        >>> print(jp2)
        File:  new-goodstuff.jp2
        JPEG 2000 Signature Box (jP  ) @ (0, 12)
        File Type Box (ftyp) @ (12, 20)
        JP2 Header Box (jp2h) @ (32, 45)
            Image Header Box (ihdr) @ (40, 22)
            Colour Specification Box (colr) @ (62, 15)
        Contiguous Codestream Box (jp2c) @ (77, 115228)
        """
        if boxes is None:
            boxes = self._get_default_jp2_boxes()

        self._validate_jp2_box_sequence(boxes)

        with open(filename, "wb") as ofile:
            for box in boxes:
                if box.box_id != "jp2c":
                    box.write(ofile)
                else:
                    self._write_wrapped_codestream(ofile, box)
            ofile.flush()

        jp2 = Jp2k(filename)
        return jp2

    def _write_wrapped_codestream(self, ofile, box):
        """Write wrapped codestream."""
        # Codestreams require a bit more care.
        # Am I a raw codestream?
        if len(self.box) == 0:
            # Yes, just write the codestream box header plus all
            # of myself out to file.
            ofile.write(struct.pack(">I", self.length + 8))
            ofile.write(b"jp2c")
            with open(self.filename, "rb") as ifile:
                ofile.write(ifile.read())
            return

        # OK, I'm a jp2/jpx file.  Need to find out where the raw codestream
        # actually starts.
        offset = box.offset
        if offset == -1:
            if self.box[1].brand == "jpx ":
                msg = (
                    "The codestream box must have its offset and length "
                    "attributes fully specified if the file type brand is JPX."
                )
                raise InvalidJp2kError(msg)

            # Find the first codestream in the file.
            jp2c = [_box for _box in self.box if _box.box_id == "jp2c"]
            offset = jp2c[0].offset

        # Ready to write the codestream.
        with open(self.filename, "rb") as ifile:
            ifile.seek(offset)

            # Verify that the specified codestream is right.
            read_buffer = ifile.read(8)
            L, T = struct.unpack_from(">I4s", read_buffer, 0)
            if T != b"jp2c":
                msg = "Unable to locate the specified codestream."
                raise InvalidJp2kError(msg)
            if L == 0:
                # The length of the box is presumed to last until the end of
                # the file.  Compute the effective length of the box.
                L = self.path.stat().st_size - ifile.tell() + 8

            elif L == 1:
                # The length of the box is in the XL field, a 64-bit value.
                read_buffer = ifile.read(8)
                (L,) = struct.unpack(">Q", read_buffer)

            ifile.seek(offset)
            read_buffer = ifile.read(L)
            ofile.write(read_buffer)

    def _get_default_jp2_boxes(self):
        """Create a default set of JP2 boxes."""
        # Try to create a reasonable default.
        boxes = [
            JPEG2000SignatureBox(),
            FileTypeBox(),
            JP2HeaderBox(),
            ContiguousCodestreamBox(),
        ]
        height = self.codestream.segment[1].ysiz
        width = self.codestream.segment[1].xsiz
        num_components = len(self.codestream.segment[1].xrsiz)
        if num_components < 3:
            colorspace = core.GREYSCALE
        else:
            if len(self.box) == 0:
                # Best guess is SRGB
                colorspace = core.SRGB
            else:
                # Take whatever the first jp2 header / color specification
                # says.
                jp2hs = [box for box in self.box if box.box_id == "jp2h"]
                colorspace = jp2hs[0].box[1].colorspace

        boxes[2].box = [
            ImageHeaderBox(
                height=height, width=width, num_components=num_components
            ),
            ColourSpecificationBox(colorspace=colorspace),
        ]

        return boxes

    def __setitem__(self, index, data):
        """Slicing protocol."""
        # Need to set this in case it is not set in the constructor.
        if self._shape is None:
            self._shape = data.shape

        if (
            isinstance(index, slice)
            and index.start is None
            and index.stop is None
            and index.step is None
        ):
            # Case of jp2[:] = data, i.e. write the entire image.
            #
            # Should have a slice object where start = stop = step = None
            self._write(data)
        elif index is Ellipsis:
            # Case of jp2[...] = data, i.e. write the entire image.
            self._write(data)
        else:
            msg = "Partial write operations are currently not allowed."
            raise ValueError(msg)

    def _remove_ellipsis(self, index, numrows, numcols, numbands):
        """resolve the first ellipsis in the index

        The intent is that it references the image

        Parameters
        ----------
        index : tuple
            tuple of index arguments, presumably one of them is the Ellipsis
        numrows, numcols, numbands : int
            image dimensions

        Returns
        -------
        tuple
            Same as index, except that the first Ellipsis is replaced with
            a proper slice whose start and stop members are not None
        """
        # Remove the first ellipsis we find.
        rows = slice(0, numrows)
        cols = slice(0, numcols)
        bands = slice(0, numbands)
        if index[0] is Ellipsis:
            if len(index) == 2:
                # jp2k[..., other_slice]
                newindex = (rows, cols, index[1])
            else:
                # jp2k[..., cols, bands]
                newindex = (rows, index[1], index[2])
        elif index[1] is Ellipsis:
            if len(index) == 2:
                # jp2k[rows, ...]
                newindex = (index[0], cols, bands)
            else:
                # jp2k[rows, ..., bands]
                newindex = (index[0], cols, index[2])
        else:
            # Assume that we don't have 4D imagery, of course.
            newindex = (index[0], index[1], bands)

        return newindex

    def _component2dtype(self, component):
        """Determin the appropriate numpy datatype for an OpenJPEG component.

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

    def _populate_image_struct(
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
            self._validate_nonzero_image_size(numrows, numcols, k)

        # set image offset and reference grid
        image.contents.x0 = self._cparams.image_offset_x0
        image.contents.y0 = self._cparams.image_offset_y0
        image.contents.x1 = (
            image.contents.x0
            + (numcols - 1) * self._cparams.subsampling_dx * tile_x_factor
            + 1
        )
        image.contents.y1 = (
            image.contents.y0
            + (numrows - 1) * self._cparams.subsampling_dy * tile_y_factor
            + 1
        )

        if tile_x_factor != 1 or tile_y_factor != 1:
            # don't stage the data if writing tiles
            return image

        # Stage the image data to the openjpeg data structure.
        for k in range(0, num_comps):
            if self._cparams.rsiz in (
                core.OPJ_PROFILE_CINEMA_2K,
                core.OPJ_PROFILE_CINEMA_4K,
            ):
                image.contents.comps[k].prec = 12
                image.contents.comps[k].bpp = 12

            layer = np.ascontiguousarray(imgdata[:, :, k], dtype=np.int32)
            dest = image.contents.comps[k].data
            src = layer.ctypes.data
            ctypes.memmove(dest, src, layer.nbytes)

        return image

    def _populate_comptparms(self, img_array):
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
            comptparms[j].dx = self._cparams.subsampling_dx
            comptparms[j].dy = self._cparams.subsampling_dy
            comptparms[j].w = numcols
            comptparms[j].h = numrows
            comptparms[j].x0 = self._cparams.image_offset_x0
            comptparms[j].y0 = self._cparams.image_offset_y0
            comptparms[j].prec = comp_prec
            comptparms[j].bpp = comp_prec
            comptparms[j].sgnd = 0

        self._comptparms = comptparms

    def _validate_jp2_box_sequence(self, boxes):
        """Run through series of tests for JP2 box legality.

        This is non-exhaustive.
        """
        JP2_IDS = [
            "colr",
            "cdef",
            "cmap",
            "jp2c",
            "ftyp",
            "ihdr",
            "jp2h",
            "jP  ",
            "pclr",
            "res ",
            "resc",
            "resd",
            "xml ",
            "ulst",
            "uinf",
            "url ",
            "uuid",
        ]

        self._validate_signature_compatibility(boxes)
        self._validate_jp2h(boxes)
        self._validate_jp2c(boxes)
        if boxes[1].brand == "jpx ":
            self._validate_jpx_box_sequence(boxes)
        else:
            # Validate the JP2 box IDs.
            count = self._collect_box_count(boxes)
            for box_id in count.keys():
                if box_id not in JP2_IDS:
                    msg = (
                        f"The presence of a '{box_id}' box requires that the "
                        f"file type brand be set to 'jpx '."
                    )
                    raise InvalidJp2kError(msg)

            self._validate_jp2_colr(boxes)

    def _validate_jp2_colr(self, boxes):
        """Validate JP2 requirements on colour specification boxes."""
        jp2h = next(filter(lambda x: x.box_id == "jp2h", boxes), None)
        for colr in [box for box in jp2h.box if box.box_id == "colr"]:
            if colr.approximation != 0:
                msg = (
                    "A JP2 colr box cannot have a non-zero approximation "
                    "field."
                )
                raise InvalidJp2kError(msg)

    def _validate_jpx_box_sequence(self, boxes):
        """Run through series of tests for JPX box legality."""
        self._validate_label(boxes)
        self._validate_jpx_compatibility(boxes, boxes[1].compatibility_list)
        self._validate_singletons(boxes)
        self._validate_top_level(boxes)

    def _validate_signature_compatibility(self, boxes):
        """Validate the file signature and compatibility status."""
        # Check for a bad sequence of boxes.
        # 1st two boxes must be 'jP  ' and 'ftyp'
        if boxes[0].box_id != "jP  " or boxes[1].box_id != "ftyp":
            msg = (
                "The first box must be the signature box and the second must "
                "be the file type box."
            )
            raise InvalidJp2kError(msg)

        # The compatibility list must contain at a minimum 'jp2 '.
        if "jp2 " not in boxes[1].compatibility_list:
            msg = "The ftyp box must contain 'jp2 ' in the compatibility list."
            raise InvalidJp2kError(msg)

    def _validate_jp2c(self, boxes):
        """Validate the codestream box in relation to other boxes."""
        # jp2c must be preceeded by jp2h
        jp2h_idx, _ = next(
            filter(lambda x: x[1].box_id == "jp2h", enumerate(boxes)),
            (None, None)
        )
        jp2c_idx, _ = next(
            filter(lambda x: x[1].box_id == "jp2c", enumerate(boxes)),
            (None, None)
        )
        if jp2c_idx is None:
            msg = (
                "A codestream box must be defined in the outermost list of "
                "boxes."
            )
            raise InvalidJp2kError(msg)

        if jp2h_idx >= jp2c_idx:
            msg = "The codestream box must be preceeded by a jp2 header box."
            raise InvalidJp2kError(msg)

    def _validate_jp2h(self, boxes):
        """Validate the JP2 Header box."""
        self._check_jp2h_child_boxes(boxes, "top-level")

        jp2h = next(filter(lambda x: x.box_id == "jp2h", boxes), None)

        # 1st jp2 header box cannot be empty.
        if len(jp2h.box) == 0:
            msg = "The JP2 header superbox cannot be empty."
            raise InvalidJp2kError(msg)

        # 1st jp2 header box must be ihdr
        if jp2h.box[0].box_id != "ihdr":
            msg = (
                "The first box in the jp2 header box must be the image header "
                "box."
            )
            raise InvalidJp2kError(msg)

        # colr must be present in jp2 header box.
        colr = next(filter(lambda x: x.box_id == "colr", jp2h.box), None)
        if colr is None:
            msg = "The jp2 header box must contain a color definition box."
            raise InvalidJp2kError(msg)

        self._validate_channel_definition(jp2h, colr)

    def _validate_channel_definition(self, jp2h, colr):
        """Validate the channel definition box."""
        cdef_lst = [
            idx for (idx, box) in enumerate(jp2h.box) if box.box_id == "cdef"
        ]
        if len(cdef_lst) > 1:
            msg = (
                "Only one channel definition box is allowed in the "
                "JP2 header."
            )
            raise InvalidJp2kError(msg)
        elif len(cdef_lst) == 1:
            cdef = jp2h.box[cdef_lst[0]]
            if colr.colorspace == core.SRGB:
                if any(
                    [
                        chan + 1 not in cdef.association
                        or cdef.channel_type[chan] != 0
                        for chan in [0, 1, 2]
                    ]
                ):
                    msg = (
                        "All color channels must be defined in the "
                        "channel definition box."
                    )
                    raise InvalidJp2kError(msg)
            elif colr.colorspace == core.GREYSCALE:
                if 0 not in cdef.channel_type:
                    msg = (
                        "All color channels must be defined in the "
                        "channel definition box."
                    )
                    raise InvalidJp2kError(msg)

    def _check_jp2h_child_boxes(self, boxes, parent_box_name):
        """Certain boxes can only reside in the JP2 header."""
        JP2H_CHILDREN = set(["bpcc", "cdef", "cmap", "ihdr", "pclr"])

        box_ids = set([box.box_id for box in boxes])
        intersection = box_ids.intersection(JP2H_CHILDREN)
        if len(intersection) > 0 and parent_box_name not in ["jp2h", "jpch"]:
            msg = (
                f"A {list(intersection)[0]} box can only be nested in a JP2 "
                f"header box."
            )
            raise InvalidJp2kError(msg)

        # Recursively check any contained superboxes.
        for box in boxes:
            if hasattr(box, "box"):
                self._check_jp2h_child_boxes(box.box, box.box_id)

    def _collect_box_count(self, boxes):
        """Count the occurences of each box type."""
        count = Counter([box.box_id for box in boxes])

        # Add the counts in the superboxes.
        for box in boxes:
            if hasattr(box, "box"):
                count.update(self._collect_box_count(box.box))

        return count

    def _check_superbox_for_top_levels(self, boxes):
        """Several boxes can only occur at the top level."""
        # We are only looking at the boxes contained in a superbox, so if any
        # of the blacklisted boxes show up here, it's an error.
        TOP_LEVEL_ONLY_BOXES = set(["dtbl"])
        box_ids = set([box.box_id for box in boxes])
        intersection = box_ids.intersection(TOP_LEVEL_ONLY_BOXES)
        if len(intersection) > 0:
            msg = (
                f"A {list(intersection)[0]} box cannot be nested in a "
                f"superbox."
            )
            raise InvalidJp2kError(msg)

        # Recursively check any contained superboxes.
        for box in boxes:
            if hasattr(box, "box"):
                self._check_superbox_for_top_levels(box.box)

    def _validate_top_level(self, boxes):
        """Several boxes can only occur at the top level."""
        # Add the counts in the superboxes.
        for box in boxes:
            if hasattr(box, "box"):
                self._check_superbox_for_top_levels(box.box)

        count = self._collect_box_count(boxes)

        # If there is one data reference box, then there must also be one ftbl.
        if "dtbl" in count and "ftbl" not in count:
            msg = (
                "The presence of a data reference box requires the "
                "presence of a fragment table box as well."
            )
            raise InvalidJp2kError(msg)

    def _validate_singletons(self, boxes):
        """Several boxes can only occur once."""
        count = self._collect_box_count(boxes)
        # Which boxes occur more than once?
        multiples = [box_id for box_id, bcount in count.items() if bcount > 1]
        if "dtbl" in multiples:
            raise InvalidJp2kError("There can only be one dtbl box in a file.")

    def _validate_jpx_compatibility(self, boxes, compatibility_list):
        """If there is a JPX box then the compatibility list must also contain
        'jpx '.
        """
        JPX_IDS = ["asoc", "nlst"]
        jpx_cl = set(compatibility_list)
        for box in boxes:
            if box.box_id in JPX_IDS:
                if len(set(["jpx ", "jpxb"]).intersection(jpx_cl)) == 0:
                    msg = (
                        "A JPX box requires that either 'jpx ' or 'jpxb' be "
                        "present in the ftype compatibility list."
                    )
                    raise InvalidJp2kError(msg)
            if hasattr(box, "box") != 0:
                # Same set of checks on any child boxes.
                self._validate_jpx_compatibility(box.box, compatibility_list)

    def _validate_label(self, boxes):
        """Label boxes can only be inside association, codestream headers, or
        compositing layer header boxes.
        """
        for box in boxes:
            if box.box_id != "asoc" and hasattr(box, "box"):
                for boxi in box.box:
                    if boxi.box_id == "lbl ":
                        msg = (
                            f"A label box cannot be nested inside a "
                            f"{box.box_id} box."
                        )
                        raise InvalidJp2kError(msg)
                # Same set of checks on any child boxes.
                    self._validate_label(box.box)


class _TileWriter(object):
    """Writes tiles to file, one by one.

    Attributes
    ----------
    jp2k : glymur.Jp2k
        Object wrapping the JPEG2000 file.
    num_tile_rows, num_tile_cols : int
        Dimensions of the image in terms of tiles.
    number_of_tiles : int
        This many tiles in the image.
    tile_index : int
        Each time the iteration protocol fires, this index will increase by
        one, as the openjpeg library requires the tiles to be processed
        sequentially.
    """

    def __init__(self, jp2k):
        self.jp2k = jp2k

        self.num_tile_rows = int(
            np.ceil(self.jp2k.shape[0] / self.jp2k.tilesize[0])
        )
        self.num_tile_cols = int(
            np.ceil(self.jp2k.shape[1] / self.jp2k.tilesize[1])
        )
        self.number_of_tiles = self.num_tile_rows * self.num_tile_cols

    def __iter__(self):
        self.tile_index = -1
        return self

    def __next__(self):
        if self.tile_index < self.number_of_tiles - 1:
            self.tile_index += 1
            return self
        else:
            # We've gone thru all the tiles by this point.
            self.jp2k.finalize(force_parse=True)
            raise StopIteration

    def __setitem__(self, index, img_array):
        """Write image data to a JP2/JPX/J2k file.  Intended usage of the
        various parameters follows that of OpenJPEG's opj_compress utility.
        """
        if version.openjpeg_version < "2.3.0":
            msg = (
                "You must have at least version 2.3.0 of OpenJPEG "
                "in order to write images."
            )
            raise RuntimeError(msg)

        if not isinstance(index, slice):
            msg = (
                "When writing tiles, the tile slice arguments must be just"
                "a single slice(None, None, None), i.e. [:]."
            )
            raise RuntimeError(msg)

        if self.tile_index == 0:
            self.setup_first_tile(img_array)

        try:
            opj2.write_tile(
                self.codec,
                self.tile_index,
                _set_planar_pixel_order(img_array),
                self.stream,
            )
        except glymur.lib.openjp2.OpenJPEGLibraryError as e:
            # properly dispose of these resources
            opj2.end_compress(self.codec, self.stream)
            opj2.stream_destroy(self.stream)
            opj2.image_destroy(self.image)
            opj2.destroy_codec(self.codec)
            raise e

        if self.tile_index == self.number_of_tiles - 1:
            # properly dispose of these resources
            opj2.end_compress(self.codec, self.stream)
            opj2.stream_destroy(self.stream)
            opj2.image_destroy(self.image)
            opj2.destroy_codec(self.codec)

    def setup_first_tile(self, img_array):
        """Only do these things for the first tile."""
        self.jp2k._determine_colorspace()
        self.jp2k._populate_cparams(img_array)
        self.jp2k._populate_comptparms(img_array)

        self.codec = opj2.create_compress(self.jp2k._cparams.codec_fmt)

        if self.jp2k.verbose:
            info_handler = opj2._INFO_CALLBACK
        else:
            info_handler = None

        opj2.set_info_handler(self.codec, info_handler)
        opj2.set_warning_handler(self.codec, opj2._WARNING_CALLBACK)
        opj2.set_error_handler(self.codec, opj2._ERROR_CALLBACK)

        self.image = opj2.image_tile_create(
            self.jp2k._comptparms, self.jp2k._colorspace
        )

        self.jp2k._populate_image_struct(
            self.image,
            img_array,
            tile_x_factor=self.num_tile_cols,
            tile_y_factor=self.num_tile_rows,
        )
        self.image.contents.x1 = self.jp2k.shape[1]
        self.image.contents.y1 = self.jp2k.shape[0]

        opj2.setup_encoder(self.codec, self.jp2k._cparams, self.image)

        if self.jp2k._plt:
            opj2.encoder_set_extra_options(self.codec, plt=self.jp2k._plt)

        self.stream = opj2.stream_create_default_file_stream(
            self.jp2k.filename,
            False
        )

        num_threads = get_option("lib.num_threads")
        if version.openjpeg_version >= "2.4.0":
            opj2.codec_set_threads(self.codec, num_threads)
        elif num_threads > 1:
            msg = (
                f"Threaded encoding is not supported in library versions "
                f"prior to 2.4.0.  Your version is "
                f"{version.openjpeg_version}."
            )
            warnings.warn(msg, UserWarning)

        opj2.start_compress(self.codec, self.image, self.stream)


def _set_planar_pixel_order(img):
    """Reorder the image pixels so that plane-0 comes first, then plane-1, etc.
    This is a requirement for using opj_write_tile.
    """
    if img.ndim == 3:
        # C-order increments along the y-axis slowest (0), then x-axis (1),
        # then z-axis (2).  We want it to go along the z-axis slowest, then
        # y-axis, then x-axis.
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)

    return img.copy()
