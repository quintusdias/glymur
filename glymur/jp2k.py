"""This file is part of glymur, a Python interface for accessing JPEG 2000.

http://glymur.readthedocs.org

Copyright 2013 John Evans

License:  MIT
"""
# Standard library imports...
from collections import Counter
from contextlib import ExitStack
from itertools import filterfalse
import ctypes
import pathlib
import re
import shutil
import struct
from uuid import UUID
import warnings

# Third party library imports
import numpy as np

# Local imports...
import glymur
from .codestream import Codestream
from . import core, version, get_option
from .jp2box import (
    Jp2kBox, JPEG2000SignatureBox, FileTypeBox, JP2HeaderBox,
    ColourSpecificationBox, ContiguousCodestreamBox, ImageHeaderBox,
    InvalidJp2kError
)
from .lib import openjp2 as opj2


class Jp2k(Jp2kBox):
    """JPEG 2000 file.

    Attributes
    ----------
    filename : str
        The path to the JPEG 2000 file.
    box : sequence
        List of top-level boxes in the file.  Each box may in turn contain
        its own list of boxes.  Will be empty if the file consists only of a
        raw codestream.
    shape : tuple
        Size of the image.

    Properties
    ----------
    codestream : glymur.codestream.Codestream
        JP2 or J2K codestream object.
    decoded_components : sequence or None
        If set, decode only these components.  The MCT will not be used.
    ignore_pclr_cmap_cdef : bool
        Whether or not to ignore the pclr, cmap, or cdef boxes during any
        color transformation, defaults to False.
    layer : int
        Zero-based number of quality layer to decode.
    verbose : bool
        Whether or not to print informational messages produced by the
        OpenJPEG library, defaults to false.

    Examples
    --------
    >>> import glymur
    >>> jfile = glymur.data.nemo()
    >>> jp2 = glymur.Jp2k(jfile)
    >>> jp2.shape
    (1456, 2592, 3)
    >>> image = jp2[:]
    >>> image.shape
    (1456, 2592, 3)

    Read a lower resolution thumbnail.

    >>> thumbnail = jp2[::2, ::2]
    >>> thumbnail.shape
    (728, 1296, 3)

    Make use of OpenJPEG's thread support

    >>> import glymur
    >>> import time
    >>> if glymur.version.openjpeg_version >= '2.2.0':
    ...     jp2file = glymur.data.nemo()
    ...     jp2 = glymur.Jp2k(jp2file)
    ...     t0 = time.time(); data = jp2[:]; t1 = time.time()
    ...     t1 - t0 #doctest: +SKIP
    0.9024193286895752
    ...     glymur.set_options('lib.num_threads', 4)
    ...     t0 = time.time(); data = jp2[:]; t1 = time.time()
    ...     t1 - t0 #doctest: +SKIP
    0.4060473537445068
    """

    def __init__(
        self, filename, data=None, shape=None, tilesize=None, verbose=False,
        capture_resolution=None, cbsize=None, cinema2k=None,
        cinema4k=None, colorspace=None, cratios=None,
        display_resolution=None, eph=None, grid_offset=None,
        irreversible=None, mct=None, modesw=None, numres=None,
        plt=False, prog=None, psizes=None, psnr=None, sop=None,
        subsam=None, tlm=False,
    ):
        """
        Parameters
        ----------
        filename : str
            The path to JPEG 2000 file.
        path : pathlib.Path
            The path to JPEG 2000 file.
        image_data : ndarray, optional
            Image data to be written to file.
        shape : tuple, optional
            Size of image data, only required when image_data is not provided.
        capture_resolution : tuple, optional
            Capture solution (VRES, HRES).  This appends a capture resolution
            box onto the end of the JP2 file when it is created.
        cbsize : tuple, optional
            Code block size (NROWS, NCOLS)
        cinema2k : int, optional
            Frames per second, either 24 or 48.
        cinema4k : bool, optional
            Set to True to specify Cinema4K mode, defaults to false.
        colorspace : {'rgb', 'gray'}
            The image color space.
        cratios : iterable, optional
            Compression ratios for successive layers.
        display_resolution : tuple, optional
            Display solution (VRES, HRES).  This appends a display resolution
            box onto the end of the JP2 file when it is created.
        eph : bool, optional
            If true, write EPH marker after each header packet.
        grid_offset : tuple, optional
            Offset (DY, DX) of the origin of the image in the reference grid.
        irreversible : bool, optional
            If true, use the irreversible DWT 9-7 transform.
        mct : bool, optional
            Usage of the multi component transform to write an image.  If not
            specified, defaults to True if the color space is RGB.
        modesw : int, optional
            mode switch
                1 = BYPASS(LAZY)
                2 = RESET
                4 = RESTART(TERMALL)
                8 = VSC
                16 = ERTERM(SEGTERM)
                32 = SEGMARK(SEGSYM)
        numres : int, optional
            Number of resolutions.
        plt : bool, optional
            Generate PLT markers.
        prog : {"LRCP" "RLCP", "RPCL", "PCRL", "CPRL"}
            Progression order.
        psnr : iterable, optional
            Different PSNR for successive layers.
        psizes : list, optional
            List of precinct sizes, each precinct size tuple is defined in
            (height x width).
        sop : bool, optional
            If true, write SOP marker before each packet.
        subsam : tuple, optional
            Subsampling factors (dy, dx).
        tilesize : tuple, optional
            Tile size in terms of (numrows, numcols), not (X, Y).
        tlm : bool, optional
            Generate TLM markers.
        verbose : bool, optional
            Print informational messages produced by the OpenJPEG library.
        """
        super().__init__()

        # In case of pathlib.Paths...
        self.filename = str(filename)
        self.path = pathlib.Path(self.filename)

        self.box = []
        self._layer = 0
        self._codestream = None
        self._decoded_components = None

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
        self._numres = numres
        self._plt = plt
        self._prog = prog
        self._psizes = psizes
        self._psnr = psnr
        self._sop = sop
        self._subsam = subsam
        self._tilesize = tilesize
        self._tlm = tlm

        self._shape = shape
        self._ndim = None
        self._dtype = None
        self._ignore_pclr_cmap_cdef = False
        self._verbose = verbose

        if self.filename[-4:].endswith(('.jp2', '.JP2')):
            self._codec_format = opj2.CODEC_JP2
        else:
            self._codec_format = opj2.CODEC_J2K

        if data is None and shape is None and self.path.exists():
            self._readonly = True
        else:
            self._readonly = False

        if data is None and tilesize is not None and shape is not None:
            self._writing_by_tiles = True
        else:
            self._writing_by_tiles = False

        if data is None and tilesize is None:
            # case of
            # j = Jp2k(filename)
            # j[:] = data
            self._expecting_to_write_by_setitem = True
        else:
            self._expecting_to_write_by_setitem = False

        if data is not None:
            self._have_data = True
            self._shape = data.shape
        else:
            self._have_data = False

        self._validate_kwargs()

        if self._readonly:
            # We must be just reading a JP2/J2K/JPX file.  Parse its
            # contents, then determine the shape.  We are then done.
            self.parse()
            self._initialize_shape()
            return

        if data is not None:
            # We are writing a JP2/J2K/JPX file where the image is
            # contained in memory.
            self._write(data)

        self.finalize()

    def finalize(self, force_parse=False):
        """
        For now, the only task remaining is to possibly write out a
        ResolutionBox if we were so instructed.  There could be other
        possibilities in the future.

        Parameters
        ----------
        force : bool
            If true, then run finalize operations
        """
        # Cases where we do NOT want to parse.
        if (
            (self._writing_by_tiles or self._expecting_to_write_by_setitem)
            and not force_parse
        ):
            # We are writing by tiles but we are not finished doing that.
            # or
            # we are writing by __setitem__ but aren't finished doing that
            # either
            return

        # So now we are basically done writing a JP2/Jp2k file ...
        if (
            self._capture_resolution is None
            and self._display_resolution is None
        ):
            # ... and we don't have any extra boxes, so go ahead and parse.
            self.parse()
            return

        # So we DO have extra boxes.  Handle them, and THEN parse.
        self.parse()
        self._insert_resolution_superbox()

    def _insert_resolution_superbox(self):
        """
        As a close-out task, insert a resolution superbox into the jp2
        header box if we were so instructed.  This requires a wrapping
        operation.
        """
        jp2h = [box for box in self.box if box.box_id == 'jp2h'][0]

        extra_boxes = []
        if self._capture_resolution is not None:
            resc = glymur.jp2box.CaptureResolutionBox(
                self._capture_resolution[0], self._capture_resolution[1],
            )
            extra_boxes.append(resc)

        if self._display_resolution is not None:
            resd = glymur.jp2box.DisplayResolutionBox(
                self._display_resolution[0], self._display_resolution[1],
            )
            extra_boxes.append(resd)

        rbox = glymur.jp2box.ResolutionBox(extra_boxes)
        jp2h.box.append(rbox)

        temp_filename = self.filename + '.tmp'
        self.wrap(temp_filename, boxes=self.box)
        shutil.move(temp_filename, self.filename)
        self.parse()

    def _validate_kwargs(self):
        """
        Validate keyword parameters passed to the constructor.
        """
        non_cinema_args = (
            self._mct, self._cratios, self._psnr, self._irreversible,
            self._cbsize, self._eph, self._grid_offset, self._modesw,
            self._numres, self._prog, self._psizes, self._sop, self._subsam
        )
        if (
            (
                self._cinema2k is not None or self._cinema4k is not None
            )
            and (not all([arg is None for arg in non_cinema_args]))
        ):
            msg = (
                "Cannot specify cinema2k/cinema4k along with any other "
                "options."
            )
            raise InvalidJp2kError(msg)

        if self._psnr is not None:
            if self._cratios is not None:
                msg = "Cannot specify cratios and psnr options together."
                raise InvalidJp2kError(msg)

            if 0 in self._psnr and self._psnr[-1] != 0:
                msg = ("If a zero value is supplied in the PSNR keyword "
                       "argument, it must be in the final position.")
                raise InvalidJp2kError(msg)

            if (
                (
                    0 in self._psnr
                    and np.any(np.diff(self._psnr[:-1]) < 0)
                )
                or (
                    0 not in self._psnr
                    and np.any(np.diff(self._psnr) < 0)
                )
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
            msg = 'Do not specify a colorspace when writing a raw codestream.'
            raise InvalidJp2kError(msg)

        if (
            self._codec_format == opj2.CODEC_J2K
            and self._capture_resolution is not None
            and self._display_resolution is not None
        ):
            msg = (
                'Do not specify capture/display resolution when writing a raw '
                'codestream.'
            )
            raise InvalidJp2kError(msg)

        if self._readonly and self._capture_resolution is not None:
            msg = (
                'Capture/Display resolution keyword parameters cannot be '
                'supplied when the intent seems to be to read an image.'
            )
            raise RuntimeError(msg)

        if (
            self._shape is not None
            and self.tilesize is not None
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

    def _initialize_shape(self):
        """
        If there was no image data provided and if no shape was
        initially provisioned, then shape must be computed AFTER we
        have parsed the input file.
        """
        if self._codec_format == opj2.CODEC_J2K:
            # get the image size from the codestream
            cstr = self.codestream
            height = cstr.segment[1].ysiz
            width = cstr.segment[1].xsiz
            num_components = len(cstr.segment[1].xrsiz)
        else:
            # try to get the image size from the IHDR box
            jp2h = [box for box in self.box if box.box_id == 'jp2h'][0]
            ihdr = [box for box in jp2h.box if box.box_id == 'ihdr'][0]

            height, width = ihdr.height, ihdr.width
            num_components = ihdr.num_components

            if num_components == 1:
                # but if there is a PCLR box, then we need to check
                # that as well, as that turns a single-channel image
                # into a multi-channel image
                pclr = [box for box in jp2h.box if box.box_id == 'pclr']
                if len(pclr) > 0:
                    num_components = len(pclr[0].signed)

        if num_components == 1:
            self.shape = (height, width)
        else:
            self.shape = (height, width, num_components)

        return self._shape

    @property
    def ignore_pclr_cmap_cdef(self):
        return self._ignore_pclr_cmap_cdef

    @ignore_pclr_cmap_cdef.setter
    def ignore_pclr_cmap_cdef(self, ignore_pclr_cmap_cdef):
        self._ignore_pclr_cmap_cdef = ignore_pclr_cmap_cdef

    @property
    def decoded_components(self):
        return self._decoded_components

    @decoded_components.setter
    def decoded_components(self, components):

        if components is None:
            # this is a special case where we are restoring the original
            # behavior of reading all bands
            self._decoded_components = components
            return

        if np.isscalar(components):
            components = [components]

        if any(x > len(self.codestream.segment[1].xrsiz) for x in components):
            msg = (
                f"{components} has at least one invalid component, "
                f"cannot be greater than "
                f"{len(self.codestream.segment[1].xrsiz)}."
            )
            raise ValueError(msg)

        self._decoded_components = components

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layer):
        # Set to the indicated value so long as it is valid.
        cod = [
            segment for segment in self.codestream.segment
            if segment.marker_id == 'COD'
        ][0]
        if layer < 0 or layer >= cod.layers:
            msg = f"Invalid layer number, must be in range [0, {cod.layers})."
            raise ValueError(msg)

        self._layer = layer

    @property
    def dtype(self):
        """
        Datatype of the image elements.
        """
        if self._dtype is None:
            c = self.get_codestream()
            bps0 = c.segment[1].bitdepth[0]
            sgnd0 = c.segment[1].signed[0]

            if (
                all(bitdepth == bps0 for bitdepth in c.segment[1].bitdepth)
                and all(signed == sgnd0 for signed in c.segment[1].signed)
            ):
                if bps0 <= 8:
                    self._dtype = np.int8 if sgnd0 else np.uint8
                else:
                    self._dtype = np.int16 if sgnd0 else np.uint16
            else:
                msg = (
                    "The dtype property is only valid when all components "
                    "have the same bitdepth and sign. "
                    "\n\n"
                    f"{c.segment[1]}"
                )
                raise TypeError(msg)

        return self._dtype

    @property
    def ndim(self):
        """
        Number of image dimensions.
        """
        if self._ndim is None:
            self._ndim = len(self.shape)

        return self._ndim

    @property
    def codestream(self):
        if self._codestream is None:
            self._codestream = self.get_codestream(header_only=True)
        return self._codestream

    @property
    def tilesize(self):
        return self._tilesize

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """Verbosity property.

        If True, print informational messages from the OPENJPEG library.

        Parameters
        ----------
        verbose : {True, False}
            Set to verbose or not.
        """
        self._verbose = verbose

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def __repr__(self):
        msg = f"glymur.Jp2k('{self.path}')"
        return msg

    def __str__(self):
        metadata = [f'File:  {self.path.name}']
        if len(self.box) > 0:
            for box in self.box:
                metadata.append(str(box))
        elif self._codestream is None and not self.path.exists():
            # No codestream either.  Empty file?  We are done.
            return metadata[0]
        else:
            metadata.append(str(self.codestream))
        return '\n'.join(metadata)

    def get_tilewriters(self):
        """
        Return an object that facilitates writing tile by tile.
        """

        if self.shape[:2] == self.tilesize:
            msg = (
                'Do not write an image tile-by-tile '
                'if there is only one tile in the first place.  '
                'See issue #586'
            )
            raise RuntimeError(msg)

        return _TileWriter(self)

    def parse(self):
        """Parses the JPEG 2000 file.

        Raises
        ------
        RuntimeError
            The file was not JPEG 2000.
        """
        self.length = self.path.stat().st_size

        with self.path.open('rb') as fptr:

            # Make sure we have a JPEG2000 file.  It could be either JP2 or
            # J2C.  Check for J2C first, single box in that case.
            read_buffer = fptr.read(2)
            signature, = struct.unpack('>H', read_buffer)
            if signature == 0xff4f:
                self._codec_format = opj2.CODEC_J2K
                # That's it, we're done.  The codestream object is only
                # produced upon explicit request.
                return

            self._codec_format = opj2.CODEC_JP2

            # Should be JP2.
            # First 4 bytes should be 12, the length of the 'jP  ' box.
            # 2nd 4 bytes should be the box ID ('jP  ').
            # 3rd 4 bytes should be the box signature (13, 10, 135, 10).
            fptr.seek(0)
            read_buffer = fptr.read(12)
            values = struct.unpack('>I4s4B', read_buffer)
            box_length = values[0]
            box_id = values[1]
            signature = values[2:]

            if (
                box_length != 12
                or box_id != b'jP  '
                or signature != (13, 10, 135, 10)
            ):
                msg = f'{self.filename} is not a JPEG 2000 file.'
                raise InvalidJp2kError(msg)

            # Back up and start again, we know we have a superbox (box of
            # boxes) here.
            fptr.seek(0)
            self.box = self.parse_superbox(fptr)
            self._validate()

    def _validate(self):
        """
        Validate the JPEG 2000 outermost superbox.  These checks must be done
        at a file level.
        """
        # A JP2 file must contain certain boxes.  The 2nd box must be a file
        # type box.
        if not isinstance(self.box[1], FileTypeBox):
            msg = f"{self.filename} does not contain a valid File Type box."
            raise InvalidJp2kError(msg)

        ftyp = self.box[1]
        if ftyp.brand != 'jp2 ':
            # Don't bother trying to validate JPX.
            return

        jp2h = [box for box in self.box if box.box_id == 'jp2h'][0]

        # An IHDR box is required as the first child box of the JP2H box.
        if jp2h.box[0].box_id != 'ihdr':
            msg = "A valid IHDR box was not found.  The JP2 file is invalid."
            raise InvalidJp2kError(msg)

        # A jp2-branded file cannot contain an "any ICC profile
        colrs = [box for box in jp2h.box if box.box_id == 'colr']
        for colr in colrs:
            if colr.method not in (core.ENUMERATED_COLORSPACE,
                                   core.RESTRICTED_ICC_PROFILE):
                msg = (
                    "Color Specification box method must specify either an "
                    "enumerated colorspace or a restricted ICC profile if the "
                    "file type box brand is 'jp2 '."
                )
                warnings.warn(msg, UserWarning)

        # We need to have one and only one JP2H box if we have a JP2 file.
        num_jp2h_boxes = len([box for box in self.box if box.box_id == 'jp2h'])
        if num_jp2h_boxes > 1:
            msg = (
                f"This file has {num_jp2h_boxes} JP2H boxes in the outermost "
                "layer of boxes.  There should only be one."
            )
            warnings.warn(msg)

        # We should have one and only one JP2C box if we have a JP2 file.
        num_jp2c_boxes = len([box for box in self.box if box.box_id == 'jp2c'])
        if num_jp2c_boxes > 1 and self.box[1].brand == 'jp2 ':
            msg = (
                f"This file claims to be JP2 but has {num_jp2c_boxes} JP2C "
                "boxes in the outermost layer of boxes.  All JP2C boxes after "
                "the first will be ignored."
            )
            warnings.warn(msg)
        elif num_jp2c_boxes == 0:
            msg = (
                "A valid JP2C box was not found in the outermost level of JP2 "
                "boxes.  The JP2 file is invalid."
            )
            raise InvalidJp2kError(msg)

        # Make sure that IHDR and SIZ conform on the dimensions.
        ihdr = jp2h.box[0]
        ihdr_dims = ihdr.height, ihdr.width, ihdr.num_components

        siz = [
            segment for segment in self.codestream.segment
            if segment.marker_id == 'SIZ'
        ][0]

        siz_dims = (siz.ysiz, siz.xsiz, len(siz.bitdepth))
        if ihdr_dims != siz_dims:
            msg = (
                f"The IHDR dimensions {ihdr_dims} do not match the codestream "
                f"dimensions {siz_dims}."
            )
            warnings.warn(msg, UserWarning)

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

        if cinema_mode == 'cinema2k':
            if fps not in [24, 48]:
                msg = 'Cinema2K frame rate must be either 24 or 48.'
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
        """
        Directs processing of write method arguments.

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
        outfile += b'0' * num_pad_bytes
        cparams.outfile = outfile

        cparams.codec_fmt = self._codec_format

        cparams.irreversible = 1 if self._irreversible else 0

        if self._cinema2k is not None:
            self._cparams = cparams
            self._set_cinema_params('cinema2k', self._cinema2k)

        if self._cinema4k is not None:
            self._cparams = cparams
            self._set_cinema_params('cinema4k', self._cinema4k)

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
            for shift in range(6):
                power_of_two = 1 << shift
                if self._modesw & power_of_two:
                    cparams.mode |= power_of_two

        if self._numres is not None:
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

        if self._tilesize is not None:
            cparams.cp_tdx = self._tilesize[1]
            cparams.cp_tdy = self._tilesize[0]
            cparams.tile_size_on = opj2.TRUE

        if self._mct is None:
            # If the multi component transform was not specified, we infer
            # that it should be used if the color space is RGB.
            cparams.tcp_mct = 1 if self._colorspace == opj2.CLRSPC_SRGB else 0
        else:
            if self._colorspace == opj2.CLRSPC_GRAY:
                msg = (
                    "Cannot specify usage of the multi component transform "
                    "if the colorspace is gray."
                )
                raise InvalidJp2kError(msg)
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
        if version.openjpeg_version < '2.3.0':
            msg = (
                "You must have at least version 2.3.0 of OpenJPEG in order to "
                "write images."
            )
            raise RuntimeError(msg)

        if hasattr(self, '_cparams'):
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

    def _validate_codeblock_size(self, cparams):
        """
        Code block dimensions must satisfy certain restrictions.

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
            if (
                np.log2(height) != np.floor(np.log2(height))
                or np.log2(width) != np.floor(np.log2(width))
            ):
                msg = (
                    f"Bad code block size ({height} x {width}).  "
                    f"The dimensions must be powers of 2."
                )
                raise InvalidJp2kError(msg)

    def _validate_precinct_size(self, cparams):
        """
        Precinct dimensions must satisfy certain restrictions if specified.

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
        """
        Images must be either 2D or 3D.
        """
        if img_array.ndim == 1 or img_array.ndim > 3:
            msg = f"{img_array.ndim}D imagery is not allowed."
            raise InvalidJp2kError(msg)

    def _validate_image_datatype(self, img_array):
        """
        Only uint8 and uint16 images are currently supported.
        """
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
        """Determine the colorspace from the supplied inputs.
        """
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
            if self._colorspace.lower() not in ('rgb', 'grey', 'gray'):
                msg = f'Invalid colorspace "{self._colorspace}".'
                raise InvalidJp2kError(msg)
            elif self._colorspace.lower() == 'rgb' and self.shape[2] < 3:
                msg = 'RGB colorspace requires at least 3 components.'
                raise InvalidJp2kError(msg)

            # Turn the colorspace from a string to the enumerated value that
            # the library expects.
            COLORSPACE_MAP = {
                'rgb': opj2.CLRSPC_SRGB,
                'gray': opj2.CLRSPC_GRAY,
                'grey': opj2.CLRSPC_GRAY,
                'ycc': opj2.CLRSPC_YCC
            }

            self._colorspace = COLORSPACE_MAP[self._colorspace.lower()]

    def _write_openjp2(self, img_array):
        """
        Write JPEG 2000 file using OpenJPEG 2.x interface.
        """
        with ExitStack() as stack:
            image = opj2.image_create(self._comptparms, self._colorspace)
            stack.callback(opj2.image_destroy, image)

            self._populate_image_struct(image, img_array)

            codec = opj2.create_compress(self._cparams.codec_fmt)
            stack.callback(opj2.destroy_codec, codec)

            if self._verbose:
                info_handler = _INFO_CALLBACK
            else:
                info_handler = None

            opj2.set_info_handler(codec, info_handler)
            opj2.set_warning_handler(codec, _WARNING_CALLBACK)
            opj2.set_error_handler(codec, _ERROR_CALLBACK)

            opj2.setup_encoder(codec, self._cparams, image)

            if self._plt:
                opj2.encoder_set_extra_options(codec, plt=self._plt)

            if self._tlm:
                opj2.encoder_set_extra_options(codec, tlm=self._tlm)

            strm = opj2.stream_create_default_file_stream(self.filename, False)

            num_threads = get_option('lib.num_threads')
            if version.openjpeg_version >= '2.4.0':
                opj2.codec_set_threads(codec, num_threads)
            elif num_threads > 1:
                msg = (
                    f'Threaded encoding is not supported in library versions '
                    f'prior to 2.4.0.  Your version is '
                    f'{version.openjpeg_version}.'
                )
                warnings.warn(msg, UserWarning)

            stack.callback(opj2.stream_destroy, strm)

            opj2.start_compress(codec, image, strm)
            opj2.encode(codec, strm)
            opj2.end_compress(codec, strm)

    def append(self, box):
        """Append a JP2 box to the file in-place.

        Parameters
        ----------
        box : Jp2Box
            Instance of a JP2 box.  Only UUID and XML boxes can currently be
            appended.
        """
        if self._codec_format == opj2.CODEC_J2K:
            msg = "Only JP2 files can currently have boxes appended to them."
            raise RuntimeError(msg)

        box_is_xml = box.box_id == 'xml '
        box_is_xmp = (
            box.box_id == 'uuid'
            and box.uuid == UUID('be7acfcb-97a9-42e8-9c71-999491e3afac')
        )
        if not (box_is_xml or box_is_xmp):
            msg = (
                "Only XML boxes and XMP UUID boxes can currently be appended."
            )
            raise RuntimeError(msg)

        # Check the last box.  If the length field is zero, then rewrite
        # the length field to reflect the true length of the box.
        with self.path.open('rb') as ifile:
            offset = self.box[-1].offset
            ifile.seek(offset)
            read_buffer = ifile.read(4)
            box_length, = struct.unpack('>I', read_buffer)
            if box_length == 0:
                # Reopen the file in write mode and rewrite the length field.
                true_box_length = self.path.stat().st_size - offset
                with self.path.open('r+b') as ofile:
                    ofile.seek(offset)
                    write_buffer = struct.pack('>I', true_box_length)
                    ofile.write(write_buffer)

        # Can now safely append the box.
        with self.path.open('ab') as ofile:
            box.write(ofile)

        self.parse()

    def wrap(self, filename, boxes=None):
        """Create a new JP2/JPX file wrapped in a new set of JP2 boxes.

        This method is primarily aimed at wrapping a raw codestream in a set of
        of JP2 boxes (turning it into a JP2 file instead of just a raw
        codestream), or rewrapping a codestream in a JP2 file in a new "jacket"
        of JP2 boxes.

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
        >>> import glymur, tempfile
        >>> jfile = glymur.data.goodstuff()
        >>> j2k = glymur.Jp2k(jfile)
        >>> tfile = tempfile.NamedTemporaryFile(suffix='jp2')
        >>> jp2 = j2k.wrap(tfile.name)
        """
        if boxes is None:
            boxes = self._get_default_jp2_boxes()

        self._validate_jp2_box_sequence(boxes)

        with open(filename, 'wb') as ofile:
            for box in boxes:
                if box.box_id != 'jp2c':
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
            ofile.write(struct.pack('>I', self.length + 8))
            ofile.write(b'jp2c')
            with open(self.filename, 'rb') as ifile:
                ofile.write(ifile.read())
            return

        # OK, I'm a jp2/jpx file.  Need to find out where the raw codestream
        # actually starts.
        offset = box.offset
        if offset == -1:
            if self.box[1].brand == 'jpx ':
                msg = (
                    "The codestream box must have its offset and length "
                    "attributes fully specified if the file type brand is JPX."
                )
                raise InvalidJp2kError(msg)

            # Find the first codestream in the file.
            jp2c = [_box for _box in self.box if _box.box_id == 'jp2c']
            offset = jp2c[0].offset

        # Ready to write the codestream.
        with open(self.filename, 'rb') as ifile:
            ifile.seek(offset)

            # Verify that the specified codestream is right.
            read_buffer = ifile.read(8)
            L, T = struct.unpack_from('>I4s', read_buffer, 0)
            if T != b'jp2c':
                msg = "Unable to locate the specified codestream."
                raise InvalidJp2kError(msg)
            if L == 0:
                # The length of the box is presumed to last until the end of
                # the file.  Compute the effective length of the box.
                L = self.path.stat().st_size - ifile.tell() + 8

            elif L == 1:
                # The length of the box is in the XL field, a 64-bit value.
                read_buffer = ifile.read(8)
                L, = struct.unpack('>Q', read_buffer)

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
            ContiguousCodestreamBox()
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
                jp2hs = [box for box in self.box if box.box_id == 'jp2h']
                colorspace = jp2hs[0].box[1].colorspace

        boxes[2].box = [
            ImageHeaderBox(
                height=height, width=width, num_components=num_components
            ),
            ColourSpecificationBox(colorspace=colorspace)
        ]

        return boxes

    def __setitem__(self, index, data):
        """
        Slicing protocol.
        """
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
        """
        resolve the first ellipsis in the index so that it references the image

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

    def __getitem__(self, pargs):
        """
        Slicing protocol.
        """
        if not self.path.exists():
            msg = f"Cannot read from {self.filename}, it does not yet exist."
            raise FileNotFoundError(msg)
        if len(self.shape) == 2:
            numrows, numcols = self.shape
            numbands = 1
        else:
            numrows, numcols, numbands = self.shape

        if isinstance(pargs, int):
            # Not a very good use of this protocol, but technically legal.
            # This retrieves a single row.
            row = pargs
            area = (row, 0, row + 1, numcols)
            return self._read(area=area).squeeze()

        if pargs is Ellipsis:
            # Case of jp2[...]
            return self._read()

        if isinstance(pargs, slice):
            if (
                pargs.start is None
                and pargs.stop is None
                and pargs.step is None
            ):
                # Case of jp2[:]
                return self._read()

            # Corner case of jp2[x] where x is a slice object with non-null
            # members.  Just augment it with an ellipsis and let the code
            # below handle it.
            pargs = (pargs, Ellipsis)

        if isinstance(pargs, tuple) and any(x is Ellipsis for x in pargs):
            newindex = self._remove_ellipsis(pargs, numrows, numcols, numbands)

            # Run once again because it is possible that there's another
            # Ellipsis object in the 2nd or 3rd position.
            return self.__getitem__(newindex)

        if isinstance(pargs, tuple) and any(isinstance(x, int) for x in pargs):
            # Replace the first such integer argument, replace it with a slice.
            lst = list(pargs)
            g = filterfalse(lambda x: not isinstance(x[1], int),
                            enumerate(pargs))
            idx = next(g)[0]
            lst[idx] = slice(pargs[idx], pargs[idx] + 1)
            newindex = tuple(lst)

            # Invoke array-based slicing again, as there may be additional
            # integer argument remaining.
            data = self.__getitem__(newindex)

            # Reduce dimensionality in the scalar dimension.
            return np.squeeze(data, axis=idx)

        # Assuming pargs is a tuple of slices from now on.
        rows = pargs[0]
        cols = pargs[1]
        if len(pargs) == 2:
            bands = slice(None, None, None)
        else:
            bands = pargs[2]

        rows_step = 1 if rows.step is None else rows.step
        cols_step = 1 if cols.step is None else cols.step
        if rows_step != cols_step:
            msg = "Row and column strides must be the same."
            raise ValueError(msg)

        # Ok, reduce layer step is the same in both xy directions, so just take
        # one of them.
        step = rows_step
        if step == -1:
            # This is a shortcut for the last decomposition (or reduce layer
            # step).
            step = 2 ** self.codestream.segment[2].num_res

        # Check if the step size is a power of 2.
        if np.abs(np.log2(step) - np.round(np.log2(step))) > 1e-6:
            msg = "Row and column strides must be powers of 2."
            raise ValueError(msg)
        rlevel = int(np.round(np.log2(step)))

        area = (
            0 if rows.start is None else rows.start,
            0 if cols.start is None else cols.start,
            numrows if rows.stop is None else rows.stop,
            numcols if cols.stop is None else cols.stop
        )
        data = self._read(area=area, rlevel=rlevel)
        if len(pargs) == 2:
            return data

        # Ok, 3 arguments in pargs.
        return data[:, :, bands]

    def _read(self, **kwargs):
        """Read a JPEG 2000 image.

        Returns
        -------
        ndarray
            The image data.

        Raises
        ------
        RuntimeError
            if the proper version of the OpenJPEG library is not available
        """
        if re.match("0|1|2.[012]", version.openjpeg_version):
            msg = (
                f"You must have a version of OpenJPEG at least as high as "
                f"2.3.0 before you can read JPEG2000 images with glymur.  "
                f"Your version is {version.openjpeg_version}"
            )
            raise RuntimeError(msg)

        img = self._read_openjp2(**kwargs)
        return img

    def read(self, **kwargs):
        """
        Read a JPEG 2000 image.

        Returns
        -------
        img_array : ndarray
            The image data.
        """

        if 'ignore_pclr_cmap_cdef' in kwargs:
            self.ignore_pclr_cmap_cdef = kwargs['ignore_pclr_cmap_cdef']
            kwargs.pop('ignore_pclr_cmap_cdef')
        warnings.warn("Use array-style slicing instead.", DeprecationWarning)
        img = self._read(**kwargs)
        return img

    def _subsampling_sanity_check(self):
        """Check for differing subsample factors.
        """
        if self._decoded_components is None:
            dxs = np.array(self.codestream.segment[1].xrsiz)
            dys = np.array(self.codestream.segment[1].yrsiz)
        else:
            dxs = np.array([
                self.codestream.segment[1].xrsiz[i]
                for i in self._decoded_components
            ])
            dys = np.array([
                self.codestream.segment[1].yrsiz[i]
                for i in self._decoded_components
            ])

        if np.any(dxs - dxs[0]) or np.any(dys - dys[0]):
            msg = (
                f"The read_bands method should be used when the subsampling "
                f"factors are different."
                f"\n\n"
                f"{self.codestream.segment[1]}"
            )
            raise RuntimeError(msg)

    def _read_openjp2(self, rlevel=0, layer=None, area=None, tile=None,
                      verbose=False):
        """Read a JPEG 2000 image using libopenjp2.

        Parameters
        ----------
        layer : int, optional
            Number of quality layer to decode.
        rlevel : int, optional
            Factor by which to rlevel output resolution.  Use -1 to get the
            lowest resolution thumbnail.
        area : tuple, optional
            Specifies decoding image area,
            (first_row, first_col, last_row, last_col)
        tile : int, optional
            Number of tile to decode.
        verbose : bool, optional
            Print informational messages produced by the OpenJPEG library.

        Returns
        -------
        ndarray
            The image data.

        Raises
        ------
        RuntimeError
            If the image has differing subsample factors.
        """
        self._subsampling_sanity_check()
        self._populate_dparams(rlevel, tile=tile, area=area)
        image = self._read_openjp2_common()
        return image

    def _read_openjp2_common(self):
        """
        Read a JPEG 2000 image using libopenjp2.

        Returns
        -------
        ndarray or lst
            Either the image as an ndarray or a list of ndarrays, each item
            corresponding to one band.
        """
        with ExitStack() as stack:
            filename = self.filename
            stream = opj2.stream_create_default_file_stream(filename, True)
            stack.callback(opj2.stream_destroy, stream)
            codec = opj2.create_decompress(self._codec_format)
            stack.callback(opj2.destroy_codec, codec)

            opj2.set_error_handler(codec, _ERROR_CALLBACK)
            opj2.set_warning_handler(codec, _WARNING_CALLBACK)

            if self._verbose:
                opj2.set_info_handler(codec, _INFO_CALLBACK)
            else:
                opj2.set_info_handler(codec, None)

            opj2.setup_decoder(codec, self._dparams)
            if version.openjpeg_version >= '2.2.0':
                opj2.codec_set_threads(codec, get_option('lib.num_threads'))

            raw_image = opj2.read_header(stream, codec)
            stack.callback(opj2.image_destroy, raw_image)

            if self._decoded_components is not None:
                opj2.set_decoded_components(codec, self._decoded_components)

            if self._dparams.nb_tile_to_decode:
                opj2.get_decoded_tile(codec, stream, raw_image,
                                      self._dparams.tile_index)
            else:
                opj2.set_decode_area(
                    codec, raw_image,
                    self._dparams.DA_x0, self._dparams.DA_y0,
                    self._dparams.DA_x1, self._dparams.DA_y1
                )
                opj2.decode(codec, stream, raw_image)

            opj2.end_decompress(codec, stream)

            image = self._extract_image(raw_image)

        return image

    def _populate_dparams(self, rlevel, tile=None, area=None):
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
        infile += b'0' * nelts
        dparam.infile = infile

        # Return raw codestream components instead of "interpolating" the
        # colormap?
        dparam.flags |= 1 if self.ignore_pclr_cmap_cdef else 0

        dparam.decod_format = self._codec_format
        dparam.cp_layer = self.layer

        # Must check the specified rlevel against the maximum.
        if rlevel != 0:
            # Must check the specified rlevel against the maximum.
            cod_seg = [
                segment for segment in self.codestream.segment
                if segment.marker_id == 'COD'
            ][0]
            max_rlevel = cod_seg.num_res
            if rlevel == -1:
                # -1 is shorthand for the largest rlevel
                rlevel = max_rlevel
            elif rlevel < -1 or rlevel > max_rlevel:
                msg = (f"rlevel must be in the range [-1, {max_rlevel}] "
                       "for this image.")
                raise ValueError(msg)

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

    def read_bands(self, rlevel=0, layer=0, area=None, tile=None,
                   verbose=False, ignore_pclr_cmap_cdef=False):
        """Read a JPEG 2000 image.

        The only time you should use this method is when the image has
        different subsampling factors across components.  Otherwise you should
        use the read method.

        Parameters
        ----------
        layer : int, optional
            Number of quality layer to decode.
        rlevel : int, optional
            Factor by which to rlevel output resolution.
        area : tuple, optional
            Specifies decoding image area,
            (first_row, first_col, last_row, last_col)
        tile : int, optional
            Number of tile to decode.
        ignore_pclr_cmap_cdef : bool
            Whether or not to ignore the pclr, cmap, or cdef boxes during any
            color transformation.  Defaults to False.
        verbose : bool, optional
            Print informational messages produced by the OpenJPEG library.

        Returns
        -------
        list
            List of the individual image components.

        See also
        --------
        read : read JPEG 2000 image

        Examples
        --------
        >>> import glymur
        >>> jfile = glymur.data.nemo()
        >>> jp = glymur.Jp2k(jfile)
        >>> components_lst = jp.read_bands(rlevel=1)
        """
        if version.openjpeg_version < '2.3.0':
            msg = (
                f"You must have at least version 2.3.0 of OpenJPEG installed "
                f"before using this method.  Your version of OpenJPEG is "
                f"{version.openjpeg_version}."
            )
            raise RuntimeError(msg)

        self.ignore_pclr_cmap_cdef = ignore_pclr_cmap_cdef
        self.layer = layer
        self._populate_dparams(rlevel, tile=tile, area=area)
        lst = self._read_openjp2_common()
        return lst

    def _extract_image(self, raw_image):
        """
        Extract unequally-sized image bands.

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
            dtypes.append(self._component2dtype(component))
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

            self._validate_nonzero_image_size(nrows[k], ncols[k], k)

            addr = ctypes.addressof(component.data.contents)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nelts = nrows[k] * ncols[k]
                band = np.ctypeslib.as_array(
                    (ctypes.c_int32 * nelts).from_address(addr))
                if is_cube:
                    image[:, :, k] = np.reshape(band.astype(dtypes[k]),
                                                (nrows[k], ncols[k]))
                else:
                    image.append(np.reshape(band.astype(dtypes[k]),
                                 (nrows[k], ncols[k])))

        if is_cube and image.shape[2] == 1:
            # The third dimension has just a single layer.  Make the image
            # data 2D instead of 3D.
            image.shape = image.shape[0:2]

        return image

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

    def get_codestream(self, header_only=True):
        """Retrieve codestream.

        Parameters
        ----------
        header_only : bool, optional
            If True, only marker segments in the main header are parsed.
            Supplying False may impose a large performance penalty.

        Returns
        -------
        Codestream
            Object describing the codestream syntax.

        Examples
        --------
        >>> import glymur
        >>> jfile = glymur.data.nemo()
        >>> jp2 = glymur.Jp2k(jfile)
        >>> codestream = jp2.get_codestream()
        >>> print(codestream.segment[1])
        SIZ marker segment @ (3233, 47)
            Profile:  no profile
            Reference Grid Height, Width:  (1456 x 2592)
            Vertical, Horizontal Reference Grid Offset:  (0 x 0)
            Reference Tile Height, Width:  (1456 x 2592)
            Vertical, Horizontal Reference Tile Offset:  (0 x 0)
            Bitdepth:  (8, 8, 8)
            Signed:  (False, False, False)
            Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
        """
        with self.path.open('rb') as fptr:
            if self._codec_format == opj2.CODEC_J2K:
                codestream = Codestream(fptr, self.length,
                                        header_only=header_only)
            else:
                box = [x for x in self.box if x.box_id == 'jp2c']
                fptr.seek(box[0].offset)
                read_buffer = fptr.read(8)
                (box_length, _) = struct.unpack('>I4s', read_buffer)
                if box_length == 0:
                    # The length of the box is presumed to last until the end
                    # of the file.  Compute the effective length of the box.
                    box_length = self.path.stat().st_size - fptr.tell() + 8
                elif box_length == 1:
                    # Seek past the XL field.
                    read_buffer = fptr.read(8)
                    box_length, = struct.unpack('>Q', read_buffer)
                codestream = Codestream(fptr, box_length - 8,
                                        header_only=header_only)

            return codestream

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
            if self._cparams.rsiz in (core.OPJ_PROFILE_CINEMA_2K,
                                      core.OPJ_PROFILE_CINEMA_4K):
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

    def _validate_nonzero_image_size(self, nrows, ncols, component_index):
        """The image cannot have area of zero.
        """
        if nrows == 0 or ncols == 0:
            # Letting this situation continue would segfault openjpeg.
            msg = (
                f"Component {component_index} has dimensions "
                f"{nrows} x {ncols}"
            )
            raise InvalidJp2kError(msg)

    def _validate_jp2_box_sequence(self, boxes):
        """Run through series of tests for JP2 box legality.

        This is non-exhaustive.
        """
        JP2_IDS = [
            'colr', 'cdef', 'cmap', 'jp2c', 'ftyp', 'ihdr', 'jp2h', 'jP  ',
            'pclr', 'res ', 'resc', 'resd', 'xml ', 'ulst', 'uinf', 'url ',
            'uuid'
        ]

        self._validate_signature_compatibility(boxes)
        self._validate_jp2h(boxes)
        self._validate_jp2c(boxes)
        if boxes[1].brand == 'jpx ':
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
        """
        Validate JP2 requirements on colour specification boxes.
        """
        lst = [box for box in boxes if box.box_id == 'jp2h']
        jp2h = lst[0]
        for colr in [box for box in jp2h.box if box.box_id == 'colr']:
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
        if boxes[0].box_id != 'jP  ' or boxes[1].box_id != 'ftyp':
            msg = (
                "The first box must be the signature box and the second must "
                "be the file type box."
            )
            raise InvalidJp2kError(msg)

        # The compatibility list must contain at a minimum 'jp2 '.
        if 'jp2 ' not in boxes[1].compatibility_list:
            msg = "The ftyp box must contain 'jp2 ' in the compatibility list."
            raise InvalidJp2kError(msg)

    def _validate_jp2c(self, boxes):
        """Validate the codestream box in relation to other boxes."""
        # jp2c must be preceeded by jp2h
        jp2h_lst = [idx for (idx, box) in enumerate(boxes)
                    if box.box_id == 'jp2h']
        jp2h_idx = jp2h_lst[0]
        jp2c_lst = [idx for (idx, box) in enumerate(boxes)
                    if box.box_id == 'jp2c']
        if len(jp2c_lst) == 0:
            msg = (
                "A codestream box must be defined in the outermost list of "
                "boxes."
            )
            raise InvalidJp2kError(msg)

        jp2c_idx = jp2c_lst[0]
        if jp2h_idx >= jp2c_idx:
            msg = "The codestream box must be preceeded by a jp2 header box."
            raise InvalidJp2kError(msg)

    def _validate_jp2h(self, boxes):
        """Validate the JP2 Header box."""
        self._check_jp2h_child_boxes(boxes, 'top-level')

        jp2h_lst = [box for box in boxes if box.box_id == 'jp2h']
        jp2h = jp2h_lst[0]

        # 1st jp2 header box cannot be empty.
        if len(jp2h.box) == 0:
            msg = "The JP2 header superbox cannot be empty."
            raise InvalidJp2kError(msg)

        # 1st jp2 header box must be ihdr
        if jp2h.box[0].box_id != 'ihdr':
            msg = (
                "The first box in the jp2 header box must be the image header "
                "box."
            )
            raise InvalidJp2kError(msg)

        # colr must be present in jp2 header box.
        colr_lst = [
            j for (j, box) in enumerate(jp2h.box) if box.box_id == 'colr'
        ]
        if len(colr_lst) == 0:
            msg = "The jp2 header box must contain a color definition box."
            raise InvalidJp2kError(msg)
        colr = jp2h.box[colr_lst[0]]

        self._validate_channel_definition(jp2h, colr)

    def _validate_channel_definition(self, jp2h, colr):
        """Validate the channel definition box."""
        cdef_lst = [j for (j, box) in enumerate(jp2h.box)
                    if box.box_id == 'cdef']
        if len(cdef_lst) > 1:
            msg = ("Only one channel definition box is allowed in the "
                   "JP2 header.")
            raise InvalidJp2kError(msg)
        elif len(cdef_lst) == 1:
            cdef = jp2h.box[cdef_lst[0]]
            if colr.colorspace == core.SRGB:
                if any([
                    chan + 1 not in cdef.association
                    or cdef.channel_type[chan] != 0
                    for chan in [0, 1, 2]
                ]):
                    msg = ("All color channels must be defined in the "
                           "channel definition box.")
                    raise InvalidJp2kError(msg)
            elif colr.colorspace == core.GREYSCALE:
                if 0 not in cdef.channel_type:
                    msg = ("All color channels must be defined in the "
                           "channel definition box.")
                    raise InvalidJp2kError(msg)

    def _check_jp2h_child_boxes(self, boxes, parent_box_name):
        """Certain boxes can only reside in the JP2 header."""
        JP2H_CHILDREN = set(['bpcc', 'cdef', 'cmap', 'ihdr', 'pclr'])

        box_ids = set([box.box_id for box in boxes])
        intersection = box_ids.intersection(JP2H_CHILDREN)
        if len(intersection) > 0 and parent_box_name not in ['jp2h', 'jpch']:
            msg = (
                f"A {list(intersection)[0]} box can only be nested in a JP2 "
                f"header box."
            )
            raise InvalidJp2kError(msg)

        # Recursively check any contained superboxes.
        for box in boxes:
            if hasattr(box, 'box'):
                self._check_jp2h_child_boxes(box.box, box.box_id)

    def _collect_box_count(self, boxes):
        """Count the occurences of each box type."""
        count = Counter([box.box_id for box in boxes])

        # Add the counts in the superboxes.
        for box in boxes:
            if hasattr(box, 'box'):
                count.update(self._collect_box_count(box.box))

        return count

    def _check_superbox_for_top_levels(self, boxes):
        """Several boxes can only occur at the top level."""
        # We are only looking at the boxes contained in a superbox, so if any
        # of the blacklisted boxes show up here, it's an error.
        TOP_LEVEL_ONLY_BOXES = set(['dtbl'])
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
            if hasattr(box, 'box'):
                self._check_superbox_for_top_levels(box.box)

    def _validate_top_level(self, boxes):
        """Several boxes can only occur at the top level."""
        # Add the counts in the superboxes.
        for box in boxes:
            if hasattr(box, 'box'):
                self._check_superbox_for_top_levels(box.box)

        count = self._collect_box_count(boxes)

        # If there is one data reference box, then there must also be one ftbl.
        if 'dtbl' in count and 'ftbl' not in count:
            msg = ('The presence of a data reference box requires the '
                   'presence of a fragment table box as well.')
            raise InvalidJp2kError(msg)

    def _validate_singletons(self, boxes):
        """Several boxes can only occur once."""
        count = self._collect_box_count(boxes)
        # Which boxes occur more than once?
        multiples = [box_id for box_id, bcount in count.items() if bcount > 1]
        if 'dtbl' in multiples:
            raise InvalidJp2kError('There can only be one dtbl box in a file.')

    def _validate_jpx_compatibility(self, boxes, compatibility_list):
        """
        If there is a JPX box then the compatibility list must also contain
        'jpx '.
        """
        JPX_IDS = ['asoc', 'nlst']
        jpx_cl = set(compatibility_list)
        for box in boxes:
            if box.box_id in JPX_IDS:
                if len(set(['jpx ', 'jpxb']).intersection(jpx_cl)) == 0:
                    msg = (
                        "A JPX box requires that either 'jpx ' or 'jpxb' be "
                        "present in the ftype compatibility list."
                    )
                    raise InvalidJp2kError(msg)
            if hasattr(box, 'box') != 0:
                # Same set of checks on any child boxes.
                self._validate_jpx_compatibility(box.box, compatibility_list)

    def _validate_label(self, boxes):
        """
        Label boxes can only be inside association, codestream headers, or
        compositing layer header boxes.
        """
        for box in boxes:
            if box.box_id != 'asoc':
                if hasattr(box, 'box'):
                    for boxi in box.box:
                        if boxi.box_id == 'lbl ':
                            msg = (
                                f"A label box cannot be nested inside a "
                                f"{box.box_id} box."
                            )
                            raise InvalidJp2kError(msg)
                    # Same set of checks on any child boxes.
                    self._validate_label(box.box)


class _TileWriter(object):
    """
    Writes tiles to file, one by one.

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
        if version.openjpeg_version < '2.3.0':
            msg = ("You must have at least version 2.3.0 of OpenJPEG "
                   "in order to write images.")
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
                self.stream
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
        """
        Only do these things for the first tile.
        """
        self.jp2k._determine_colorspace()
        self.jp2k._populate_cparams(img_array)
        self.jp2k._populate_comptparms(img_array)

        self.codec = opj2.create_compress(self.jp2k._cparams.codec_fmt)

        if self.jp2k.verbose:
            info_handler = _INFO_CALLBACK
        else:
            info_handler = None

        opj2.set_info_handler(self.codec, info_handler)
        opj2.set_warning_handler(self.codec, _WARNING_CALLBACK)
        opj2.set_error_handler(self.codec, _ERROR_CALLBACK)

        self.image = opj2.image_tile_create(
            self.jp2k._comptparms, self.jp2k._colorspace
        )

        self.jp2k._populate_image_struct(
            self.image, img_array,
            tile_x_factor=self.num_tile_cols,
            tile_y_factor=self.num_tile_rows
        )
        self.image.contents.x1 = self.jp2k.shape[1]
        self.image.contents.y1 = self.jp2k.shape[0]

        opj2.setup_encoder(self.codec, self.jp2k._cparams, self.image)

        if self.jp2k._plt:
            opj2.encoder_set_extra_options(self.codec, plt=self.jp2k._plt)

        self.stream = opj2.stream_create_default_file_stream(
            self.jp2k.filename, False
        )

        num_threads = get_option('lib.num_threads')
        if version.openjpeg_version >= '2.4.0':
            opj2.codec_set_threads(self.codec, num_threads)
        elif num_threads > 1:
            msg = (
                f'Threaded encoding is not supported in library versions '
                f'prior to 2.4.0.  Your version is '
                f'{version.openjpeg_version}.'
            )
            warnings.warn(msg, UserWarning)

        opj2.start_compress(self.codec, self.image, self.stream)


def _set_planar_pixel_order(img):
    """
    Reorder the image pixels so that plane-0 comes first, then plane-1, etc.
    This is a requirement for using opj_write_tile.
    """
    if img.ndim == 3:
        # C-order increments along the y-axis slowest (0), then x-axis (1),
        # then z-axis (2).  We want it to go along the z-axis slowest, then
        # y-axis, then x-axis.
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)

    return img.copy()


# Setup the default callback handlers.  See the callback functions subsection
# in the ctypes section of the Python documentation for a solid explanation of
# what's going on here.
_CMPFUNC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)


def _default_error_handler(msg, _):
    """Default error handler callback for libopenjp2."""
    msg = "OpenJPEG library error:  {0}".format(msg.decode('utf-8').rstrip())
    opj2.set_error_message(msg)


def _default_info_handler(msg, _):
    """Default info handler callback."""
    print("[INFO] {0}".format(msg.decode('utf-8').rstrip()))


def _default_warning_handler(library_msg, _):
    """Default warning handler callback."""
    library_msg = library_msg.decode('utf-8').rstrip()
    msg = "OpenJPEG library warning:  {0}".format(library_msg)
    warnings.warn(msg, UserWarning)


_ERROR_CALLBACK = _CMPFUNC(_default_error_handler)
_INFO_CALLBACK = _CMPFUNC(_default_info_handler)
_WARNING_CALLBACK = _CMPFUNC(_default_warning_handler)
