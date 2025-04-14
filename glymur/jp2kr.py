"""This file is part of glymur, a Python interface for accessing JPEG 2000.

http://glymur.readthedocs.org

Copyright 2013 John Evans

License:  MIT
"""

# Standard library imports...
from __future__ import annotations
from contextlib import ExitStack
import ctypes
import pathlib
import re
import struct
import sys
import warnings

# Third party library imports
import numpy as np

# Local imports...
from .codestream import Codestream
from . import core, version, get_option
from .jp2box import Jp2kBox, FileTypeBox, InvalidJp2kError, InvalidJp2kWarning
from .lib import openjp2 as opj2


class Jp2kr(Jp2kBox):
    """Read JPEG 2000 files.

    Attributes
    ----------
    box : sequence
        List of top-level boxes in the file.  Each box may in turn contain
        its own list of boxes.  Will be empty if the file consists only of a
        raw codestream.
    filename : str
        The path to the JPEG 2000 file.
    path : Path
        The path to the JPEG 2000 file.

    Examples
    --------
    >>> jfile = glymur.data.nemo()
    >>> jp2 = glymur.Jp2kr(jfile)
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

    >>> import time
    >>> jp2file = glymur.data.nemo()
    >>> jp2 = glymur.Jp2k(jp2file)
    >>> t0 = time.time(); data = jp2[:]; t1 = time.time()
    >>> t1 - t0 #doctest: +SKIP
    0.9024193286895752
    >>> glymur.set_option('lib.num_threads', 4)
    >>> t0 = time.time(); data = jp2[:]; t1 = time.time()
    >>> t1 - t0 #doctest: +SKIP
    0.4060473537445068
    """

    def __init__(
        self,
        filename: str | pathlib.Path,
        verbose: bool = False,
        **kwargs
    ):
        """
        Parameters
        ----------
        filename : str or Path
            Interpreted as a path to a JPEG 2000 file.
        verbose : bool
            If true, print informational messages produced by the OpenJPEG
            library.
        """
        super().__init__()

        # In case of pathlib.Paths...
        self.filename = str(filename)
        self.path = pathlib.Path(self.filename)

        # Setup some default attributes
        self.box = []
        self._codestream = None
        self._decoded_components = None
        self._dtype = None
        self._ignore_pclr_cmap_cdef = False
        self._layer = 0
        self._ndim = None
        self._parse_count = 0
        self._verbose = verbose
        self._tilesize_r = None

        if not self.path.exists():
            raise FileNotFoundError(f"{self.filename} does not exist.")

        self._parse()
        self._initialize_shape()

    def _initialize_shape(self):
        """If there was no image data provided and if no shape was
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
            jp2h = next(filter(lambda x: x.box_id == "jp2h", self.box), None)
            ihdr = next(filter(lambda x: x.box_id == "ihdr", jp2h.box), None)

            height, width = ihdr.height, ihdr.width
            num_components = ihdr.num_components

            if num_components == 1:
                # but if there is a PCLR box, then we need to check
                # that as well, as that turns a single-channel image
                # into a multi-channel image
                pclr = [box for box in jp2h.box if box.box_id == "pclr"]
                if len(pclr) > 0:
                    num_components = len(pclr[0].signed)

        if num_components == 1:
            self.shape = (height, width)
        else:
            self.shape = (height, width, num_components)

        return self._shape

    @property
    def ignore_pclr_cmap_cdef(self):
        """If true, ignore the pclr, cmap, or cdef boxes during any
        color transformation.  Why would you wish to do that?  In the immortal
        words of Critical Drinker, don't know!

        Defaults to false.

        Examples
        --------
        >>> from glymur import Jp2kr
        >>> jpxfile = glymur.data.jpxfile()
        >>> j = Jp2kr(jpxfile)
        >>> d = j[:]
        >>> print(d.shape)
        (1024, 1024, 3)
        >>> j.ignore_pclr_cmap_cdef = True
        >>> d = j[:]
        >>> print(d.shape)
        (1024, 1024)
        """
        return self._ignore_pclr_cmap_cdef

    @ignore_pclr_cmap_cdef.setter
    def ignore_pclr_cmap_cdef(self, ignore_pclr_cmap_cdef):
        self._ignore_pclr_cmap_cdef = ignore_pclr_cmap_cdef

    @property
    def decoded_components(self):
        """If true, decode only these components.  The MCT will not be used.
        List or scalar or None (default).

        Examples
        --------
        >>> from glymur import Jp2kr
        >>> j = Jp2kr(glymur.data.nemo())
        >>> rgb = j[:]
        >>> print(rgb.shape)
        (1456, 2592, 3)
        >>> j.decoded_components = 0
        >>> comp0 = j[:]
        >>> print(comp0.shape)
        (1456, 2592)
        """
        return self._decoded_components

    @decoded_components.setter
    def decoded_components(self, components):

        if components is None:
            # This is ok.  It is a special case where we are restoring the
            # original behavior of reading all bands.
            self._decoded_components = components
            return

        if np.isscalar(components):
            # turn it into a list to be general
            components = [components]

        if any(x > len(self.codestream.segment[1].xrsiz) for x in components):

            msg = (
                f"{components} has at least one invalid component, "
                f"cannot be greater than "
                f"{len(self.codestream.segment[1].xrsiz)}."
            )
            raise ValueError(msg)

        elif any(x < 0 for x in components):

            msg = (
                f"{components} has at least one invalid component, "
                f"components cannot be negative."
            )
            raise ValueError(msg)

        self._decoded_components = components

    @property
    def layer(self):
        """Zero-based number of quality layer to decode.  Defaults to 0, the
        highest quality layer.
        """
        return self._layer

    @layer.setter
    def layer(self, layer):
        # Set to the indicated value so long as it is valid.
        cod = next(
            filter(lambda x: x.marker_id == "COD", self.codestream.segment),
            None
        )
        if layer < 0 or layer >= cod.layers:
            msg = f"Invalid layer number, must be in range [0, {cod.layers})."
            raise ValueError(msg)

        self._layer = layer

    @property
    def dtype(self):
        """Datatype of the image.

        Examples
        --------
        >>> from glymur import Jp2kr
        >>> jp2file = glymur.data.nemo()
        >>> j = Jp2kr(jp2file)
        >>> j.dtype
        <class 'numpy.uint8'>
        """

        if self._dtype is None:
            c = self.get_codestream()
            bps0 = c.segment[1].bitdepth[0]
            sgnd0 = c.segment[1].signed[0]

            if (
                not all(bitdepth == bps0 for bitdepth in c.segment[1].bitdepth)
                or not all(signed == sgnd0 for signed in c.segment[1].signed)
            ):
                msg = (
                    "The dtype property is only valid when all components "
                    "have the same bitdepth and sign. "
                    "\n\n"
                    f"{c.segment[1]}"
                )
                raise TypeError(msg)

            if bps0 <= 8:
                self._dtype = np.int8 if sgnd0 else np.uint8
            else:
                self._dtype = np.int16 if sgnd0 else np.uint16

        return self._dtype

    @property
    def ndim(self):
        """Number of image dimensions.

        Examples
        --------
        >>> from glymur import Jp2kr
        >>> jp2file = glymur.data.nemo()
        >>> j = Jp2kr(jp2file)
        >>> j.ndim
        3
        """
        return len(self.shape)

    @property
    def codestream(self):
        """Metadata for JP2 or J2K codestream header.

        Examples
        --------
        >>> from glymur import Jp2kr
        >>> jp2file = glymur.data.nemo()
        >>> c = Jp2kr(jp2file).codestream
        >>> print(c.segment[0])
        SOC marker segment @ (85, 0)
        >>> len(c.segment)
        5
        """
        if self._codestream is None:
            self._codestream = self.get_codestream(header_only=True)
        return self._codestream

    @property
    def tilesize(self):
        """Height and width of the image tiles.

        Examples
        --------
        >>> jp = glymur.Jp2kr(glymur.data.nemo())
        >>> print(jp.shape)
        (1456, 2592, 3)
        >>> print(jp.tilesize)
        (1456, 2592)
        """

        if not hasattr(self, '_tilesize_w') and self._tilesize_r is None:
            # file was opened as read-only case
            segment = self.codestream.segment[1]
            tilesize = segment.ytsiz, segment.xtsiz
        elif self._tilesize_w is None:
            # read-write case, but we are reading not writing
            segment = self.codestream.segment[1]
            tilesize = segment.ytsiz, segment.xtsiz
        else:
            # write-only case
            tilesize = self._tilesize_w

        return tilesize

    @property
    def verbose(self):
        """If true, print informational messages produced by the
        OpenJPEG library.  Defaults to false.

        Examples
        --------
        >>> import skimage
        >>> j = glymur.Jp2k('moon.jp2', tilesize=[256, 256], verbose=True)
        >>> j[:] = skimage.data.moon()
        [INFO] tile number 1 / 4
        [INFO] tile number 2 / 4
        [INFO] tile number 3 / 4
        [INFO] tile number 4 / 4
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @property
    def shape(self):
        """Dimensions of full resolution image.

        Examples
        --------
        >>> jp = glymur.Jp2kr(glymur.data.nemo())
        >>> print(jp.shape)
        (1456, 2592, 3)
        """
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    def __repr__(self):
        msg = f"glymur.Jp2kr('{self.path}')"
        return msg

    def __str__(self):
        metadata = [f"File:  {self.path.name}"]
        if len(self.box) > 0:
            for box in self.box:
                metadata.append(str(box))
        elif self._codestream is None and not self.path.exists():
            # No codestream either.  Empty file?  We are done.
            return metadata[0]
        else:
            # Just a codestream, so J2K
            metadata.append(str(self.codestream))
        return "\n".join(metadata)

    def parse(self, force=False):
        """
        .. deprecated:: 0.15.0
        """
        breakpoint()
        msg = "Deprecated, do not use."
        warnings.warn(msg, DeprecationWarning)
        self._parse(force=force)

    def _parse(self, force=False):
        """Parses the JPEG 2000 file.

        Parameters
        ----------
        force : bool
            If true, parse the file even if it has already been parsed once.

        Raises
        ------
        RuntimeError
            The file was not JPEG 2000.
        """
        if self._parse_count > 0 and not force:
            # don't parse more than once if we can help it
            return

        self.length = self.path.stat().st_size

        with self.path.open("rb") as fptr:

            # Make sure we have a JPEG2000 file.  It could be either JP2 or
            # J2C.  Check for J2C first, single box in that case.
            read_buffer = fptr.read(2)
            (signature,) = struct.unpack(">H", read_buffer)
            if signature == 0xFF4F:
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
            values = struct.unpack(">I4s4B", read_buffer)
            box_length = values[0]
            box_id = values[1]
            signature = values[2:]

            if (
                box_length != 12
                or box_id != b"jP  "
                or signature != (13, 10, 135, 10)
            ):
                msg = f"{self.filename} is not a JPEG 2000 file."
                raise InvalidJp2kError(msg)

            # Back up and start again, we know we have a superbox (box of
            # boxes) here.
            fptr.seek(0)
            self.box = self.parse_superbox(fptr)
            self._validate()

        self._parse_count += 1

    def _validate(self):
        """Validate the JPEG 2000 outermost superbox.  These checks must be
        done at a file level.
        """
        # A JP2 file must contain certain boxes.  The 2nd box must be a file
        # type box.
        if not isinstance(self.box[1], FileTypeBox):
            msg = f"{self.filename} does not contain a valid File Type box."
            raise InvalidJp2kError(msg)

        ftyp = self.box[1]
        if ftyp.brand != "jp2 ":
            # Don't bother trying to validate JPX.
            return

        jp2h = next(filter(lambda x: x.box_id == "jp2h", self.box), None)
        if jp2h is None:
            msg = (
                "No JP2 header box was located in the outermost jacket of "
                "boxes."
            )
            raise InvalidJp2kError(msg)

        # An IHDR box is required as the first child box of the JP2H box.
        if jp2h.box[0].box_id != "ihdr":
            msg = "A valid IHDR box was not found.  The JP2 file is invalid."
            raise InvalidJp2kError(msg)

        # A jp2-branded file cannot contain an "any ICC profile
        colrs = [box for box in jp2h.box if box.box_id == "colr"]
        for colr in colrs:
            if colr.method not in (
                core.ENUMERATED_COLORSPACE,
                core.RESTRICTED_ICC_PROFILE,
            ):
                msg = (
                    "Color Specification box method must specify either an "
                    "enumerated colorspace or a restricted ICC profile if the "
                    "file type box brand is 'jp2 '."
                )
                warnings.warn(msg, InvalidJp2kWarning)

        # We need to have one and only one JP2H box if we have a JP2 file.
        num_jp2h_boxes = len([box for box in self.box if box.box_id == "jp2h"])
        if num_jp2h_boxes > 1:
            msg = (
                f"This file has {num_jp2h_boxes} JP2H boxes in the outermost "
                "layer of boxes.  There should only be one."
            )
            warnings.warn(msg, InvalidJp2kWarning)

        # We should have one and only one JP2C box if we have a JP2 file.
        num_jp2c_boxes = len([box for box in self.box if box.box_id == "jp2c"])
        if num_jp2c_boxes > 1 and self.box[1].brand == "jp2 ":
            msg = (
                f"This file has {num_jp2c_boxes} JP2C boxes (images) in the "
                "outermost layer of boxes.  All JP2C boxes after the first "
                "will be ignored."
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

        siz = next(
            filter(lambda x: x.marker_id == "SIZ", self.codestream.segment),
            None
        )

        siz_dims = (siz.ysiz, siz.xsiz, len(siz.bitdepth))
        if ihdr_dims != siz_dims:
            msg = (
                f"The IHDR dimensions {ihdr_dims} do not match the codestream "
                f"dimensions {siz_dims}."
            )
            warnings.warn(msg, UserWarning)

    def __getitem__(self, pargs):
        """Slicing protocol."""
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
            idx, _ = next(
                filter(lambda x: isinstance(x[1], int), enumerate(lst)), None
            )
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
            numcols if cols.stop is None else cols.stop,
        )
        data = self._read(area=area, rlevel=rlevel)
        if len(pargs) == 2:
            return data

        # Ok, 3 arguments in pargs.
        return data[:, :, bands]

    def _subsampling_sanity_check(self):
        """Check for differing subsample factors."""
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

    def _read(self, rlevel=0, layer=None, area=None, tile=None, verbose=False):
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
        if re.match("0|1|2.[012]", version.openjpeg_version):
            msg = (
                f"You must have a version of OpenJPEG at least as high as "
                f"2.3.0 before you can read JPEG2000 images with glymur.  "
                f"Your version is {version.openjpeg_version}"
            )
            raise RuntimeError(msg)

        self._subsampling_sanity_check()
        self._populate_dparams(rlevel, tile=tile, area=area)
        image = self._read_openjp2()
        return image

    def _read_openjp2(self):
        """Read a JPEG 2000 image using libopenjp2.

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

            opj2.set_error_handler(codec, opj2._ERROR_CALLBACK)
            opj2.set_warning_handler(codec, opj2._WARNING_CALLBACK)

            if self._verbose:
                opj2.set_info_handler(codec, opj2._INFO_CALLBACK)
            else:
                opj2.set_info_handler(codec, None)

            opj2.setup_decoder(codec, self._dparams)
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
        infile += b"0" * nelts
        dparam.infile = infile

        # Return raw codestream components instead of "interpolating" the
        # colormap?
        dparam.flags |= 1 if self.ignore_pclr_cmap_cdef else 0

        dparam.decod_format = self._codec_format
        dparam.cp_layer = self.layer

        # Must check the specified rlevel against the maximum.
        if rlevel != 0:
            # Must check the specified rlevel against the maximum.
            cod_seg = next(
                filter(
                    lambda x: x.marker_id == "COD", self.codestream.segment
                ),
                None
            )
            max_rlevel = cod_seg.num_res
            if rlevel == -1:
                # -1 is shorthand for the largest rlevel
                rlevel = max_rlevel
            elif rlevel < -1 or rlevel > max_rlevel:
                msg = (
                    f"rlevel must be in the range [-1, {max_rlevel}] "
                    "for this image."
                )
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

    def read_bands(
        self,
        rlevel=0,
        layer=0,
        area=None,
        tile=None,
        verbose=False,
        ignore_pclr_cmap_cdef=False,
    ):
        """Read a JPEG 2000 image.

        The only time you should ever use this method is when the image has
        different subsampling factors across components.  Otherwise you should
        use numpy-style slicing.

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
        >>> jfile = glymur.data.nemo()
        >>> jp = glymur.Jp2k(jfile)
        >>> components_lst = jp.read_bands(rlevel=1)
        """
        if version.openjpeg_version < "2.3.0":
            msg = (
                f"You must have at least version 2.3.0 of OpenJPEG installed "
                f"before using this method.  Your version of OpenJPEG is "
                f"{version.openjpeg_version}."
            )
            raise RuntimeError(msg)

        self.ignore_pclr_cmap_cdef = ignore_pclr_cmap_cdef
        self.layer = layer
        self._populate_dparams(rlevel, tile=tile, area=area)
        lst = self._read_openjp2()
        return lst

    def _extract_image(self, raw_image):
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

    def _component2dtype(self, component):
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

    def get_codestream(self, header_only=True):
        """Retrieve codestream.

        This differs from the codestream property in that segment
        metadata that lies past the end of the codestream header
        can be retrieved.

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
        >>> jfile = glymur.data.nemo()
        >>> jp2 = glymur.Jp2k(jfile)
        >>> codestream = jp2.get_codestream(header_only=False)
        >>> print(codestream.segment[1])
        SIZ marker segment @ (87, 47)
            Profile:  no profile
            Reference Grid Height, Width:  (1456 x 2592)
            Vertical, Horizontal Reference Grid Offset:  (0 x 0)
            Reference Tile Height, Width:  (1456 x 2592)
            Vertical, Horizontal Reference Tile Offset:  (0 x 0)
            Bitdepth:  (8, 8, 8)
            Signed:  (False, False, False)
            Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
        >>> print(len(codestream.segment))
        12
        >>> print(codestream.segment[-1])
        EOC marker segment @ (1132371, 0)
        """
        with self.path.open("rb") as fptr:

            # if it's just a raw codestream file, it's easy
            if self._codec_format == opj2.CODEC_J2K:
                return self._get_codestream(fptr, self.length, header_only)

            # continue assuming JP2, must seek to the JP2C box and past its
            # header
            box = next(filter(lambda x: x.box_id == "jp2c", self.box), None)

            fptr.seek(box.offset)
            read_buffer = fptr.read(8)
            (box_length, _) = struct.unpack(">I4s", read_buffer)
            if box_length == 0:
                # The length of the box is presumed to last until the end
                # of the file.  Compute the effective length of the box.
                box_length = self.path.stat().st_size - fptr.tell() + 8
            elif box_length == 1:
                # Seek past the XL field.
                read_buffer = fptr.read(8)
                (box_length,) = struct.unpack(">Q", read_buffer)

            return self._get_codestream(fptr, box_length - 8, header_only)

    def _get_codestream(self, fptr, length, header_only):
        """
        Parsing errors can make for confusing errors sometimes, so catch any
        such error and add context to it.
        """

        try:
            codestream = Codestream(fptr, length, header_only=header_only)
        except Exception:
            _, value, traceback = sys.exc_info()
            msg = (
                f"The file is invalid "
                f'because the codestream could not be parsed:  "{value}"'
            )
            raise InvalidJp2kError(msg).with_traceback(traceback)
        else:
            return codestream

    def _validate_nonzero_image_size(self, nrows, ncols, component_index):
        """The image cannot have area of zero."""
        if nrows == 0 or ncols == 0:
            # Letting this situation continue would segfault openjpeg.
            msg = (
                f"Component {component_index} has dimensions "
                f"{nrows} x {ncols}"
            )
            raise InvalidJp2kError(msg)
