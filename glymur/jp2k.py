"""This file is part of glymur, a Python interface for accessing JPEG 2000.

http://glymur.readthedocs.org

Copyright 2013 John Evans

License:  MIT
"""

import sys

# Exitstack not found in contextlib in 2.7
# pylint: disable=E0611
if sys.hexversion >= 0x03030000:
    from contextlib import ExitStack
else:
    from contextlib2 import ExitStack

from collections import Counter
import ctypes
import math
import os
import re
import struct
from uuid import UUID
import warnings

import numpy as np

from .codestream import Codestream
from .core import SRGB, GREYSCALE
from .core import PROGRESSION_ORDER
from .core import ENUMERATED_COLORSPACE, RESTRICTED_ICC_PROFILE
from . import core
from .jp2box import Jp2kBox
from .jp2box import JPEG2000SignatureBox, FileTypeBox, JP2HeaderBox
from .jp2box import ColourSpecificationBox, ContiguousCodestreamBox
from .jp2box import ImageHeaderBox
from .lib import openjpeg as opj
from .lib import openjp2 as opj2
from . import version
from .lib import c as libc

JP2_IDS = ['colr', 'cdef', 'cmap', 'jp2c', 'ftyp', 'ihdr', 'jp2h', 'jP  ',
           'pclr', 'res ', 'resc', 'resd', 'xml ', 'ulst', 'uinf', 'url ',
           'uuid']
JPX_IDS = ['asoc', 'nlst']

class Jp2k(Jp2kBox):
    """JPEG 2000 file.

    Attributes
    ----------
    filename : str
        The path to the JPEG 2000 file.
    mode : str
        The mode used to open the file.
    box : sequence
        List of top-level boxes in the file.  Each box may in turn contain
        its own list of boxes.  Will be empty if the file consists only of a
        raw codestream.
    """

    def __init__(self, filename, mode='rb'):
        """
        Parameters
        ----------
        filename : str or file
            The path to JPEG 2000 file.
        mode : str, optional
            The mode used to open the file.
        """
        Jp2kBox.__init__(self)
        self.filename = filename
        self.mode = mode
        self.box = []
        self._codec_format = None

        # Parse the file for JP2/JPX contents only if we are reading it.
        if mode == 'rb':
            self.parse()

    def __repr__(self):
        msg = "glymur.Jp2k('{0}')".format(self.filename)
        return msg

    def __str__(self):
        metadata = ['File:  ' + os.path.basename(self.filename)]
        if len(self.box) > 0:
            for box in self.box:
                metadata.append(str(box))
        else:
            codestream = self.get_codestream()
            metadata.append(str(codestream))
        return '\n'.join(metadata)

    def parse(self):
        """Parses the JPEG 2000 file.

        Raises
        ------
        IOError
            The file was not JPEG 2000.
        """
        self.length = os.path.getsize(self.filename)

        with open(self.filename, 'rb') as fptr:

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
            if (((box_length != 12) or (box_id != b'jP  ') or
                 (signature != (13, 10, 135, 10)))):
                msg = '{0} is not a JPEG 2000 file.'.format(self.filename)
                raise IOError(msg)

            # Back up and start again, we know we have a superbox (box of
            # boxes) here.
            fptr.seek(0)
            self.box = self.parse_superbox(fptr)
            self._validate()

    def _validate(self):
        """Validate the JPEG 2000 outermost superbox.
        """
        # A jp2-branded file cannot contain an "any ICC profile
        ftyp = self.box[1]
        if ftyp.brand == 'jp2 ':
            jp2h = [box for box in self.box if box.box_id == 'jp2h'][0]
            colrs = [box for box in jp2h.box if box.box_id == 'colr']
            for colr in colrs:
                if colr.method not in (ENUMERATED_COLORSPACE,
                                       RESTRICTED_ICC_PROFILE):
                    msg = "Color Specification box method must specify either "
                    msg += "an enumerated colorspace or a restricted ICC "
                    msg += "profile if the file type box brand is 'jp2 '."
                    warnings.warn(msg)

    def _set_cinema_params(self, cparams, cinema_mode, fps):
        """Populate compression parameters structure for cinema2K.

        Parameters
        ----------
        params : ctypes struct
            Corresponds to compression parameters structure used by the
            library.
        cinema_mode : str
            Either 'cinema2k' or 'cinema4k'
        fps : int
            Frames per second, should be either 24 or 48.
        """
        if re.match("(1.5|2.0.0)", version.openjpeg_version) is not None:
            msg = "Writing Cinema2K or Cinema4K files is not supported with "
            msg += 'openjpeg library versions less than 2.0.1.'
            raise IOError(msg)

        if cinema_mode == 'cinema2k':
            if fps not in [24, 48]:
                raise IOError('Cinema2K frame rate must be either 24 or 48.')

            if re.match("2.0", version.openjpeg_version) is not None:
                # 2.0 API
                if fps == 24:
                    cparams.cp_cinema = core.OPJ_CINEMA2K_24
                else:
                    cparams.cp_cinema = core.OPJ_CINEMA2K_48
            else:
                # 2.1 API
                if fps == 24:
                    cparams.rsiz = core.OPJ_PROFILE_CINEMA_2K
                    cparams.max_comp_size = core.OPJ_CINEMA_24_COMP
                    cparams.max_cs_size = core.OPJ_CINEMA_24_CS
                else:
                    cparams.rsiz = core.OPJ_PROFILE_CINEMA_2K
                    cparams.max_comp_size = core.OPJ_CINEMA_48_COMP
                    cparams.max_cs_size = core.OPJ_CINEMA_48_CS

        else:
            # cinema4k
            if re.match("2.0", version.openjpeg_version) is not None:
                # 2.0 API
                cparams.cp_cinema = core.OPJ_CINEMA4K_24
            else:
                # 2.1 API
                cparams.rsiz = core.OPJ_PROFILE_CINEMA_4K

        return

    def _populate_cparams(self, **kwargs):
        """Populate compression parameters structure from input arguments.

        Parameters
        ----------
        cbsize : tuple, optional
            Code block size (DY, DX).
        cratios : iterable
            Compression ratios for successive layers.
        eph : bool, optional
            If true, write SOP marker after each header packet.
        grid_offset : tuple, optional
            Offset (DY, DX) of the origin of the image in the reference grid.
        mct : bool, optional
            Specifies usage of the multi component transform.  If not
            specified, defaults to True if the colorspace is RGB.
        modesw : int, optional
            Mode switch.
                1 = BYPASS(LAZY)
                2 = RESET
                4 = RESTART(TERMALL)
                8 = VSC
                16 = ERTERM(SEGTERM)
                32 = SEGMARK(SEGSYM)
        numres : int, optional
            Number of resolutions.
        prog : str, optional
            Progression order, one of "LRCP" "RLCP", "RPCL", "PCRL", "CPRL".
        psnr : iterable, optional
            Different PSNR for successive layers.
        psizes : list, optional
            List of precinct sizes.  Each precinct size tuple is defined in
            (height x width).
        sop : bool, optional
            If true, write SOP marker before each packet.
        subsam : tuple, optional
            Subsampling factors (dy, dx).
        tilesize : tuple, optional
            Numeric tuple specifying tile size in terms of (numrows, numcols),
            not (X, Y).

        Returns
        -------
        cparams : CompressionParametersType(ctypes.Structure)
            Corresponds to cparameters_t type in openjp2 headers.
        """
        if version.openjpeg_version_tuple[0] == 1:
            cparams = opj.set_default_encoder_parameters()
        else:
            cparams = opj2.set_default_encoder_parameters()

        outfile = self.filename.encode()
        num_pad_bytes = opj2.PATH_LEN - len(outfile)
        outfile += b'0' * num_pad_bytes
        cparams.outfile = outfile

        if self.filename[-4:].endswith(('.jp2', '.JP2')):
            cparams.codec_fmt = opj2.CODEC_JP2
        else:
            cparams.codec_fmt = opj2.CODEC_J2K

        # Set defaults to lossless to begin.
        cparams.tcp_rates[0] = 0
        cparams.tcp_numlayers = 1
        cparams.cp_disto_alloc = 1

        if 'irreversible' in kwargs and kwargs['irreversible'] is True:
            cparams.irreversible = 1

        if 'cinema2k' in kwargs:
            self._set_cinema_params(cparams, 'cinema2k', kwargs['cinema2k'])
            return cparams

        if 'cinema4k' in kwargs:
            self._set_cinema_params(cparams, 'cinema4k', kwargs['cinema4k'])
            return cparams

        if 'cbsize' in kwargs:
            cparams.cblockw_init = kwargs['cbsize'][1]
            cparams.cblockh_init = kwargs['cbsize'][0]

        if 'cratios' in kwargs:
            cparams.tcp_numlayers = len(kwargs['cratios'])
            for j, cratio in enumerate(kwargs['cratios']):
                cparams.tcp_rates[j] = cratio
            cparams.cp_disto_alloc = 1

        if 'eph' in kwargs:
            cparams.csty |= 0x04

        if 'grid_offset' in kwargs:
            cparams.image_offset_x0 = kwargs['grid_offset'][1]
            cparams.image_offset_y0 = kwargs['grid_offset'][0]

        if 'modesw' in kwargs:
            for shift in range(6):
                power_of_two = 1 << shift
                if kwargs['modesw'] & power_of_two:
                    cparams.mode |= power_of_two

        if 'numres' in kwargs:
            cparams.numresolution = kwargs['numres']

        if 'prog' in kwargs:
            prog = kwargs['prog'].upper()
            cparams.prog_order = PROGRESSION_ORDER[prog]

        if 'psnr' in kwargs:
            cparams.tcp_numlayers = len(kwargs['psnr'])
            for j, snr_layer in enumerate(kwargs['psnr']):
                cparams.tcp_distoratio[j] = snr_layer
            cparams.cp_fixed_quality = 1

        if 'psizes' in kwargs:
            for j, (prch, prcw) in enumerate(kwargs['psizes']):
                cparams.prcw_init[j] = prcw
                cparams.prch_init[j] = prch
            cparams.csty |= 0x01
            cparams.res_spec = len(kwargs['psizes'])

        if 'sop' in kwargs:
            cparams.csty |= 0x02

        if 'subsam' in kwargs:
            cparams.subsampling_dy = kwargs['subsam'][0]
            cparams.subsampling_dx = kwargs['subsam'][1]

        if 'tilesize' in kwargs:
            cparams.cp_tdx = kwargs['tilesize'][1]
            cparams.cp_tdy = kwargs['tilesize'][0]
            cparams.tile_size_on = opj2.TRUE

        return cparams

    def _process_write_inputs(self, img_array, colorspace=None, **kwargs):
        """Directs processing of write method arguments.

        It's somewhat awkward to process all the kwargs arguments at once.
        The "colorspace" is not a parameter that gets processed into the
        compression parameters structure, and it unfortunately must be handled
        in the middle of the compression parameter processing.

        Parameters
        ----------
        img_array : ndarray
            Image data to be written to file.
        colorspace : str, optional
            Either 'rgb' or 'gray'.

        Returns
        -------
        cparams : CompressionParametersType(ctypes.Structure)
            Corresponds to cparameters_t type in openjp2 headers.
        colorspace : int
            Either CLRSPC_SRGB or CLRSPC_GRAY
        """
        if (('cinema2k' in kwargs or 'cinema4k' in kwargs)  and
                (len(set(kwargs)) > 1)):
            msg = "Cannot specify cinema2k/cinema4k along with other options."
            raise IOError(msg)

        if 'cratios' in kwargs and 'psnr' in kwargs:
            msg = "Cannot specify cratios and psnr together."
            raise IOError(msg)

        cparams = self._populate_cparams(**kwargs)
        _validate_compression_params(img_array, cparams)

        colorspace = _unpack_colorspace(colorspace, img_array, cparams)

        try:
            mct = kwargs['mct']
            if mct and colorspace == opj2.CLRSPC_GRAY:
                # Cannot check for this in the validate routine, as we need
                # to know what the target colorspace has been determined to be.
                msg = "Cannot specify usage of the multi component transform "
                msg += "if the colorspace is gray."
                raise IOError(msg)
            cparams.tcp_mct = 1 if mct else 0
        except KeyError:
            # If the multi component transform was not specified, we infer
            # that it should be used if the color space is RGB.
            if colorspace == opj2.CLRSPC_SRGB:
                cparams.tcp_mct = 1
            else:
                cparams.tcp_mct = 0

        return cparams, colorspace

    def write(self, img_array, verbose=False, **kwargs):
        """Write image data to a JP2/JPX/J2k file.  Intended usage of the
        various parameters follows that of OpenJPEG's opj_compress utility.

        This method can only be used to create JPEG 2000 images that can fit
        in memory.

        Parameters
        ----------
        img_array : ndarray
            Image data to be written to file.
        cbsize : tuple, optional
            Code block size (DY, DX).
        cinema2k : int, optional
            frames per second, either 24 or 48
        cinema4k : bool, optional
            Set to True to specify Cinema4K mode, defaults to false.
        colorspace : str, optional
            Either 'rgb' or 'gray'.
        cratios : iterable
            Compression ratios for successive layers.
        eph : bool, optional
            If true, write SOP marker after each header packet.
        grid_offset : tuple, optional
            Offset (DY, DX) of the origin of the image in the reference grid.
        irreversible : bool, optional
            If true, use the irreversible DWT 9-7 transform. 
        mct : bool, optional
            Specifies usage of the multi component transform.  If not
            specified, defaults to True if the colorspace is RGB.
        modesw : int, optional
            Mode switch.
                1 = BYPASS(LAZY)
                2 = RESET
                4 = RESTART(TERMALL)
                8 = VSC
                16 = ERTERM(SEGTERM)
                32 = SEGMARK(SEGSYM)
        numres : int, optional
            Number of resolutions.
        prog : str, optional
            Progression order, one of "LRCP" "RLCP", "RPCL", "PCRL", "CPRL".
        psnr : iterable, optional
            Different PSNR for successive layers.
        psizes : list, optional
            List of precinct sizes.  Each precinct size tuple is defined in
            (height x width).
        sop : bool, optional
            If true, write SOP marker before each packet.
        subsam : tuple, optional
            Subsampling factors (dy, dx).
        tilesize : tuple, optional
            Numeric tuple specifying tile size in terms of (numrows, numcols),
            not (X, Y).
        verbose : bool, optional
            Print informational messages produced by the OpenJPEG library.

        Examples
        --------
        >>> import glymur
        >>> jfile = glymur.data.nemo()
        >>> jp2 = glymur.Jp2k(jfile)
        >>> data = jp2.read(rlevel=1)
        >>> from tempfile import NamedTemporaryFile
        >>> tfile = NamedTemporaryFile(suffix='.jp2', delete=False)
        >>> j = Jp2k(tfile.name, mode='wb')
        >>> j.write(data.astype(np.uint8))

        Raises
        ------
        glymur.LibraryNotFoundError
            If glymur is unable to load the openjp2 library.
        """
        if opj2.OPENJP2 is not None:
            self._write_openjp2(img_array, verbose=verbose, **kwargs)
        elif opj.OPENJPEG is not None:
            self._write_openjpeg(img_array, verbose=verbose, **kwargs)
        else:
            raise LibraryNotFoundError("You must have at least version 1.5 of "
                                       "OpenJPEG before using this "
                                       "functionality.")

    def _write_openjpeg(self, img_array, verbose=False, **kwargs):
        """
        Write JPEG 2000 file using OpenJPEG 1.5 interface.
        """
        cparams, colorspace = self._process_write_inputs(img_array, **kwargs)

        if img_array.ndim == 2:
            # Force the image to be 3D.  Just makes things easier later on.
            img_array = img_array.reshape(img_array.shape[0],
                                          img_array.shape[1],
                                          1)

        comptparms = _populate_comptparms(img_array, cparams)

        with ExitStack() as stack:
            image = opj.image_create(comptparms, colorspace)
            stack.callback(opj.image_destroy, image)

            numrows, numcols, numlayers = img_array.shape

            # set image offset and reference grid
            image.contents.x0 = cparams.image_offset_x0
            image.contents.y0 = cparams.image_offset_y0
            image.contents.x1 = image.contents.x0 \
                              + (numcols - 1) * cparams.subsampling_dx + 1
            image.contents.y1 = image.contents.y0 \
                              + (numrows - 1) * cparams.subsampling_dy + 1

            # Stage the image data to the openjpeg data structure.
            for k in range(0, numlayers):
                layer = np.ascontiguousarray(img_array[:, :, k],
                                             dtype=np.int32)
                dest = image.contents.comps[k].data
                src = layer.ctypes.data
                ctypes.memmove(dest, src, layer.nbytes)

            cinfo = opj.create_compress(cparams.codec_fmt)
            stack.callback(opj.destroy_compress, cinfo)

            # Setup the info, warning, and error handlers.
            # Always use the warning and error handler.  Use of an info
            # handler is optional.
            event_mgr = opj.EventMgrType()
            _info_handler = _INFO_CALLBACK if verbose else None
            event_mgr.info_handler = _info_handler
            event_mgr.warning_handler = ctypes.cast(_WARNING_CALLBACK,
                                                    ctypes.c_void_p)
            event_mgr.error_handler = ctypes.cast(_ERROR_CALLBACK,
                                                  ctypes.c_void_p)

            opj.setup_encoder(cinfo, ctypes.byref(cparams), image)

            cio = opj.cio_open(cinfo)
            stack.callback(opj.cio_close, cio)

            if not opj.encode(cinfo, cio, image):
                raise IOError("Encode error.")

            pos = opj.cio_tell(cio)

            blob = ctypes.string_at(cio.contents.buffer, pos)
            fptr = open(self.filename, 'wb')
            stack.callback(fptr.close)
            fptr.write(blob)

        self.parse()


    def _write_openjp2(self, img_array, verbose=False, **kwargs):
        """
        Write JPEG 2000 file using OpenJPEG 2.0 interface.
        """
        cparams, colorspace = self._process_write_inputs(img_array, **kwargs)

        if img_array.ndim == 2:
            # Force the image to be 3D.  Just makes things easier later on.
            numrows, numcols = img_array.shape
            img_array = img_array.reshape(numrows, numcols, 1)

        comptparms = _populate_comptparms(img_array, cparams)

        with ExitStack() as stack:
            image = opj2.image_create(comptparms, colorspace)
            stack.callback(opj2.image_destroy, image)

            _populate_image_struct(cparams, image, img_array)

            codec = opj2.create_compress(cparams.codec_fmt)
            stack.callback(opj2.destroy_codec, codec)

            info_handler = _INFO_CALLBACK if verbose else None
            opj2.set_info_handler(codec, info_handler)
            opj2.set_warning_handler(codec, _WARNING_CALLBACK)
            opj2.set_error_handler(codec, _ERROR_CALLBACK)

            opj2.setup_encoder(codec, cparams, image)

            if re.match("2.0", version.openjpeg_version) is not None:
                fptr = libc.fopen(self.filename, 'wb')
                strm = opj2.stream_create_default_file_stream(fptr, False)
                stack.callback(opj2.stream_destroy, strm)
                stack.callback(libc.fclose, fptr)
            else:
                # Introduced in 2.1 devel series.
                strm = opj2.stream_create_default_file_stream(self.filename,
                                                              False)
                stack.callback(opj2.stream_destroy, strm)

            opj2.start_compress(codec, image, strm)
            opj2.encode(codec, strm)
            opj2.end_compress(codec, strm)

        # Refresh the metadata.
        self.parse()

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
            raise IOError(msg)

        if not ((box.box_id == 'xml ') or
                (box.box_id == 'uuid' and
                 box.uuid == UUID('be7acfcb-97a9-42e8-9c71-999491e3afac'))):
            msg = "Only XML boxes and XMP UUID boxes can currently be appended."
            raise IOError(msg)

        # Check the last box.  If the length field is zero, then rewrite
        # the length field to reflect the true length of the box.
        with open(self.filename, 'rb') as ifile:
            offset = self.box[-1].offset
            ifile.seek(offset)
            read_buffer = ifile.read(4)
            box_length, = struct.unpack('>I', read_buffer)
            if box_length == 0:
                # Reopen the file in write mode and rewrite the length field.
                true_box_length = os.path.getsize(ifile.name) - offset
                with open(self.filename, 'r+b') as ofile:
                    ofile.seek(offset)
                    write_buffer = struct.pack('>I', true_box_length)
                    ofile.write(write_buffer)

        # Can now safely append the box.
        with open(self.filename, 'ab') as ofile:
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
        jp2 : Jp2k object
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

        _validate_jp2_box_sequence(boxes)

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
                msg = "The codestream box must have its offset and "
                msg += "length attributes fully specified if the file "
                msg += "type brand is JPX."
                raise IOError(msg)

            # Find the first codestream in the file.
            jp2c = [box for box in self.box if box.box_id == 'jp2c']
            offset = jp2c[0].offset

        # Ready to write the codestream.
        with open(self.filename, 'rb') as ifile:
            ifile.seek(offset)

            # Verify that the specified codestream is right.
            read_buffer = ifile.read(8)
            L, T = struct.unpack_from('>I4s', read_buffer, 0)
            if T != b'jp2c':
                msg = "Unable to locate the specified codestream."
                raise IOError(msg)
            if L == 0:
                # The length of the box is presumed to last until the end of
                # the file.  Compute the effective length of the box.
                L = os.path.getsize(ifile.name) - ifile.tell() + 8

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
        boxes = [JPEG2000SignatureBox(),
                 FileTypeBox(),
                 JP2HeaderBox(),
                 ContiguousCodestreamBox()]
        codestream = self.get_codestream()
        height = codestream.segment[1].ysiz
        width = codestream.segment[1].xsiz
        num_components = len(codestream.segment[1].xrsiz)
        if num_components < 3:
            colorspace = GREYSCALE
        else:
            if len(self.box) == 0:
                # Best guess is SRGB
                colorspace = SRGB
            else:
                # Take whatever the first jp2 header / color specification
                # says.
                jp2hs = [box for box in self.box if box.box_id == 'jp2h']
                colorspace = jp2hs[0].box[1].colorspace

        boxes[2].box = [ImageHeaderBox(height=height, width=width,
                                       num_components=num_components),
                        ColourSpecificationBox(colorspace=colorspace)]

        return boxes

    def read(self, **kwargs):
        """Read a JPEG 2000 image.

        Parameters
        ----------
        rlevel : int, optional
            Factor by which to rlevel output resolution.  Use -1 to get the
            lowest resolution thumbnail.  This is the only keyword option
            available to use when only the OpenJPEG version 1.5.1 is present.
        layer : int, optional
            Number of quality layer to decode.
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
        img_array : ndarray
            The image data.

        Raises
        ------
        glymur.LibraryNotFoundError
            If glymur is unable to load either the openjpeg or openjp2
            libraries.
        IOError
            If the image has differing subsample factors.

        Examples
        --------
        >>> import glymur
        >>> jfile = glymur.data.nemo()
        >>> jp = glymur.Jp2k(jfile)
        >>> image = jp.read()
        >>> image.shape
        (1456, 2592, 3)

        Read the lowest resolution thumbnail.

        >>> thumbnail = jp.read(rlevel=-1)
        >>> thumbnail.shape
        (728, 1296, 3)
        """
        if opj2.OPENJP2 is not None:
            img = self._read_openjp2(**kwargs)
        elif opj.OPENJPEG is not None:
            img = self._read_openjpeg(**kwargs)
        else:
            raise LibraryNotFoundError("You must have either a recent version "
                                       "of OpenJPEG or the development "
                                       "version of OpenJP2 installed before "
                                       "using this functionality.")
        return img

    def _subsampling_sanity_check(self):
        """Check for differing subsample factors.
        """
        codestream = self.get_codestream(header_only=True)
        dxs = np.array(codestream.segment[1].xrsiz)
        dys = np.array(codestream.segment[1].yrsiz)
        if np.any(dxs - dxs[0]) or np.any(dys - dys[0]):
            msg = "Components must all have the same subsampling factors "
            msg += "to use this method.  Please consider using OPENJP2 and "
            msg += "the read_bands method instead."
            raise RuntimeError(msg)

    def _read_openjpeg(self, rlevel=0, ignore_pclr_cmap_cdef=False,
                       verbose=False):
        """Read a JPEG 2000 image using libopenjpeg.

        Parameters
        ----------
        rlevel : int, optional
            Factor by which to rlevel output resolution.  Use -1 to get the
            lowest resolution thumbnail.
        ignore_pclr_cmap_cdef : bool
            Whether or not to ignore the pclr, cmap, or cdef boxes during any
            color transformation.  Defaults to False.
        verbose : bool, optional
            Print informational messages produced by the OpenJPEG library.

        Returns
        -------
        img_array : ndarray
            The image data.

        Raises
        ------
        RuntimeError
            If the image has differing subsample factors.
        """
        self._subsampling_sanity_check()

        dparameters = self._populate_dparam(rlevel, ignore_pclr_cmap_cdef)

        with ExitStack() as stack:
            try:
                dparameters.decod_format = self._codec_format

                dinfo = opj.create_decompress(dparameters.decod_format)

                event_mgr = opj.EventMgrType()
                info_handler = ctypes.cast(_INFO_CALLBACK, ctypes.c_void_p)
                event_mgr.info_handler = info_handler if verbose else None
                event_mgr.warning_handler = ctypes.cast(_WARNING_CALLBACK,
                                                        ctypes.c_void_p)
                event_mgr.error_handler = ctypes.cast(_ERROR_CALLBACK,
                                                      ctypes.c_void_p)
                opj.set_event_mgr(dinfo, ctypes.byref(event_mgr))

                opj.setup_decoder(dinfo, dparameters)

                with open(self.filename, 'rb') as fptr:
                    src = fptr.read()
                cio = opj.cio_open(dinfo, src)

                image = opj.decode(dinfo, cio)

                stack.callback(opj.image_destroy, image)
                stack.callback(opj.destroy_decompress, dinfo)
                stack.callback(opj.cio_close, cio)

                data = extract_image_cube(image)

            except ValueError:
                opj2.check_error(0)

        if data.shape[2] == 1:
            # The third dimension has just a single layer.  Make the image
            # data 2D instead of 3D.
            data.shape = data.shape[0:2]

        return data

    def _read_openjp2(self, rlevel=0, layer=0, area=None, tile=None,
                      verbose=False, ignore_pclr_cmap_cdef=False):
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
        img_array : ndarray
            The image data.

        Raises
        ------
        RuntimeError
            If the image has differing subsample factors.
        """
        self._subsampling_sanity_check()

        dparam = self._populate_dparam(rlevel, ignore_pclr_cmap_cdef,
                                       layer=layer, tile=tile, area=area)

        with ExitStack() as stack:
            if re.match("2.1", version.openjpeg_version):
                filename = self.filename
                stream = opj2.stream_create_default_file_stream(filename, True)
                stack.callback(opj2.stream_destroy, stream)
            else:
                fptr = libc.fopen(self.filename, 'rb')
                stack.callback(libc.fclose, fptr)
                stream = opj2.stream_create_default_file_stream(fptr, True)
                stack.callback(opj2.stream_destroy, stream)
            codec = opj2.create_decompress(self._codec_format)
            stack.callback(opj2.destroy_codec, codec)

            opj2.set_error_handler(codec, _ERROR_CALLBACK)
            opj2.set_warning_handler(codec, _WARNING_CALLBACK)
            if verbose:
                opj2.set_info_handler(codec, _INFO_CALLBACK)
            else:
                opj2.set_info_handler(codec, None)

            opj2.setup_decoder(codec, dparam)
            image = opj2.read_header(stream, codec)
            stack.callback(opj2.image_destroy, image)

            if dparam.nb_tile_to_decode:
                opj2.get_decoded_tile(codec, stream, image, dparam.tile_index)
            else:
                opj2.set_decode_area(codec, image,
                                     dparam.DA_x0, dparam.DA_y0,
                                     dparam.DA_x1, dparam.DA_y1)
                opj2.decode(codec, stream, image)
                opj2.end_decompress(codec, stream)

            img_array = extract_image_cube(image)

        if img_array.shape[2] == 1:
            img_array.shape = img_array.shape[0:2]

        return img_array

    def _populate_dparam(self, rlevel, ignore_pclr_cmap_cdef, tile=None,
                         layer=None, area=None):
        """Populate decompression structure with appropriate input parameters.

        Parameters
        ----------
        layer : int
            Number of quality layer to decode.
        rlevel : int
            Factor by which to rlevel output resolution.
        area : tuple
            Specifies decoding image area,
            (first_row, first_col, last_row, last_col)
        tile : int
            Number of tile to decode.
        ignore_pclr_cmap_cdef : bool
            Whether or not to ignore the pclr, cmap, or cdef boxes during any
            color transformation.  Defaults to False.

        Returns
        -------
        dparam : DecompressionParametersType (ctypes)
            Corresponds to openjp2 decompression parameters structure.
        """
        if opj2.OPENJP2 is not None:
            dparam = opj2.set_default_decoder_parameters()
        else:
            dparam = opj.DecompressionParametersType()
            opj.set_default_decoder_parameters(ctypes.byref(dparam))

        infile = self.filename.encode()
        nelts = opj2.PATH_LEN - len(infile)
        infile += b'0' * nelts
        dparam.infile = infile

        dparam.decod_format = self._codec_format

        if layer is not None:
            dparam.cp_layer = layer

        # Must check the specified rlevel against the maximum.
        if rlevel != 0:
            # Must check the specified rlevel against the maximum.
            codestream = self.get_codestream()
            max_rlevel = codestream.segment[2].spcod[4]
            if rlevel == -1:
                # -1 is shorthand for the largest rlevel
                rlevel = max_rlevel
            elif rlevel < -1 or rlevel > max_rlevel:
                msg = "rlevel must be in the range [-1, {0}] for this image."
                msg = msg.format(max_rlevel)
                raise IOError(msg)

        dparam.cp_reduce = rlevel

        if area is not None:
            if area[0] < 0 or area[1] < 0 or area[2] <= 0 or area[3] <= 0:
                msg = "Upper left corner coordinates must be nonnegative and "
                msg += "lower right corner coordinates must be positive:  {0}"
                raise IOError(msg.format(area))
            dparam.DA_y0 = area[0]
            dparam.DA_x0 = area[1]
            dparam.DA_y1 = area[2]
            dparam.DA_x1 = area[3]

        if tile is not None:
            dparam.tile_index = tile
            dparam.nb_tile_to_decode = 1

        if ignore_pclr_cmap_cdef is True:
            # Return raw codestream components.
            dparam.flags |= 1

        return dparam

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
        lst : list
            The individual image components.

        See also
        --------
        read : read JPEG 2000 image

        Examples
        --------
        >>> import glymur
        >>> jfile = glymur.data.nemo()
        >>> jp = glymur.Jp2k(jfile)
        >>> components_lst = jp.read_bands(rlevel=1)

        Raises
        ------
        glymur.LibraryNotFoundError
            If glymur is unable to load the openjp2 library.
        """
        if version.openjpeg_version_tuple[0] < 2:
            raise LibraryNotFoundError("You must have at least version 2.0.0 "
                                       "of OpenJP2 installed before using "
                                       "this functionality.")

        dparam = self._populate_dparam(rlevel, ignore_pclr_cmap_cdef,
                                       layer=layer, tile=tile, area=area)

        with ExitStack() as stack:
            if re.match("2.1", version.openjpeg_version):
                # API change in 2.1
                filename = self.filename
                stream = opj2.stream_create_default_file_stream(filename, True)
                stack.callback(opj2.stream_destroy, stream)
            else:
                fptr = libc.fopen(self.filename, 'rb')
                stack.callback(libc.fclose, fptr)
                stream = opj2.stream_create_default_file_stream(fptr, True)
                stack.callback(opj2.stream_destroy, stream)
            codec = opj2.create_decompress(self._codec_format)
            stack.callback(opj2.destroy_codec, codec)

            opj2.set_error_handler(codec, _ERROR_CALLBACK)
            opj2.set_warning_handler(codec, _WARNING_CALLBACK)
            if verbose:
                opj2.set_info_handler(codec, _INFO_CALLBACK)
            else:
                opj2.set_info_handler(codec, None)

            opj2.setup_decoder(codec, dparam)
            image = opj2.read_header(stream, codec)
            stack.callback(opj2.image_destroy, image)

            if dparam.nb_tile_to_decode:
                opj2.get_decoded_tile(codec, stream, image, dparam.tile_index)
            else:
                opj2.set_decode_area(codec, image,
                                     dparam.DA_x0, dparam.DA_y0,
                                     dparam.DA_x1, dparam.DA_y1)
                opj2.decode(codec, stream, image)
                opj2.end_decompress(codec, stream)

            lst = extract_image_bands(image)

        return lst

    def get_codestream(self, header_only=True):
        """Returns a codestream object.

        Parameters
        ----------
        header_only : bool, optional
            If True, only marker segments in the main header are parsed.
            Supplying False may impose a large performance penalty.

        Returns
        -------
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
        with open(self.filename, 'rb') as fptr:
            if self._codec_format == opj2.CODEC_J2K:
                codestream = Codestream(fptr, self.length,
                                        header_only=header_only)
            else:
                ftyp = self.box[1]
                box = [x for x in self.box if x.box_id == 'jp2c']
                fptr.seek(box[0].offset)
                read_buffer = fptr.read(8)
                (box_length, _) = struct.unpack('>I4s', read_buffer)
                if box_length == 0:
                    # The length of the box is presumed to last until the end
                    # of the file.  Compute the effective length of the box.
                    box_length = os.path.getsize(fptr.name) - fptr.tell() + 8
                elif box_length == 1:
                    # Seek past the XL field.
                    read_buffer = fptr.read(8)
                    box_length, = struct.unpack('>Q', read_buffer)
                codestream = Codestream(fptr, box_length - 8,
                                        header_only=header_only)

            return codestream


def _component2dtype(component):
    """Take an OpenJPEG component structure and determine the numpy datatype.

    Parameters
    ----------
    component : ctypes pointer to ImageCompType (image_comp_t)
        single image component structure.

    Returns
    -------
    dtype : builtins.type
        numpy datatype to be used to construct an image array.
    """
    if component.sgnd:
        if component.prec <= 8:
            dtype = np.int8
        elif component.prec <= 16:
            dtype = np.int16
        else:
            raise RuntimeError("Unhandled precision, datatype")
    else:
        if component.prec <= 8:
            dtype = np.uint8
        elif component.prec <= 16:
            dtype = np.uint16
        else:
            raise RuntimeError("Unhandled precision, datatype")

    return dtype


def _validate_nonzero_image_size(nrows, ncols, component_index):
    """The image cannot have area of zero.
    """
    if nrows == 0 or ncols == 0:
        # Letting this situation continue would segfault Python.
        msg = "Component {0} has dimensions {1} x {2}"
        msg = msg.format(component_index, nrows, ncols)
        raise IOError(msg)


JP2_IDS = ['colr', 'cdef', 'cmap', 'jp2c', 'ftyp', 'ihdr', 'jp2h', 'jP  ',
           'pclr', 'res ', 'resc', 'resd', 'xml ', 'ulst', 'uinf', 'url ',
           'uuid']

def _validate_jp2_box_sequence(boxes):
    """Run through series of tests for JP2 box legality.

    This is non-exhaustive.
    """
    _validate_signature_compatibility(boxes)
    _validate_jp2h(boxes)
    _validate_jp2c(boxes)
    if boxes[1].brand == 'jpx ':
        _validate_jpx_box_sequence(boxes)
    else:
        # Validate the JP2 box IDs.
        count = _collect_box_count(boxes)
        for box_id in count.keys():
            if box_id not in JP2_IDS:
                msg = "The presence of a '{0}' box requires that the file type "
                msg += "brand be set to 'jpx '."
                raise IOError(msg.format(box_id))

        _validate_jp2_colr(boxes)

def _validate_jp2_colr(boxes):
    """
    Validate JP2 requirements on colour specification boxes.
    """
    lst = [box for box in boxes if box.box_id == 'jp2h']
    jp2h = lst[0]
    for colr in [box for box in jp2h.box if box.box_id == 'colr']:
        if colr.approximation != 0:
            msg = "A JP2 colr box cannot have a non-zero approximation field."
            raise IOError(msg)

def _validate_jpx_box_sequence(boxes):
    """Run through series of tests for JPX box legality."""
    _validate_label(boxes)
    _validate_jpx_brand(boxes, boxes[1].brand)
    _validate_jpx_compatibility(boxes, boxes[1].compatibility_list)
    _validate_singletons(boxes)
    _validate_top_level(boxes)

def _validate_signature_compatibility(boxes):
    """Validate the file signature and compatibility status."""
    # Check for a bad sequence of boxes.
    # 1st two boxes must be 'jP  ' and 'ftyp'
    if boxes[0].box_id != 'jP  ' or boxes[1].box_id != 'ftyp':
        msg = "The first box must be the signature box and the second "
        msg += "must be the file type box."
        raise IOError(msg)

    # The compatibility list must contain at a minimum 'jp2 '.
    if 'jp2 ' not in boxes[1].compatibility_list:
        msg = "The ftyp box must contain 'jp2 ' in the compatibility list."
        raise IOError(msg)


def _validate_jp2c(boxes):
    """Validate the codestream box in relation to other boxes."""
    # jp2c must be preceeded by jp2h
    jp2h_lst = [idx for (idx, box) in enumerate(boxes)
                if box.box_id == 'jp2h']
    jp2h_idx = jp2h_lst[0]
    jp2c_lst = [idx for (idx, box) in enumerate(boxes)
                if box.box_id == 'jp2c']
    if len(jp2c_lst) == 0:
        msg = "A codestream box must be defined in the outermost "
        msg += "list of boxes."
        raise IOError(msg)

    jp2c_idx = jp2c_lst[0]
    if jp2h_idx >= jp2c_idx:
        msg = "The codestream box must be preceeded by a jp2 header box."
        raise IOError(msg)


def _validate_jp2h(boxes):
    """Validate the JP2 Header box."""
    _check_jp2h_child_boxes(boxes, 'top-level')

    jp2h_lst = [box for box in boxes if box.box_id == 'jp2h']
    jp2h = jp2h_lst[0]

    # 1st jp2 header box cannot be empty.
    if len(jp2h.box) == 0:
        msg = "The JP2 header superbox cannot be empty."
        raise IOError(msg)

    # 1st jp2 header box must be ihdr
    if jp2h.box[0].box_id != 'ihdr':
        msg = "The first box in the jp2 header box must be the image "
        msg += "header box."
        raise IOError(msg)

    # colr must be present in jp2 header box.
    colr_lst = [j for (j, box) in enumerate(jp2h.box)
                if box.box_id == 'colr']
    if len(colr_lst) == 0:
        msg = "The jp2 header box must contain a color definition box."
        raise IOError(msg)
    colr = jp2h.box[colr_lst[0]]

    _validate_channel_definition(jp2h, colr)


def _validate_channel_definition(jp2h, colr):
    """Validate the channel definition box."""
    cdef_lst = [j for (j, box) in enumerate(jp2h.box) if box.box_id == 'cdef']
    if len(cdef_lst) > 1:
        msg = "Only one channel definition box is allowed in the "
        msg += "JP2 header."
        raise IOError(msg)
    elif len(cdef_lst) == 1:
        cdef = jp2h.box[cdef_lst[0]]
        if colr.colorspace == SRGB:
            if any([chan + 1 not in cdef.association
                    or cdef.channel_type[chan] != 0
                    for chan in [0, 1, 2]]):
                msg = "All color channels must be defined in the "
                msg += "channel definition box."
                raise IOError(msg)
        elif colr.colorspace == GREYSCALE:
            if 0 not in cdef.channel_type:
                msg = "All color channels must be defined in the "
                msg += "channel definition box."
                raise IOError(msg)


JP2H_CHILDREN = set(['bpcc', 'cdef', 'cmap', 'ihdr', 'pclr'])
def _check_jp2h_child_boxes(boxes, parent_box_name):
    """Certain boxes can only reside in the JP2 header."""
    box_ids = set([box.box_id for box in boxes])
    intersection = box_ids.intersection(JP2H_CHILDREN)
    if len(intersection) > 0 and parent_box_name not in ['jp2h', 'jpch']:
        msg = "A '{0}' box can only be nested in a JP2 header box."
        raise IOError(msg.format(list(intersection)[0]))

    # Recursively check any contained superboxes.
    for box in boxes:
        if hasattr(box, 'box'):
            _check_jp2h_child_boxes(box.box, box.box_id)


def _collect_box_count(boxes):
    """Count the occurences of each box type."""
    count = Counter([box.box_id for box in boxes])

    # Add the counts in the superboxes.
    for box in boxes:
        if hasattr(box, 'box'):
            count.update(_collect_box_count(box.box))

    return count

TOP_LEVEL_ONLY_BOXES = set(['dtbl'])

def _check_superbox_for_top_levels(boxes):
    """Several boxes can only occur at the top level."""
    # We are only looking at the boxes contained in a superbox, so if any of
    # the blacklisted boxes show up here, it's an error.
    box_ids = set([box.box_id for box in boxes])
    intersection = box_ids.intersection(TOP_LEVEL_ONLY_BOXES)
    if len(intersection) > 0:
        msg = "A '{0}' box cannot be nested in a superbox."
        raise IOError(msg.format(list(intersection)[0]))

    # Recursively check any contained superboxes.
    for box in boxes:
        if hasattr(box, 'box'):
            _check_superbox_for_top_levels(box.box)

def _validate_top_level(boxes):
    """Several boxes can only occur at the top level."""
    # Add the counts in the superboxes.
    for box in boxes:
        if hasattr(box, 'box'):
            _check_superbox_for_top_levels(box.box)

    count = _collect_box_count(boxes)
    # Which boxes occur more than once?
    multiples = [box_id for box_id, bcount in count.items() if bcount > 1]
    if 'dtbl' in multiples:
        raise IOError('There can only be one dtbl box in a file.')

    # If there is one data reference box, then there must also be one ftbl.
    if 'dtbl' in count and 'ftbl' not in count:
        msg = 'The presence of a data reference box requires the presence of '
        msg += 'a fragment table box as well.'
        raise IOError(msg)

def _validate_singletons(boxes):
    """Several boxes can only occur once."""
    count = _collect_box_count(boxes)
    # Which boxes occur more than once?
    multiples = [box_id for box_id, bcount in count.items() if bcount > 1]
    if 'dtbl' in multiples:
        raise IOError('There can only be one dtbl box in a file.')

def _validate_jpx_brand(boxes, brand):
    """
    If there is a JPX box then the brand must be 'jpx '.
    """
    for box in boxes:
        if box.box_id in JPX_IDS:
            if brand != 'jpx ':
                msg = "A JPX box requires that the file type box brand be "
                msg += "'jpx '."
                raise RuntimeError(msg)
        if hasattr(box, 'box') != 0:
            # Same set of checks on any child boxes.
            _validate_jpx_brand(box.box, brand)

def _validate_jpx_compatibility(boxes, compatibility_list):
    """
    If there is a JPX box then the compatibility list must also contain 'jpx '.
    """
    jpx_cl = set(compatibility_list)
    for box in boxes:
        if box.box_id in JPX_IDS:
            if len(set(['jpx ', 'jpxb']).intersection(jpx_cl)) == 0:
                msg = "A JPX box requires that either 'jpx ' or 'jpxb' be "
                msg += "present in the ftype compatibility list."
                raise RuntimeError(msg)
        if hasattr(box, 'box') != 0:
            # Same set of checks on any child boxes.
            _validate_jpx_compatibility(box.box, compatibility_list)

def _validate_label(boxes):
    """
    Label boxes can only be inside association, codestream headers, or
    compositing layer header boxes.
    """
    for box in boxes:
        if box.box_id != 'asoc':
            if hasattr(box, 'box'):
                for boxi in box.box:
                    if boxi.box_id == 'lbl ':
                        msg = "A label box cannot be nested inside a {0} box."
                        msg = msg.format(box.box_id)
                        raise IOError(msg)
                # Same set of checks on any child boxes.
                _validate_label(box.box)

def extract_image_cube(image):
    """Extract 3D image from openjpeg data structure.
    """
    ncomps = image.contents.numcomps
    component = image.contents.comps[0]
    dtype = _component2dtype(component)

    nrows = component.h
    ncols = component.w
    data = np.zeros((nrows, ncols, ncomps), dtype)

    for k in range(image.contents.numcomps):
        component = image.contents.comps[k]
        nrows = component.h
        ncols = component.w

        _validate_nonzero_image_size(nrows, ncols, k)

        addr = ctypes.addressof(component.data.contents)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nelts = nrows * ncols
            band = np.ctypeslib.as_array(
                (ctypes.c_int32 * nelts).from_address(addr))
            data[:, :, k] = np.reshape(band.astype(dtype), (nrows, ncols))

    return data


def extract_image_bands(image):
    """Extract unequally-sized image bands.

    This routine need only be called when subsampling differs across image
    components, such as is often the case with YCbCr imagery.
    """
    data = []
    for k in range(image.contents.numcomps):
        component = image.contents.comps[k]

        dtype = _component2dtype(component)
        nrows = component.h
        ncols = component.w

        _validate_nonzero_image_size(nrows, ncols, k)

        addr = ctypes.addressof(component.data.contents)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            band = np.ctypeslib.as_array(
                (ctypes.c_int32 * nrows * ncols).from_address(addr))
        data.append(np.reshape(band.astype(dtype), (nrows, ncols)))

    return data


def _unpack_colorspace(colorspace, img_array, cparams):
    """Determine the colorspace from the supplied inputs.

    Parameters
    ----------
    colorspace : int
        Either CLRSPC_SRGB or CLRSPC_GRAY
    img_array : ndarray
        Image data to be written to file.
    cparams : CompressionParametersType(ctypes.Structure)
        Corresponds to cparameters_t type in openjp2 headers.
    """
    if colorspace is None:
        # Must infer the colorspace from the image dimensions.
        if img_array.ndim < 3:
            # A single channel image is grayscale.
            colorspace = opj2.CLRSPC_GRAY
        elif img_array.shape[2] == 1 or img_array.shape[2] == 2:
            # A single channel image or an image with two channels is going
            # to be greyscale.
            colorspace = opj2.CLRSPC_GRAY
        else:
            # Anything else must be RGB, right?
            colorspace = opj2.CLRSPC_SRGB
    else:
        # Supplied a string colorspace, so we must validate it.
        if cparams.codec_fmt == opj2.CODEC_J2K:
            msg = 'Do not specify a colorspace when writing a raw '
            msg += 'codestream.'
            raise IOError(msg)
        if colorspace.lower() not in ('rgb', 'grey', 'gray'):
            msg = 'Invalid colorspace "{0}"'.format(colorspace)
            raise IOError(msg)
        elif colorspace.lower() == 'rgb' and img_array.shape[2] < 3:
            msg = 'RGB colorspace requires at least 3 components.'
            raise IOError(msg)

        # Turn the colorspace from a string to the enumerated value that
        # the library expects.
        colorspace = _COLORSPACE_MAP[colorspace.lower()]

    return colorspace


def _populate_comptparms(img_array, cparams):
    """Instantiate and populate comptparms structure.

    This structure defines the image components.

    Parameters
    ----------
    img_array : ndarray
        Image data to be written to file.
    cparams : CompressionParametersType(ctypes.Structure)
        Corresponds to cparameters_t type in openjp2 headers.

    Returns
    -------
    comptparms : ImageCompType(ctypes.Structure)
        Corresponds to image_comp_t type in openjp2 headers.
    """
    # Only two precisions are possible.
    if img_array.dtype == np.uint8:
        comp_prec = 8
    else:
        comp_prec = 16

    numrows, numcols, num_comps = img_array.shape
    if version.openjpeg_version_tuple[0] == 1:
        comptparms = (opj.ImageComptParmType * num_comps)()
    else:
        comptparms = (opj2.ImageComptParmType * num_comps)()
    for j in range(num_comps):
        comptparms[j].dx = cparams.subsampling_dx
        comptparms[j].dy = cparams.subsampling_dy
        comptparms[j].w = numcols
        comptparms[j].h = numrows
        comptparms[j].x0 = cparams.image_offset_x0
        comptparms[j].y0 = cparams.image_offset_y0
        comptparms[j].prec = comp_prec
        comptparms[j].bpp = comp_prec
        comptparms[j].sgnd = 0

    return comptparms


def _populate_image_struct(cparams, image, imgdata):
    """Populates image struct needed for compression.

    Parameters
    ----------
    cparams : CompressionParametersType(ctypes.Structure)
        Corresponds to cparameters_t type in openjp2 headers.
    image : ImageType(ctypes.Structure)
        Corresponds to image_t type in openjp2 headers.
    imgarray : ndarray
        Image data to be written to file.
    """

    numrows, numcols, num_comps = imgdata.shape

    # set image offset and reference grid
    image.contents.x0 = cparams.image_offset_x0
    image.contents.y0 = cparams.image_offset_y0
    image.contents.x1 = (image.contents.x0 +
                         (numcols - 1) * cparams.subsampling_dx + 1)
    image.contents.y1 = (image.contents.y0 +
                         (numrows - 1) * cparams.subsampling_dy + 1)

    # Stage the image data to the openjpeg data structure.
    for k in range(0, num_comps):
        if re.match("2.0", version.openjpeg_version) is not None:
            # 2.0 API
            if cparams.cp_cinema:
                image.contents.comps[k].prec = 12
                image.contents.comps[k].bpp = 12
        else:
            # 2.1 API
            if cparams.rsiz in (core.OPJ_PROFILE_CINEMA_2K,
                                core.OPJ_PROFILE_CINEMA_4K):
                image.contents.comps[k].prec = 12
                image.contents.comps[k].bpp = 12

        layer = np.ascontiguousarray(imgdata[:, :, k], dtype=np.int32)
        dest = image.contents.comps[k].data
        src = layer.ctypes.data
        ctypes.memmove(dest, src, layer.nbytes)

    return image


def _validate_compression_params(img_array, cparams):
    """Check that the compression parameters are valid.

    Parameters
    ----------
    img_array : ndarray
        Image data to be written to file.
    cparams : CompressionParametersType(ctypes.Structure)
        Corresponds to cparameters_t type in openjp2 headers.
    """

    # Code block size
    code_block_specified = False
    if cparams.cblockw_init != 0 and cparams.cblockh_init != 0:
        # These fields ARE zero if uninitialized.
        width = cparams.cblockw_init
        height = cparams.cblockh_init
        code_block_specified = True
        if height * width > 4096 or height < 4 or width < 4:
            msg = "Code block area cannot exceed 4096.  "
            msg += "Code block height and width must be larger than 4."
            raise IOError(msg)
        if ((math.log(height, 2) != math.floor(math.log(height, 2)) or
             math.log(width, 2) != math.floor(math.log(width, 2)))):
            msg = "Bad code block size ({0}, {1}), "
            msg += "must be powers of 2."
            raise IOError(msg.format(height, width))

    # Precinct size
    if cparams.res_spec != 0:
        # precinct size was not specified if this field is zero.
        for j in range(cparams.res_spec):
            prch = cparams.prch_init[j]
            prcw = cparams.prcw_init[j]
            if j == 0 and code_block_specified:
                height, width = cparams.cblockh_init, cparams.cblockw_init
                if height * 2 > prch or width * 2 > prcw:
                    msg = "Highest Resolution precinct size must be at "
                    msg += "least twice that of the code block dimensions."
                    raise IOError(msg)
            if ((math.log(prch, 2) != math.floor(math.log(prch, 2)) or
                 math.log(prcw, 2) != math.floor(math.log(prcw, 2)))):
                msg = "Bad precinct sizes ({0}, {1}), "
                msg += "must be powers of 2."
                raise IOError(msg.format(prch, prcw))

    # What would the point of 1D images be?
    if img_array.ndim == 1 or img_array.ndim > 3:
        msg = "{0}D imagery is not allowed.".format(img_array.ndim)
        raise IOError(msg)

    if re.match("2.0.0", version.openjpeg_version) is not None:
        if (((img_array.ndim != 2) and
             (img_array.shape[2] != 1 and img_array.shape[2] != 3))):
            msg = "Writing images is restricted to single-channel "
            msg += "greyscale images or three-channel RGB images when "
            msg += "the OpenJPEG library version is the official 2.0.0 "
            msg += "release."
            raise IOError(msg)

    if img_array.dtype != np.uint8 and img_array.dtype != np.uint16:
        msg = "Only uint8 and uint16 images are currently supported."
        raise RuntimeError(msg)

_COLORSPACE_MAP = {'rgb': opj2.CLRSPC_SRGB,
                   'gray': opj2.CLRSPC_GRAY,
                   'grey': opj2.CLRSPC_GRAY,
                   'ycc': opj2.CLRSPC_YCC}

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
    warnings.warn(msg)

_ERROR_CALLBACK = _CMPFUNC(_default_error_handler)
_INFO_CALLBACK = _CMPFUNC(_default_info_handler)
_WARNING_CALLBACK = _CMPFUNC(_default_warning_handler)


class LibraryNotFoundError(IOError):
    """Raised if functionality is requested without the necessary library.
    """
    def __init__(self, msg):
        IOError.__init__(self, msg)
