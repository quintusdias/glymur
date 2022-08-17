# standard library imports
import io
import logging
import struct
import warnings

# 3rd party library imports
import numpy as np
from uuid import UUID

# local imports
from glymur import Jp2k
from .lib import tiff as libtiff
from . import jp2box
from ._tiff import TAGNUM2NAME

# Create a mapping of tag names to tag numbers.  Make the tag names to be
# lower-case because we need to compare against user-supplied tag names.
TAGNAME2NUM = {v.lower(): k for k, v in TAGNUM2NAME.items()}


# Map the numeric TIFF datatypes to the format string used by the struct module
# and keep track of how wide they are.
tag_dtype = {
    1: {'format': 'B', 'nbytes': 1},
    2: {'format': 'B', 'nbytes': 1},
    3: {'format': 'H', 'nbytes': 2},
    4: {'format': 'I', 'nbytes': 4},
    5: {'format': 'II', 'nbytes': 8},
    7: {'format': 'B', 'nbytes': 1},
    9: {'format': 'i', 'nbytes': 4},
    10: {'format': 'ii', 'nbytes': 8},
    11: {'format': 'f', 'nbytes': 4},
    12: {'format': 'd', 'nbytes': 8},
    13: {'format': 'I', 'nbytes': 4},
    16: {'format': 'Q', 'nbytes': 8},
    17: {'format': 'q', 'nbytes': 8},
    18: {'format': 'Q', 'nbytes': 8}
}

# Mnemonics for the two TIFF format version numbers.
_TIFF = 42
_BIGTIFF = 43


class Tiff2Jp2k(object):
    """
    Attributes
    ----------
    found_geotiff_tags : bool
        If true, then this TIFF must be a GEOTIFF
    tiff_filename : path or str
        Path to TIFF file.
    jp2_filename : path or str
        Path to JPEG 2000 file to be written.
    jp2_kwargs : dict
        Keyword arguments to pass along to the Jp2k constructor.
    tilesize : tuple
        The dimensions of a tile in the JP2K file.
    create_exif_uuid : bool
        Create a UUIDBox for the TIFF IFD metadata.
    version : int
        Identifies the TIFF as 32-bit TIFF or 64-bit TIFF.
    """

    def __init__(
        self, tiff_filename, jp2_filename,
        create_exif_uuid=True, create_xmp_uuid=False, exclude_tags=None,
        tilesize=None, verbosity=logging.CRITICAL,
        **kwargs
    ):
        """
        Parameters
        ----------
        create_exif_uuid : bool
            If true, create an EXIF UUID out of the TIFF metadata (tags)
        exclude_tags : list or None
            If not None and if create_exif_uuid is True, exclude any listed
            tags from the EXIF UUID.
        tiff_filename : path or str
            Path to TIFF file.
        jp2_filename : path or str
            Path to JPEG 2000 file to be written.
        tilesize : tuple
            The dimensions of a tile in the JP2K file.
        verbosity : int
            Set the level of logging, i.e. WARNING, INFO, etc.
        create_xmp_uuid : bool
            If true and if there is an XMLPacket (700) tag in the TIFF main
            IFD, it will be removed from the IFD and placed in a UUID box.
        """

        self.tiff_filename = tiff_filename
        if not self.tiff_filename.exists():
            raise FileNotFoundError(f'{tiff_filename} does not exist')

        self.jp2_filename = jp2_filename
        self.tilesize = tilesize
        self.create_exif_uuid = create_exif_uuid
        self.create_xmp_uuid = create_xmp_uuid

        self.exclude_tags = self._process_exclude_tags(exclude_tags)

        self.jp2_kwargs = kwargs

        self.setup_logging(verbosity)

    def _process_exclude_tags(self, exclude_tags):
        """
        The list of tags to exclude may be mixed type (str or integer).  There
        is also the possibility that they may be capitalized differently than
        our internal list, so the goal here is to convert them all to integer
        values.

        Parameters
        ----------
        exclude_tags : list or None
            List of tags that are meant to be excluded from the EXIF UUID.

        Returns
        -------
        list of numeric tag values
        """
        if exclude_tags is None:
            return exclude_tags

        lst = []

        # first, make the tags all str datatype
        # compare the tags as lower case for consistency's sake
        exclude_tags = [
            tag.lower() if isinstance(tag, str) else str(tag)
            for tag in exclude_tags
        ]

        # If any tags were specified as strings, we need to convert them
        # into tag numbers.
        for tag in exclude_tags:

            # convert from str to numeric
            #
            # is it a string like '325'?  then convert to integer 325
            try:
                tag_num = int(tag)
            except ValueError:
                # tag wasn't '325', but rather 'tilebytecounts',
                # hopefully?  Try to map from the name back to the tag
                # number.
                try:
                    tag_num = TAGNAME2NUM[tag]
                except KeyError:
                    msg = f"{tag} is not a recognized TIFF tag"
                    warnings.warn(msg)
                else:
                    lst.append(tag_num)
            else:
                # tag really was something like '325', so we keep the
                # numeric value
                lst.append(tag_num)

        exclude_tags = lst

        return exclude_tags

    def setup_logging(self, verbosity):
        self.logger = logging.getLogger('tiff2jp2')
        self.logger.setLevel(verbosity)
        ch = logging.StreamHandler()
        ch.setLevel(verbosity)
        self.logger.addHandler(ch)

    def __enter__(self):
        self.tiff_fp = libtiff.open(self.tiff_filename)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        libtiff.close(self.tiff_fp)

    def run(self):

        self.get_main_ifd()
        self.copy_image()
        self.append_extra_jp2_boxes()

    def append_extra_jp2_boxes(self):
        """
        Copy over the TIFF IFD.  Place it in a UUID box.  Append to the JPEG
        2000 file.
        """
        if not self.create_exif_uuid:
            return

        # create a bytesio object for the IFD
        b = io.BytesIO()

        # write this 32-bit header into the UUID, no matter if we had bigtiff
        # or regular tiff or big endian
        data = struct.pack('<BBHI', 73, 73, 42, 8)
        b.write(data)

        if 700 in self.tags and self.create_xmp_uuid:
            # remove the XMLPacket data from the IFD dictionary
            xmp_data = self.tags.pop(700)
        else:
            xmp_data = None

        self._write_ifd(b, self.tags)

        # create the Exif UUID
        if self.found_geotiff_tags:
            # geotiff UUID
            the_uuid = UUID('b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03')
            payload = b.getvalue()
        else:
            # Make it an exif UUID.
            the_uuid = UUID(bytes=b'JpgTiffExif->JP2')
            payload = b'EXIF\0\0' + b.getvalue()

        # the length of the box is the length of the payload plus 8 bytes
        # to store the length of the box and the box ID
        box_length = len(payload) + 8

        uuid_box = jp2box.UUIDBox(the_uuid, payload, box_length)
        with open(self.jp2_filename, mode='ab') as f:
            uuid_box.write(f)

        if xmp_data is None:
            return

        # create the XMP UUID
        the_uuid = jp2box.UUID('be7acfcb-97a9-42e8-9c71-999491e3afac')
        payload = bytes(xmp_data['payload'])
        box_length = len(payload) + 8
        uuid_box = jp2box.UUIDBox(the_uuid, payload, box_length)
        with open(self.jp2_filename, mode='ab') as f:
            uuid_box.write(f)

    def get_main_ifd(self):
        """
        Read all the tags in the main IFD.  We do it this way because of the
        difficulty in using TIFFGetFieldDefaulted when the datatype of a tag
        can differ.
        """

        with open(self.tiff_filename, 'rb') as tfp:

            self.read_tiff_header(tfp)

            self.tags = self.read_ifd(tfp)

            if 34665 in self.tags:
                # we have an EXIF IFD
                offset = self.tags[34665]['payload'][0]
                tfp.seek(offset)
                exif_ifd = self.read_ifd(tfp)

                self.tags[34665]['payload'] = exif_ifd

    def read_ifd(self, tfp):
        """
        Process either the main IFD or an Exif IFD

        Parameters
        ----------
        tfp : file-like
            FILE pointer for TIFF

        Returns
        -------
        dictionary of the TIFF IFD
        """

        self.found_geotiff_tags = False

        tag_length = 20 if self.version == _BIGTIFF else 12

        # how many tags?
        if self.version == _BIGTIFF:
            buffer = tfp.read(8)
            num_tags, = struct.unpack(self.endian + 'Q', buffer)
        else:
            buffer = tfp.read(2)
            num_tags, = struct.unpack(self.endian + 'H', buffer)

        # Ok, so now we have the IFD main body, but following that we have
        # the tag payloads that cannot fit into 4 bytes.

        # the IFD main body in the TIFF.  As it might be big endian, we
        # cannot just process it as one big chunk.
        buffer = tfp.read(num_tags * tag_length)

        if self.version == _BIGTIFF:
            tag_format_str = self.endian + 'HHQQ'
            tag_payload_offset = 12
            max_tag_payload_length = 8
        else:
            tag_format_str = self.endian + 'HHII'
            tag_payload_offset = 8
            max_tag_payload_length = 4

        tags = {}

        for idx in range(num_tags):

            self.logger.debug(f'tag #: {idx}')

            tag_data = buffer[idx * tag_length:(idx + 1) * tag_length]

            tag, dtype, nvalues, offset = struct.unpack(tag_format_str, tag_data)  # noqa : E501

            if tag == 34735:
                self.found_geotiff_tags = True

            payload_length = tag_dtype[dtype]['nbytes'] * nvalues

            if payload_length > max_tag_payload_length:
                # the payload does not fit into the tag entry, so use the
                # offset to seek to that position
                current_position = tfp.tell()
                tfp.seek(offset)
                payload_buffer = tfp.read(payload_length)
                tfp.seek(current_position)

                # read the payload from the TIFF
                payload_format = tag_dtype[dtype]['format'] * nvalues
                payload = struct.unpack(
                    self.endian + payload_format, payload_buffer
                )

            else:
                # the payload DOES fit into the TIFF tag entry
                payload_buffer = tag_data[tag_payload_offset:]

                # read ALL of the payload buffer
                fmt = tag_dtype[dtype]['format']
                num_items = (
                    int(max_tag_payload_length / tag_dtype[dtype]['nbytes'])
                )
                payload_format = self.endian + fmt * num_items
                payload = struct.unpack(payload_format, payload_buffer)

                # Extract the actual payload.  Two things going
                # on here.  First of all, not all of the items may
                # be used.  For example, if the payload length is
                # 4 bytes but the format string was HHH, the that
                # last 16 bit value is not wanted, so we should
                # discard it.  Second thing is that the signed and
                # unsigned rational datatypes effectively have twice
                # the number of values so we need to account for that.
                if dtype in [5, 10]:
                    payload = payload[:2 * nvalues]
                else:
                    payload = payload[:nvalues]

            tags[tag] = {
                'dtype': dtype,
                'nvalues': nvalues,
                'payload': payload
            }

        return tags

    def _write_ifd(self, b, tags):

        # keep this for writing to the UUID, which will always be for 32-bit
        # TIFFs
        little_tiff_tag_length = 12

        if self.exclude_tags is not None:
            for tag in self.exclude_tags:
                if tag in tags:
                    tags.pop(tag)

        num_tags = len(tags)

        write_buffer = struct.pack('<H', num_tags)
        b.write(write_buffer)

        # Ok, so now we have the IFD main body, but following that we have
        # the tag payloads that cannot fit into 4 bytes.

        # the IFD main body in the TIFF.  As it might be big endian, we cannot
        # just process it as one big chunk.

        ifd_start_loc = b.tell()
        after_ifd_position = ifd_start_loc + num_tags * little_tiff_tag_length

        # We write a little-TIFF IFD
        max_tag_payload_length = 4

        for idx, tag in enumerate(tags):

            tag_offset = ifd_start_loc + idx * little_tiff_tag_length
            self.logger.debug(f'tag #: {tag}, writing to {tag_offset}')
            self.logger.debug(f'tag #: {tag}, after IFD {after_ifd_position}')

            b.seek(tag_offset)

            dtype = tags[tag]['dtype']
            nvalues = tags[tag]['nvalues']
            payload = tags[tag]['payload']

            payload_length = tag_dtype[dtype]['nbytes'] * nvalues

            if payload_length > max_tag_payload_length:
                # the payload does not fit into the tag entry

                # read the payload from the TIFF
                payload_format = tag_dtype[dtype]['format'] * nvalues

                # write the tag entry to the UUID
                new_offset = after_ifd_position
                outbuffer = struct.pack(
                    '<HHII', tag, dtype, nvalues, new_offset
                )
                b.write(outbuffer)

                # now write the payload at the outlying position and then come
                # back to the same position in the file stream
                cpos = b.tell()
                b.seek(new_offset)

                out_format = '<' + tag_dtype[dtype]['format'] * nvalues
                outbuffer = struct.pack(out_format, *payload)
                b.write(outbuffer)

                # keep track of the next position to write out-of-IFD data
                after_ifd_position = b.tell()
                b.seek(cpos)

            else:
                # the payload DOES fit into the TIFF tag entry

                payload_format = (
                    tag_dtype[dtype]['format']
                    * int(max_tag_payload_length / tag_dtype[dtype]['nbytes'])
                )

                # the payload DOES fit into the UUID tag entry

                # so write it back into the tag entry in the UUID
                outbuffer = struct.pack('<HHI', tag, dtype, nvalues)
                b.write(outbuffer)

                payload_format = tag_dtype[dtype]['format'] * nvalues

                # we may need to alter the output format
                if payload_format in ['H', 'B', 'I']:
                    # just write it as an integer
                    payload_format = 'I'

                if tag == 34665:
                    # special case for an EXIF IFD
                    outbuffer = struct.pack('<I', after_ifd_position)
                    b.write(outbuffer)
                    b.seek(after_ifd_position)
                    self._write_ifd(b, payload)
                else:
                    # write a normal tag
                    outbuffer = struct.pack('<' + payload_format, *payload)
                    b.write(outbuffer)

    def read_tiff_header(self, tfp):
        """
        Get the endian-ness of the TIFF, seek to the main IFD
        """

        buffer = tfp.read(4)
        data = struct.unpack('BB', buffer[:2])

        # big endian or little endian?
        if data[0] == 73 and data[1] == 73:
            # little endian
            self.endian = '<'
        elif data[0] == 77 and data[1] == 77:
            # big endian
            self.endian = '>'
        else:
            msg = (
                f"The byte order indication in the TIFF header "
                f"({data}) is invalid.  It should be either "
                f"{bytes([73, 73])} or {bytes([77, 77])}."
            )
            raise RuntimeError(msg)

        # version number and offset to the first IFD
        version, = struct.unpack(self.endian + 'H', buffer[2:4])
        self.version = _TIFF if version == 42 else _BIGTIFF

        if self.version == _BIGTIFF:
            buffer = tfp.read(12)
            _, _, offset = struct.unpack(self.endian + 'HHQ', buffer)
        else:
            buffer = tfp.read(4)
            offset, = struct.unpack(self.endian + 'I', buffer)
        tfp.seek(offset)

    def get_tag_value(self, tagnum):
        """
        Return the value associated with the tag.  Some tags are not actually
        written into the IFD, but are instead "defaulted".

        Returns
        -------
        tag value
        """

        if tagnum not in self.tags and tagnum == 284:
            # PlanarConfig is not always written into the IFD, defaults to 1
            return 1

        if tagnum not in self.tags and tagnum == 339:
            # SampleFormat is not always written into the IFD, defaults to 1
            return 1

        # The tag value is always stored as a tuple with at least one member.
        return self.tags[tagnum]['payload'][0]

    def copy_image(self):
        """
        Transfer the image data from the TIFF to the JPEG 2000 file.  If the
        TIFF has a stripped configuration, this may be somewhat inefficient.
        """

        if libtiff.isTiled(self.tiff_fp):
            isTiled = True
        else:
            isTiled = False

        photo = self.get_tag_value(262)
        imagewidth = self.get_tag_value(256)
        imageheight = self.get_tag_value(257)
        spp = self.get_tag_value(277)
        sf = self.get_tag_value(339)
        bps = self.get_tag_value(258)
        planar = self.get_tag_value(284)

        if sf not in [libtiff.SampleFormat.UINT, libtiff.SampleFormat.VOID]:
            sampleformat_str = self.tagvalue2str(libtiff.SampleFormat, sf)

            msg = (
                f"The TIFF SampleFormat is {sampleformat_str}.  Only UINT "
                "and VOID are supported."
            )
            raise RuntimeError(msg)

        if bps not in [8, 16]:
            msg = (
                f"The TIFF BitsPerSample is {bps}.  Only 8 and 16 bits per "
                "sample are supported."
            )
            raise RuntimeError(msg)

        if bps == 8 and sf == libtiff.SampleFormat.UINT:
            dtype = np.uint8
        if bps == 16 and sf == libtiff.SampleFormat.UINT:
            dtype = np.uint16

        if (
            planar == libtiff.PlanarConfig.SEPARATE
            and self.tilesize is not None
        ):
            msg = (
                "A separated planar configuration is not supported when a "
                "tile size is specified."
            )
            raise RuntimeError(msg)

        if libtiff.isTiled(self.tiff_fp):
            tw = self.get_tag_value(322)
            th = self.get_tag_value(323)
        else:
            tw = imagewidth
            rps = self.get_tag_value(278)

        if self.tilesize is not None:
            # The JP2K tile size was specified.  Compute the number of JP2K
            # tile rows and columns.
            jth, jtw = self.tilesize

            num_jp2k_tile_rows = int(np.ceil(imagewidth / jtw))
            num_jp2k_tile_cols = int(np.ceil(imagewidth / jtw))

        if photo == libtiff.Photometric.YCBCR:
            # Using the RGBA interface is the only reasonable way to deal with
            # this.
            use_rgba_interface = True
        elif photo == libtiff.Photometric.PALETTE:
            # Using the RGBA interface is the only reasonable way to deal with
            # this.  The single plane gets turned into RGB.
            use_rgba_interface = True
            spp = 3
        else:
            use_rgba_interface = False

        jp2 = Jp2k(
            self.jp2_filename,
            shape=(imageheight, imagewidth, spp),
            tilesize=self.tilesize,
            **self.jp2_kwargs
        )

        if not libtiff.RGBAImageOK(self.tiff_fp):
            photometric_string = self.tagvalue2str(libtiff.Photometric, photo)
            msg = (
                f"The TIFF Photometric tag is {photometric_string}.  It is "
                "not supported by this program."
            )
            raise RuntimeError(msg)

        elif self.tilesize is None:

            # this handles both cases of a striped TIFF and a tiled TIFF

            self._write_rgba_single_tile(
                photo, imagewidth, imageheight, spp, jp2
            )
            jp2.finalize(force_parse=True)

        elif isTiled and self.tilesize is not None:

            self._write_tiled_tiff_to_tiled_jp2k(
                imagewidth, imageheight, spp,
                jtw, jth, tw, th,
                num_jp2k_tile_cols, num_jp2k_tile_rows,
                dtype,
                use_rgba_interface,
                jp2
            )

        elif not isTiled and self.tilesize is not None:

            self._write_striped_tiff_to_tiled_jp2k(
                imagewidth, imageheight, spp,
                jtw, jth, rps,
                num_jp2k_tile_cols, num_jp2k_tile_rows,
                dtype,
                use_rgba_interface,
                jp2
            )

    def tagvalue2str(self, cls, tag_value):
        """
        Take a class that encompasses all of a tag's allowed values and find
        the name of that value.
        """

        tag_value_string = [
            key for key in dir(cls) if getattr(cls, key) == tag_value
        ][0]

        return tag_value_string

    def _write_rgba_single_tile(
        self, photo, imagewidth, imageheight, spp, jp2
    ):
        """
        If no jp2k tiling was specified and if the image is ok to read
        via the RGBA interface, then just do that.  The image will be
        written with the tilesize equal to the image size, so it will
        be written using a single write operation.

        Parameters
        ----------
        photo, imagewidth, imageheight, spp : int
            TIFF tag values corresponding to the photometric interpretation,
            image width and height, and samples per pixel
        jp2 : JP2K object
            Write to this JPEG2000 file
        """
        msg = (
            "Reading using the RGBA interface, writing as a single tile "
            "image."
        )
        self.logger.info(msg)

        if photo not in [
            libtiff.Photometric.MINISWHITE,
            libtiff.Photometric.MINISBLACK,
            libtiff.Photometric.PALETTE,
            libtiff.Photometric.YCBCR,
            libtiff.Photometric.RGB
        ]:
            photostr = self.tagvalue2str(libtiff.Photometric, photo)
            msg = (
                "Beware, the RGBA interface to attempt to read this TIFF "
                f"when it has a PhotometricInterpretation of {photostr}."
            )
            warnings.warn(msg)

        image = libtiff.readRGBAImageOriented(
            self.tiff_fp, imagewidth, imageheight
        )

        if spp < 4:
            image = image[:, :, :3]

        jp2[:] = image

    def _write_tiled_tiff_to_tiled_jp2k(
        self, imagewidth, imageheight, spp, jtw, jth, tw, th,
        num_jp2k_tile_cols, num_jp2k_tile_rows, dtype, use_rgba_interface, jp2
    ):
        """
        The input TIFF image is tiled and we are to create the output JPEG2000
        image with specific tile dimensions.

        Parameters
        ----------
        imagewidth, imageheight, spp : int
            TIFF tag values corresponding to the photometric interpretation,
            image width and height, and samples per pixel.
        jtw, jth : int
            The tile dimensions for the JPEG2000 image.
        tw, th : int
            The tile dimensions for the TIFF image.
        num_jp2k_tile_cols, num_jp2k_tile_rows
            The number of tiles down the rows and across the columns for the
            JPEG2000 image
        dtype : np.dtype
            Datatype of the image.
        use_rgba_interface : bool
            If true, use the RGBA interface to read the TIFF image data.
        jp2 : JP2K object
            Write to this JPEG2000 file
        """

        num_tiff_tile_cols = int(np.ceil(imagewidth / tw))

        partial_jp2_tile_rows = (imageheight / jth) != (imageheight // jth)
        partial_jp2_tile_cols = (imagewidth / jtw) != (imagewidth // jtw)

        rgba_tile = np.zeros((th, tw, 4), dtype=np.uint8)

        self.logger.debug(f'image:  {imageheight} x {imagewidth}')
        self.logger.debug(f'jptile:  {jth} x {jtw}')
        self.logger.debug(f'ttile:  {th} x {tw}')
        for idx, tilewriter in enumerate(jp2.get_tilewriters()):

            # populate the jp2k tile with tiff tiles
            self.logger.info(f'Tile:  #{idx}')
            self.logger.debug(f'J tile row:  #{idx // num_jp2k_tile_cols}')
            self.logger.debug(f'J tile col:  #{idx % num_jp2k_tile_cols}')

            jp2k_tile = np.zeros((jth, jtw, spp), dtype=dtype)
            tiff_tile = np.zeros((th, tw, spp), dtype=dtype)

            jp2k_tile_row = int(np.ceil(idx // num_jp2k_tile_cols))
            jp2k_tile_col = int(np.ceil(idx % num_jp2k_tile_cols))

            # the coordinates of the upper left pixel of the jp2k tile
            julr, julc = jp2k_tile_row * jth, jp2k_tile_col * jtw

            # loop while the upper left corner of the current tiff file is
            # less than the lower left corner of the jp2k tile
            r = julr
            while (r // th) * th < min(julr + jth, imageheight):
                c = julc

                tilenum = libtiff.computeTile(self.tiff_fp, c, r, 0, 0)
                self.logger.debug(f'TIFF tile # {tilenum}')

                tiff_tile_row = int(np.ceil(tilenum // num_tiff_tile_cols))
                tiff_tile_col = int(np.ceil(tilenum % num_tiff_tile_cols))

                # the coordinates of the upper left pixel of the TIFF tile
                tulr = tiff_tile_row * th
                tulc = tiff_tile_col * tw

                # loop while the left corner of the current tiff tile is
                # less than the right hand corner of the jp2k tile
                while ((c // tw) * tw) < min(julc + jtw, imagewidth):

                    if use_rgba_interface:
                        libtiff.readRGBATile(
                            self.tiff_fp, tulc, tulr, rgba_tile
                        )

                        # flip the tile upside down!!
                        tiff_tile = np.flipud(rgba_tile[:, :, :3])
                    else:
                        libtiff.readEncodedTile(
                            self.tiff_fp, tilenum, tiff_tile
                        )

                    # determine how to fit this tiff tile into the jp2k
                    # tile
                    #
                    # these are the section coordinates in image space
                    ulr = max(julr, tulr)
                    llr = min(julr + jth, tulr + th)

                    ulc = max(julc, tulc)
                    urc = min(julc + jtw, tulc + tw)

                    # convert to JP2K tile coordinates
                    jrows = slice(ulr % jth, (llr - 1) % jth + 1)
                    jcols = slice(ulc % jtw, (urc - 1) % jtw + 1)

                    # convert to TIFF tile coordinates
                    trows = slice(ulr % th, (llr - 1) % th + 1)
                    tcols = slice(ulc % tw, (urc - 1) % tw + 1)

                    jp2k_tile[jrows, jcols, :] = tiff_tile[trows, tcols, :]

                    # move exactly one tiff tile over
                    c += tw

                    tilenum = libtiff.computeTile(self.tiff_fp, c, r, 0, 0)

                    tiff_tile_row = int(
                        np.ceil(tilenum // num_tiff_tile_cols)
                    )
                    tiff_tile_col = int(
                        np.ceil(tilenum % num_tiff_tile_cols)
                    )

                    # the coordinates of the upper left pixel of the TIFF
                    # tile
                    tulr = tiff_tile_row * th
                    tulc = tiff_tile_col * tw

                r += th

            # last tile column?  If so, we may have a partial tile.
            if (
                partial_jp2_tile_cols
                and jp2k_tile_col == num_jp2k_tile_cols - 1
            ):
                last_j2k_cols = slice(0, imagewidth - julc)
                jp2k_tile = jp2k_tile[:, last_j2k_cols, :].copy()
            if (
                partial_jp2_tile_rows
                and jp2k_tile_row == num_jp2k_tile_rows - 1
            ):
                last_j2k_rows = slice(0, imageheight - julr)
                jp2k_tile = jp2k_tile[last_j2k_rows, :, :].copy()

            tilewriter[:] = jp2k_tile

    def _write_striped_tiff_to_tiled_jp2k(
        self, imagewidth, imageheight, spp, jtw, jth, rps,
        num_jp2k_tile_cols, num_jp2k_tile_rows, dtype, use_rgba_interface, jp2
    ):
        """
        The input TIFF image is striped and we are to create the output
        JPEG2000 image as a tiled JP2K.

        Parameters
        ----------
        imagewidth, imageheight, spp : int
            TIFF tag values corresponding to the photometric interpretation,
            image width and height, and samples per pixel.
        jtw, jth : int
            The tile dimensions for the JPEG2000 image.
        rps : int
            The number of rows per strip in the TIFF.
        num_jp2k_tile_cols, num_jp2k_tile_rows
            The number of tiles down the rows and across the columns for the
            JPEG2000 image
        dtype : np.dtype
            Datatype of the image.
        use_rgba_interface : bool
            If true, use the RGBA interface to read the TIFF image data.
        jp2 : JP2K object
            Write to this JPEG2000 file
        """

        self.logger.debug(f'image:  {imageheight} x {imagewidth}')
        self.logger.debug(f'jptile:  {jth} x {jtw}')
        num_strips = libtiff.numberOfStrips(self.tiff_fp)

        num_jp2k_tile_cols = int(np.ceil(imagewidth / jtw))

        for idx, tilewriter in enumerate(jp2.get_tilewriters()):

            self.logger.info(f'Tile: #{idx}')

            jp2k_tile_row = idx // num_jp2k_tile_cols
            jp2k_tile_col = idx % num_jp2k_tile_cols

            # the coordinates of the upper left pixel of the jp2k tile
            julr, julc = jp2k_tile_row * jth, jp2k_tile_col * jtw

            if jp2k_tile_col == 0:

                tiff_multi_strip = self._construct_multi_strip(
                    julr, num_strips, jth, imageheight, imagewidth, rps, spp,
                    dtype, use_rgba_interface
                )

            # Get the multi-strip coordinates of the upper left jp2k corner
            stripnum = libtiff.computeStrip(self.tiff_fp, julr, 0)
            tulr = stripnum * rps

            ms_ulr = julr - tulr
            ms_ulc = julc

            # ms_lrr = ms_ulr + min(ms_ulr + jth, imageheight)
            # ms_lrc = ms_ulc + min(ms_ulc + jtw, imagewidth)
            ms_lrr = min(ms_ulr + jth, imageheight - tulr)
            ms_lrc = min(ms_ulc + jtw, imagewidth)

            rows = slice(ms_ulr, ms_lrr)
            cols = slice(ms_ulc, ms_lrc)

            tilewriter[:] = tiff_multi_strip[rows, cols, :]

    def _construct_multi_strip(
        self, julr, num_strips, jth, imageheight, imagewidth, rps, spp, dtype,
        use_rgba_interface
    ):
        """
        TIFF strips are pretty inefficient.  If our I/O was stupidly focused
        solely on each JP2K tile, we would read in each TIFF strip multiple
        times.  If instead, we read in ALL the strips that will encompass that
        current row of JP2K tiles, we will save ourselves from pushing around
        a lot of electrons.

        Parameters
        ----------
        imagewidth, imageheight, spp : int
            TIFF tag values corresponding to the photometric interpretation,
            image width and height, and samples per pixel.
        jth : int
            The number of rows in a JP2K tile.
        rps : int
            The number of rows per strip in the TIFF.
        julr : int
            The top row of the current JP2K tile row.
        dtype : np.dtype
            Datatype of the image.
        use_rgba_interface : bool
            If true, use the RGBA interface to read the TIFF image data.

        Returns
        -------
        tiff_multi_strip : np.array
            Holds all the TIFF strips that tightly encompass the current JP2K
            tile row.
        """
        # We need to create a TIFF "multi-strip" that can hold all of
        # the JP2K tiles in a JP2K tile row.
        r = julr
        top_strip_num = libtiff.computeStrip(self.tiff_fp, r, 0)

        # advance thru to the next tile row
        while (r // rps) * rps < min(julr + jth, imageheight):
            r += rps
        bottom_strip_num = libtiff.computeStrip(self.tiff_fp, r, 0)

        # compute the number of rows contained between the top strip
        # and the bottom strip
        num_rows = (bottom_strip_num - top_strip_num) * rps

        if use_rgba_interface:
            tiff_strip = np.zeros((rps, imagewidth, 4), dtype=np.uint8)
            tiff_multi_strip = np.zeros(
                (num_rows, imagewidth, spp), dtype=np.uint8
            )
        else:
            tiff_strip = np.zeros((rps, imagewidth, spp), dtype=dtype)
            tiff_multi_strip = np.zeros(
                (num_rows, imagewidth, spp), dtype=dtype
            )

        # fill the multi-strip
        for stripnum in range(top_strip_num, bottom_strip_num):

            if use_rgba_interface:

                libtiff.readRGBAStrip(
                    self.tiff_fp, stripnum * rps, tiff_strip
                )

                # must flip the rows (!!) and get rid of the alpha
                # plane
                tiff_strip = np.flipud(tiff_strip[:, :, :spp])

            else:

                libtiff.readEncodedStrip(
                    self.tiff_fp, stripnum, tiff_strip
                )

            # push the strip into the multi-strip
            top_row = (stripnum - top_strip_num) * rps
            bottom_row = (stripnum - top_strip_num + 1) * rps
            rows = slice(top_row, bottom_row)
            tiff_multi_strip[rows, :, :] = tiff_strip

        return tiff_multi_strip
