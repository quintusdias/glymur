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
from .jp2box import UUIDBox


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
    tilesize : tuple
        The dimensions of a tile in the JP2K file.
    create_uuid : bool
        Create a UUIDBox for the TIFF IFD metadata.
    version : int
        Identifies the TIFF as 32-bit TIFF or 64-bit TIFF.
    """

    def __init__(
        self, tiff_filename, jp2_filename, tilesize=None,
        verbosity=logging.CRITICAL, create_uuid=True, **kwargs
    ):

        self.tiff_filename = tiff_filename
        if not self.tiff_filename.exists():
            raise FileNotFoundError(f'{tiff_filename} does not exist')

        self.jp2_filename = jp2_filename
        self.tilesize = tilesize
        self.create_uuid = create_uuid

        self.kwargs = kwargs

        self.setup_logging(verbosity)

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

        if self.create_uuid:
            self.copy_metadata()

    def copy_metadata(self):
        """
        Copy over the TIFF IFD.  Place it in a UUID box.  Append to the JPEG
        2000 file.
        """
        # create a bytesio object for the IFD
        b = io.BytesIO()

        # write this 32-bit header into the UUID, no matter if we had bigtiff
        # or regular tiff or big endian
        data = struct.pack('<BBHI', 73, 73, 42, 8)
        b.write(data)

        self._process_tags(b)

        if self.found_geotiff_tags:
            # geotiff UUID
            uuid = UUID('b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03')
            payload = b.getvalue()
        else:
            # Make it an exif UUID.
            uuid = UUID(bytes=b'JpgTiffExif->JP2')
            payload = b'EXIF\0\0' + b.getvalue()

        # the length of the box is the length of the payload plus 8 bytes
        # to store the length of the box and the box ID
        box_length = len(payload) + 8

        uuid_box = UUIDBox(uuid, payload, box_length)
        with open(self.jp2_filename, mode='ab') as f:
            uuid_box.write(f)

    def get_main_ifd(self):
        """
        Read all the tags in the main IFD.  We do it this way because of the
        difficulty in using TIFFGetFieldDefaulted when the datatype of a tag
        can differ.
        """

        with open(self.tiff_filename, 'rb') as tfp:

            self.get_endianness(tfp)

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

            self.tags = {}

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
                    payload_format = (
                        tag_dtype[dtype]['format']
                        * int(max_tag_payload_length / tag_dtype[dtype]['nbytes'])  # noqa : E501
                    )

                    payload = struct.unpack(
                        self.endian + payload_format, payload_buffer
                    )

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

                self.tags[tag] = {
                    'dtype': dtype,
                    'nvalues': nvalues,
                    'payload': payload
                }

    def _process_tags(self, b):

        # keep this for writing to the UUID, which will always be 32-bit
        little_tiff_tag_length = 12

        num_tags = len(self.tags)

        write_buffer = struct.pack('<H', num_tags)
        b.write(write_buffer)

        # Ok, so now we have the IFD main body, but following that we have
        # the tag payloads that cannot fit into 4 bytes.

        # the IFD main body in the TIFF.  As it might be big endian, we cannot
        # just process it as one big chunk.

        tag_start_loc = b.tell()
        after_ifd_position = tag_start_loc + num_tags * little_tiff_tag_length

        # We write a little-TIFF IFD
        max_tag_payload_length = 4

        for idx, tag in enumerate(self.tags):

            self.logger.debug(f'tag #: {tag}')

            b.seek(tag_start_loc + idx * little_tiff_tag_length)

            dtype = self.tags[tag]['dtype']
            nvalues = self.tags[tag]['nvalues']
            payload = self.tags[tag]['payload']

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

                # read ALL of the payload buffer
                payload_format = (
                    tag_dtype[dtype]['format']
                    * int(max_tag_payload_length / tag_dtype[dtype]['nbytes'])
                )

                # Does it fit into the UUID tag entry (4 bytes)?
                if payload_length <= 4:

                    # so write it back into the tag entry in the UUID
                    outbuffer = struct.pack('<HHI', tag, dtype, nvalues)
                    b.write(outbuffer)

                    payload_format = tag_dtype[dtype]['format'] * nvalues

                    # we may need to alter the output format
                    if payload_format in ['H', 'B', 'I']:
                        # just write it as an integer
                        payload_format = 'I'

                    outbuffer = struct.pack('<' + payload_format, *payload)
                    b.write(outbuffer)

                else:

                    # UUID:  write the tag entry after the IFD
                    new_offset = after_ifd_position
                    outbuffer = struct.pack(
                        '<HHII', tag, dtype, nvalues, new_offset
                    )
                    b.write(outbuffer)

                    # now write the payload at the outlying position and then
                    # come back to the same position in the file stream
                    cpos = b.tell()
                    b.seek(new_offset)

                    out_format = '<' + tag_dtype[dtype]['format'] * nvalues
                    outbuffer = struct.pack(out_format, *payload)
                    b.write(outbuffer)

                    # keep track of the next position to write out-of-IFD data
                    after_ifd_position = b.tell()
                    b.seek(cpos)

    def get_endianness(self, tfp):
        """
        Set the endian-ness of the TIFF
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

    def _process_header(self, b, tfp):

        buffer = tfp.read(4)
        data = struct.unpack('BB', buffer[:2])

        # big endian or little endian?
        if data[0] == 73 and data[1] == 73:
            # little endian
            endian = '<'
        elif data[0] == 77 and data[1] == 77:
            # big endian
            endian = '>'
        else:
            msg = (
                f"The byte order indication in the TIFF header "
                f"({data}) is invalid.  It should be either "
                f"{bytes([73, 73])} or {bytes([77, 77])}."
            )
            raise RuntimeError(msg)

        # version number and offset to the first IFD
        version, = struct.unpack(endian + 'H', buffer[2:4])
        self.version = _TIFF if version == 42 else _BIGTIFF

        if self.version == _BIGTIFF:
            buffer = tfp.read(12)
            _, _, offset = struct.unpack(endian + 'HHQ', buffer)
        else:
            buffer = tfp.read(4)
            offset, = struct.unpack(endian + 'I', buffer)
        tfp.seek(offset)

        # write this 32-bit header into the UUID, no matter if we had bigtiff
        # or regular tiff or big endian
        data = struct.pack('<BBHI', 73, 73, 42, 8)
        b.write(data)

        return endian

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
            num_strips = libtiff.numberOfStrips(self.tiff_fp)

        if self.tilesize is not None:
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
            **self.kwargs
        )

        if not libtiff.RGBAImageOK(self.tiff_fp):
            photometric_string = self.tagvalue2str(libtiff.Photometric, photo)
            msg = (
                f"The TIFF Photometric tag is {photometric_string} and is "
                "not supported."
            )
            raise RuntimeError(msg)

        elif self.tilesize is None and libtiff.RGBAImageOK(self.tiff_fp):

            # if no jp2k tiling was specified and if the image is ok to read
            # via the RGBA interface, then just do that.
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

        elif isTiled and self.tilesize is not None:

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

        elif not isTiled and self.tilesize is not None:

            num_strips = libtiff.numberOfStrips(self.tiff_fp)

            num_jp2k_tile_cols = int(np.ceil(imagewidth / jtw))

            partial_jp2_tile_rows = (imageheight / jth) != (imageheight // jth)
            partial_jp2_tile_cols = (imagewidth / jtw) != (imagewidth // jtw)

            tiff_strip = np.zeros((rps, imagewidth, spp), dtype=dtype)
            rgba_strip = np.zeros((rps, imagewidth, 4), dtype=np.uint8)

            for idx, tilewriter in enumerate(jp2.get_tilewriters()):
                self.logger.info(f'Tile: #{idx}')

                jp2k_tile = np.zeros((jth, jtw, spp), dtype=dtype)

                jp2k_tile_row = idx // num_jp2k_tile_cols
                jp2k_tile_col = idx % num_jp2k_tile_cols

                # the coordinates of the upper left pixel of the jp2k tile
                julr, julc = jp2k_tile_row * jth, jp2k_tile_col * jtw

                # Populate the jp2k tile with tiff strips.
                # Move by strips from the start of the jp2k tile to the bottom
                # of the jp2k tile.  That last strip may be partially empty,
                # worry about that later.
                #
                # loop while the upper left corner of the current tiff file is
                # less than the lower left corner of the jp2k tile
                r = julr
                while (r // rps) * rps < min(julr + jth, imageheight):

                    stripnum = libtiff.computeStrip(self.tiff_fp, r, 0)

                    if stripnum >= num_strips:
                        # we've moved past the end of the tiff
                        break

                    if use_rgba_interface:

                        # must use the first row in the strip
                        libtiff.readRGBAStrip(
                            self.tiff_fp, stripnum * rps, rgba_strip
                        )
                        # must flip the rows (!!) and get rid of the alpha
                        # plane
                        tiff_strip = np.flipud(rgba_strip[:, :, :spp])

                    else:
                        libtiff.readEncodedStrip(
                            self.tiff_fp, stripnum, tiff_strip
                        )

                    # the coordinates of the upper left pixel of the TIFF
                    # strip
                    tulr = stripnum * rps
                    tulc = 0

                    # determine how to fit this tiff strip into the jp2k
                    # tile
                    #
                    # these are the section coordinates in image space
                    ulr = max(julr, tulr)
                    llr = min(julr + jth, tulr + rps)

                    ulc = max(julc, tulc)
                    urc = min(julc + jtw, tulc + tw)

                    # convert to JP2K tile coordinates
                    jrows = slice(ulr % jth, (llr - 1) % jth + 1)
                    jcols = slice(ulc % jtw, (urc - 1) % jtw + 1)

                    # convert to TIFF strip coordinates
                    trows = slice(ulr % rps, (llr - 1) % rps + 1)
                    tcols = slice(ulc % tw, (urc - 1) % tw + 1)

                    jp2k_tile[jrows, jcols, :] = tiff_strip[trows, tcols, :]

                    r += rps

                # last tile column?  If so, we may have a partial tile.
                # j2k_cols is not sufficient here, must shorten it from 250
                # to 230
                if (
                    partial_jp2_tile_cols
                    and jp2k_tile_col == num_jp2k_tile_cols - 1
                ):
                    # decrease the number of columns by however many it sticks
                    # over the image width
                    last_j2k_cols = slice(0, imagewidth - julc)
                    jp2k_tile = jp2k_tile[:, last_j2k_cols, :].copy()

                if (
                    partial_jp2_tile_rows
                    and stripnum == num_strips - 1
                ):
                    # decrease the number of rows by however many it sticks
                    # over the image height
                    last_j2k_rows = slice(0, imageheight - julr)
                    jp2k_tile = jp2k_tile[last_j2k_rows, :, :].copy()

                tilewriter[:] = jp2k_tile

    def tagvalue2str(self, cls, tag_value):
        """
        Take a class that encompasses all of a tag's allowed values and find
        the name of that value.
        """

        tag_value_string = [
            key for key in dir(cls) if getattr(cls, key) == tag_value
        ][0]

        return tag_value_string
