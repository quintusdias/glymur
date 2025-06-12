"""Core definitions to be shared amongst the modules."""
# standard library imports
import io
import logging
import shutil
import struct
from typing import Tuple
from uuid import UUID

# local imports
from . import jp2box
from .lib._tiff import DATATYPE2FMT
from .jp2k import Jp2k
from glymur.core import RESTRICTED_ICC_PROFILE

# Mnemonics for the two TIFF format version numbers.
TIFF = 42
BIGTIFF = 43


class _2JP2Converter(object):
    """
    This private class is used by both the TIFF2JP2 and the JPEG2JP2
    converters.

    Attributes
    ----------
    create_exif_uuid : bool
        Create a UUIDBox for the TIFF IFD metadata.
    tilesize : tuple
        The dimensions of a tile in the JP2K file.
    verbosity : int
        Set the level of logging, i.e. WARNING, INFO, etc.
    """

    def __init__(
        self,
        create_exif_uuid: bool,
        create_xmp_uuid: bool,
        include_icc_profile: bool,
        tilesize: Tuple[int, int] | None,
        verbosity: int
    ):

        self.create_exif_uuid = create_exif_uuid
        self.create_xmp_uuid = create_xmp_uuid
        self.include_icc_profile = include_icc_profile
        self.tilesize = tilesize

        # Assume that there is no ICC profile tag until we know otherwise.
        self.icc_profile = None

        self.setup_logging(verbosity)

    def read_ifd(self, tfp):
        """Process either the main IFD or an Exif IFD

        Parameters
        ----------
        tfp : file-like
            FILE pointer for TIFF

        Returns
        -------
        dictionary of the TIFF IFD
        """

        self.found_geotiff_tags = False

        tag_length = 20 if self.version == BIGTIFF else 12

        # how many tags?
        if self.version == BIGTIFF:
            buffer = tfp.read(8)
            (num_tags,) = struct.unpack(self.endian + "Q", buffer)
        else:
            buffer = tfp.read(2)
            (num_tags,) = struct.unpack(self.endian + "H", buffer)

        # Ok, so now we have the IFD main body, but following that we have
        # the tag payloads that cannot fit into 4 bytes.

        # the IFD main body in the TIFF.  As it might be big endian, we
        # cannot just process it as one big chunk.
        buffer = tfp.read(num_tags * tag_length)

        if self.version == BIGTIFF:
            tag_format_str = self.endian + "HHQQ"
            tag_payload_offset = 12
            max_tag_payload_length = 8
        else:
            tag_format_str = self.endian + "HHII"
            tag_payload_offset = 8
            max_tag_payload_length = 4

        tags = {}

        for idx in range(num_tags):

            self.logger.debug(f"tag #: {idx}")

            tag_data = buffer[idx * tag_length:(idx + 1) * tag_length]

            tag, dtype, nvalues, offset = struct.unpack(
                tag_format_str, tag_data
            )  # noqa : E501

            if tag == 34735:
                self.found_geotiff_tags = True

            payload_length = DATATYPE2FMT[dtype]["nbytes"] * nvalues

            if tag in (34665, 34853):

                # found exif or gps ifd
                # save our location, go get that IFD, and come on back
                orig_pos = tfp.tell()
                tfp.seek(offset)
                payload = self.read_ifd(tfp)
                tfp.seek(orig_pos)

            elif payload_length > max_tag_payload_length:
                # the payload does not fit into the tag entry, so use the
                # offset to seek to that position
                current_position = tfp.tell()
                tfp.seek(offset)
                payload_buffer = tfp.read(payload_length)
                tfp.seek(current_position)

                # read the payload from the TIFF
                payload_format = DATATYPE2FMT[dtype]["format"] * nvalues
                payload = struct.unpack(
                    self.endian + payload_format,
                    payload_buffer
                )

            else:
                # the payload DOES fit into the TIFF tag entry
                payload_buffer = tag_data[tag_payload_offset:]

                # read ALL of the payload buffer
                fmt = DATATYPE2FMT[dtype]["format"]
                nelts = max_tag_payload_length / DATATYPE2FMT[dtype]["nbytes"]
                num_items = int(nelts)
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
                    payload = payload[: 2 * nvalues]
                else:
                    payload = payload[:nvalues]

            tags[tag] = {
                "dtype": dtype, "nvalues": nvalues, "payload": payload
            }

        return tags

    def setup_logging(self, verbosity):

        self.logger = logging.getLogger("tiff2jp2")
        self.logger.setLevel(verbosity)
        ch = logging.StreamHandler()
        ch.setLevel(verbosity)
        self.logger.addHandler(ch)

    def read_tiff_header(self, tfp):
        """Get the endian-ness of the TIFF, seek to the main IFD"""

        buffer = tfp.read(4)
        data = struct.unpack("BB", buffer[:2])

        # big endian or little endian?
        if data[0] == 73 and data[1] == 73:
            # little endian
            self.endian = "<"
        elif data[0] == 77 and data[1] == 77:
            # big endian
            self.endian = ">"
        # no other option is possible, libtiff.open would have errored out
        # else:
        #     msg = (
        #         f"The byte order indication in the TIFF header "
        #         f"({data}) is invalid.  It should be either "
        #         f"{bytes([73, 73])} or {bytes([77, 77])}."
        #     )
        #     raise RuntimeError(msg)

        # version number and offset to the first IFD
        (version,) = struct.unpack(self.endian + "H", buffer[2:4])
        self.version = TIFF if version == 42 else BIGTIFF

        if self.version == BIGTIFF:
            buffer = tfp.read(12)
            _, _, offset = struct.unpack(self.endian + "HHQ", buffer)
        else:
            buffer = tfp.read(4)
            (offset,) = struct.unpack(self.endian + "I", buffer)
        tfp.seek(offset)

    def append_exif_uuid_box(self):
        """Append an EXIF UUID box onto the end of the JPEG 2000 file.  It will
        contain metadata from the TIFF IFD.
        """
        if not self.create_exif_uuid:
            return

        # create a bytesio object for the IFD
        b = io.BytesIO()

        # write this 32-bit header into the UUID, no matter if we had bigtiff
        # or regular tiff or big endian
        data = struct.pack("<BBHI", 73, 73, 42, 8)
        b.write(data)

        self.write_ifd(b, self.tags)

        # create the Exif UUID
        if self.found_geotiff_tags:
            # geotiff UUID
            the_uuid = UUID("b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03")
            payload = b.getvalue()
        else:
            # Make it an exif UUID.
            the_uuid = UUID(bytes=b"JpgTiffExif->JP2")
            payload = b"EXIF\0\0" + b.getvalue()

        # the length of the box is the length of the payload plus 8 bytes
        # to store the length of the box and the box ID
        box_length = len(payload) + 8

        uuid_box = jp2box.UUIDBox(the_uuid, payload, box_length)
        with self.jp2_path.open(mode="ab") as f:
            uuid_box.write(f)

        self.jp2.finalize(force_parse=True)

    def write_ifd(self, b, tags):
        """Write the IFD out to the UUIDBox.  We will always write IFDs
        for 32-bit TIFFs, i.e. 12 byte tags, meaning just 4 bytes within
        the tag for the tag data
        """

        little_tiff_tag_length = 12
        max_tag_payload_length = 4

        # exclude any unwanted tags
        if self.exclude_tags is not None:
            for tag in self.exclude_tags:
                if tag in tags:
                    tags.pop(tag)

        num_tags = len(tags)
        write_buffer = struct.pack("<H", num_tags)
        b.write(write_buffer)

        # Ok, so now we have the IFD main body, but following that we have
        # the tag payloads that cannot fit into 4 bytes.

        ifd_start_loc = b.tell()
        after_ifd_position = ifd_start_loc + num_tags * little_tiff_tag_length

        for idx, tag in enumerate(tags):

            tag_offset = ifd_start_loc + idx * little_tiff_tag_length
            self.logger.debug(f"tag #: {tag}, writing to {tag_offset}")
            self.logger.debug(f"tag #: {tag}, after IFD {after_ifd_position}")

            b.seek(tag_offset)

            try:
                dtype = tags[tag]["dtype"]
            except IndexError:
                breakpoint()
                pass

            nvalues = tags[tag]["nvalues"]
            payload = tags[tag]["payload"]

            payload_length = DATATYPE2FMT[dtype]["nbytes"] * nvalues

            if payload_length > max_tag_payload_length:

                # the payload does not fit into the tag entry

                # read the payload from the TIFF
                payload_format = DATATYPE2FMT[dtype]["format"] * nvalues

                # write the tag entry to the UUID
                new_offset = after_ifd_position
                buffer = struct.pack("<HHII", tag, dtype, nvalues, new_offset)
                b.write(buffer)

                # now write the payload at the outlying position and then come
                # back to the same position in the file stream
                cpos = b.tell()
                b.seek(new_offset)

                format = "<" + DATATYPE2FMT[dtype]["format"] * nvalues
                buffer = struct.pack(format, *payload)
                b.write(buffer)

                # keep track of the next position to write out-of-IFD data
                after_ifd_position = b.tell()
                b.seek(cpos)

            else:

                # the payload DOES fit into the TIFF tag entry
                # write the tag metadata
                buffer = struct.pack("<HHI", tag, dtype, nvalues)
                b.write(buffer)

                payload_format = DATATYPE2FMT[dtype]["format"] * nvalues

                # we may need to alter the output format
                if payload_format in ["H", "B", "I"]:
                    # just write it as an integer
                    payload_format = "I"

                if tag in (34665, 34853):

                    # special case for an EXIF or GPS IFD
                    buffer = struct.pack("<I", after_ifd_position)
                    b.write(buffer)
                    b.seek(after_ifd_position)
                    after_ifd_position = self.write_ifd(b, payload)

                else:

                    buffer = struct.pack("<" + payload_format, *payload)
                    b.write(buffer)

        return after_ifd_position

    def append_xmp_uuid_box(self):
        """Append an XMP UUID box onto the end of the JPEG 2000 file if there
        was an XMP tag in the TIFF IFD.
        """

        if self.xmp_data is None:
            return

        if not self.create_xmp_uuid:
            return

        # create the XMP UUID
        the_uuid = jp2box.UUID("be7acfcb-97a9-42e8-9c71-999491e3afac")
        box_length = len(self.xmp_data) + 8
        uuid_box = jp2box.UUIDBox(the_uuid, self.xmp_data, box_length)
        with self.jp2_path.open(mode="ab") as f:
            uuid_box.write(f)

    def rewrap_for_icc_profile(self):
        """Consume an ICC profile, if one is there."""
        if self.icc_profile is None and self.include_icc_profile:
            self.logger.warning("No ICC profile was found.")

        if self.icc_profile is None or not self.include_icc_profile:
            return

        self.logger.info(
            "Consuming an ICC profile into JP2 color specification box."
        )

        colr = jp2box.ColourSpecificationBox(
            method=RESTRICTED_ICC_PROFILE,
            precedence=0,
            icc_profile=self.icc_profile
        )

        # construct the new set of JP2 boxes, insert the color specification
        # box with the ICC profile
        jp2 = Jp2k(self.jp2_path)
        boxes = jp2.box
        boxes[2].box = [boxes[2].box[0], colr]

        # re-wrap the codestream, involves a file copy
        tmp_filename = str(self.jp2_path) + ".tmp"

        with open(tmp_filename, mode="wb") as tfile:
            jp2.wrap(tfile.name, boxes=boxes)

        shutil.move(tmp_filename, self.jp2_path)
