# standard library imports
from __future__ import annotations
import logging
import pathlib
import shutil
import sys
from typing import List, Tuple
import warnings

# 3rd party library imports
import numpy as np

# local imports
from glymur import Jp2k, set_option
from glymur.core import SRGB
from ._core_converter import _2JP2Converter
from .lib import _tiff as libtiff
from . import jp2box

# we need a lower case mapping from the tag name to the tag number
TAGNAME2NUM = {k.lower(): v["number"] for k, v in libtiff.TAGS.items()}


class Tiff2Jp2k(_2JP2Converter):
    """
    Transform a TIFF image into a JP2 image.

    Attributes
    ----------
    create_exif_uuid : bool
        Create a UUIDBox for the TIFF IFD metadata.
    dtype : np.dtype
        Datatype of the image.
    found_geotiff_tags : bool
        If true, then this TIFF must be a GEOTIFF
    imageheight, imagewidth : int
        Dimensions of the image.
    jp2 : JP2K object
        Write to this JPEG2000 file
    jp2_path : path
        Path to JPEG 2000 file to be written.
    jp2_kwargs : dict
        Keyword arguments to pass along to the Jp2k constructor.
    photo : int
        The photometric interpretation of the image.
    rps : int
        The number of rows per strip in the TIFF.
    spp : int
        Samples Per Pixel TIFF tag value
    tiff_path : path
        Path to TIFF file.
    tilesize : tuple
        The dimensions of a tile in the JP2K file.
    tw, th : int
        The tile dimensions for the TIFF image.
    version : int
        Identifies the TIFF as 32-bit TIFF or 64-bit TIFF.
    xmp_data : bytes
        Encoded bytes from XML_PACKET tag (700), or None if not present.
    """

    def __init__(
        self,
        tiff_path: pathlib.Path,
        jp2_path: pathlib.Path,
        create_exif_uuid: bool = True,
        create_xmp_uuid: bool = True,
        exclude_tags: List[int | str] | None = None,
        num_threads: int = 1,
        tilesize: Tuple[int, int] | None = None,
        include_icc_profile: bool = False,
        verbosity: int = logging.CRITICAL,
        **kwargs,
    ):
        """
        Construct the object.

        Parameters
        ----------
        create_exif_uuid : bool
            If true, create an EXIF UUID out of the TIFF metadata (tags)
        create_xmp_uuid : bool
            If true and if there is an XMLPacket (700) tag in the TIFF main
            IFD, it will be removed from the IFD and placed in a UUID box.
        include_icc_profile : bool
            If true, consume any ICC profile tag (34765) into the colour
            specification box.
        exclude_tags : list or None
            If not None and if create_exif_uuid is True, exclude any listed
            tags from the EXIF UUID.
        jp2_filename : path or str
            Path to JPEG 2000 file to be written.
        tiff_filename : path or str
            Path to TIFF file.
        tilesize : tuple
            The dimensions of a tile in the JP2K file.
        verbosity : int
            Set the level of logging, i.e. WARNING, INFO, etc.
        """
        super().__init__(
            create_exif_uuid, create_xmp_uuid, include_icc_profile, tilesize,
            verbosity
        )

        self.tiff_path = pathlib.Path(tiff_path)
        if not self.tiff_path.exists():
            raise FileNotFoundError(f"{tiff_path} does not exist")

        self.jp2_path = pathlib.Path(jp2_path)
        if self.jp2_path.exists():
            msg = (
                f'{str(self.jp2_path)} already exists, ',
                'please delete if you wish to overwrite.'
            )
            raise FileExistsError(msg)

        self.create_exif_uuid = create_exif_uuid
        self.create_xmp_uuid = create_xmp_uuid

        if exclude_tags is None:
            exclude_tags = []
        self.exclude_tags = self._process_exclude_tags(exclude_tags)

        self.jp2 = None
        self.jp2_kwargs = kwargs

        # Assume that there is no ColorMap tag until we know otherwise.
        self._colormap = None

        # Assume no XML_PACKET tag until we know otherwise.
        self.xmp_data = None

        self.setup_logging(verbosity)

        if num_threads > 1:
            set_option("lib.num_threads", num_threads)

    def _process_exclude_tags(self, exclude_tags):
        """The list of tags to exclude may be mixed type (str or integer).
        There is also the possibility that they may be capitalized differently
        than our internal list, so the goal here is to convert them all to
        integer values.

        Parameters
        ----------
        exclude_tags : list
            List of tags that are meant to be excluded from the EXIF UUID.

        Returns
        -------
        list of numeric tag values
        """
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

        return lst

    def __enter__(self):
        """The Tiff2Jp2k must be used with a context manager."""
        self.tiff_fp = libtiff.open(str(self.tiff_path))
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        libtiff.close(self.tiff_fp)

    def run(self):

        self.get_main_ifd()
        self.copy_image()
        self.append_extra_jp2_boxes()
        self.rewrap_jp2()

    def rewrap_jp2(self):
        """These re-wrap operations should be mutually exclusive.  An ICC
        profile should not exist in a TIFF with a colormap.
        """
        self.rewrap_for_colormap()
        self.rewrap_for_icc_profile()

    def rewrap_for_colormap(self):
        """If the photometric interpretation was PALETTE, then we need to
        insert a pclr box and a cmap (component mapping box).
        """

        photo = self.get_tag_value(262)
        if photo != libtiff.Photometric.PALETTE:
            return

        jp2h = [box for box in self.jp2.box if box.box_id == "jp2h"][0]

        bps = (8, 8, 8)
        pclr = jp2box.PaletteBox(
            palette=self._colormap,
            bits_per_component=bps,
            signed=(False, False, False)
        )
        jp2h.box.append(pclr)

        # append the component mapping box
        cmap = jp2box.ComponentMappingBox(
            component_index=(0, 0, 0),
            mapping_type=(1, 1, 1),
            palette_index=(0, 1, 2)
        )
        jp2h.box.append(cmap)

        # fix the colr box.  the colorspace needs to be changed from greyscale
        # to rgb
        colr = [box for box in jp2h.box if box.box_id == "colr"][0]
        colr.colorspace = SRGB

        temp_filename = str(self.jp2_path) + ".tmp"
        self.jp2.wrap(temp_filename, boxes=self.jp2.box)
        shutil.move(temp_filename, self.jp2_path)
        self.jp2._parse()

    def append_extra_jp2_boxes(self):
        """Copy over the TIFF IFD.  Place it in a UUID box.  Append to the JPEG
        2000 file.
        """
        self.append_exif_uuid_box()
        self.append_xmp_uuid_box()

    def get_main_ifd(self):
        """Read all the tags in the main IFD.  We do it this way because of the
        difficulty in using TIFFGetFieldDefaulted when the datatype of a tag
        can differ.
        """

        with self.tiff_path.open(mode="rb") as tfp:

            self.read_tiff_header(tfp)

            self.tags = self.read_ifd(tfp)

            if 320 in self.tags:

                # the TIFF must have PALETTE photometric interpretation
                data = np.array(self.tags[320]["payload"])
                self._colormap = data.reshape(len(data) // 3, 3)
                self._colormap = self._colormap / 65535
                self._colormap = (self._colormap * 255).astype(np.uint8)

            if 700 in self.tags:

                # XMLPacket
                self.xmp_data = bytes(self.tags[700]["payload"])

            else:
                self.xmp_data = None

            if 34675 in self.tags:
                # ICC profile
                self.icc_profile = bytes(self.tags[34675]["payload"])

            else:
                self.icc_profile = None

    def get_tag_value(self, tagnum):
        """Return the value associated with the tag.  Some tags are not
        actually written into the IFD, but are instead "defaulted".

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
        return self.tags[tagnum]["payload"][0]

    def copy_image(self):
        """Transfer the image data from the TIFF to the JPEG 2000 file."""

        if libtiff.isTiled(self.tiff_fp):
            isTiled = True
        else:
            isTiled = False

        self.photo = self.get_tag_value(262)
        self.imagewidth = self.get_tag_value(256)
        self.imageheight = self.get_tag_value(257)
        self.spp = self.get_tag_value(277)
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

        if bps == 8 and sf == libtiff.SampleFormat.UINT:
            self.dtype = np.uint8
        elif bps == 16 and sf == libtiff.SampleFormat.UINT:
            self.dtype = np.uint16
        else:
            msg = (
                f"Only unsigned sample formats and bits per sample values of "
                f"8 or 16 are supported.  The values in the TIFF are {bps} "
                f"{sf}."
            )
            raise RuntimeError(msg)

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
            self.tw = self.get_tag_value(322)
            self.th = self.get_tag_value(323)
        else:
            self.tw = self.imagewidth
            try:
                self.rps = self.get_tag_value(278)
            except KeyError:
                # stripped but no RowsPerStrip tag?  default to the image
                # height
                self.rps = self.imageheight

        if self.spp == 1:
            shape = (self.imageheight, self.imagewidth)
        else:
            shape = (self.imageheight, self.imagewidth, self.spp)

        self.jp2 = Jp2k(
            self.jp2_path,
            shape=shape,
            tilesize=self.tilesize,
            **self.jp2_kwargs
        )

        if not libtiff.RGBAImageOK(self.tiff_fp):
            photometric_string = self.tagvalue2str(
                libtiff.Photometric, self.photo
            )
            msg = (
                f"The TIFF Photometric tag is {photometric_string}.  It is "
                "not supported by this program."
            )
            raise RuntimeError(msg)
        elif self.tilesize is None and self.photo == libtiff.Photometric.YCBCR:
            # this handles both YCbCr cases of a striped TIFF and a tiled TIFF
            self._write_rgba_single_tile()
            self.jp2.finalize(force_parse=True)
        elif self.tilesize is None and isTiled:
            self._write_tiled_tiff_to_single_tile_jp2k()
        elif self.tilesize is None and not isTiled:
            self._write_stripped_tiff_to_single_tile_jp2k()
        elif isTiled and self.tilesize is not None:
            self._write_tiled_tiff_to_tiled_jp2k()
        elif not isTiled and self.tilesize is not None:
            self._write_striped_tiff_to_tiled_jp2k()

    def tagvalue2str(self, cls, tag_value):
        """Take a class that encompasses all of a tag's allowed values and find
        the name of that value.
        """

        tag_value_string = [
            key for key in dir(cls) if getattr(cls, key) == tag_value
        ][0]

        return tag_value_string

    def _write_rgba_single_tile(self):
        """If no jp2k tiling was specified and if the image is ok to read
        via the RGBA interface, then just do that.  The image will be
        written with the tilesize equal to the image size, so it will
        be written using a single write operation.
        """
        msg = (
            "Reading using the RGBA interface, writing as a single tile "
            "image."
        )
        self.logger.info(msg)

        image = libtiff.readRGBAImageOriented(
            self.tiff_fp, self.imagewidth, self.imageheight
        )

        # must reorder image planes on big-endian
        if sys.byteorder == "big":
            image = np.flip(image, axis=2)

        # potentially get rid of the alpha plane
        if self.spp < 4:
            image = image[:, :, :3]

        self.jp2[:] = image

    def _write_stripped_tiff_to_single_tile_jp2k(self):
        """The input TIFF image is stripped and we are to create the output
        JPEG2000 image as a single tile.
        """
        num_tiff_strip_rows = int(np.ceil(self.imageheight / self.rps))

        # This might be a bit bigger than the actual image because of a
        # possibly partial last strip.
        stripped_shape = (
            num_tiff_strip_rows * self.rps,
            self.imagewidth,
            self.spp
        )
        image = np.zeros(stripped_shape, dtype=self.dtype)

        tiff_strip = np.zeros(
            (self.rps, self.imagewidth, self.spp), dtype=self.dtype
        )

        # manually collect all the tiff strips, stuff into the image
        for stripnum in range(num_tiff_strip_rows):
            rows = slice(stripnum * self.rps, (stripnum + 1) * self.rps)
            libtiff.readEncodedStrip(self.tiff_fp, stripnum, tiff_strip)
            image[rows, :, :] = tiff_strip

        if self.imageheight != stripped_shape[0]:
            # cut the image down due to a partial last strip
            image = image[: self.imageheight, :, :]

        self.jp2[:] = image

    def _write_tiled_tiff_to_single_tile_jp2k(self):
        """The input TIFF image is tiled and we are to create the output
        JPEG2000 image as a single tile.
        """
        num_tiff_tile_cols = int(np.ceil(self.imagewidth / self.tw))
        num_tiff_tile_rows = int(np.ceil(self.imageheight / self.th))

        # tiled shape might differ from the final image shape if we have
        # partial tiles on the bottom and on the right
        final_shape = self.imageheight, self.imagewidth, self.spp
        tiled_shape = (
            num_tiff_tile_rows * self.th,
            num_tiff_tile_cols * self.tw,
            self.spp,
        )

        image = np.zeros(tiled_shape, dtype=self.dtype)
        tiff_tile = np.zeros((self.th, self.tw, self.spp), dtype=self.dtype)

        # manually collect all the tiff tiles, stuff into the image
        for tr in range(num_tiff_tile_rows):

            rows = slice(tr * self.th, (tr + 1) * self.th)

            for tc in range(num_tiff_tile_cols):
                ttile_num = num_tiff_tile_cols * tr + tc
                libtiff.readEncodedTile(self.tiff_fp, ttile_num, tiff_tile)

                cols = slice(tc * self.tw, (tc + 1) * self.tw)

                image[rows, cols, :] = tiff_tile

        if final_shape != tiled_shape:
            image = image[: final_shape[0], : final_shape[1], :]

        self.jp2[:] = image

    def _write_tiled_tiff_to_tiled_jp2k(self):
        """The input TIFF image is tiled and we are to create the output
        JPEG2000 image with specific tile dimensions.
        """
        for jp2k_tilenum, tilewriter in enumerate(self.jp2.get_tilewriters()):
            tiff_tiles = self._get_covering_tiles(jp2k_tilenum)
            jp2k_tile = self._cover_tile(jp2k_tilenum, tiff_tiles)
            self.logger.info(f"Writing tile {jp2k_tilenum}")
            tilewriter[:] = jp2k_tile

    def _cover_tile(self, jp2k_tile_num, tiff_tile_nums):
        """
        Fill in the jp2k tile with image data from the TIFF tiles.

        Parameters
        ----------
        jp2k_tilenum : int
            number of the JPEG2000 tile
        tiff_tile_nums : set
            all TIFF tile identifiers that cover the jpeg2000 tile
        """
        jth, jtw = self.tilesize

        # Does the JP2K have partial tiles on the far right and bottom of the
        # image.
        partial_jp2_tile_rows = (self.imageheight / jth) != (
            self.imageheight // jth
        )  # noqa : E501
        partial_jp2_tile_cols = (self.imagewidth / jtw) != (
            self.imagewidth // jtw
        )  # noqa : E501

        num_jp2k_tile_rows = int(np.ceil(self.imageheight / jth))
        num_jp2k_tile_cols = int(np.ceil(self.imagewidth / jtw))

        jp2k_tile_row = int(np.ceil(jp2k_tile_num // num_jp2k_tile_cols))
        jp2k_tile_col = int(np.ceil(jp2k_tile_num % num_jp2k_tile_cols))

        num_tiff_tile_cols = int(np.ceil(self.imagewidth / self.tw))

        jp2k_tile = np.zeros((jth, jtw, self.spp), dtype=self.dtype)

        # coordinates of jp2k upper left corner
        jp2k_ulx = jtw * jp2k_tile_col
        jp2k_uly = jth * jp2k_tile_row

        for ttile_num in tiff_tile_nums:

            # compute the coordinates of the upper left pixel of the TIFF tile
            tiff_tile_row = int(np.ceil(ttile_num // num_tiff_tile_cols))
            tiff_tile_col = int(np.ceil(ttile_num % num_tiff_tile_cols))
            tile_uly = tiff_tile_row * self.th
            tile_ulx = tiff_tile_col * self.tw

            if self.photo == libtiff.Photometric.YCBCR:

                rgba_tile = np.zeros((self.th, self.tw, 4), dtype=np.uint8)

                tiff_tile_row = int(np.ceil(ttile_num // num_tiff_tile_cols))
                tiff_tile_col = int(np.ceil(ttile_num % num_tiff_tile_cols))
                x = tiff_tile_col * self.tw
                y = tiff_tile_row * self.th

                libtiff.readRGBATile(self.tiff_fp, x, y, rgba_tile)

                # The RGBA interface requires some reordering.
                if sys.byteorder == "little":
                    # image is upside down
                    dims = [0]
                else:
                    # image is upside down, but in addition,
                    # if big-endian, must also flip the image planes
                    dims = [0, 2]
                rgba_tile = np.flip(rgba_tile, axis=dims)

                # We may need to remove the alpha plane.
                tiff_tile = rgba_tile[:, :, :3]

            else:

                tiff_tile = np.zeros(
                    (self.th, self.tw, self.spp),
                    dtype=self.dtype
                )
                libtiff.readEncodedTile(self.tiff_fp, ttile_num, tiff_tile)

            # determine how to fit this tiff tile into the jp2k
            # tile
            #
            # these are the section coordinates in image space
            uly = max(jp2k_uly, tile_uly)
            lly = min(jp2k_uly + jth, tile_uly + self.th)

            ulx = max(jp2k_ulx, tile_ulx)
            urx = min(jp2k_ulx + jtw, tile_ulx + self.tw)

            # convert to JP2K tile coordinates
            jrows = slice(uly % jth, (lly - 1) % jth + 1)
            jcols = slice(ulx % jtw, (urx - 1) % jtw + 1)

            # convert to TIFF tile coordinates
            trows = slice(uly % self.th, (lly - 1) % self.th + 1)
            tcols = slice(ulx % self.tw, (urx - 1) % self.tw + 1)

            jp2k_tile[jrows, jcols, :] = tiff_tile[trows, tcols, :]

        # last tile column?  last tile row?  If so, we may have a partial tile.
        if partial_jp2_tile_cols and jp2k_tile_col == num_jp2k_tile_cols - 1:
            last_j2k_cols = slice(0, self.imagewidth - jp2k_ulx)
            jp2k_tile = jp2k_tile[:, last_j2k_cols, :].copy()

        if partial_jp2_tile_rows and jp2k_tile_row == num_jp2k_tile_rows - 1:
            last_j2k_rows = slice(0, self.imageheight - jp2k_uly)
            jp2k_tile = jp2k_tile[last_j2k_rows, :, :].copy()

        return jp2k_tile

    def _get_covering_tiles(self, jp2k_tile_num):
        """
        Construct the set of TIFF tiles that completely cover the jpeg2000
        tile.
        """
        jth, jtw = self.tilesize

        num_jp2k_tile_cols = int(np.ceil(self.imagewidth / jtw))

        jp2k_tile_row = int(np.ceil(jp2k_tile_num // num_jp2k_tile_cols))
        jp2k_tile_col = int(np.ceil(jp2k_tile_num % num_jp2k_tile_cols))

        num_tiff_tile_cols = int(np.ceil(self.imagewidth / self.tw))

        # Upper left corner of the jp2k tile
        ulx = jtw * jp2k_tile_col
        uly = jth * jp2k_tile_row
        ul_tiff_tilenum = libtiff.computeTile(self.tiff_fp, ulx, uly, 0, 0)
        left_tiff_tile_col = int(np.ceil(ulx // self.tw))

        # Upper right corner of the jp2k tile
        urx = min(ulx + jtw - 1, self.imagewidth - 1)
        right_tiff_tile_col = int(np.ceil(urx // self.tw))

        # lower left corner
        llx = ulx
        lly = min(uly + jth - 1, self.imageheight - 1)

        # lower right corner
        lrx = min(llx + jtw - 1, self.imagewidth - 1)
        lry = lly
        lr_tiff_tilenum = libtiff.computeTile(self.tiff_fp, lrx, lry, 0, 0)

        # collect the tiles
        tiles = set()

        for tile_num in range(ul_tiff_tilenum, lr_tiff_tilenum + 1):

            tiff_tile_col = int(np.ceil(tile_num % num_tiff_tile_cols))

            # The tile rows should always be good.  But we need to exclude
            # any tiles based on the tile column.
            if (
                tiff_tile_col < left_tiff_tile_col
                or tiff_tile_col > right_tiff_tile_col
            ):
                continue

            # otherwise, this is a tile that intersects the jp2k tile
            tiles.add(tile_num)

        return tiles

    def _write_striped_tiff_to_tiled_jp2k(self):
        """The input TIFF image is striped and we are to create the output
        JPEG2000 image as a tiled JP2K.
        """

        jth, jtw = self.tilesize

        self.logger.debug(f"image:  {self.imageheight} x {self.imagewidth}")
        self.logger.debug(f"jptile:  {jth} x {jtw}")
        num_strips = libtiff.numberOfStrips(self.tiff_fp)

        num_jp2k_tile_cols = int(np.ceil(self.imagewidth / jtw))

        for idx, tilewriter in enumerate(self.jp2.get_tilewriters()):

            jp2k_tile_row = idx // num_jp2k_tile_cols
            jp2k_tile_col = idx % num_jp2k_tile_cols

            msg = f"Tile:  #{idx} row #{jp2k_tile_row} col #{jp2k_tile_col}"
            self.logger.info(msg)

            # the coordinates of the upper left pixel of the jp2k tile
            july, julx = jp2k_tile_row * jth, jp2k_tile_col * jtw

            # If we are starting a new row of jp2k tiles, we want to allocate
            # space for all the TIFF strips that encompass this jp2k row, and
            # then go ahead and read them in.  For all other cases, just assign
            # jp2k tiles from this same TIFF multi-strip.
            if jp2k_tile_col == 0:
                tiff_multi_strip = self._construct_multi_strip(
                    july,
                    num_strips,
                    jth,
                )

            # construct the TIFF row and column slices from the multi-strip,
            # assign to the jp2k tile.
            stripnum = libtiff.computeStrip(self.tiff_fp, july, 0)
            tile_uly = stripnum * self.rps

            ms_uly = july - tile_uly
            ms_ulx = julx

            ms_lry = min(ms_uly + jth, self.imageheight - tile_uly)
            ms_lrx = min(ms_ulx + jtw, self.imagewidth)

            rows = slice(ms_uly, ms_lry)
            cols = slice(ms_ulx, ms_lrx)

            tilewriter[:] = tiff_multi_strip[rows, cols, :]

    def _construct_multi_strip(self, july, num_strips, jth):
        """TIFF strips are pretty inefficient.  If our I/O was stupidly focused
        solely on each JP2K tile, we would read in each TIFF strip multiple
        times, once for each JP2K tile in the JP2K tile row.  If instead, we
        read in ALL the strips that will encompass that current row of JP2K
        tiles, we will save ourselves a lot of disk I/O.

        Parameters
        ----------
        jth : int
            The number of rows in a JP2K tile.
        july : int
            The top row of the current JP2K tile row.

        Returns
        -------
        tiff_multi_strip : np.array
            Holds all the TIFF strips that tightly encompass the current JP2K
            tile row.
        """
        num_strips = libtiff.numberOfStrips(self.tiff_fp)

        # We need to create a TIFF "multi-strip" that can hold all of
        # the JP2K tiles in a JP2K tile row.
        y = july
        top_strip_num = libtiff.computeStrip(self.tiff_fp, y, 0)

        # Find the strip number that is ONE MORE than the bottom strip.
        while (y // self.rps) * self.rps < min(july + jth, self.imageheight):
            y += self.rps
        bottom_strip_num = libtiff.computeStrip(self.tiff_fp, y, 0)

        # compute the number of rows contained between the top strip
        # and the bottom strip
        num_rows = (bottom_strip_num - top_strip_num) * self.rps

        if self.photo == libtiff.Photometric.YCBCR:
            # always single byte samples of R, G, B, and A
            dtype = np.uint8
        else:
            dtype = self.dtype
        spp = self.spp

        # This may result in a multi-strip that has more rows than the jp2k
        # tile
        tiff_multi_strip = np.zeros(
            (num_rows, self.imagewidth, spp),
            dtype=dtype
        )

        # Fill the multi-strip
        for stripnum in range(top_strip_num, bottom_strip_num):

            if self.photo == libtiff.Photometric.YCBCR:

                tiff_rgba_strip = np.zeros(
                    (self.rps, self.imagewidth, 4),
                    dtype=dtype
                )

                libtiff.readRGBAStrip(
                    self.tiff_fp, stripnum * self.rps, tiff_rgba_strip
                )

                # If a partial last strip...
                if (
                    stripnum == num_strips - 1
                    and self.imageheight // self.rps != num_strips
                ):
                    # According to the man page:
                    #
                    # When reading a partial last strip in the file the last
                    # line of the image  will  begin at the beginning of the
                    # buffer.
                    #
                    # move the top strips down to the bottom, otherwise the
                    # following flipping logic doesn't work.
                    bottom_row = self.imageheight % self.rps
                    irows = slice(0, bottom_row)
                    orows = slice(self.rps - bottom_row, self.rps)
                    tiff_rgba_strip[orows, :, :] = tiff_rgba_strip[irows, :, :]

                # The rgba interface requires at least flipping the image
                # upside down, and also reordering the planes on big endian
                if sys.byteorder == "little":
                    dims = [0]
                else:
                    dims = [0, 2]
                tiff_rgba_strip = np.flip(tiff_rgba_strip, axis=dims)

                # potentially get rid of alpha plane
                tiff_strip = tiff_rgba_strip[:, :, : self.spp]

            else:

                tiff_strip = np.zeros(
                    (self.rps, self.imagewidth, spp),
                    dtype=dtype
                )

                libtiff.readEncodedStrip(self.tiff_fp, stripnum, tiff_strip)

            # push the strip into the multi-strip
            top_row = (stripnum - top_strip_num) * self.rps
            bottom_row = (stripnum - top_strip_num + 1) * self.rps
            rows = slice(top_row, bottom_row)

            tiff_multi_strip[rows, :, :] = tiff_strip

        return tiff_multi_strip
