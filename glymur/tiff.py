# 3rd party library imports
import logging
import numpy as np

# local imports
from glymur import Jp2k
from .lib import tiff as libtiff


class Tiff2Jp2k(object):
    """
    Attributes
    ----------
    tiff_filename : path or str
        Path to TIFF file.
    jp2_filename : path or str
        Path to JPEG 2000 file to be written.
    tilesize : tuple
        The dimensions of a tile in the JP2K file.
    """

    def __init__(
        self, tiff_filename, jp2_filename, tilesize=None,
        verbosity=logging.CRITICAL
    ):

        self.tiff_filename = tiff_filename
        if not self.tiff_filename.exists():
            raise FileNotFoundError(f'{tiff_filename} does not exist')

        self.jp2_filename = jp2_filename
        self.tilesize = tilesize

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

        if libtiff.isTiled(self.tiff_fp):
            isTiled = True
        else:
            isTiled = False

        photometric = libtiff.getFieldDefaulted(self.tiff_fp, 'Photometric')
        imagewidth = libtiff.getFieldDefaulted(self.tiff_fp, 'ImageWidth')
        imageheight = libtiff.getFieldDefaulted(self.tiff_fp, 'ImageLength')
        spp = libtiff.getFieldDefaulted(self.tiff_fp, 'SamplesPerPixel')
        sf = libtiff.getFieldDefaulted(self.tiff_fp, 'SampleFormat')
        bps = libtiff.getFieldDefaulted(self.tiff_fp, 'BitsPerSample')

        if sf != libtiff.SampleFormat.UINT:
            sf_string = [
                key for key in dir(libtiff.SampleFormat)
                if getattr(libtiff.SampleFormat, key) == sf
            ][0]
            msg = (
                f"The TIFF SampleFormat is {sf_string}.  Only UINT is "
                "supported."
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

        if libtiff.isTiled(self.tiff_fp):
            tw = libtiff.getFieldDefaulted(self.tiff_fp, 'TileWidth')
            th = libtiff.getFieldDefaulted(self.tiff_fp, 'TileLength')
            num_tiles = libtiff.numberOfTiles(self.tiff_fp)
        else:
            tw = imagewidth
            rps = libtiff.getFieldDefaulted(self.tiff_fp, 'RowsPerStrip')
            num_strips = libtiff.numberOfStrips(self.tiff_fp)

        if self.tilesize is not None:
            jth, jtw = self.tilesize

            num_jp2k_tile_rows = int(np.ceil(imagewidth / jtw))
            num_jp2k_tile_cols = int(np.ceil(imagewidth / jtw))

        # Using the RGBA interface is the only reasonable way to deal with
        # them.
        if photometric in [
            libtiff.Photometric.YCBCR, libtiff.Photometric.PALETTE
        ]:
            use_rgba_interface = True
        else:
            use_rgba_interface = False

        if self.tilesize is None and libtiff.RGBAImageOK(self.tiff_fp):

            # if no jp2k tiling was specified and if the image is ok to read
            # via the RGBA interface, then just do that.
            msg = (
                "Reading using the RGBA interface, writing as a single tile "
                "image."
            )
            self.logger.info(msg)

            image = libtiff.readRGBAImageOriented(self.tiff_fp)

            if spp < 4:
                image = image[:, :, :3]

            Jp2k(self.jp2_filename, data=image)

        elif isTiled and self.tilesize is not None:

            jp2 = Jp2k(
                self.jp2_filename,
                shape=(imageheight, imagewidth, spp),
                tilesize=self.tilesize,
            )

            num_tiff_tile_cols = int(np.ceil(imagewidth / tw))

            partial_jp2_tile_rows = (imageheight / jth) != (imageheight // jth)
            partial_jp2_tile_cols = (imagewidth / jtw) != (imagewidth // jtw)

            rgba_tile = np.zeros((th, tw, 4), dtype=np.uint8)

            import logging
            logging.warning(f'image:  {imageheight} x {imagewidth}')
            logging.warning(f'jptile:  {jth} x {jtw}')
            logging.warning(f'ttile:  {th} x {tw}')
            for idx, tilewriter in enumerate(jp2.get_tilewriters()):

                # populate the jp2k tile with tiff tiles
                logging.warning(f'IDX:  {idx}')

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
                    last_j2k_cols = slice(0, jtw - (ulc - imagewidth))
                    jp2k_tile = jp2k_tile[:, jcols, :].copy()
                if (
                    partial_jp2_tile_rows
                    and jp2k_tile_row == num_jp2k_tile_rows - 1
                ):
                    last_j2k_rows = slice(0, jth - (llr - imageheight))
                    jp2k_tile = jp2k_tile[jrows, :, :].copy()

                tilewriter[:] = jp2k_tile

        elif not isTiled and self.tilesize is not None:

            jp2 = Jp2k(
                self.jp2_filename,
                shape=(imageheight, imagewidth, spp),
                tilesize=self.tilesize,
            )

            num_strips = libtiff.numberOfStrips(self.tiff_fp)

            import logging
            logging.warning(f'Image size:  {imageheight} x {imagewidth}')
            logging.warning(f'Jp2k tile size:  {jth} x {jtw}')
            logging.warning(f'TIFF strip length:  {rps}')

            num_jp2k_tile_cols = int(np.ceil(imagewidth / jtw))

            partial_jp2_tile_rows = (imageheight / jth) != (imageheight // jth)
            partial_jp2_tile_cols = (imagewidth / jtw) != (imagewidth // jtw)

            tiff_strip = np.zeros((rps, imagewidth, spp), dtype=dtype)
            rgba_strip = np.zeros((rps, imagewidth, 4), dtype=np.uint8)

            for idx, tilewriter in enumerate(jp2.get_tilewriters()):
                logging.warning(f'jp2k tile idx: {idx}')

                jp2k_tile = np.zeros((jth, jtw, spp), dtype=dtype)

                jp2k_tile_row = idx // num_jp2k_tile_cols
                jp2k_tile_col = idx % num_jp2k_tile_cols

                # the coordinates of the upper left pixel of the jp2k tile
                julr, julc = jp2k_tile_row * jth, jp2k_tile_col * jtw
                logging.warning(f'Upper left j coord:  {julr}, {julc}')

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

                    logging.warning(f'row: {r}')
                    logging.warning(f'strip: {stripnum}')

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

                    logging.warning(f'strip size is {tiff_strip.shape}')
                    logging.warning(f'upper left coord of intersection:  ({ulr}, {ulc})')
                    logging.warning(f'lower right coord of intersection:  ({llr}, {urc})')
                    logging.warning(f'j rows received are {jrows}')
                    logging.warning(f'j cols received are {jcols}')
                    logging.warning(f't rows transferred are {trows}')
                    logging.warning(f't cols transferred are {tcols}')

                    r += rps

                # last tile column?  If so, we may have a partial tile.
                # j2k_cols is not sufficient here, must shorten it from 250
                # to 230
                if (
                    partial_jp2_tile_cols
                    and jp2k_tile_col == num_jp2k_tile_cols - 1
                ):
                    logging.warning('Hit a partial jp2k tile on right side')
                    # decrease the number of columns by however many it sticks
                    # over the image width
                    last_j2k_cols = slice(0, jtw - (ulc + jtw - imagewidth))
                    jp2k_tile = jp2k_tile[:, last_j2k_cols, :].copy()

                if (
                    partial_jp2_tile_rows
                    and stripnum == num_strips - 1
                ):
                    logging.warning('Hit a partial jp2k tile on the bottom')
                    # decrease the number of rows by however many it sticks
                    # over the image height
                    last_j2k_rows = slice(0, imageheight - julr)
                    jp2k_tile = jp2k_tile[last_j2k_rows, :, :].copy()

                tilewriter[:] = jp2k_tile
