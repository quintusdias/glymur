# 3rd party library imports
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

    def __init__(self, tiff_filename, jp2_filename, tilesize=None):
        self.tiff_filename = tiff_filename
        self.jp2_filename = jp2_filename
        self.tilesize = tilesize

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
        else:
            tw = imagewidth
            rps = libtiff.getFieldDefaulted(self.tiff_fp, 'RowsPerStrip')

        if self.tilesize is not None:
            jth, jtw = self.tilesize

            num_jp2k_tile_rows = int(np.ceil(imagewidth / jtw))
            num_jp2k_tile_cols = int(np.ceil(imagewidth / jtw))

        if self.tilesize is None and libtiff.RGBAImageOK(self.tiff_fp):

            # if no jp2k tiling was specified and if the image is ok to read
            # via the RGBA interface, then just do that.
            image = libtiff.readRGBAImageOriented(self.tiff_fp)

            if spp < 4:
                image = image[:, :, :spp]

            Jp2k(self.jp2_filename, data=image)

        elif isTiled and self.tilesize is not None:

            tile = np.zeros((th, tw, spp), dtype=dtype)
            jp2 = Jp2k(
                self.jp2_filename,
                shape=(imageheight, imagewidth, spp),
                tilesize=self.tilesize,
            )

            num_tiff_tile_cols = int(np.ceil(imagewidth / tw))

            partial_jp2k_tile_rows = (imageheight / jth) != (imageheight // jth) 
            partial_jp2k_tile_cols = (imagewidth / jtw) != (imagewidth // jtw) 

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
                    #for y in range(julr, min(julr + jth, imageheight), th):
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
                        j2k_rows = slice(ulr % jth, (llr - 1) % jth + 1)
                        j2k_cols = slice(ulc % jtw, (urc - 1) % jtw + 1)

                        # convert to TIFF tile coordinates
                        tiff_rows = slice(ulr % th, (llr - 1) % th + 1)
                        tiff_cols = slice(ulc % tw, (urc - 1) % tw + 1)

                        try:
                            jp2k_tile[j2k_rows, j2k_cols, :] = tiff_tile[tiff_rows, tiff_cols, :]
                        except ValueError:
                            breakpoint()
                            raise

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
                if partial_jp2k_tile_cols and jp2k_tile_col == num_jp2k_tile_cols - 1:
                    last_j2k_cols = slice(0, jtw - (ulc - imagewidth))
                    jp2k_tile = jp2k_tile[:, j2k_cols, :].copy()
                if partial_jp2k_tile_rows and jp2k_tile_row == num_jp2k_tile_rows - 1:
                    last_j2k_rows = slice(0, jth - (llr - imageheight))
                    jp2k_tile = jp2k_tile[j2k_rows, :, :].copy()

                try:
                    tilewriter[:] = jp2k_tile
                except Exception as e:
                    breakpoint()
                    pass

        elif not isTiled and self.tilesize is not None:

            jp2 = Jp2k(
                self.jp2_filename,
                shape=(imageheight, imagewidth, spp),
                tilesize=self.tilesize,
            )

            import logging
            logging.warning(f'Image size:  {imageheight} x {imagewidth}')
            logging.warning(f'Jp2k tile size:  {jth} x {jtw}')
            logging.warning(f'TIFF strip length:  {rps}')

            num_jp2k_tile_cols = int(np.ceil(imagewidth / jtw))

            partial_jp2k_tile_rows = (imageheight / jth) != (imageheight // jth) 
            partial_jp2k_tile_cols = (imagewidth / jtw) != (imagewidth // jtw) 

            tiff_strip = np.zeros((rps, imagewidth, spp), dtype=dtype)

            for idx, tilewriter in enumerate(jp2.get_tilewriters()):
                logging.warning(f'jp2k tile idx: {idx}')

                jp2k_tile = np.zeros((jth, jtw, spp), dtype=dtype)

                jp2k_tile_row = idx // num_jp2k_tile_cols
                jp2k_tile_col = idx % num_jp2k_tile_cols

                # the coordinates of the upper left pixel of the jp2k tile
                julr, julc = jp2k_tile_row * jth, jp2k_tile_col * jtw
                logging.warning(f'Upper left j coord:  {julr}, {julc}')

                # populate the jp2k tile with tiff strips
                for r in range(julr, min(julr + jth, imageheight), rps):

                    stripnum = libtiff.computeStrip(self.tiff_fp, r, 0)
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
                    j2k_rows = slice(ulr % jth, (llr - 1) % jth + 1)
                    j2k_cols = slice(ulc % jtw, (urc - 1) % jtw + 1)

                    # convert to TIFF strip coordinates
                    tiff_rows = slice(ulr % rps, (llr - 1) % rps + 1)
                    tiff_cols = slice(ulc % tw, (urc - 1) % tw + 1)

                    try:
                        jp2k_tile[j2k_rows, j2k_cols, :] = tiff_strip[tiff_rows, tiff_cols, :]
                    except ValueError as e:
                        breakpoint()
                        raise

                # last tile column?  If so, we may have a partial tile.
                # j2k_cols is not sufficient here, must shorten it from 250
                # to 230
                if partial_jp2k_tile_cols and jp2k_tile_col == num_jp2k_tile_cols - 1:
                    # decrease the number of columns by however many it sticks
                    # over the image width
                    last_j2k_cols = slice(0, jtw - (ulc + jtw - imagewidth))
                    jp2k_tile = jp2k_tile[:, last_j2k_cols, :].copy()
                if partial_jp2k_tile_rows and jp2k_tile_row == num_jp2k_tile_rows - 1:
                    # decrease the number of rows by however many it sticks
                    # over the image height
                    last_j2k_rows = slice(0, jth - (llr - imageheight))
                    jp2k_tile = jp2k_tile[last_j2k_rows, :, :].copy()
                try:
                    tilewriter[:] = jp2k_tile
                except Exception as e:
                    raise
