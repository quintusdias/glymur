# 3rd party library imports
import numpy as np

# local imports
from glymur import Jp2k
from .lib import tiff as libtiff


class Tiff2Jp2(object):

    def __init__(self, tiff_filename, jp2_filename):
        self.tiff_filename = tiff_filename
        self.jp2_filename = jp2_filename

    def __enter__(self):
        self.tiff_fp = libtiff.open(self.tiff_filename)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        libtiff.close(self.tiff_fp)

    def run(self):

        if not libtiff.isTiled(self.tiff_fp):
            raise NotImplementedError('Not supported for stripped TIFFs')

        width = libtiff.getFieldDefaulted(self.tiff_fp, 'ImageWidth')
        height = libtiff.getFieldDefaulted(self.tiff_fp, 'ImageLength')
        spp = libtiff.getFieldDefaulted(self.tiff_fp, 'SamplesPerPixel')
        sf = libtiff.getFieldDefaulted(self.tiff_fp, 'SampleFormat')
        bps = libtiff.getFieldDefaulted(self.tiff_fp, 'BitsPerSample')
        tw = libtiff.getFieldDefaulted(self.tiff_fp, 'TileWidth')
        th = libtiff.getFieldDefaulted(self.tiff_fp, 'TileLength')

        if tw > width and th > height and libtiff.RGBAImageOK(self.tiff_fp):
            image = libtiff.readRGBAImageOriented(self.tiff_fp)

            if spp < 4:
                image = image[:, :, :3]

            Jp2k(self.jp2_filename, data=image)

        elif (
            (width % tw) == 0
            and (height % th) == 0
            and bps == 8
            and sf == libtiff.SampleFormat.UINT
        ):

            # The image is evenly tiled uint8.  This is ideal.
            tile = np.zeros((th, tw, spp), dtype=np.uint8)
            jp2 = Jp2k(
                self.jp2_filename, shape=(height, width, spp), tilesize=(th, tw)
            )
            for idx, tilewriter in enumerate(jp2.get_tilewriters()):
                libtiff.readEncodedTile(self.tiff_fp, idx, tile)
                tilewriter[:] = tile
