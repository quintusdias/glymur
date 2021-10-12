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
        tw = libtiff.getFieldDefaulted(self.tiff_fp, 'TileWidth')
        th = libtiff.getFieldDefaulted(self.tiff_fp, 'TileLength')

        if tw > width and th > height and libtiff.RGBAImageOK(self.tiff_fp):
            image = libtiff.readRGBAImageOriented(self.tiff_fp)

            if spp < 4:
                image = image[:, :, :3]

        self.write(image)

    def write(self, image):

        Jp2k(self.jp2_filename, data=image)
