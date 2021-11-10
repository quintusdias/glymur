# standard library imports
import unittest

# 3rd party library imports
import numpy as np

# local imports
from . import fixtures
from glymur.lib import tiff as libtiff


@unittest.skipIf(
    not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
)
@unittest.skipIf(fixtures.TIFF_NOT_AVAILABLE, fixtures.TIFF_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):

    def test_simple_tile(self):
        """
        SCENARIO:  create a simple monochromatic 2x2 tiled image
        """
        data = fixtures.skimage.data.moon()
        h, w = data.shape
        th, tw = h // 2, w // 2

        fp = libtiff.open(self.temp_tiff_filename, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.MINISBLACK)
        libtiff.setField(fp, 'Compression', libtiff.Compression.DEFLATE)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'TileLength', th)
        libtiff.setField(fp, 'TileWidth', tw)
        libtiff.setField(fp, 'BitsPerSample', 8)
        libtiff.setField(fp, 'SamplesPerPixel', 1)

        libtiff.writeEncodedTile(fp, 0, data[:th, :tw].copy())
        libtiff.writeEncodedTile(fp, 1, data[:th, tw:w].copy())
        libtiff.writeEncodedTile(fp, 2, data[th:h, :tw].copy())
        libtiff.writeEncodedTile(fp, 3, data[th:h, tw:w].copy())

        libtiff.close(fp)

        fp = libtiff.open(self.temp_tiff_filename)

        tile = np.zeros((th, tw), dtype=np.uint8)
        actual_data = np.zeros((h, w), dtype=np.uint8)

        libtiff.readEncodedTile(fp, 0, tile)
        actual_data[:th, :tw] = tile

        libtiff.readEncodedTile(fp, 1, tile)
        actual_data[:th, tw:w] = tile

        libtiff.readEncodedTile(fp, 2, tile)
        actual_data[th:h, :tw] = tile

        libtiff.readEncodedTile(fp, 3, tile)
        actual_data[th:h, tw:w] = tile

        libtiff.close(fp)

        np.testing.assert_array_equal(data, actual_data)
