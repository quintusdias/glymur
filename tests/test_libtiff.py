# standard library imports
import platform
import unittest

# 3rd party library imports
import numpy as np

# local imports
from . import fixtures
from glymur.lib import tiff as libtiff


@unittest.skipIf(
    platform.system() == 'Darwin' and platform.machine() == 'arm64',
    'See issue #593'
)
@unittest.skipIf(
    not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
)
@unittest.skipIf(fixtures.TIFF_NOT_AVAILABLE, fixtures.TIFF_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):

    def test_simple_tile(self):
        """
        SCENARIO:  create a simple monochromatic 2x2 tiled image

        Expected result:  The image matches.  The number of tiles checks out.
        The tile width and height checks out.
        """
        data = fixtures.skimage.data.moon()
        h, w = data.shape
        th, tw = h // 2, w // 2

        fp = libtiff.open(self.temp_tiff_filename, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.MINISBLACK)
        libtiff.setField(fp, 'Compression', libtiff.Compression.ADOBE_DEFLATE)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'TileLength', th)
        libtiff.setField(fp, 'TileWidth', tw)
        libtiff.setField(fp, 'BitsPerSample', 8)
        libtiff.setField(fp, 'SamplesPerPixel', 1)
        libtiff.setField(fp, 'PlanarConfig', libtiff.PlanarConfig.CONTIG)

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

        np.testing.assert_array_equal(data, actual_data)

        n = libtiff.numberOfTiles(fp)
        self.assertEqual(n, 4)

        actual_th = libtiff.getFieldDefaulted(fp, 'TileLength')
        self.assertEqual(actual_th, th)

        actual_tw = libtiff.getFieldDefaulted(fp, 'TileWidth')
        self.assertEqual(actual_tw, tw)

        libtiff.close(fp)

    def test_simple_strip(self):
        """
        Scenario:  create a simple monochromatic 2x1 stripped image

        Expected result:  The image matches.  The number of strips checks out.
        """
        data = fixtures.skimage.data.moon()
        h, w = data.shape
        rps = h // 2

        fp = libtiff.open(self.temp_tiff_filename, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.MINISBLACK)
        libtiff.setField(fp, 'Compression', libtiff.Compression.ADOBE_DEFLATE)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'RowsPerStrip', rps)
        libtiff.setField(fp, 'BitsPerSample', 8)
        libtiff.setField(fp, 'SamplesPerPixel', 1)
        libtiff.setField(fp, 'PlanarConfig', libtiff.PlanarConfig.CONTIG)

        libtiff.writeEncodedStrip(fp, 0, data[:rps, :].copy())
        libtiff.writeEncodedStrip(fp, 1, data[rps:h, :].copy())

        libtiff.close(fp)

        fp = libtiff.open(self.temp_tiff_filename)

        strip = np.zeros((rps, w), dtype=np.uint8)
        actual_data = np.zeros((h, w), dtype=np.uint8)

        libtiff.readEncodedStrip(fp, 0, strip)
        actual_data[:rps, :] = strip

        libtiff.readEncodedStrip(fp, 1, strip)
        actual_data[rps:, :] = strip

        np.testing.assert_array_equal(data, actual_data)

        n = libtiff.numberOfStrips(fp)
        self.assertEqual(n, 2)

        libtiff.close(fp)
