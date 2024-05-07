# standard library imports
import importlib.resources as ir
import platform
import sys
import unittest
from unittest.mock import patch
import warnings

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

    def test_simple_2x2_tiled(self):
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
        libtiff.setField(fp, 'Software', 'glymur')
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

    def test_bigtiff_ycbcr_2x2_tiled(self):
        """
        SCENARIO:  create a YCbCr/JPEG 2x2 tiled image

        Expected result:  The data is subject to lossy JPEG compression, so it
        will not match exactly, but should be reasonably close.
        """
        expected = fixtures.skimage.data.astronaut()
        h, w, nz = expected.shape
        th, tw = h // 2, w // 2

        fp = libtiff.open(self.temp_tiff_filename, mode='w8')
        libtiff.setField(fp, 'Photometric', libtiff.Photometric.YCBCR)
        libtiff.setField(fp, 'Compression', libtiff.Compression.JPEG)
        libtiff.setField(fp, 'JPEGColorMode', libtiff.JPEGColorMode.RGB)
        libtiff.setField(fp, 'PlanarConfig', libtiff.PlanarConfig.CONTIG)
        libtiff.setField(fp, 'JPEGQuality', 90)
        libtiff.setField(fp, 'YCbCrSubsampling', 1, 1)
        libtiff.setField(fp, 'ImageWidth', w)
        libtiff.setField(fp, 'ImageLength', h)
        libtiff.setField(fp, 'TileWidth', tw)
        libtiff.setField(fp, 'TileLength', th)
        libtiff.setField(fp, 'BitsPerSample', 8)
        libtiff.setField(fp, 'SamplesPerPixel', nz)
        libtiff.setField(fp, 'Software', libtiff.getVersion())
        libtiff.writeEncodedTile(fp, 0, expected[:th, :tw].copy())
        libtiff.writeEncodedTile(fp, 1, expected[:th, tw:w].copy())
        libtiff.writeEncodedTile(fp, 2, expected[th:h, :tw].copy())
        libtiff.writeEncodedTile(fp, 3, expected[th:h, tw:w].copy())
        libtiff.close(fp)

        fp = libtiff.open(self.temp_tiff_filename)
        actual = libtiff.readRGBAImageOriented(fp)
        libtiff.close(fp)

        # Adjust for big-endian if necessary
        actual = np.flip(actual, 2) if sys.byteorder == 'big' else actual

        error = fixtures.skimage.metrics.mean_squared_error(
            actual[:, :, :3], expected
        )
        self.assertTrue(error < 9)

    def test_simple_strip(self):
        """
        SCENARIO:  create a simple monochromatic 2 strip image

        Expected result:  The image matches.  The number of tiles checks out.
        The tile width and height checks out.
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
        actual_data[rps:h, :] = strip

        np.testing.assert_array_equal(data, actual_data)

        n = libtiff.numberOfStrips(fp)
        self.assertEqual(n, 2)

        libtiff.close(fp)

    def test_warning(self):
        """
        SCENARIO:  open a geotiff with just the regular tiff library

        Expected result:  the library will warn about geotiff tags being
        unrecognized
        """
        path = ir.files('tests.data.tiff').joinpath('warning.tif')
        with warnings.catch_warnings(record=True) as w:
            fp = libtiff.open(path)
            libtiff.close(fp)
        self.assertTrue(len(w) > 0)

    def test_read_rgba_without_image_length_width(self):
        """
        SCENARIO:  open a CMYK tiff, read via rgba interface without supplying
        the width or height.

        Expected result:  the image is read as expected
        """
        path = ir.files('tests.data.tiff').joinpath('cmyk.tif')
        fp = libtiff.open(path)

        # need to set the inkset appropriately, multi-ink won't cut it
        libtiff.setField(fp, 'InkSet', libtiff.InkSet.CMYK)

        image = libtiff.readRGBAImageOriented(fp)
        libtiff.close(fp)

        self.assertEqual(image.shape, (512, 512, 4))

    def test_tiff_version_when_not_installed(self):
        """
        SCENARIO:  access the tiff library version when the library is not
        installed

        Expected result:  '0.0.0'
        """
        with patch.object(libtiff, '_LIBTIFF', new=None):
            actual = libtiff.getVersion()
        self.assertEqual(actual, '0.0.0')
