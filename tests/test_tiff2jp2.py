# standard library imports
import pathlib
import shutil
import tempfile
import warnings

# 3rd party library imports
import numpy as np
import skimage.data

# Local imports
from glymur import Jp2k, Tiff2Jp2
from . import fixtures
from glymur.lib import tiff as libtiff


class TestSuite(fixtures.TestCommon):

    @classmethod
    def setup_moon(cls, path):
        """
        SCENARIO:  create a simple monochromatic 2x2 tiled image
        """
        data = skimage.data.moon()
        h, w = data.shape
        th, tw = h // 2, w // 2

        fp = libtiff.open(path, mode='w')

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

        # now read it back
        fp = libtiff.open(path)

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

        cls.moon_data = actual_data
        cls.moon_tif = path

    @classmethod
    def setUpClass(cls):
        cls.test_tiff_dir = tempfile.mkdtemp()
        cls.test_tiff_path = pathlib.Path(cls.test_tiff_dir)

        cls.setup_moon(cls.test_tiff_path / 'moon.tif')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_tiff_dir)

    def test_smoke(self):
        """
        SCENARIO:  Convert TIFF file to JP2

        EXPECTED RESULT:  data matches
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with Tiff2Jp2(self.tiff_file, self.temp_jp2_filename) as j:
                j.run()

        actual = Jp2k(self.temp_jp2_filename)[:]

        self.assertEqual(actual.shape, (213, 234, 3))

    def test_moon(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with warnings.catch_warnings():
            with Tiff2Jp2(self.moon_tif, self.temp_jp2_filename) as j:
                j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.moon_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 256)
        self.assertEqual(c.segment[1].ytsiz, 256)
