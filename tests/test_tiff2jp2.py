# standard library imports
import importlib.resources as ir
import pathlib
import shutil
import tempfile
import warnings

# 3rd party library imports
import numpy as np
import skimage.data

# Local imports
from glymur import Jp2k, Tiff2Jp2k
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
    def setup_moon3(cls, path):
        """
        SCENARIO:  create a simple monochromatic 3x3 tiled image
        """
        data = skimage.data.moon()
        data = data[:480, :480]

        h, w = data.shape
        th, tw = h // 3, w // 3

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
        libtiff.writeEncodedTile(fp, 1, data[:th, tw:tw * 2].copy())
        libtiff.writeEncodedTile(fp, 2, data[:th, tw * 2:w].copy())
        libtiff.writeEncodedTile(fp, 3, data[th:th * 2, :tw].copy())
        libtiff.writeEncodedTile(fp, 4, data[th:th * 2, tw:tw * 2].copy())
        libtiff.writeEncodedTile(fp, 5, data[th:th * 2, tw * 2:w].copy())
        libtiff.writeEncodedTile(fp, 6, data[2 * th:h, :tw].copy())
        libtiff.writeEncodedTile(fp, 7, data[2 * th:h, tw:tw * 2].copy())
        libtiff.writeEncodedTile(fp, 8, data[2 * th:h, tw * 2:w].copy())

        libtiff.close(fp)

        cls.moon3_data = data
        cls.moon3_tif = path

    @classmethod
    def setup_moon3_stripped(cls, path):
        """
        SCENARIO:  create a simple monochromatic 3-strip image
        """
        data = skimage.data.moon()
        data = data[:480, :480]

        h, w = data.shape
        rps = h // 3

        fp = libtiff.open(path, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.MINISBLACK)
        libtiff.setField(fp, 'Compression', libtiff.Compression.DEFLATE)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'RowsPerStrip', rps)
        libtiff.setField(fp, 'BitsPerSample', 8)
        libtiff.setField(fp, 'SamplesPerPixel', 1)
        libtiff.setField(fp, 'PlanarConfig', libtiff.PlanarConfig.CONTIG)

        libtiff.writeEncodedStrip(fp, 0, data[:rps, :].copy())
        libtiff.writeEncodedStrip(fp, 1, data[rps:rps * 2, :].copy())
        libtiff.writeEncodedStrip(fp, 2, data[rps * 2:rps * 3, :].copy())

        libtiff.close(fp)

        cls.moon3_stripped_tif = path

    @classmethod
    def setup_astronaut(cls, path):
        """
        SCENARIO:  create a simple color 2x2 tiled image
        """
        data = skimage.data.astronaut()
        h, w, z = data.shape
        th, tw = h // 2, w // 2

        fp = libtiff.open(path, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.RGB)
        libtiff.setField(fp, 'Compression', libtiff.Compression.DEFLATE)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'TileLength', th)
        libtiff.setField(fp, 'TileWidth', tw)
        libtiff.setField(fp, 'BitsPerSample', 8)
        libtiff.setField(fp, 'SamplesPerPixel', 3)
        libtiff.setField(fp, 'PlanarConfig', libtiff.PlanarConfig.CONTIG)

        libtiff.writeEncodedTile(fp, 0, data[:th, :tw, :].copy())
        libtiff.writeEncodedTile(fp, 1, data[:th, tw:w, :].copy())
        libtiff.writeEncodedTile(fp, 2, data[th:h, :tw, :].copy())
        libtiff.writeEncodedTile(fp, 3, data[th:h, tw:w, :].copy())

        libtiff.close(fp)

        # now read it back
        fp = libtiff.open(path)

        tile = np.zeros((th, tw, 3), dtype=np.uint8)
        actual_data = np.zeros((h, w, 3), dtype=np.uint8)

        libtiff.readEncodedTile(fp, 0, tile)
        actual_data[:th, :tw, :] = tile

        libtiff.readEncodedTile(fp, 1, tile)
        actual_data[:th, tw:w, :] = tile

        libtiff.readEncodedTile(fp, 2, tile)
        actual_data[th:h, :tw, :] = tile

        libtiff.readEncodedTile(fp, 3, tile)
        actual_data[th:h, tw:w, :] = tile

        libtiff.close(fp)

        cls.astronaut_data = actual_data
        cls.astronaut_tif = path

    @classmethod
    def setUpClass(cls):
        cls.test_tiff_dir = tempfile.mkdtemp()
        cls.test_tiff_path = pathlib.Path(cls.test_tiff_dir)

        with ir.path('tests.data', 'zackthecat.tif') as filename:
            cls.zackthecat = filename

        # uint8 spp=1 image
        cls.setup_moon(cls.test_tiff_path / 'moon.tif')

        # uint8 spp=1 image with 3x3 tiles
        cls.setup_moon3(cls.test_tiff_path / 'moon3.tif')

        # uint8 spp=1 image with 3 strips
        cls.setup_moon3_stripped(cls.test_tiff_path / 'moon3_stripped.tif')

        # uint8 spp=3 image
        cls.setup_astronaut(cls.test_tiff_path / 'astronaut.tif')

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
            with Tiff2Jp2k(self.zackthecat, self.temp_jp2_filename) as j:
                j.run()

        actual = Jp2k(self.temp_jp2_filename)[:]

        self.assertEqual(actual.shape, (213, 234, 3))

    def test_moon(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(self.moon_tif, self.temp_jp2_filename) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.moon_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 256)
        self.assertEqual(c.segment[1].ytsiz, 256)

    def test_moon__smaller_tilesize_specified(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 2x2, but we want 4x4.

        EXPECTED RESULT:  The data matches.  The JP2 file has 16 tiles.
        """
        with Tiff2Jp2k(
            self.moon_tif, self.temp_jp2_filename,
            tilesize=(128, 128)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.moon_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 128)
        self.assertEqual(c.segment[1].ytsiz, 128)

    def test_moon3_stripped(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        stripped by 3, but we want 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.moon3_stripped_tif, self.temp_jp2_filename,
            tilesize=(240, 240)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.moon3_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 480)
        self.assertEqual(c.segment[1].ysiz, 480)
        self.assertEqual(c.segment[1].xtsiz, 240)
        self.assertEqual(c.segment[1].ytsiz, 240)

    def test_moon3__larger_tilesize_specified(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 3x3, but we want 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.moon3_tif, self.temp_jp2_filename,
            tilesize=(240, 240)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.moon3_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 480)
        self.assertEqual(c.segment[1].ysiz, 480)
        self.assertEqual(c.segment[1].xtsiz, 240)
        self.assertEqual(c.segment[1].ytsiz, 240)

    def test_astronaut(self):
        """
        SCENARIO:  Convert RGB TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(self.astronaut_tif, self.temp_jp2_filename) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.astronaut_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 256)
        self.assertEqual(c.segment[1].ytsiz, 256)
