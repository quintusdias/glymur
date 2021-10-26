# standard library imports
import logging
import pathlib
import shutil
import sys
import tempfile

# 3rd party library imports
import numpy as np
import skimage.data

# Local imports
import glymur
from glymur import Jp2k, Tiff2Jp2k, command_line
from . import fixtures
from glymur.lib import tiff as libtiff


class TestSuite(fixtures.TestCommon):

    @classmethod
    def setup_rgb_evenly_stripped(cls, path):
        """
        SCENARIO:  create a simple RGB stripped image, stripsize of 32
        """
        j = Jp2k(glymur.data.goodstuff())
        data = j[:]
        h, w, spp = data.shape
        rps = 32

        fp = libtiff.open(path, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.RGB)
        libtiff.setField(fp, 'Compression', libtiff.Compression.DEFLATE)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'RowsPerStrip', rps)
        libtiff.setField(fp, 'BitsPerSample', 8)
        libtiff.setField(fp, 'SamplesPerPixel', spp)
        libtiff.setField(fp, 'PlanarConfig', libtiff.PlanarConfig.CONTIG)

        for stripnum in range(25):
            row = rps * stripnum
            stripdata = data[row:row + rps, :, :].copy()
            libtiff.writeEncodedStrip(fp, stripnum, stripdata)

        libtiff.close(fp)

        cls.goodstuff_data = data
        cls.goodstuff_path = path

    @classmethod
    def setup_minisblack_spp1(cls, path):
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
    def setup_moon_partial_tiles(cls, path):
        """
        SCENARIO:  create a simple monochromatic 2x2 tiled image with partial
        tiles.
        """
        data = skimage.data.moon()
        h, w = 480, 480
        th, tw = 256, 256

        fp = libtiff.open(path, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.MINISBLACK)
        libtiff.setField(fp, 'Compression', libtiff.Compression.DEFLATE)
        libtiff.setField(fp, 'ImageLength', h)
        libtiff.setField(fp, 'ImageWidth', w)
        libtiff.setField(fp, 'TileLength', th)
        libtiff.setField(fp, 'TileWidth', tw)
        libtiff.setField(fp, 'BitsPerSample', 8)
        libtiff.setField(fp, 'SamplesPerPixel', 1)

        libtiff.writeEncodedTile(fp, 0, data[:th, :tw].copy())
        libtiff.writeEncodedTile(fp, 1, data[:th, tw:w].copy())
        libtiff.writeEncodedTile(fp, 2, data[th:h, :tw].copy())
        libtiff.writeEncodedTile(fp, 3, data[th:h, tw:w].copy())

        libtiff.close(fp)

        cls.moon_partial_tiles_data = data[:h, :w]
        cls.moon_partial_tiles_path = path

    @classmethod
    def setup_minisblack_3x3(cls, path):
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

        cls.minisblack_3x3_data = data
        cls.minisblack_3x3_tif = path

    @classmethod
    def setup_minisblack_3strip(cls, path):
        """
        SCENARIO:  create a simple monochromatic 3-strip image.  The strips
        evenly divide the image.
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
    def setup_moon_partial_last_strip(cls, path):
        """
        SCENARIO:  create a simple monochromatic 3-strip image
        """
        data = skimage.data.moon()
        data = data[:480, :480]

        h, w = data.shape

        # instead of 160, this will cause a partially empty last strip
        rps = 170

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
        libtiff.writeEncodedStrip(fp, 1, data[rps:, :].copy())

        data2 = np.vstack((
            data[340:480, :], np.zeros((30, 480), dtype=np.uint8)
        ))
        libtiff.writeEncodedStrip(fp, 2, data2)

        libtiff.close(fp)

        cls.moon_partial_last_strip = path

    @classmethod
    def setup_rgb_uint16(cls, path):
        """
        SCENARIO:  create a simple color 2x2 tiled 16bit image
        """
        data = skimage.data.astronaut().astype(np.uint16)
        h, w, z = data.shape
        th, tw = h // 2, w // 2

        fp = libtiff.open(path, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.RGB)
        libtiff.setField(fp, 'Compression', libtiff.Compression.DEFLATE)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'TileLength', th)
        libtiff.setField(fp, 'TileWidth', tw)
        libtiff.setField(fp, 'BitsPerSample', 16)
        libtiff.setField(fp, 'SamplesPerPixel', 3)
        libtiff.setField(fp, 'SampleFormat', libtiff.SampleFormat.UINT)
        libtiff.setField(fp, 'PlanarConfig', libtiff.PlanarConfig.CONTIG)

        libtiff.writeEncodedTile(fp, 0, data[:th, :tw, :].copy())
        libtiff.writeEncodedTile(fp, 1, data[:th, tw:w, :].copy())
        libtiff.writeEncodedTile(fp, 2, data[th:h, :tw, :].copy())
        libtiff.writeEncodedTile(fp, 3, data[th:h, tw:w, :].copy())

        libtiff.close(fp)

        # now read it back
        fp = libtiff.open(path)

        tile = np.zeros((th, tw, 3), dtype=np.uint16)
        actual_data = np.zeros((h, w, 3), dtype=np.uint16)

        libtiff.readEncodedTile(fp, 0, tile)
        actual_data[:th, :tw, :] = tile

        libtiff.readEncodedTile(fp, 1, tile)
        actual_data[:th, tw:w, :] = tile

        libtiff.readEncodedTile(fp, 2, tile)
        actual_data[th:h, :tw, :] = tile

        libtiff.readEncodedTile(fp, 3, tile)
        actual_data[th:h, tw:w, :] = tile

        libtiff.close(fp)

        cls.astronaut_uint16_data = actual_data
        cls.astronaut_uint16_filename = path

    @classmethod
    def setup_ycbcr_jpeg(cls, path):
        """
        SCENARIO:  create a simple color 2x2 tiled image
        """
        data = skimage.data.astronaut()
        h, w, z = data.shape
        th, tw = h // 2, w // 2

        fp = libtiff.open(path, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.YCBCR)
        libtiff.setField(fp, 'Compression', libtiff.Compression.JPEG)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'TileLength', th)
        libtiff.setField(fp, 'TileWidth', tw)
        libtiff.setField(fp, 'BitsPerSample', 8)
        libtiff.setField(fp, 'SamplesPerPixel', 3)
        libtiff.setField(fp, 'PlanarConfig', libtiff.PlanarConfig.CONTIG)
        libtiff.setField(fp, 'JPEGColorMode', libtiff.PlanarConfig.CONTIG)
        libtiff.setField(fp, 'JPEGQuality', 100)

        libtiff.writeEncodedTile(fp, 0, data[:th, :tw, :].copy())
        libtiff.writeEncodedTile(fp, 1, data[:th, tw:w, :].copy())
        libtiff.writeEncodedTile(fp, 2, data[th:h, :tw, :].copy())
        libtiff.writeEncodedTile(fp, 3, data[th:h, tw:w, :].copy())

        libtiff.close(fp)

        # now read it back
        fp = libtiff.open(path)

        tile = np.zeros((th, tw, 4), dtype=np.uint8)
        actual_data = np.zeros((h, w, 3), dtype=np.uint8)

        libtiff.readRGBATile(fp, 0, 0, tile)
        actual_data[:th, :tw, :] = tile[::-1, :, :3]

        libtiff.readRGBATile(fp, 256, 0, tile)
        actual_data[:th, tw:w, :] = tile[::-1, :, :3]

        libtiff.readRGBATile(fp, 0, 256, tile)
        actual_data[th:h, :tw, :] = tile[::-1, :, :3]

        libtiff.readRGBATile(fp, 256, 256, tile)
        actual_data[th:h, tw:w, :] = tile[::-1, :, :3]

        libtiff.close(fp)

        cls.astronaut_ycbcr_jpeg_data = actual_data
        cls.astronaut_ycbcr_jpeg_tif = path

    @classmethod
    def setup_rgb(cls, path):
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

        cls.setup_rgb_evenly_stripped(cls.test_tiff_path / 'goodstuff.tif')

        cls.setup_minisblack_spp1(cls.test_tiff_path / 'moon.tif')

        cls.setup_minisblack_3x3(cls.test_tiff_path / 'minisblack_3x3.tif')

        cls.setup_minisblack_3strip(cls.test_tiff_path / 'moon3_stripped.tif')

        path = cls.test_tiff_path / 'moon3_partial_last_strip.tif'
        cls.setup_moon_partial_last_strip(path)

        path = cls.test_tiff_path / 'moon_partial_tiles.tif'
        cls.setup_moon_partial_tiles(path)

        cls.setup_rgb(cls.test_tiff_path / 'astronaut.tif')

        cls.setup_ycbcr_jpeg(
            cls.test_tiff_path / 'astronaut_ycbcr_jpeg_tiled.tif'
        )

        cls.setup_rgb_uint16(cls.test_tiff_path / 'astronaut_uint16.tif')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_tiff_dir)

    def test_smoke(self):
        """
        SCENARIO:  Convert TIFF file to JP2

        EXPECTED RESULT:  data matches, number of resolution is the default.
        There should be just one layer.  The number of resolutions should be
        the default (5).  There are not PLT segments.  There are no EPH
        markers.  There are no SOP markers.
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        self.assertEqual(actual.shape, (512, 512, 3))

        c = j.get_codestream(header_only=False)

        actual = c.segment[2].code_block_size
        expected = (64, 64)
        self.assertEqual(actual, expected)

        self.assertEqual(c.segment[2].layers, 1)
        self.assertEqual(c.segment[2].num_res, 5)

        at_least_one_eph = any(
            isinstance(seg, glymur.codestream.EPHsegment)
            for seg in c.segment
        )
        self.assertFalse(at_least_one_eph)

        at_least_one_plt = any(
            isinstance(seg, glymur.codestream.PLTsegment)
            for seg in c.segment
        )
        self.assertFalse(at_least_one_plt)

        at_least_one_sop = any(
            isinstance(seg, glymur.codestream.SOPsegment)
            for seg in c.segment
        )
        self.assertFalse(at_least_one_sop)

    def test_sop(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with SOP markers.

        EXPECTED RESULT:  data matches, sop markers confirmed
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename, sop=True
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)
        c = j.get_codestream(header_only=False)

        at_least_one_sop = any(
            isinstance(seg, glymur.codestream.SOPsegment)
            for seg in c.segment
        )
        self.assertTrue(at_least_one_sop)

    def test_eph(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with EPH markers.

        EXPECTED RESULT:  data matches, plt markers confirmed
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename, eph=True
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)
        c = j.get_codestream(header_only=False)

        at_least_one_eph = any(
            isinstance(seg, glymur.codestream.EPHsegment)
            for seg in c.segment
        )
        self.assertTrue(at_least_one_eph)

    def test_plt(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with PLT markers.

        EXPECTED RESULT:  data matches, plt markers confirmed
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename, plt=True
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)
        c = j.get_codestream(header_only=False)

        at_least_one_plt = any(
            isinstance(seg, glymur.codestream.PLTsegment)
            for seg in c.segment
        )
        self.assertTrue(at_least_one_plt)

    def test_resolutions(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with 4 resolution layers instead
        of the default, which is 5.

        EXPECTED RESULT:  data matches, number of resolution layers is 4.
        """
        expected = 4
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename,
            numres=expected
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        self.assertEqual(actual.shape, (512, 512, 3))

        c = j.get_codestream()
        actual = c.segment[2].num_res
        self.assertEqual(actual, expected - 1)

    def test_layers(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with multiple compression layers

        EXPECTED RESULT:  data matches, number of layers is 3
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename,
            cratios=[200, 50, 10]
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        self.assertEqual(actual.shape, (512, 512, 3))

        c = j.get_codestream()
        self.assertEqual(c.segment[2].layers, 3)

    def test_codeblock_size(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with a specific code block size

        EXPECTED RESULT:  data matches, number of resolution is the default
        """
        expected = (32, 32)
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename,
            cbsize=expected
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        self.assertEqual(actual.shape, (512, 512, 3))

        c = j.get_codestream()
        actual = c.segment[2].code_block_size
        self.assertEqual(actual, expected)

    def test_smoke_verbosity(self):
        """
        SCENARIO:  Convert TIFF file to JP2, use INFO log level.

        EXPECTED RESULT:  data matches
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename,
            verbosity=logging.INFO
        ) as j:
            with self.assertLogs(logger='tiff2jp2', level=logging.INFO) as cm:
                j.run()

                self.assertEqual(len(cm.output), 1)

    def test_partial_strip_and_partial_tiles(self):
        """
        SCENARIO:  Convert monochromatic stripped TIFF file to JP2.  The TIFF
        has a partial last strip.  The JP2K will have partial tiles.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.moon_partial_last_strip, self.temp_jp2_filename,
            tilesize=(250, 250)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.moon_partial_tiles_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 480)
        self.assertEqual(c.segment[1].ysiz, 480)
        self.assertEqual(c.segment[1].xtsiz, 250)
        self.assertEqual(c.segment[1].ytsiz, 250)

    def test_partial_last_strip(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF has a
        partial last strip.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.moon_partial_last_strip, self.temp_jp2_filename,
            tilesize=(240, 240)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.minisblack_3x3_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 480)
        self.assertEqual(c.segment[1].ysiz, 480)
        self.assertEqual(c.segment[1].xtsiz, 240)
        self.assertEqual(c.segment[1].ytsiz, 240)

    def test_32bit(self):
        """
        SCENARIO:  The sample format is 32bit integer.

        EXPECTED RESULT:  RuntimeError
        """
        data = skimage.data.moon().astype(np.uint32)

        h, w = data.shape
        th, tw = h // 2, w // 2

        fp = libtiff.open(self.temp_tiff_filename, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.MINISBLACK)
        libtiff.setField(fp, 'Compression', libtiff.Compression.DEFLATE)
        libtiff.setField(fp, 'SampleFormat', libtiff.SampleFormat.UINT)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'TileLength', th)
        libtiff.setField(fp, 'TileWidth', tw)
        libtiff.setField(fp, 'BitsPerSample', 32)
        libtiff.setField(fp, 'SamplesPerPixel', 1)

        libtiff.writeEncodedTile(fp, 0, data[:th, :tw].copy())
        libtiff.writeEncodedTile(fp, 1, data[:th, tw:w].copy())
        libtiff.writeEncodedTile(fp, 2, data[th:h, :tw].copy())
        libtiff.writeEncodedTile(fp, 3, data[th:h, tw:w].copy())

        libtiff.close(fp)

        with Tiff2Jp2k(self.temp_tiff_filename, self.temp_jp2_filename) as j:
            with self.assertRaises(RuntimeError):
                j.run()

    def test_floating_point(self):
        """
        SCENARIO:  The sample format is 32bit floating point.

        EXPECTED RESULT:  RuntimeError
        """
        data = skimage.data.moon().astype(np.float32)

        h, w = data.shape
        th, tw = h // 2, w // 2

        fp = libtiff.open(self.temp_tiff_filename, mode='w')

        libtiff.setField(fp, 'Photometric', libtiff.Photometric.MINISBLACK)
        libtiff.setField(fp, 'Compression', libtiff.Compression.DEFLATE)
        libtiff.setField(fp, 'SampleFormat', libtiff.SampleFormat.IEEEFP)
        libtiff.setField(fp, 'ImageLength', data.shape[0])
        libtiff.setField(fp, 'ImageWidth', data.shape[1])
        libtiff.setField(fp, 'TileLength', th)
        libtiff.setField(fp, 'TileWidth', tw)
        libtiff.setField(fp, 'BitsPerSample', 32)
        libtiff.setField(fp, 'SamplesPerPixel', 1)

        libtiff.writeEncodedTile(fp, 0, data[:th, :tw].copy())
        libtiff.writeEncodedTile(fp, 1, data[:th, tw:w].copy())
        libtiff.writeEncodedTile(fp, 2, data[th:h, :tw].copy())
        libtiff.writeEncodedTile(fp, 3, data[th:h, tw:w].copy())

        libtiff.close(fp)

        with Tiff2Jp2k(self.temp_tiff_filename, self.temp_jp2_filename) as j:
            with self.assertRaises(RuntimeError):
                j.run()

    def test_evenly_tiled(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.moon_tif, self.temp_jp2_filename, tilesize=(256, 256)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.moon_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 256)
        self.assertEqual(c.segment[1].ytsiz, 256)

    def test_tiled_logging(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.  Logging is turned on.

        EXPECTED RESULT:  there are four messages logged, one for each tile
        """
        with Tiff2Jp2k(
            self.moon_tif, self.temp_jp2_filename, tilesize=(256, 256)
        ) as j:
            with self.assertLogs(logger='tiff2jp2', level=logging.INFO) as cm:
                j.run()

                self.assertEqual(len(cm.output), 4)

    def test_minisblack__smaller_tilesize_specified(self):
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

    def test_minisblack_3strip_to_2x2(self):
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

        np.testing.assert_array_equal(actual, self.minisblack_3x3_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 480)
        self.assertEqual(c.segment[1].ysiz, 480)
        self.assertEqual(c.segment[1].xtsiz, 240)
        self.assertEqual(c.segment[1].ytsiz, 240)

    def test_minisblack_3x3__larger_tilesize_specified(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 3x3, but we want 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.minisblack_3x3_tif, self.temp_jp2_filename,
            tilesize=(240, 240)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.minisblack_3x3_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 480)
        self.assertEqual(c.segment[1].ysiz, 480)
        self.assertEqual(c.segment[1].xtsiz, 240)
        self.assertEqual(c.segment[1].ytsiz, 240)

    def test_rgb_tiled_tiff(self):
        """
        SCENARIO:  Convert RGB TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.astronaut_tif, self.temp_jp2_filename, tilesize=(256, 256)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.astronaut_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 256)
        self.assertEqual(c.segment[1].ytsiz, 256)

    def test_ycbcr_jpeg_unevenly_tiled(self):
        """
        SCENARIO:  Convert YCBCR/JPEG TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.  The JPEG 2000 file will be tiled 75x75.

        EXPECTED RESULT:  The data matches.  No errors
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename,
            tilesize=(75, 75)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.astronaut_ycbcr_jpeg_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 75)
        self.assertEqual(c.segment[1].ytsiz, 75)

    def test_ycbcr_jpeg_tiff(self):
        """
        SCENARIO:  Convert YCBCR/JPEG TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename,
            tilesize=(256, 256)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.astronaut_ycbcr_jpeg_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 256)
        self.assertEqual(c.segment[1].ytsiz, 256)

    def test_ycbcr_jpeg_single_tile(self):
        """
        SCENARIO:  Convert YCBCR/JPEG TIFF file to JP2.  The TIFF is evenly
        tiled 2x2, but no tilesize is specified.

        EXPECTED RESULT:  The data matches.
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tif, self.temp_jp2_filename,
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.astronaut_ycbcr_jpeg_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 512)
        self.assertEqual(c.segment[1].ytsiz, 512)

    def test_tiff_file_not_there(self):
        """
        Scenario:  The input TIFF file is not present.

        Expected Result:  FileNotFoundError
        """

        with self.assertRaises(FileNotFoundError):
            Tiff2Jp2k(
                self.test_dir_path / 'not_there.tif', self.temp_jp2_filename
            )

    def test_rgb_uint16(self):
        """
        SCENARIO:  Convert RGB TIFF file to JP2.  The TIFF is evenly
        tiled 2x2 and uint16.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.astronaut_uint16_filename, self.temp_jp2_filename,
            tilesize=(256, 256)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.astronaut_uint16_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 256)
        self.assertEqual(c.segment[1].ytsiz, 256)

    def test_commandline_tiff2jp2(self):
        """
        Scenario:  patch sys such that we can run the command line tiff2jp2
        script.

        Expected Results:  Same as test_astronaut.
        """
        sys.argv = [
            '', str(self.astronaut_tif), str(self.temp_jp2_filename),
            '--tilesize', '256', '256'
        ]
        command_line.tiff2jp2()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.astronaut_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 256)
        self.assertEqual(c.segment[1].ytsiz, 256)

    def test_stripped_logging(self):
        """
        Scenario:  input TIFF is organized by strips and logging is turned on.

        Expected result:  there are 104 log messages, one for each tile
        """
        with Tiff2Jp2k(
            self.goodstuff_path, self.temp_jp2_filename, tilesize=(64, 64),
            verbosity=logging.INFO
        ) as j:
            with self.assertLogs(logger='tiff2jp2', level=logging.INFO) as cm:
                j.run()

                self.assertEqual(len(cm.output), 104)

    def test_rgb_stripped(self):
        """
        Scenario:  input TIFF is evenly divided into strips, but the tile size
        does not evenly divide either dimension.
        """
        with Tiff2Jp2k(
            self.goodstuff_path, self.temp_jp2_filename, tilesize=(64, 64)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.goodstuff_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 480)
        self.assertEqual(c.segment[1].ysiz, 800)
        self.assertEqual(c.segment[1].xtsiz, 64)
        self.assertEqual(c.segment[1].ytsiz, 64)

    def test_rgb_stripped_bottom_of_tile_coincides_with_bottom_of_strip(self):
        """
        Scenario:  input TIFF is evenly divided into strips, but the tile size
        does not evenly divide either dimension.  The strip size is 32.  The
        tile size is 13x13, so the jp2k tile in tile row 4 and column 0 will
        have it's last row only one pixel past the last row of the tiff tile
        in row 2 and column 0.

        Expected Result:  no errors
        """
        with Tiff2Jp2k(
            self.goodstuff_path, self.temp_jp2_filename, tilesize=(75, 75)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        np.testing.assert_array_equal(actual, self.goodstuff_data)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 480)
        self.assertEqual(c.segment[1].ysiz, 800)
        self.assertEqual(c.segment[1].xtsiz, 75)
        self.assertEqual(c.segment[1].ytsiz, 75)
