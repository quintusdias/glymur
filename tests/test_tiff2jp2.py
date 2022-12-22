# standard library imports
import importlib.resources as ir
import logging
import pathlib
import platform
import shutil
import struct
import sys
import tempfile
import unittest
from uuid import UUID
import warnings

# 3rd party library imports
import numpy as np
import skimage

# Local imports
import glymur
from glymur import Jp2k, Tiff2Jp2k, command_line
from glymur.core import SRGB
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


def _file_helper(filename, module='tests.data.skimage'):
    """
    Mask importlib.resources differences between >=3.9 and below.
    """
    if sys.version_info[1] >= 9:
        return ir.files(module).joinpath(filename)
    else:
        with ir.path(module, filename) as path:
            return path


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):

    @classmethod
    def setUpClass(cls):

        cls.astronaut8 = _file_helper('astronaut8.tif')
        cls.astronaut_u16 = _file_helper('astronaut_uint16.tif')
        cls.astronaut8_stripped = _file_helper('astronaut8_stripped.tif')
        cls.astronaut_ycbcr_jpeg_tiled = _file_helper('astronaut_ycbcr_jpeg_tiled.tif')  # noqa : E501
        cls.moon = _file_helper('moon.tif')
        cls.moon_3x3 = _file_helper('moon_3x3.tif')
        cls.moon_3stripped = _file_helper('moon3_stripped.tif')
        cls.moon3_partial_last_strip = _file_helper('moon3_partial_last_strip.tif')  # noqa : E501
        cls.ycbcr_bg = _file_helper('ycbcr_bg.tif')
        cls.stripped = _file_helper('stripped.tif')

        test_tiff_dir = tempfile.mkdtemp()
        cls.test_tiff_path = pathlib.Path(test_tiff_dir)

        cls.setup_exif(cls.test_tiff_path / 'exif.tif')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_tiff_path)

    @classmethod
    def setup_exif(cls, path):
        """
        Create a simple TIFF file that is constructed to contain an EXIF IFD.
        """

        # main TIFF header @ 0
        # image data @ 8
        # main IFD @ 65544 = 256*256 + 8 (2 + 12*12 = 146 bytes)
        # main IDF data @ 65690  = main_ifd + 2 + 12 * 12 + 4
        #
        # strip offsets @ 65694 (16 bytes)
        # strip byte counts @ 65710 (16 bytes)
        # xmp data @ 65726 (12532 bytes)
        # camera ID data @ 78258 (8 bytes)
        #
        # exif IFD @ 78266 (2 + 2*12 + 4 = 30 bytes)
        # exif IFD data @ 78296 (6 bytes)
        with path.open(mode='wb') as f:

            w = 256
            h = 256
            rps = 64
            header_length = 8

            # write the header (8 bytes).  The IFD will follow the image data
            # (256x256 bytes), so the offset to the IFD will be 8 + h * w.
            main_ifd_offset = header_length + h * w
            buffer = struct.pack('<BBHI', 73, 73, 42, main_ifd_offset)
            f.write(buffer)

            # write the image data, 4 64x256 strips of all zeros
            strip = bytes([0] * rps * w)
            f.write(strip)
            f.write(strip)
            f.write(strip)
            f.write(strip)

            # write an IFD with 12 tags
            main_ifd_data_offset = main_ifd_offset + 2 + 12 * 12 + 4

            buffer = struct.pack('<H', 12)
            f.write(buffer)

            # width and length and bitspersample
            buffer = struct.pack('<HHII', 256, 4, 1, w)
            f.write(buffer)
            buffer = struct.pack('<HHII', 257, 4, 1, h)
            f.write(buffer)
            buffer = struct.pack('<HHII', 258, 4, 1, 8)
            f.write(buffer)

            # photometric
            buffer = struct.pack('<HHII', 262, 4, 1, 1)
            f.write(buffer)

            # strip offsets
            buffer = struct.pack('<HHII', 273, 4, 4, main_ifd_data_offset)
            f.write(buffer)

            # spp
            buffer = struct.pack('<HHII', 277, 4, 1, 1)
            f.write(buffer)

            # rps
            buffer = struct.pack('<HHII', 278, 4, 1, 64)
            f.write(buffer)

            # strip byte counts
            buffer = struct.pack('<HHII', 279, 4, 4, main_ifd_data_offset + 16)
            f.write(buffer)

            # pagenumber
            buffer = struct.pack('<HHIHH', 297, 3, 2, 1, 0)
            f.write(buffer)

            # XMP
            xmp_path = fixtures._path_to('issue555.xmp')
            with xmp_path.open() as f2:
                xmp = f2.read()
                xmp = xmp + '\0'
            buffer = struct.pack(
                '<HHII', 700, 1, len(xmp), main_ifd_data_offset + 32
            )
            f.write(buffer)

            # exif tag
            # write it AFTER lensinfo, which is 8 chars
            exif_ifd_offset = main_ifd_data_offset + 32 + len(xmp) + 8
            buffer = struct.pack('<HHII', 34665, 4, 1, exif_ifd_offset)
            f.write(buffer)

            # lensmodel
            offset = main_ifd_data_offset + 32 + len(xmp)
            buffer = struct.pack('<HHII', 50708, 2, 8, offset)
            f.write(buffer)

            # terminate the IFD
            buffer = struct.pack('<I', 0)
            f.write(buffer)

            # write the strip offsets here
            buffer = struct.pack(
                '<IIII', 8, 8 + rps*w, 8 + 2*rps*w, 8 + 3*rps*w
            )
            f.write(buffer)

            # write the strip byte counts
            buffer = struct.pack('<IIII', rps*w, rps*w, rps*w, rps*w)
            f.write(buffer)

            # write the XMP data
            f.write(xmp.encode('utf-8'))

            # write the camera ID
            f.write("abcdefg\x00".encode('utf-8'))

            # write a minimal Exif IFD
            buffer = struct.pack('<H', 2)
            f.write(buffer)

            # exposure program
            buffer = struct.pack('<HHIHH', 34850, 3, 1, 2, 0)
            f.write(buffer)

            # lens model
            data_location = exif_ifd_offset + 2 + 2*12 + 4
            buffer = struct.pack('<HHII', 42036, 2, 6, data_location)
            f.write(buffer)

            # terminate the IFD
            buffer = struct.pack('<I', 0)
            f.write(buffer)

            data = 'Canon\0'.encode('utf-8')
            buffer = struct.pack('<BBBBBB', *data)
            f.write(buffer)

        cls.exif = path

    def test_smoke(self):
        """
        SCENARIO:  Convert TIFF file to JP2

        EXPECTED RESULT:  data matches, number of resolution is the default.
        There should be just one layer.  The number of resolutions should be
        the default (5).  There are not PLT segments.  There are no EPH
        markers.  There are no SOP markers.  The progression order is LRCP.
        The irreversible transform will NOT be used.  PSNR cannot be tested
        if it is not applied.

        There is a UUID box appended at the end containing the metadata.
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename
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

        self.assertEqual(c.segment[2].prog_order, glymur.core.LRCP)

        self.assertEqual(
            c.segment[2].xform, glymur.core.WAVELET_XFORM_5X3_REVERSIBLE
        )

        self.assertEqual(j.box[-1].box_id, 'uuid')
        self.assertEqual(j.box[-1].data['ImageWidth'], 512)
        self.assertEqual(j.box[-1].data['ImageLength'], 512)

    def test_smoke_rgba(self):
        """
        SCENARIO:  Convert RGBA TIFF file to JP2

        EXPECTED RESULT:  data matches, number of resolution is the default.
        There should be just one layer.  The number of resolutions should be
        the default (5).  There are not PLT segments.  There are no EPH
        markers.  There are no SOP markers.  The progression order is LRCP.
        The irreversible transform will NOT be used.  PSNR cannot be tested
        if it is not applied.

        There is a UUID box appended at the end containing the metadata.
        """
        with Tiff2Jp2k(
            self.astronaut8_stripped, self.temp_jp2_filename,
            tilesize=[32, 32]
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        self.assertEqual(actual.shape, (64, 64, 3))

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

        self.assertEqual(c.segment[2].prog_order, glymur.core.LRCP)

        self.assertEqual(
            c.segment[2].xform, glymur.core.WAVELET_XFORM_5X3_REVERSIBLE
        )

        self.assertEqual(j.box[-1].box_id, 'uuid')
        self.assertEqual(j.box[-1].data['ImageWidth'], 64)
        self.assertEqual(j.box[-1].data['ImageLength'], 64)

    def test_no_uuid(self):
        """
        SCENARIO:  Convert TIFF file to JP2, but do not include the UUID box
        for the TIFF IFD.

        EXPECTED RESULT:  data matches, no UUID box
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
            create_exif_uuid=False
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        self.assertEqual(actual.shape, (512, 512, 3))

        at_least_one_uuid = any(
            isinstance(box, glymur.jp2box.UUIDBox) for box in j.box
        )
        self.assertFalse(at_least_one_uuid)

    @unittest.skipIf(
        platform.machine() == 's390x', 'See issue #546'
    )
    def test_psnr(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with the irreversible transform.

        EXPECTED RESULT:  data matches, the irreversible transform is confirmed
        """
        with Tiff2Jp2k(
            self.moon, self.temp_jp2_filename, psnr=(30, 35, 40, 0)
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)

        d = {}
        for layer in range(4):
            j.layer = layer
            d[layer] = j[:]

        truth = skimage.io.imread(self.moon)

        with warnings.catch_warnings():
            # MSE is zero for that first image, resulting in a divide-by-zero
            # warning
            warnings.simplefilter('ignore')
            psnr = [
                skimage.metrics.peak_signal_noise_ratio(truth, d[j])
                for j in range(4)
            ]

        # That first image should be lossless.
        self.assertTrue(np.isinf(psnr[0]))

        # None of the subsequent images should have inf PSNR.
        self.assertTrue(not np.any(np.isinf(psnr[1:])))

        # PSNR should increase for the remaining images.
        self.assertTrue(np.all(np.diff(psnr[1:])) > 0)

    def test_irreversible(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with the irreversible transform.

        EXPECTED RESULT:  data matches, the irreversible transform is confirmed
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
            irreversible=True
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)
        c = j.get_codestream(header_only=False)

        self.assertEqual(
            c.segment[2].xform, glymur.core.WAVELET_XFORM_9X7_IRREVERSIBLE
        )

    def test_sop(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with SOP markers.

        EXPECTED RESULT:  data matches, sop markers confirmed
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename, sop=True
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)
        c = j.get_codestream(header_only=False)

        at_least_one_sop = any(
            isinstance(seg, glymur.codestream.SOPsegment)
            for seg in c.segment
        )
        self.assertTrue(at_least_one_sop)

    def test_progression_order(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with EPH markers.

        EXPECTED RESULT:  data matches, plt markers confirmed
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
            prog='rlcp'
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)
        c = j.get_codestream(header_only=False)

        self.assertEqual(c.segment[2].prog_order, glymur.core.RLCP)

    def test_eph(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with EPH markers.

        EXPECTED RESULT:  data matches, plt markers confirmed
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename, eph=True
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)
        c = j.get_codestream(header_only=False)

        at_least_one_eph = any(
            isinstance(seg, glymur.codestream.EPHsegment)
            for seg in c.segment
        )
        self.assertTrue(at_least_one_eph)

    @unittest.skipIf(
        glymur.version.openjpeg_version < '2.4.0', "Requires as least v2.4.0"
    )
    def test_plt(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with PLT markers.

        EXPECTED RESULT:  data matches, plt markers confirmed
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
            plt=True
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
        expected_numres = 4

        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
            numres=expected_numres
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        self.assertEqual(actual.shape, (512, 512, 3))

        c = j.get_codestream()
        actual = c.segment[2].num_res
        self.assertEqual(actual, expected_numres - 1)

    def test_layers(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with multiple compression layers

        EXPECTED RESULT:  data matches, number of layers is 3
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
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
        cbsize = (32, 32)
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
            cbsize=cbsize
        ) as j:
            j.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        self.assertEqual(actual.shape, (512, 512, 3))

        c = j.get_codestream()
        actual = c.segment[2].code_block_size
        self.assertEqual(actual, cbsize)

    def test_verbosity(self):
        """
        SCENARIO:  Convert TIFF file to JP2, use INFO log level.

        EXPECTED RESULT:  data matches
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
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
            self.moon3_partial_last_strip, self.temp_jp2_filename,
            tilesize=(48, 48)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(
            self.moon3_partial_last_strip, plugin='pil'
        )
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 90)
        self.assertEqual(c.segment[1].ysiz, 90)
        self.assertEqual(c.segment[1].xtsiz, 48)
        self.assertEqual(c.segment[1].ytsiz, 48)

    def test_partial_last_strip(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF has a
        partial last strip.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.moon3_partial_last_strip,
            self.temp_jp2_filename,
            tilesize=(48, 48), verbose='DEBUG'
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]

        expected = skimage.io.imread(
            self.moon3_partial_last_strip, plugin='pil'
        )
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 90)
        self.assertEqual(c.segment[1].ysiz, 90)
        self.assertEqual(c.segment[1].xtsiz, 48)
        self.assertEqual(c.segment[1].ytsiz, 48)

    def test_32bit(self):
        """
        SCENARIO:  The sample format is 32bit integer.

        EXPECTED RESULT:  RuntimeError
        """
        infile = _file_helper('uint32.tif', module='tests.data.tiff')
        with Tiff2Jp2k(infile, self.temp_jp2_filename) as j:
            with self.assertRaises(RuntimeError):
                j.run()

    def test_floating_point(self):
        """
        SCENARIO:  The sample format is 32bit floating point.

        EXPECTED RESULT:  RuntimeError
        """
        infile = _file_helper('ieeefp32.tif', module='tests.data.tiff')
        with Tiff2Jp2k(infile, self.temp_jp2_filename) as j:
            with self.assertRaises(RuntimeError):
                j.run()

    def test_evenly_tiled(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.moon, self.temp_jp2_filename, tilesize=(64, 64)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.moon)
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 128)
        self.assertEqual(c.segment[1].ysiz, 128)
        self.assertEqual(c.segment[1].xtsiz, 64)
        self.assertEqual(c.segment[1].ytsiz, 64)

    def test_tiled_logging(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.  Logging is turned on.

        EXPECTED RESULT:  there are four messages logged, one for each tile
        """
        with Tiff2Jp2k(
            self.moon, self.temp_jp2_filename, tilesize=(64, 64)
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
            self.moon, self.temp_jp2_filename, tilesize=(32, 32)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.moon)
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 128)
        self.assertEqual(c.segment[1].ysiz, 128)
        self.assertEqual(c.segment[1].xtsiz, 32)
        self.assertEqual(c.segment[1].ytsiz, 32)

    def test_minisblack_3strip_to_2x2(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        stripped by 3, but we want 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.moon_3stripped, self.temp_jp2_filename, tilesize=(48, 48)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.moon_3stripped)
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 96)
        self.assertEqual(c.segment[1].ysiz, 96)
        self.assertEqual(c.segment[1].xtsiz, 48)
        self.assertEqual(c.segment[1].ytsiz, 48)

    def test_minisblack_3x3__larger_tilesize_specified(self):
        """
        SCENARIO:  Convert monochromatic TIFF file to JP2.  The TIFF is evenly
        tiled 3x3, but we want 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.moon_3x3, self.temp_jp2_filename, tilesize=(48, 48)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.moon_3x3)
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 96)
        self.assertEqual(c.segment[1].ysiz, 96)
        self.assertEqual(c.segment[1].xtsiz, 48)
        self.assertEqual(c.segment[1].ytsiz, 48)

    def test_rgb_tiled_bigtiff(self):
        """
        SCENARIO:  Convert RGB BigTIFF file to JP2.  The TIFF is evenly
        tiled 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.ycbcr_bg, self.temp_jp2_filename, tilesize=(256, 256),
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.ycbcr_bg, plugin='pil')
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 256)
        self.assertEqual(c.segment[1].ytsiz, 256)

    def test_rgb_tiled_tiff(self):
        """
        SCENARIO:  Convert RGB TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.astronaut8, self.temp_jp2_filename, tilesize=(32, 32)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.astronaut8)
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 64)
        self.assertEqual(c.segment[1].ysiz, 64)
        self.assertEqual(c.segment[1].xtsiz, 32)
        self.assertEqual(c.segment[1].ytsiz, 32)

    def test_ycbcr_jpeg_unevenly_tiled(self):
        """
        SCENARIO:  Convert YCBCR/JPEG TIFF file to JP2.  The TIFF is evenly
        tiled 2x2.  The JPEG 2000 file will be tiled 75x75.

        EXPECTED RESULT:  The data matches.  No errors
        """
        with Tiff2Jp2k(
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
            tilesize=(75, 75)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(
            self.astronaut_ycbcr_jpeg_tiled, plugin='pil'
        )
        np.testing.assert_array_equal(actual, expected)

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
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename,
            tilesize=(256, 256)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(
            self.astronaut_ycbcr_jpeg_tiled, plugin='pil'
        )
        np.testing.assert_array_equal(actual, expected)

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
            self.astronaut_ycbcr_jpeg_tiled, self.temp_jp2_filename
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(
            self.astronaut_ycbcr_jpeg_tiled, plugin='pil'
        )
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 512)
        self.assertEqual(c.segment[1].ysiz, 512)
        self.assertEqual(c.segment[1].xtsiz, 512)
        self.assertEqual(c.segment[1].ytsiz, 512)

    def test_rgb_uint16(self):
        """
        SCENARIO:  Convert RGB TIFF file to JP2.  The TIFF is evenly
        tiled 2x2 and uint16.

        EXPECTED RESULT:  The data matches.  The JP2 file has 4 tiles.
        """
        with Tiff2Jp2k(
            self.astronaut_u16, self.temp_jp2_filename, tilesize=(32, 32)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.astronaut_u16)
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 64)
        self.assertEqual(c.segment[1].ysiz, 64)
        self.assertEqual(c.segment[1].xtsiz, 32)
        self.assertEqual(c.segment[1].ytsiz, 32)

    def test_commandline_tiff2jp2(self):
        """
        Scenario:  patch sys such that we can run the command line tiff2jp2
        script.

        Expected Results:  Same as test_astronaut.
        """
        sys.argv = [
            '', str(self.astronaut8), str(self.temp_jp2_filename),
            '--tilesize', '32', '32'
        ]
        command_line.tiff2jp2()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.astronaut8)

        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 64)
        self.assertEqual(c.segment[1].ysiz, 64)
        self.assertEqual(c.segment[1].xtsiz, 32)
        self.assertEqual(c.segment[1].ytsiz, 32)

    def test_commandline_tiff2jp2_exclude_tags_numeric(self):
        """
        Scenario:  patch sys such that we can run the command line tiff2jp2
        script.  Exclude TileByteCounts and TileByteOffsets, but provide those
        tags as numeric values.

        Expected Results:  Same as test_astronaut.
        """
        sys.argv = [
            '', str(self.astronaut8), str(self.temp_jp2_filename),
            '--tilesize', '32', '32',
            '--exclude-tags', '324', '325'
        ]
        command_line.tiff2jp2()

        jp2 = Jp2k(self.temp_jp2_filename)
        tags = jp2.box[-1].data

        self.assertNotIn('TileByteCounts', tags)
        self.assertNotIn('TileOffsets', tags)

    def test_cmyk(self):
        """
        Scenario:  CMYK (or separated) is not a supported colorspace.

        Expected result:  RuntimeError
        """
        infile = _file_helper('cmyk.tif', module='tests.data.tiff')
        with Tiff2Jp2k(infile, self.temp_jp2_filename) as j:
            with self.assertRaises(RuntimeError):
                j.run()

    def test_commandline_tiff2jp2_exclude_tags(self):
        """
        Scenario:  patch sys such that we can run the command line tiff2jp2
        script.  Exclude TileByteCounts and TileByteOffsets

        Expected Results:  TileByteCounts and TileOffsets are not in the EXIF
        UUID.
        """
        sys.argv = [
            '', str(self.astronaut8), str(self.temp_jp2_filename),
            '--tilesize', '32', '32',
            '--exclude-tags', 'tilebytecounts', 'tileoffsets'
        ]
        command_line.tiff2jp2()

        jp2 = Jp2k(self.temp_jp2_filename)
        tags = jp2.box[-1].data

        self.assertNotIn('TileByteCounts', tags)
        self.assertNotIn('TileOffsets', tags)

    def test_numeric_exclude_keyword_argument(self):
        """
        Scenario:  specify exclude_tags keyword argument as list of integer
        keyword argument is set to True.

        Expected result:  The tags are not included in the exif IFD.
        """
        with Tiff2Jp2k(
            self.stripped, self.temp_jp2_filename,
            exclude_tags=[273, 279]
        ) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        self.assertNotIn('StripOffsets', j.box[-1].data)
        self.assertNotIn('StripByteCounts', j.box[-1].data)

    def test_string_exclude_keyword_argument(self):
        """
        Scenario:  specify exclude_tags keyword argument as list of integer
        keyword argument is set to True.

        Expected result:  The tags are not included in the exif IFD.
        """
        with Tiff2Jp2k(
            self.stripped, self.temp_jp2_filename,
            exclude_tags=['StripOffsets', 'StripByteCounts']
        ) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        self.assertNotIn('StripOffsets', j.box[-1].data)
        self.assertNotIn('StripByteCounts', j.box[-1].data)

    def test_tiff_has_no_icc_profile(self):
        """
        Scenario:  input TIFF has no ICC profile, yet the include_icc_profile
        keyword argument is set to True.

        Expected result:  a warning is issued
        """
        with Tiff2Jp2k(
            self.stripped, self.temp_jp2_filename, tilesize=(64, 64),
            include_icc_profile=True, verbosity=logging.INFO
        ) as j:
            with self.assertLogs(
                logger='tiff2jp2', level=logging.WARNING
            ) as cm:
                j.run()

                self.assertEqual(len(cm.output), 1)

    def test_stripped_logging(self):
        """
        Scenario:  input TIFF is organized by strips and logging is turned on.

        Expected result:  there are 104 log messages. These messages come from
        the tiles (a 13x8 grid of tiles).
        """
        with Tiff2Jp2k(
            self.stripped, self.temp_jp2_filename, tilesize=(64, 64),
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
            self.stripped, self.temp_jp2_filename, tilesize=(64, 64)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.stripped)
        np.testing.assert_array_equal(actual, expected)

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
            self.stripped, self.temp_jp2_filename, tilesize=(75, 75)
        ) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)
        actual = jp2[:]
        expected = skimage.io.imread(self.stripped)
        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xsiz, 480)
        self.assertEqual(c.segment[1].ysiz, 800)
        self.assertEqual(c.segment[1].xtsiz, 75)
        self.assertEqual(c.segment[1].ytsiz, 75)

    def test_exclude_tags(self):
        """
        Scenario:  Convert TIFF to JP2, but exclude the StripByteCounts and
        StripOffsets tags.

        Expected Result:  The Exif UUID box prints without error.
        The StripByteCounts and StripOffsets tags are not present.
        """
        with Tiff2Jp2k(
            self.exif, self.temp_jp2_filename,
            exclude_tags=[273, 'stripbytecounts']
        ) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        tags = j.box[-2].data
        self.assertNotIn('StripByteCounts', tags)
        self.assertNotIn('StripOffsets', tags)

        str(j.box[-1])

    def test_exclude_tags_but_specify_a_bad_tag(self):
        """
        Scenario:  Convert TIFF to JP2, but exclude the StripByteCounts and
        StripOffsets tags.  In addition, specify a tag that is not recognized.

        Expected Result:  The results should be the same as the previous
        test except that a warning is issued due to the bad tag.
        """
        with self.assertWarns(UserWarning):
            with Tiff2Jp2k(
                self.exif, self.temp_jp2_filename,
                exclude_tags=[273, 'stripbytecounts', 'gdalstuff']
            ) as p:
                p.run()

        j = Jp2k(self.temp_jp2_filename)

        tags = j.box[-2].data
        self.assertNotIn('StripByteCounts', tags)
        self.assertNotIn('StripOffsets', tags)

    def test_exclude_tags_camelcase(self):
        """
        Scenario:  Convert TIFF to JP2, but exclude the StripByteCounts and
        StripOffsets tags.  Supply the argments as camel-case.

        Expected Result:  No warnings, no errors.  The Exif LensModel tag is
        recoverable from the UUIDbox.
        """
        with Tiff2Jp2k(
            self.exif, self.temp_jp2_filename,
            exclude_tags=['StripOffsets', 'StripByteCounts']
        ) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        tags = j.box[-2].data
        self.assertNotIn('StripByteCounts', tags)
        self.assertNotIn('StripOffsets', tags)

    def test_exif(self):
        """
        Scenario:  Convert TIFF with Exif IFD to JP2

        Expected Result:  No warnings, no errors.  The Exif LensModel tag is
        recoverable from the UUIDbox.
        """
        with Tiff2Jp2k(self.exif, self.temp_jp2_filename) as p:
            with warnings.catch_warnings(record=True) as w:
                p.run()
                self.assertEqual(len(w), 0)

        j = Jp2k(self.temp_jp2_filename)

        tags = j.box[-2].data

        self.assertEqual(tags['UniqueCameraModel'], 'abcdefg')
        self.assertEqual(tags['ExifTag']['LensModel'], 'Canon')

        str(j.box[-1])

    def test_xmp(self):
        """
        Scenario:  Convert TIFF with Exif IFD to JP2.  The main IFD has an
        XML Packet tag (700).  Supply the 'xmp_uuid' keyword as True.

        Expected Result:  An Exif UUID is appended to the end of the
        JP2 file, and then an XMP UUID is appended.  The XMLPacket tag is still
        present in the UUID IFD.
        """
        with Tiff2Jp2k(
            self.exif, self.temp_jp2_filename, create_xmp_uuid=True
        ) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        # first we find the Exif UUID, then maybe the XMP UUID.  The Exif UUID
        # data should still have have the XMLPacket tag as only the exclude
        # tags keyword argument can do that.
        box = j.box[-2]
        actual = box.uuid
        expected = UUID(bytes=b'JpgTiffExif->JP2')
        self.assertEqual(actual, expected)
        self.assertIn('XMLPacket', box.data)

        # ok so the xmp UUID is the last box
        xmp_box = j.box[-1]
        actual = xmp_box.uuid
        expected = UUID('be7acfcb-97a9-42e8-9c71-999491e3afac')
        self.assertEqual(actual, expected)
        self.assertEqual(
            xmp_box.data.getroot().values(), ['Public XMP Toolkit Core 3.5']
        )

    def test_xmp_false(self):
        """
        Scenario:  Convert TIFF with Exif IFD to JP2.  The main IFD has an
        XML Packet tag (700).  Supply the 'xmp_uuid' keyword as False.

        Expected Result:  An Exif UUID is appended to the end of the
        JP2 file, but no XMP UUID is appended.  The XMLPacket tag is still
        present in the UUID data.
        """
        with Tiff2Jp2k(
            self.exif, self.temp_jp2_filename, create_xmp_uuid=False
        ) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        # we find the Exif UUID at the end.
        box = j.box[-1]
        actual = box.uuid
        expected = UUID(bytes=b'JpgTiffExif->JP2')
        self.assertEqual(actual, expected)
        self.assertIn('XMLPacket', box.data)

    def test_xmp__exclude_XMLPacket(self):
        """
        Scenario:  Convert TIFF with Exif IFD to JP2.  The main IFD has an
        XML Packet tag (700).  Supply the 'create_xmp_uuid' keyword.  Supply
        the exclude_tags keyword, but don't supply XMLPacket.

        Expected Result:  The XMLPacket tag is not removed from the main IFD.
        An Exif UUID is appended to the end of the JP2 file, and then an XMP
        UUID is appended.
        """
        kwargs = {'create_xmp_uuid': True, 'exclude_tags': ['StripOffsets']}
        with Tiff2Jp2k(self.exif, self.temp_jp2_filename, **kwargs) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        # first we find the Exif UUID, then the XMP UUID.  The Exif UUID
        # data should not have the XMLPacket tag.
        actual = j.box[-2].uuid
        expected = UUID(bytes=b'JpgTiffExif->JP2')
        self.assertEqual(actual, expected)
        self.assertIn('XMLPacket', j.box[-2].data)

        actual = j.box[-1].uuid
        expected = UUID('be7acfcb-97a9-42e8-9c71-999491e3afac')
        self.assertEqual(actual, expected)
        self.assertEqual(
            j.box[-1].data.getroot().values(), ['Public XMP Toolkit Core 3.5']
        )

    def test_commandline_capture_display_resolution(self):
        """
        Scenario:  patch sys such that we can run the command
        line tiff2jp2 script.  Supply the --capture-resolution and
        --display-resolution arguments.

        Expected Result:  The last box is a ResolutionBox.
        """
        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4

        sys.argv = [
            '', str(self.exif), str(self.temp_jp2_filename),
            '--capture-resolution', str(vresc), str(hresc),
            '--display-resolution', str(vresd), str(hresd),
        ]
        command_line.tiff2jp2()

        j = Jp2k(self.temp_jp2_filename)

        # the resolution superbox is appended in the jp2 header box.
        # the exit uuid comes later
        self.assertEqual(j.box[-1].box_id, 'uuid')

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resc')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresc)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresc)

        self.assertEqual(j.box[2].box[2].box[1].box_id, 'resd')
        self.assertEqual(j.box[2].box[2].box[1].vertical_resolution, vresd)
        self.assertEqual(j.box[2].box[2].box[1].horizontal_resolution, hresd)

    def test_commandline__capture_display_resolution__tilesize(self):
        """
        Scenario:  patch sys such that we can run the command line
        tiff2jp2 script.  Supply the --tilesize, --capture-resolution
        and --display-resolution arguments.

        Expected Result:  The last box is a ResolutionBox.
        """
        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4

        sys.argv = [
            '', str(self.exif), str(self.temp_jp2_filename),
            '--tilesize', '64', '64',
            '--capture-resolution', str(vresc), str(hresc),
            '--display-resolution', str(vresd), str(hresd),
        ]
        command_line.tiff2jp2()

        j = Jp2k(self.temp_jp2_filename)

        # the resolution superbox is appended in the jp2 header box.
        # the exit uuid comes later
        self.assertEqual(j.box[-1].box_id, 'uuid')

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resc')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresc)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresc)

        self.assertEqual(j.box[2].box[2].box[1].box_id, 'resd')
        self.assertEqual(j.box[2].box[2].box[1].vertical_resolution, vresd)
        self.assertEqual(j.box[2].box[2].box[1].horizontal_resolution, hresd)

    def test_commandline_tiff2jp2_xmp_uuid(self):
        """
        Scenario:  patch sys such that we can run the command line tiff2jp2
        script.  Use the --create-xmp-uuid option.

        Expected Result:  An Exif UUID is appended to the end of the
        JP2 file, and then an XMP UUID is appended.
        """
        sys.argv = [
            '', str(self.exif), str(self.temp_jp2_filename),
            '--tilesize', '64', '64',
            '--create-xmp-uuid'
        ]
        command_line.tiff2jp2()

        j = Jp2k(self.temp_jp2_filename)

        # first we find the Exif UUID, then the XMP UUID.
        actual = j.box[-2].uuid
        expected = UUID(bytes=b'JpgTiffExif->JP2')
        self.assertEqual(actual, expected)

        actual = j.box[-1].uuid
        expected = UUID('be7acfcb-97a9-42e8-9c71-999491e3afac')
        self.assertEqual(actual, expected)
        self.assertEqual(
            j.box[-1].data.getroot().values(), ['Public XMP Toolkit Core 3.5']
        )

    def test_one_component_no_tilesize(self):
        """
        Scenario:  The jp2 tilesize is the same as the image size.

        Expected Result:  No errors.
        """
        with Tiff2Jp2k(
            self.exif, self.temp_jp2_filename,
        ) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)
        self.assertEqual(j.box[2].box[0].num_components, 1)

    @unittest.skip('segfaulting')
    def test_one_component_tilesize(self):
        """
        Scenario:  The jp2 tilesize is the same as the image size,
        and the tilesize is specified.

        Expected Result:  No errors.
        """
        with Tiff2Jp2k(
            self.exif, self.temp_jp2_filename, tilesize=[256, 256]
        ) as p:
            p.run()

        Jp2k(self.temp_jp2_filename)

    def test_icc_profile(self):
        """
        Scenario:  The input TIFF has the ICC profile tag.  Provide the
        include_icc_profile keyword as True.

        Expected Result.  The ICC profile is verified in the
        ColourSpecificationBox.  There is a logging message at the info
        level stating that a color profile was consumed.
        """
        path = fixtures._path_to('basn6a08.tif')

        with path.open(mode='rb') as f:
            buffer = f.read()
            ifd = glymur.lib.tiff.tiff_header(buffer)
        icc_profile = bytes(ifd['ICCProfile'])

        with Tiff2Jp2k(
            path, self.temp_jp2_filename, include_icc_profile=True
        ) as p:

            with self.assertLogs(
                logger='tiff2jp2', level=logging.INFO
            ) as cm:
                p.run()

            self.assertEqual(
                sum('ICC profile' in msg for msg in cm.output), 1
            )

        j = Jp2k(self.temp_jp2_filename)

        # The colour specification box has the profile
        self.assertEqual(j.box[2].box[1].icc_profile, bytes(icc_profile))

    def test_icc_profile_commandline(self):
        """
        Scenario:  The input TIFF has the ICC profile tag.  Provide the
        --include-icc-profile argument.

        Expected Result.  The ICC profile is verified in the
        ColourSpecificationBox.
        """
        path = fixtures._path_to('basn6a08.tif')

        with path.open(mode='rb') as f:
            buffer = f.read()
            ifd = glymur.lib.tiff.tiff_header(buffer)
        icc_profile = bytes(ifd['ICCProfile'])

        sys.argv = [
            '', str(path), str(self.temp_jp2_filename),
            '--include-icc-profile'
        ]
        command_line.tiff2jp2()

        j = Jp2k(self.temp_jp2_filename)

        # The colour specification box has the profile
        self.assertEqual(j.box[2].box[1].icc_profile, bytes(icc_profile))

    def test_exclude_icc_profile_commandline(self):
        """
        Scenario:  The input TIFF has the ICC profile tag.  Do not provide the
        --include-icc-profile flag.

        Expected Result.  The ColourSpecificationBox is normal (no ICC
        profile).  The ICC profile tag will be present in the
        JpgTiffExif->JP2 UUID box.
        """
        path = fixtures._path_to('basn6a08.tif')

        sys.argv = ['', str(path), str(self.temp_jp2_filename)]
        command_line.tiff2jp2()

        j = Jp2k(self.temp_jp2_filename)

        # The colour specification box does not have the profile
        colr = j.box[2].box[1]
        self.assertEqual(colr.method, glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(colr.precedence, 0)
        self.assertEqual(colr.approximation, 0)
        self.assertEqual(colr.colorspace, SRGB)
        self.assertIsNone(colr.icc_profile)

    def test_exclude_icc_profile_commandline__exclude_from_uuid(self):
        """
        Scenario:  The input TIFF has the ICC profile tag.  Do not specify
        the --include-icc-profile flag.  Specify the 34675 (ICCProfile) tag
        in the --exclude-tags flag.

        Expected Result.  The ICC profile is verified to not be present in the
        ColourSpecificationBox.  The ICC profile tag will be not present in the
        JpgTiffExif->JP2 UUID box.
        """
        path = fixtures._path_to('basn6a08.tif')

        sys.argv = [
            '', str(path), str(self.temp_jp2_filename),
            '--exclude-tags', 'ICCProfile',
        ]
        command_line.tiff2jp2()

        j = Jp2k(self.temp_jp2_filename)

        # The colour specification box does not have the profile
        colr = j.box[2].box[1]
        self.assertEqual(colr.method, glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(colr.precedence, 0)
        self.assertEqual(colr.approximation, 0)
        self.assertEqual(colr.colorspace, SRGB)
        self.assertIsNone(colr.icc_profile)

        # the exif UUID box does not have the profile
        self.assertNotIn('ICCProfile', j.box[-1].data)

    def test_not_a_tiff(self):
        """
        Scenario:  The input "TIFF" is not actually a TIFF.  This used to
        segfault.

        Expected Result:  no segfault
        """
        with self.assertRaises(RuntimeError):
            path = fixtures._path_to('simple_rdf.txt')
            with Tiff2Jp2k(path, self.temp_jp2_filename):
                pass

    def test_colormap(self):
        """
        Scenario:  The input "TIFF" has a colormap tag.

        Expected Result:  The output JP2 has a single layer and the jp2h box
        has a pclr box.
        """
        for tag in ['ColorMap', 'StripOffsets']:
            with self.subTest(tag=tag):
                self._test_colormap(tag=tag)

    def _test_colormap(self, tag):

        kwargs = {'tilesize': (32, 32), 'exclude_tags': [tag]}
        path = fixtures._path_to('issue572.tif')
        with Tiff2Jp2k(path, self.temp_jp2_filename, **kwargs) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        # the image header box shows just a single layer
        shape = (
            j.box[2].box[0].height,
            j.box[2].box[0].width,
            j.box[2].box[0].num_components,
        )
        self.assertEqual(shape, (64, 64, 1))

        # the colr box says sRGB, not greyscale
        self.assertEqual(j.box[2].box[1].colorspace, SRGB)

        # a pclr box exists
        self.assertEqual(j.box[2].box[2].box_id, 'pclr')

        # a component mapping box exists
        self.assertEqual(j.box[2].box[3].box_id, 'cmap')
        self.assertEqual(j.box[2].box[3].component_index, (0, 0, 0))
        self.assertEqual(j.box[2].box[3].mapping_type, (1, 1, 1))
        self.assertEqual(j.box[2].box[3].palette_index, (0, 1, 2))

        # The last box should be the exif uuid.  It may or may not have the
        # colormap tag depending on what was specified.
        exif_box = j.box[-1]
        actual = exif_box.uuid
        expected = UUID(bytes=b'JpgTiffExif->JP2')
        self.assertEqual(actual, expected)
        if tag == 'ColorMap':
            self.assertNotIn('ColorMap', exif_box.data)
        else:
            self.assertIn('ColorMap', exif_box.data)

    def test_excluded_tags_is_none(self):
        """
        Scenario:  Convert TIFF to JP2, but provide None for the exclude_tags
        argument.

        Expected Result:  The UUIDbox has StripOffsets, StripByteCounts, and
        ICCProfile.
        """
        path = fixtures._path_to('basn6a08.tif')
        with Tiff2Jp2k(path, self.temp_jp2_filename, exclude_tags=None) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        # last box is exif
        tags = j.box[-1].data
        self.assertIn('StripByteCounts', tags)
        self.assertIn('StripOffsets', tags)
        self.assertIn('ICCProfile', tags)

    def test_geotiff(self):
        """
        SCENARIO:  Convert a one-component GEOTIFF file to JP2

        EXPECTED RESULT:  there is a geotiff UUID.  The JP2 file has only one
        component.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            path = fixtures._path_to('albers27.tif')
            with Tiff2Jp2k(path, self.temp_jp2_filename) as j:
                j.run()

        j = Jp2k(self.temp_jp2_filename)

        self.assertEqual(j.box[-1].box_id, 'uuid')
        self.assertEqual(
            j.box[-1].uuid, UUID('b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03')
        )
        self.assertEqual(j.box[2].box[0].num_components, 1)

    def test_separated_configuration(self):
        """
        SCENARIO:  The TIFF has a planar configuration of SEPARATE which is
        not supported if a tilesize is specified.

        EXPECTED RESULT:  RuntimeError
        """
        with self.assertRaises(RuntimeError):
            path = fixtures._path_to('flower-separated-planar-08.tif')
            with Tiff2Jp2k(
                path, self.temp_jp2_filename, tilesize=(64, 64)
            ) as j:
                j.run()

    def test_bad_tile_size(self):
        """
        SCENARIO:  Specify a tilesize that exceeds the image size.  This will
        cause a segfault unless caught.

        EXPECTED RESULT:  RuntimeError
        """
        with self.assertRaises(RuntimeError):
            path = fixtures._path_to('albers27-8.tif')
            with Tiff2Jp2k(
                path, self.temp_jp2_filename, tilesize=(256, 256),
            ) as j:
                j.run()

    def test_minisblack_spp1_bigtiff(self):
        """
        SCENARIO:  Convert minisblack BigTIFF file to JP2.  The TIFF has tag
        XResolution.

        EXPECTED RESULT:  no errors.
        """
        path = fixtures._path_to('albers27-8.tif')
        with Tiff2Jp2k(path, self.temp_jp2_filename) as j:
            j.run()

    def test_tiff_file_not_there(self):
        """
        Scenario:  The input TIFF file is not present.

        Expected Result:  FileNotFoundError
        """

        with self.assertRaises(FileNotFoundError):
            Tiff2Jp2k(
                self.test_dir_path / 'not_there.tif', self.temp_jp2_filename
            )
