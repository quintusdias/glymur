# standard library imports
import importlib.resources as ir
import pathlib
import shutil
import struct
import sys
import tempfile
import unittest
from uuid import UUID
import warnings

# 3rd party library imports
import imageio.v3 as iio
import numpy as np
import skimage

# Local imports
import glymur
from glymur import Jp2k, command_line
from glymur.core import SRGB
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):

    @classmethod
    def setUpClass(cls):

        root = ir.files('tests.data.skimage')

        cls.astronaut8 = root.joinpath('astronaut8.tif')
        cls.astronaut_u16 = root.joinpath('astronaut_uint16.tif')
        cls.astronaut_s_u16 = root.joinpath('astronaut_s_uint16.tif')
        cls.astronaut8_stripped = root.joinpath('astronaut8_stripped.tif')
        cls.astronaut_ycbcr_jpeg_tiled = root.joinpath('astronaut_ycbcr_jpeg_tiled.tif')  # noqa : E501
        cls.moon = root.joinpath('moon.tif')
        cls.moon_3x3 = root.joinpath('moon_3x3.tif')
        cls.moon_3stripped = root.joinpath('moon3_stripped.tif')
        cls.moon3_partial_last_strip = root.joinpath('moon3_partial_last_strip.tif')  # noqa : E501
        cls.ycbcr_bg = root.joinpath('ycbcr_bg.tif')
        cls.ycbcr_stripped = root.joinpath('ycbcr_stripped.tif')
        cls.stripped = root.joinpath('stripped.tif')
        cls.moon63 = root.joinpath('moon63.tif')

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
            xmp_path = ir.files('tests.data.misc').joinpath('issue555.xmp')
            txt = xmp_path.read_text()
            xmp = txt + '\0'
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

    def test_psnr(self):
        """
        SCENARIO:  Convert TIFF file to JP2, specify psnr via the command line

        EXPECTED RESULT:  data matches
        """
        sys.argv = [
            '', str(self.moon), str(self.temp_jp2_filename),
            '--psnr', '30', '35', '40', '0'
        ]
        command_line.tiff2jp2()

        j = Jp2k(self.temp_jp2_filename)

        d = {}
        for layer in range(4):
            j.layer = layer
            d[layer] = j[:]

        truth = iio.imread(self.moon)

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

    def test_layers(self):
        """
        SCENARIO:  Convert TIFF file to JP2 with multiple compression layers
        using the command line.

        EXPECTED RESULT:  data matches, number of layers is 3
        """
        sys.argv = [
            '',
            str(self.astronaut_ycbcr_jpeg_tiled), str(self.temp_jp2_filename),
            '--cratio', '200', '50', '10'
        ]
        command_line.tiff2jp2()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        self.assertEqual(actual.shape, (512, 512, 3))

        c = j.get_codestream()
        self.assertEqual(c.segment[2].layers, 3)

    def test_tiff2jp2(self):
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

    def test_tiff2jp2_exclude_tags_numeric(self):
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

    def test_tiff2jp2_exclude_tags(self):
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

    def test_capture_display_resolution(self):
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

    def test_capture_display_resolution__tilesize(self):
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

    def test_tiff2jp2_xmp_uuid(self):
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

    def test_icc_profile(self):
        """
        Scenario:  The input TIFF has the ICC profile tag.  Provide the
        --include-icc-profile argument.

        Expected Result.  The ICC profile is verified in the
        ColourSpecificationBox.
        """
        path = ir.files('tests.data.tiff').joinpath('basn6a08.tif')
        buffer = path.read_bytes()
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

    def test_exclude_icc_profile(self):
        """
        Scenario:  The input TIFF has the ICC profile tag.  Do not provide the
        --include-icc-profile flag.

        Expected Result.  The ColourSpecificationBox is normal (no ICC
        profile).  The ICC profile tag will be present in the
        JpgTiffExif->JP2 UUID box.
        """
        path = ir.files('tests.data.tiff').joinpath('basn6a08.tif')

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

    def test_exclude_icc_profile__exclude_from_uuid(self):
        """
        Scenario:  The input TIFF has the ICC profile tag.  Do not specify
        the --include-icc-profile flag.  Specify the 34675 (ICCProfile) tag
        in the --exclude-tags flag.

        Expected Result.  The ICC profile is verified to not be present in the
        ColourSpecificationBox.  The ICC profile tag will be not present in the
        JpgTiffExif->JP2 UUID box.
        """
        path = ir.files('tests.data.tiff').joinpath('basn6a08.tif')

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
