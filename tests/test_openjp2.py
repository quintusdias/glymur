"""
Tests for libopenjp2 wrapping functions.
"""
# Standard library imports ...
from contextlib import ExitStack
from io import StringIO
import unittest
from unittest.mock import patch

# Third party library imports ...
import numpy as np

# Local imports ...
import glymur
from glymur.lib import openjp2
from glymur.jp2k import _INFO_CALLBACK, _WARNING_CALLBACK, _ERROR_CALLBACK
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestOpenJP2(fixtures.TestCommon):
    """Test openjp2 library functionality.

    Some tests correspond to those in the openjpeg test suite.
    """
    def test_no_openjp2_library(self):
        """
        SCENARIO:  There is no openjp2 library.

        EXPECTED RESPONSE:  The version method should return "0.0.0"
        """
        with patch.object(openjp2, 'OPENJP2', new=None):
            actual = openjp2.version()
        self.assertEqual(actual, '0.0.0')

    @unittest.skipIf(glymur.lib.openjp2.version() < '2.2.0', 'Not implemented')
    def test_get_num_cpus(self):
        """
        SCENARIO:  Hard to test this.  Values will be different across
        different machines, all we can do is test that the function runs.

        EXPECTED VALUE:  an integer
        """
        num_cpus = openjp2.get_num_cpus()
        self.assertTrue(isinstance(num_cpus, int))

    def test_default_encoder_parameters(self):
        """Ensure that the encoder structure is clean upon init."""
        cparams = openjp2.set_default_encoder_parameters()

        self.assertEqual(cparams.res_spec, 0)
        self.assertEqual(cparams.cblockw_init, 64)
        self.assertEqual(cparams.cblockh_init, 64)
        self.assertEqual(cparams.numresolution, 6)
        self.assertEqual(cparams.subsampling_dx, 1)
        self.assertEqual(cparams.subsampling_dy, 1)
        self.assertEqual(cparams.mode, 0)
        self.assertEqual(cparams.prog_order, glymur.core.LRCP)
        self.assertEqual(cparams.roi_shift, 0)
        self.assertEqual(cparams.cp_tx0, 0)
        self.assertEqual(cparams.cp_ty0, 0)

        self.assertEqual(cparams.irreversible, 0)

    def test_default_decoder_parameters(self):
        """Tests that the structure is clean upon initialization"""
        dparams = openjp2.set_default_decoder_parameters()

        self.assertEqual(dparams.DA_x0, 0)
        self.assertEqual(dparams.DA_y0, 0)
        self.assertEqual(dparams.DA_x1, 0)
        self.assertEqual(dparams.DA_y1, 0)

    def test_tte0(self):
        """
        SCENARIO:  create a J2K file with the following characteristics:

            8-bit
            3 components
            irreversible wavelet transform
            200x200 image
            2x2 tiling

        This corresponds to test tte0 in OpenJPEG test suite.

        EXPECTED RESULT:  no exceptions are raised
        """
        ttx0_setup(str(self.temp_j2k_filename))
        self.assertTrue(True)

    def test_ttd0(self):
        """Runs test designated ttd0 in OpenJPEG test suite."""
        filename = str(self.temp_j2k_filename)
        ttx0_setup(filename)

        kwargs = {'x0': 0,
                  'y0': 0,
                  'x1': 1000,
                  'y1': 1000,
                  'filename': filename,
                  'codec_format': openjp2.CODEC_J2K}
        tile_decoder(**kwargs)
        self.assertTrue(True)

    def xtx1_setup(self, filename):
        """Runs tests tte1, rta1."""
        kwargs = {'filename': filename,
                  'codec': openjp2.CODEC_J2K,
                  'comp_prec': 8,
                  'irreversible': 1,
                  'num_comps': 3,
                  'image_height': 256,
                  'image_width': 256,
                  'tile_height': 128,
                  'tile_width': 128}
        tile_encoder(**kwargs)
        self.assertTrue(True)

    def test_tte1(self):
        """Runs test designated tte1 in OpenJPEG test suite."""
        filename = str(self.temp_j2k_filename)
        self.xtx1_setup(filename)

    def test_ttd1(self):
        """Runs test designated ttd1 in OpenJPEG test suite."""
        filename = str(self.temp_j2k_filename)

        # Produce the tte0 output file for ttd0 input.
        self.xtx1_setup(filename)

        kwargs = {'x0': 0,
                  'y0': 0,
                  'x1': 128,
                  'y1': 128,
                  'filename': filename,
                  'codec_format': openjp2.CODEC_J2K}
        tile_decoder(**kwargs)
        self.assertTrue(True)

    def test_tte2(self):
        """Runs test designated tte2 in OpenJPEG test suite."""
        filename = str(self.temp_j2k_filename)
        xtx2_setup(filename)
        self.assertTrue(True)

    def test_ttd2(self):
        """Runs test designated ttd2 in OpenJPEG test suite."""
        filename = str(self.temp_j2k_filename)

        xtx2_setup(filename)

        kwargs = {'x0': 0,
                  'y0': 0,
                  'x1': 128,
                  'y1': 128,
                  'filename': filename,
                  'codec_format': openjp2.CODEC_JP2}
        tile_decoder(**kwargs)
        self.assertTrue(True)

    def test_tte3(self):
        """Runs test designated tte3 in OpenJPEG test suite."""
        filename = str(self.temp_j2k_filename)
        xtx3_setup(filename)
        self.assertTrue(True)

    def test_tte4(self):
        """Runs test designated tte4 in OpenJPEG test suite."""
        filename = str(self.temp_j2k_filename)
        xtx4_setup(filename)
        self.assertTrue(True)

    def test_tte5(self):
        """Runs test designated tte5 in OpenJPEG test suite."""
        filename = str(self.temp_j2k_filename)
        xtx5_setup(filename)
        self.assertTrue(True)

    def test_tte5_short_write_tile_signature(self):
        """Runs test designated tte5 in OpenJPEG test suite."""
        filename = str(self.temp_j2k_filename)
        xtx5_setup(filename, short_sig=True)
        self.assertTrue(True)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_tile_write_moon(self):
        """
        Test writing tiles for a 2D image.
        """
        img = fixtures.skimage.data.moon()

        num_comps = 1
        image_height, image_width = img.shape

        tile_height, tile_width = 256, 256

        comp_prec = 8
        irreversible = True

        cblockh_init, cblockw_init = 64, 64

        numresolution = 6

        cparams = openjp2.set_default_encoder_parameters()

        cparams.tile_size_on = openjp2.TRUE
        cparams.cp_tdx = tile_width
        cparams.cp_tdy = tile_height

        cparams.cblockw_init, cparams.cblockh_init = cblockw_init, cblockh_init

        cparams.irreversible = 1 if irreversible else 0

        cparams.numresolution = numresolution
        cparams.prog_order = glymur.core.PROGRESSION_ORDER['LRCP']

        # greyscale so no mct
        cparams.tcp_mct = 0

        # comptparms == l_params
        comptparms = (openjp2.ImageComptParmType * num_comps)()
        for j in range(num_comps):
            comptparms[j].dx = 1
            comptparms[j].dy = 1
            comptparms[j].w = tile_width
            comptparms[j].h = tile_height
            comptparms[j].x0 = 0
            comptparms[j].y0 = 0
            comptparms[j].prec = comp_prec
            comptparms[j].bpp = comp_prec
            comptparms[j].sgnd = 0

        with ExitStack() as stack:

            codec = openjp2.create_compress(openjp2.CODEC_J2K)
            stack.callback(openjp2.destroy_codec, codec)

            info_handler = _INFO_CALLBACK

            openjp2.set_info_handler(codec, info_handler)
            openjp2.set_warning_handler(codec, _WARNING_CALLBACK)
            openjp2.set_error_handler(codec, _ERROR_CALLBACK)

            # l_params == comptparms
            # l_image == tile
            image = openjp2.image_tile_create(comptparms, openjp2.CLRSPC_GRAY)
            stack.callback(openjp2.image_destroy, image)

            image.contents.x0, image.contents.y0 = 0, 0
            image.contents.x1, image.contents.y1 = image_width, image_height
            image.contents.color_space = openjp2.CLRSPC_GRAY

            openjp2.setup_encoder(codec, cparams, image)

            filename = str(self.temp_j2k_filename)
            strm = openjp2.stream_create_default_file_stream(filename, False)
            stack.callback(openjp2.stream_destroy, strm)

            openjp2.start_compress(codec, image, strm)

            openjp2.write_tile(codec, 0, img[0:256, 0:256].copy(), strm)
            openjp2.write_tile(codec, 1, img[0:256, 256:512].copy(), strm)
            openjp2.write_tile(codec, 2, img[256:512, 0:256].copy(), strm)
            openjp2.write_tile(codec, 3, img[256:512, 256:512].copy(), strm)

            openjp2.end_compress(codec, strm)

    @unittest.skipIf(
        not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
    )
    def test_write_tiles_3D(self):
        """
        Test writing tiles for an RGB image.
        """

        img = fixtures.skimage.data.astronaut()

        image_height, image_width, num_comps = img.shape

        tile_height, tile_width = 256, 256

        comp_prec = 8
        irreversible = False

        cblockh_init, cblockw_init = 64, 64

        numresolution = 6

        cparams = openjp2.set_default_encoder_parameters()

        outfile = str(self.temp_j2k_filename).encode()
        num_pad_bytes = openjp2.PATH_LEN - len(outfile)
        outfile += b'0' * num_pad_bytes
        cparams.outfile = outfile

        # not from openjpeg test file
        cparams.cp_disto_alloc = 1

        cparams.tile_size_on = openjp2.TRUE
        cparams.cp_tdx = tile_width
        cparams.cp_tdy = tile_height

        cparams.cblockw_init, cparams.cblockh_init = cblockw_init, cblockh_init

        # not from openjpeg test file
        cparams.mode = 0

        cparams.irreversible = 1 if irreversible else 0

        cparams.numresolution = numresolution
        cparams.prog_order = glymur.core.PROGRESSION_ORDER['LRCP']

        cparams.tcp_mct = 1

        cparams.tcp_numlayers = 1
        cparams.tcp_rates[0] = 0
        cparams.tcp_distoratio[0] = 0

        # comptparms == l_params
        comptparms = (openjp2.ImageComptParmType * num_comps)()
        for j in range(num_comps):
            comptparms[j].dx = 1
            comptparms[j].dy = 1
            comptparms[j].w = image_width
            comptparms[j].h = image_height
            comptparms[j].x0 = 0
            comptparms[j].y0 = 0
            comptparms[j].prec = comp_prec
            comptparms[j].bpp = comp_prec
            comptparms[j].sgnd = 0

        with ExitStack() as stack:
            codec = openjp2.create_compress(openjp2.CODEC_J2K)
            stack.callback(openjp2.destroy_codec, codec)

            info_handler = _INFO_CALLBACK

            openjp2.set_info_handler(codec, info_handler)
            openjp2.set_warning_handler(codec, _WARNING_CALLBACK)
            openjp2.set_error_handler(codec, _ERROR_CALLBACK)

            image = openjp2.image_tile_create(comptparms, openjp2.CLRSPC_SRGB)
            stack.callback(openjp2.image_destroy, image)

            image.contents.x0, image.contents.y0 = 0, 0
            image.contents.x1, image.contents.y1 = image_width, image_height
            image.contents.color_space = openjp2.CLRSPC_SRGB

            openjp2.setup_encoder(codec, cparams, image)

            filename = str(self.temp_j2k_filename)
            strm = openjp2.stream_create_default_file_stream(filename, False)
            stack.callback(openjp2.stream_destroy, strm)

            openjp2.start_compress(codec, image, strm)

            # have to change the memory layout of 3D images in order to use
            # opj_write_tile
            openjp2.write_tile(
                codec, 0, _set_planar_pixel_order(img[0:256, 0:256, :]), strm
            )
            openjp2.write_tile(
                codec, 1, _set_planar_pixel_order(img[0:256, 256:512, :]), strm
            )
            openjp2.write_tile(
                codec, 2, _set_planar_pixel_order(img[256:512, 0:256, :]), strm
            )
            openjp2.write_tile(
                codec, 3, _set_planar_pixel_order(img[256:512, 256:512, :]),
                strm
            )

            openjp2.end_compress(codec, strm)


def _set_planar_pixel_order(img):
    """
    Reorder the image pixels so that plane-0 comes first, then plane-1, etc.
    This is a requirement for using opj_write_tile.
    """
    if img.ndim == 3:
        # C-order increments along the y-axis slowest (0), then x-axis (1),
        # then z-axis (2).  We want it to go along the z-axis slowest, then
        # y-axis, then x-axis.
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 0, 1)

    return img.copy()


def tile_encoder(**kwargs):
    """Fixture used by many tests."""
    num_tiles = ((kwargs['image_width'] / kwargs['tile_width'])
                 * (kwargs['image_height'] / kwargs['tile_height']))
    tile_size = ((kwargs['tile_width'] * kwargs['tile_height'])
                 * (kwargs['num_comps'] * kwargs['comp_prec'] / 8))

    data = np.random.random((kwargs['tile_height'],
                             kwargs['tile_width'],
                             kwargs['num_comps']))
    data = (data * 255).astype(np.uint8)

    l_param = openjp2.set_default_encoder_parameters()

    l_param.tcp_numlayers = 1
    l_param.cp_fixed_quality = 1
    l_param.tcp_distoratio[0] = 20

    # position of the tile grid aligned with the image
    l_param.cp_tx0 = 0
    l_param.cp_ty0 = 0

    # tile size, we are using tile based encoding
    l_param.tile_size_on = 1
    l_param.cp_tdx = kwargs['tile_width']
    l_param.cp_tdy = kwargs['tile_height']

    # use irreversible encoding
    l_param.irreversible = kwargs['irreversible']

    l_param.numresolution = 6

    l_param.prog_order = glymur.core.LRCP

    l_params = (openjp2.ImageComptParmType * kwargs['num_comps'])()
    for j in range(kwargs['num_comps']):
        l_params[j].dx = 1
        l_params[j].dy = 1
        l_params[j].h = kwargs['image_height']
        l_params[j].w = kwargs['image_width']
        l_params[j].sgnd = 0
        l_params[j].prec = kwargs['comp_prec']
        l_params[j].x0 = 0
        l_params[j].y0 = 0

    codec = openjp2.create_compress(kwargs['codec'])

    openjp2.set_info_handler(codec, None)
    openjp2.set_warning_handler(codec, None)
    openjp2.set_error_handler(codec, None)

    cspace = openjp2.CLRSPC_SRGB
    l_image = openjp2.image_tile_create(l_params, cspace)

    l_image.contents.x0 = 0
    l_image.contents.y0 = 0
    l_image.contents.x1 = kwargs['image_width']
    l_image.contents.y1 = kwargs['image_height']
    l_image.contents.color_space = openjp2.CLRSPC_SRGB

    openjp2.setup_encoder(codec, l_param, l_image)

    stream = openjp2.stream_create_default_file_stream(kwargs['filename'],
                                                       False)
    openjp2.start_compress(codec, l_image, stream)

    for j in np.arange(num_tiles):
        if 'short_sig' in kwargs and kwargs['short_sig']:
            openjp2.write_tile(codec, j, data, stream)
        else:
            openjp2.write_tile(codec, j, data, tile_size, stream)

    openjp2.end_compress(codec, stream)
    openjp2.stream_destroy(stream)
    openjp2.destroy_codec(codec)
    openjp2.image_destroy(l_image)


def tile_decoder(**kwargs):
    """Fixture called with various configurations by many tests.

    Reads a tile.  That's all it does.
    """
    stream = openjp2.stream_create_default_file_stream(kwargs['filename'],
                                                       True)
    dparam = openjp2.set_default_decoder_parameters()

    dparam.decod_format = kwargs['codec_format']

    # Do not use layer decoding limitation.
    dparam.cp_layer = 0

    # do not use resolution reductions.
    dparam.cp_reduce = 0

    codec = openjp2.create_decompress(kwargs['codec_format'])

    openjp2.set_info_handler(codec, None)
    openjp2.set_warning_handler(codec, None)
    openjp2.set_error_handler(codec, None)

    openjp2.setup_decoder(codec, dparam)
    image = openjp2.read_header(stream, codec)
    openjp2.set_decode_area(codec, image,
                            kwargs['x0'], kwargs['y0'],
                            kwargs['x1'], kwargs['y1'])

    data = np.zeros((1150, 2048, 3), dtype=np.uint8)
    while True:
        rargs = openjp2.read_tile_header(codec, stream)
        tidx = rargs[0]
        size = rargs[1]
        go_on = rargs[-1]
        if not go_on:
            break
        openjp2.decode_tile_data(codec, tidx, data, size, stream)

    openjp2.end_decompress(codec, stream)
    openjp2.destroy_codec(codec)
    openjp2.stream_destroy(stream)
    openjp2.image_destroy(image)


def ttx0_setup(filename):
    """Runs tests tte0, tte0."""
    kwargs = {'filename': filename,
              'codec': openjp2.CODEC_J2K,
              'comp_prec': 8,
              'irreversible': 1,
              'num_comps': 3,
              'image_height': 200,
              'image_width': 200,
              'tile_height': 100,
              'tile_width': 100}
    tile_encoder(**kwargs)


def xtx2_setup(filename):
    """Runs tests rta2, tte2, ttd2."""
    kwargs = {'filename': filename,
              'codec': openjp2.CODEC_JP2,
              'comp_prec': 8,
              'irreversible': 1,
              'num_comps': 3,
              'image_height': 256,
              'image_width': 256,
              'tile_height': 128,
              'tile_width': 128}
    tile_encoder(**kwargs)


def xtx3_setup(filename):
    """Runs tests tte3, rta3."""
    kwargs = {'filename': filename,
              'codec': openjp2.CODEC_J2K,
              'comp_prec': 8,
              'irreversible': 1,
              'num_comps': 1,
              'image_height': 256,
              'image_width': 256,
              'tile_height': 128,
              'tile_width': 128}
    tile_encoder(**kwargs)


def xtx4_setup(filename):
    """Runs tests rta4, tte4."""
    kwargs = {'filename': filename,
              'codec': openjp2.CODEC_J2K,
              'comp_prec': 8,
              'irreversible': 0,
              'num_comps': 1,
              'image_height': 256,
              'image_width': 256,
              'tile_height': 128,
              'tile_width': 128}
    tile_encoder(**kwargs)


def xtx5_setup(filename, short_sig=False):
    """Runs tests rta5, tte5."""
    kwargs = {
        'filename': filename,
        'codec': openjp2.CODEC_J2K,
        'comp_prec': 8,
        'irreversible': 0,
        'num_comps': 1,
        'image_height': 512,
        'image_width': 512,
        'tile_height': 256,
        'tile_width': 256,
        'short_sig': short_sig
    }
    tile_encoder(**kwargs)


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestPrintingOpenjp2(unittest.TestCase):
    """Tests for verifying how printing works on openjp2 library structures."""
    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def test_decompression_parameters(self):
        """printing DecompressionParametersType"""
        dparams = glymur.lib.openjp2.set_default_decoder_parameters()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(dparams)
            actual = fake_out.getvalue().strip()
        expected = fixtures.DECOMPRESSION_PARAMETERS_TYPE
        self.assertEqual(actual, expected)

    def test_progression_order_changes(self):
        """printing PocType"""
        ptype = glymur.lib.openjp2.PocType()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(ptype)
            actual = fake_out.getvalue().strip()
        expected = fixtures.DEFAULT_PROGRESSION_ORDER_CHANGES_TYPE
        self.assertEqual(actual, expected)

    def test_default_compression_parameters(self):
        """printing default compression parameters"""
        cparams = glymur.lib.openjp2.set_default_encoder_parameters()
        with patch('sys.stdout', new=StringIO()) as fake_out:
            print(cparams)
            actual = fake_out.getvalue().strip()
        expected = fixtures.DEFAULT_COMPRESSION_PARAMETERS_TYPE
        self.assertEqual(actual, expected)
