"""
Tests for libopenjp2 wrapping functions.
"""
# Standard library imports ...
import os
import re
import sys
import tempfile
import unittest
if sys.hexversion >= 0x03000000:
    from unittest.mock import patch
else:
    from mock import patch

# Third party library imports ...
import numpy as np

# Local imports ...
import glymur
from glymur.lib import openjp2


@unittest.skipIf(os.name == "nt", "Temporary file issue on window.")
@unittest.skipIf(re.match(r'''0|1.5|2.0''',
                          glymur.version.openjpeg_version) is not None,
                 "Not to be run until 2.1.0")
class TestOpenJP2(unittest.TestCase):
    """Test openjp2 library functionality.

    Some tests correspond to those in the openjpeg test suite.
    """

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
        """Runs test designated tte0 in OpenJPEG test suite."""
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            ttx0_setup(tfile.name)
        self.assertTrue(True)

    def test_ttd0(self):
        """Runs test designated ttd0 in OpenJPEG test suite."""
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:

            # Produce the tte0 output file for ttd0 input.
            ttx0_setup(tfile.name)

            kwargs = {'x0': 0,
                      'y0': 0,
                      'x1': 1000,
                      'y1': 1000,
                      'filename': tfile.name,
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
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            self.xtx1_setup(tfile.name)

    def test_ttd1(self):
        """Runs test designated ttd1 in OpenJPEG test suite."""
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:

            # Produce the tte0 output file for ttd0 input.
            self.xtx1_setup(tfile.name)

            kwargs = {'x0': 0,
                      'y0': 0,
                      'x1': 128,
                      'y1': 128,
                      'filename': tfile.name,
                      'codec_format': openjp2.CODEC_J2K}
            tile_decoder(**kwargs)
        self.assertTrue(True)

    def test_tte2(self):
        """Runs test designated tte2 in OpenJPEG test suite."""
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            xtx2_setup(tfile.name)
        self.assertTrue(True)

    def test_ttd2(self):
        """Runs test designated ttd2 in OpenJPEG test suite."""
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            # Produce the tte0 output file for ttd0 input.
            xtx2_setup(tfile.name)

            kwargs = {'x0': 0,
                      'y0': 0,
                      'x1': 128,
                      'y1': 128,
                      'filename': tfile.name,
                      'codec_format': openjp2.CODEC_JP2}
            tile_decoder(**kwargs)
        self.assertTrue(True)

    def test_tte3(self):
        """Runs test designated tte3 in OpenJPEG test suite."""
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            xtx3_setup(tfile.name)
        self.assertTrue(True)

    def test_tte4(self):
        """Runs test designated tte4 in OpenJPEG test suite."""
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            xtx4_setup(tfile.name)
        self.assertTrue(True)

    def test_tte5(self):
        """Runs test designated tte5 in OpenJPEG test suite."""
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            xtx5_setup(tfile.name)
        self.assertTrue(True)


def tile_encoder(**kwargs):
    """Fixture used by many tests."""
    num_tiles = ((kwargs['image_width'] / kwargs['tile_width']) *
                 (kwargs['image_height'] / kwargs['tile_height']))
    tile_size = ((kwargs['tile_width'] * kwargs['tile_height']) *
                 (kwargs['num_comps'] * kwargs['comp_prec'] / 8))

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


def xtx5_setup(filename):
    """Runs tests rta5, tte5."""
    kwargs = {'filename': filename,
              'codec': openjp2.CODEC_J2K,
              'comp_prec': 8,
              'irreversible': 0,
              'num_comps': 1,
              'image_height': 512,
              'image_width': 512,
              'tile_height': 256,
              'tile_width': 256}
    tile_encoder(**kwargs)


@unittest.skipIf(sys.hexversion < 0x03000000, "do not care about 2.7 here")
@unittest.skipIf(re.match('0|1|2.0', glymur.version.openjpeg_version),
                 "Requires openjpeg 2.1.0 or higher")
class TestPrintingOpenjp2(unittest.TestCase):
    """Tests for verifying how printing works on openjp2 library structures."""
    def setUp(self):
        self.jp2file = glymur.data.nemo()

    def tearDown(self):
        pass

    def test_decompression_parameters(self):
        """printing DecompressionParametersType"""
        dparams = glymur.lib.openjp2.set_default_decoder_parameters()
        actual = str(dparams)
        exp = ("<class 'glymur.lib.openjp2.DecompressionParametersType'>:\n"
               "    cp_reduce: 0\n"
               "    cp_layer: 0\n"
               "    infile: b''\n"
               "    outfile: b''\n"
               "    decod_format: -1\n"
               "    cod_format: -1\n"
               "    DA_x0: 0\n"
               "    DA_x1: 0\n"
               "    DA_y0: 0\n"
               "    DA_y1: 0\n"
               "    m_verbose: 0\n"
               "    tile_index: 0\n"
               "    nb_tile_to_decode: 0\n"
               "    jpwl_correct: 0\n"
               "    jpwl_exp_comps: 0\n"
               "    jpwl_max_tiles: 0\n"
               "    flags: 0\n")
        self.assertEqual(actual, exp)

    def test_progression_order_changes(self):
        """printing PocType"""
        ptype = glymur.lib.openjp2.PocType()
        actual = str(ptype)
        expected = ("<class 'glymur.lib.openjp2.PocType'>:\n"
                    "    resno0: 0\n"
                    "    compno0: 0\n"
                    "    layno1: 0\n"
                    "    resno1: 0\n"
                    "    compno1: 0\n"
                    "    layno0: 0\n"
                    "    precno0: 0\n"
                    "    precno1: 0\n"
                    "    prg1: 0\n"
                    "    prg: 0\n"
                    "    progorder: b''\n"
                    "    tile: 0\n"
                    "    tx0: 0\n"
                    "    tx1: 0\n"
                    "    ty0: 0\n"
                    "    ty1: 0\n"
                    "    layS: 0\n"
                    "    resS: 0\n"
                    "    compS: 0\n"
                    "    prcS: 0\n"
                    "    layE: 0\n"
                    "    resE: 0\n"
                    "    compE: 0\n"
                    "    prcE: 0\n"
                    "    txS: 0\n"
                    "    txE: 0\n"
                    "    tyS: 0\n"
                    "    tyE: 0\n"
                    "    dx: 0\n"
                    "    dy: 0\n"
                    "    lay_t: 0\n"
                    "    res_t: 0\n"
                    "    comp_t: 0\n"
                    "    prec_t: 0\n"
                    "    tx0_t: 0\n"
                    "    ty0_t: 0\n")
        self.assertEqual(actual, expected)

    def test_default_compression_parameters(self):
        """printing default compression parameters"""
        cparams = glymur.lib.openjp2.set_default_encoder_parameters()
        actual = str(cparams)

        expected = ("<class 'glymur.lib.openjp2.CompressionParametersType'>:\n"
                    "    tile_size_on: 0\n"
                    "    cp_tx0: 0\n"
                    "    cp_ty0: 0\n"
                    "    cp_tdx: 0\n"
                    "    cp_tdy: 0\n"
                    "    cp_disto_alloc: 0\n"
                    "    cp_fixed_alloc: 0\n"
                    "    cp_fixed_quality: 0\n"
                    "    cp_matrice: None\n"
                    "    cp_comment: None\n"
                    "    csty: 0\n"
                    "    prog_order: 0\n"
                    "    numpocs: 0\n"
                    "    numpocs: 0\n"
                    "    tcp_numlayers: 0\n"
                    "    tcp_rates: []\n"
                    "    tcp_distoratio: []\n"
                    "    numresolution: 6\n"
                    "    cblockw_init: 64\n"
                    "    cblockh_init: 64\n"
                    "    mode: 0\n"
                    "    irreversible: 0\n"
                    "    roi_compno: -1\n"
                    "    roi_shift: 0\n"
                    "    res_spec: 0\n"
                    "    prch_init: []\n"
                    "    prcw_init: []\n"
                    "    infile: b''\n"
                    "    outfile: b''\n"
                    "    index_on: 0\n"
                    "    index: b''\n"
                    "    image_offset_x0: 0\n"
                    "    image_offset_y0: 0\n"
                    "    subsampling_dx: 1\n"
                    "    subsampling_dy: 1\n"
                    "    decod_format: -1\n"
                    "    cod_format: -1\n"
                    "    jpwl_epc_on: 0\n"
                    "    jpwl_hprot_mh: 0\n"
                    "    jpwl_hprot_tph_tileno: "
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
                    "    jpwl_hprot_tph: "
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
                    "    jpwl_pprot_tileno: "
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
                    "    jpwl_pprot_packno: "
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
                    "    jpwl_pprot: "
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
                    "    jpwl_sens_size: 0\n"
                    "    jpwl_sens_addr: 0\n"
                    "    jpwl_sens_range: 0\n"
                    "    jpwl_sens_mh: 0\n"
                    "    jpwl_sens_tph_tileno: "
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
                    "    jpwl_sens_tph: "
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
                    "    cp_cinema: 0\n"
                    "    max_comp_size: 0\n"
                    "    cp_rsiz: 0\n"
                    "    tp_on: 0\n"
                    "    tp_flag: 0\n"
                    "    tcp_mct: 0\n"
                    "    jpip_on: 0\n"
                    "    mct_data: None\n"
                    "    max_cs_size: 0\n"
                    "    rsiz: 0\n")
        self.assertEqual(actual, expected)

    def test_default_component_parameters(self):
        """printing default image component parameters"""
        icpt = glymur.lib.openjp2.ImageComptParmType()
        actual = str(icpt)

        expected = ("<class 'glymur.lib.openjp2.ImageComptParmType'>:\n"
                    "    dx: 0\n"
                    "    dy: 0\n"
                    "    w: 0\n"
                    "    h: 0\n"
                    "    x0: 0\n"
                    "    y0: 0\n"
                    "    prec: 0\n"
                    "    bpp: 0\n"
                    "    sgnd: 0\n")
        self.assertEqual(actual, expected)

    def test_default_image_type(self):
        """printing default image type"""
        it = glymur.lib.openjp2.ImageType()
        actual = str(it)

        # The "icc_profile_buf" field is problematic as it is a pointer value.
        # Easiest to do this as a regular expression.
        expected = ("<class 'glymur.lib.openjp2.ImageType'>:\n"
                    "    x0: 0\n"
                    "    y0: 0\n"
                    "    x1: 0\n"
                    "    y1: 0\n"
                    "    numcomps: 0\n"
                    "    color_space: 0\n"
                    "    icc_profile_buf: <glymur.lib.openjp2.LP_c_ubyte "
                    "object at 0x[0-9A-Fa-f]*>\n"
                    "    icc_profile_len: 0")
        self.assertRegex(actual, expected)
