import doctest
import os
import pkg_resources
import shutil
import struct
import sys
import tempfile
import unittest

import numpy as np

import glymur


# Doc tests should be run as well.
def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite('glymur.lib.openjp2'))
    return tests


class TestOpenJP2(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_set_default_encoder_parameters(self):
        cparams = glymur.lib.openjp2._set_default_encoder_parameters()

        self.assertEqual(cparams.res_spec, 0)
        self.assertEqual(cparams.cblockw_init, 64)
        self.assertEqual(cparams.cblockh_init, 64)
        self.assertEqual(cparams.numresolution, 6)
        self.assertEqual(cparams.subsampling_dx, 1)
        self.assertEqual(cparams.subsampling_dy, 1)
        self.assertEqual(cparams.mode, 0)
        self.assertEqual(cparams.prog_order, glymur.lib.openjp2.LRCP)
        self.assertEqual(cparams.roi_shift, 0)
        self.assertEqual(cparams.cp_tx0, 0)
        self.assertEqual(cparams.cp_ty0, 0)

        self.assertEqual(cparams.irreversible, 0)

    def test_set_default_decoder_parameters(self):
        dparams = glymur.lib.openjp2._set_default_decoder_parameters()

        self.assertEqual(dparams.DA_x0, 0)
        self.assertEqual(dparams.DA_y0, 0)
        self.assertEqual(dparams.DA_x1, 0)
        self.assertEqual(dparams.DA_y1, 0)

    def tile_macro(self, codec, stream, imagep, tidx):
        # called only by j2k_random_tile_access
        glymur.lib.openjp2._get_decoded_tile(codec, stream, imagep, tidx)
        for j in range(imagep.contents.numcomps):
            self.assertIsNotNone(imagep.contents.comps[j].data)

    def j2k_random_tile_access(self, filename, codec_format=None):
        # called by the test_rtaX methods
        dparam = glymur.lib.openjp2._set_default_decoder_parameters()

        infile = filename.encode()
        nelts = glymur.lib.openjp2._PATH_LEN - len(infile)
        infile += b'0' * nelts
        dparam.infile = infile

        dparam.decod_format = codec_format

        codec = glymur.lib.openjp2._create_decompress(codec_format)

        glymur.lib.openjp2._set_info_handler(codec, None)
        glymur.lib.openjp2._set_warning_handler(codec, None)
        glymur.lib.openjp2._set_error_handler(codec, None)

        x = (filename, True)
        stream = glymur.lib.openjp2._stream_create_default_file_stream_v3(*x)

        glymur.lib.openjp2._setup_decoder(codec, dparam)
        image = glymur.lib.openjp2._read_header(stream, codec)

        cstr_info = glymur.lib.openjp2._get_cstr_info(codec)

        tile_ul = 0
        tile_ur = cstr_info.contents.tw - 1
        tile_lr = cstr_info.contents.tw * cstr_info.contents.th - 1
        tile_ll = tile_lr - cstr_info.contents.tw

        self.tile_macro(codec, stream, image, tile_ul)
        self.tile_macro(codec, stream, image, tile_ur)
        self.tile_macro(codec, stream, image, tile_lr)
        self.tile_macro(codec, stream, image, tile_ll)

        glymur.lib.openjp2._destroy_cstr_info(cstr_info)

        glymur.lib.openjp2._end_decompress(codec, stream)
        glymur.lib.openjp2._destroy_codec(codec)
        glymur.lib.openjp2._stream_destroy_v3(stream)
        glymur.lib.openjp2._image_destroy(image)

    def tile_decoder(self, x0=None, y0=None, x1=None, y1=None, filename=None,
                     codec_format=None):
        x = (filename, True)
        stream = glymur.lib.openjp2._stream_create_default_file_stream_v3(*x)
        dparam = glymur.lib.openjp2._set_default_decoder_parameters()

        dparam.decod_format = codec_format

        # Do not use layer decoding limitation.
        dparam.cp_layer = 0

        # do not use resolution reductions.
        dparam.cp_reduce = 0

        codec = glymur.lib.openjp2._create_decompress(codec_format)

        glymur.lib.openjp2._set_info_handler(codec, None)
        glymur.lib.openjp2._set_warning_handler(codec, None)
        glymur.lib.openjp2._set_error_handler(codec, None)

        glymur.lib.openjp2._setup_decoder(codec, dparam)
        image = glymur.lib.openjp2._read_header(stream, codec)
        glymur.lib.openjp2._set_decode_area(codec, image, x0, y0, x1, y1)

        data = np.zeros((1150, 2048, 3), dtype=np.uint8)
        while True:
            rargs = glymur.lib.openjp2._read_tile_header(codec, stream)
            tidx = rargs[0]
            sz = rargs[1]
            go_on = rargs[-1]
            if not go_on:
                break
            glymur.lib.openjp2._decode_tile_data(codec, tidx, data, sz, stream)

        glymur.lib.openjp2._end_decompress(codec, stream)
        glymur.lib.openjp2._destroy_codec(codec)
        glymur.lib.openjp2._stream_destroy_v3(stream)
        glymur.lib.openjp2._image_destroy(image)

    def tile_encoder(self, num_comps=None, tile_width=None, tile_height=None,
                     filename=None, codec=None, comp_prec=None,
                     image_width=None, image_height=None,
                     irreversible=None):
        num_tiles = (image_width / tile_width) * (image_height / tile_height)
        tile_size = tile_width * tile_height * num_comps * comp_prec / 8

        data = np.random.random((tile_height, tile_width, num_comps))
        data = (data * 255).astype(np.uint8)

        l_param = glymur.lib.openjp2._set_default_encoder_parameters()

        l_param.tcp_numlayers = 1
        l_param.cp_fixed_quality = 1
        l_param.tcp_distoratio[0] = 20

        # position of the tile grid aligned with the image
        l_param.cp_tx0 = 0
        l_param.cp_ty0 = 0

        # tile size, we are using tile based encoding
        l_param.tile_size_on = 1
        l_param.cp_tdx = tile_width
        l_param.cp_tdy = tile_height

        # use irreversible encoding
        l_param.irreversible = irreversible

        l_param.numresolution = 6

        l_param.prog_order = glymur.lib.openjp2.LRCP

        l_params = (glymur.lib.openjp2._image_comptparm_t * num_comps)()
        for j in range(num_comps):
            l_params[j].dx = 1
            l_params[j].dy = 1
            l_params[j].h = image_height
            l_params[j].w = image_width
            l_params[j].sgnd = 0
            l_params[j].prec = comp_prec
            l_params[j].x0 = 0
            l_params[j].y0 = 0

        codec = glymur.lib.openjp2._create_compress(codec)

        glymur.lib.openjp2._set_info_handler(codec, None)
        glymur.lib.openjp2._set_warning_handler(codec, None)
        glymur.lib.openjp2._set_error_handler(codec, None)

        cspace = glymur.lib.openjp2._CLRSPC_SRGB
        l_image = glymur.lib.openjp2._image_tile_create(l_params, cspace)

        l_image.contents.x0 = 0
        l_image.contents.y0 = 0
        l_image.contents.x1 = image_width
        l_image.contents.y1 = image_height
        l_image.contents.color_space = glymur.lib.openjp2._CLRSPC_SRGB

        glymur.lib.openjp2._setup_encoder(codec, l_param, l_image)

        x = (filename, False)
        stream = glymur.lib.openjp2._stream_create_default_file_stream_v3(*x)
        glymur.lib.openjp2._start_compress(codec, l_image, stream)

        for j in np.arange(num_tiles):
            glymur.lib.openjp2._write_tile(codec, j, data, tile_size, stream)

        glymur.lib.openjp2._end_compress(codec, stream)
        glymur.lib.openjp2._stream_destroy_v3(stream)
        glymur.lib.openjp2._destroy_codec(codec)
        glymur.lib.openjp2._image_destroy(l_image)

    def tte0_setup(self, filename):
        kwargs = {'filename': filename,
                  'codec': glymur.lib.openjp2._CODEC_J2K,
                  'comp_prec': 8,
                  'irreversible': 1,
                  'num_comps': 3,
                  'image_height': 200,
                  'image_width': 200,
                  'tile_height': 100,
                  'tile_width': 100}
        self.tile_encoder(**kwargs)

    def test_tte0(self):
        # Runs test designated tte0 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            self.tte0_setup(tfile.name)

    def test_ttd0(self):
        # Runs test designated ttd0 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:

            # Produce the tte0 output file for ttd0 input.
            self.tte0_setup(tfile.name)

            kwargs = {'x0': 0,
                      'y0': 0,
                      'x1': 1000,
                      'y1': 1000,
                      'filename': tfile.name,
                      'codec_format': glymur.lib.openjp2._CODEC_J2K}
            self.tile_decoder(**kwargs)

    def tte1_setup(self, filename):
        kwargs = {'filename': filename,
                  'codec': glymur.lib.openjp2._CODEC_J2K,
                  'comp_prec': 8,
                  'irreversible': 1,
                  'num_comps': 3,
                  'image_height': 256,
                  'image_width': 256,
                  'tile_height': 128,
                  'tile_width': 128}
        self.tile_encoder(**kwargs)

    def test_tte1(self):
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            # Runs test designated tte1 in OpenJPEG test suite.
            self.tte1_setup(tfile.name)

    def test_ttd1(self):
        # Runs test designated ttd1 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:

            # Produce the tte0 output file for ttd0 input.
            self.tte1_setup(tfile.name)

            kwargs = {'x0': 0,
                      'y0': 0,
                      'x1': 128,
                      'y1': 128,
                      'filename': tfile.name,
                      'codec_format': glymur.lib.openjp2._CODEC_J2K}
            self.tile_decoder(**kwargs)

    def test_rta1(self):
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            # Runs test designated rta1 in OpenJPEG test suite.
            self.tte1_setup(tfile.name)

            kwargs = {'codec_format':  glymur.lib.openjp2._CODEC_J2K}
            self.j2k_random_tile_access(tfile.name, **kwargs)

    def tte2_setup(self, filename):
        kwargs = {'filename': filename,
                  'codec': glymur.lib.openjp2._CODEC_JP2,
                  'comp_prec': 8,
                  'irreversible': 1,
                  'num_comps': 3,
                  'image_height': 256,
                  'image_width': 256,
                  'tile_height': 128,
                  'tile_width': 128}
        self.tile_encoder(**kwargs)

    def test_tte2(self):
        # Runs test designated tte2 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            self.tte2_setup(tfile.name)

    def test_ttd2(self):
        # Runs test designated ttd2 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            # Produce the tte0 output file for ttd0 input.
            self.tte2_setup(tfile.name)

            kwargs = {'x0': 0,
                      'y0': 0,
                      'x1': 128,
                      'y1': 128,
                      'filename': tfile.name,
                      'codec_format': glymur.lib.openjp2._CODEC_JP2}
            self.tile_decoder(**kwargs)

    def test_rta2(self):
        with tempfile.NamedTemporaryFile(suffix=".jp2") as tfile:
            # Runs test designated rta2 in OpenJPEG test suite.
            self.tte2_setup(tfile.name)

            kwargs = {'codec_format':  glymur.lib.openjp2._CODEC_JP2}
            self.j2k_random_tile_access(tfile.name, **kwargs)

    def tte3_setup(self, filename):
        kwargs = {'filename': filename,
                  'codec': glymur.lib.openjp2._CODEC_J2K,
                  'comp_prec': 8,
                  'irreversible': 1,
                  'num_comps': 1,
                  'image_height': 256,
                  'image_width': 256,
                  'tile_height': 128,
                  'tile_width': 128}
        self.tile_encoder(**kwargs)

    def test_tte3(self):
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            # Runs test designated tte3 in OpenJPEG test suite.
            self.tte3_setup(tfile.name)

    def test_rta3(self):
        # Runs test designated rta3 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            self.tte3_setup(tfile.name)

            kwargs = {'codec_format':  glymur.lib.openjp2._CODEC_J2K}
            self.j2k_random_tile_access(tfile.name, **kwargs)

    def tte4_setup(self, filename):
        kwargs = {'filename': filename,
                  'codec': glymur.lib.openjp2._CODEC_J2K,
                  'comp_prec': 8,
                  'irreversible': 0,
                  'num_comps': 1,
                  'image_height': 256,
                  'image_width': 256,
                  'tile_height': 128,
                  'tile_width': 128}
        self.tile_encoder(**kwargs)

    def test_tte4(self):
        # Runs test designated tte4 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            self.tte4_setup(tfile.name)

    def test_rta4(self):
        # Runs test designated rta4 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            self.tte4_setup(tfile.name)

            kwargs = {'codec_format':  glymur.lib.openjp2._CODEC_J2K}
            self.j2k_random_tile_access(tfile.name, **kwargs)

    def tte5_setup(self, filename):
        kwargs = {'filename': filename,
                  'codec': glymur.lib.openjp2._CODEC_J2K,
                  'comp_prec': 8,
                  'irreversible': 0,
                  'num_comps': 1,
                  'image_height': 512,
                  'image_width': 512,
                  'tile_height': 256,
                  'tile_width': 256}
        self.tile_encoder(**kwargs)

    def test_tte5(self):
        # Runs test designated tte5 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            self.tte5_setup(tfile.name)

    def test_rta5(self):
        # Runs test designated rta5 in OpenJPEG test suite.
        with tempfile.NamedTemporaryFile(suffix=".j2k") as tfile:
            self.tte5_setup(tfile.name)

            kwargs = {'codec_format':  glymur.lib.openjp2._CODEC_J2K}
            self.j2k_random_tile_access(tfile.name, **kwargs)

if __name__ == "__main__":
    unittest.main()
