# standard library imports
from contextlib import ExitStack
import unittest

# 3rd party library imports
import skimage.data

# local imports
from glymur.lib import openjp2 as opj2
import glymur.lib
from glymur import core
from glymur.jp2k import _INFO_CALLBACK, _WARNING_CALLBACK, _ERROR_CALLBACK


class TestSuite(unittest.TestCase):

    def test_moon(self):

        img = skimage.data.moon()

        num_comps = 1
        image_height, image_width = img.shape

        tile_height, tile_width = 256, 256

        comp_prec = 8
        irreversible = True

        quality_loss = 0
        cblockh_init, cblockw_init = 64, 64

        numresolution = 6
        offsetx, offsety = 0, 0

        filename = "test.j2k"

        nb_tiles_width, nb_tiles_height = 2, 2
        nb_tiles = nb_tiles_width * nb_tiles_height

        data_size = tile_width * tile_height * num_comps * comp_prec / 8

        cparams = opj2.set_default_encoder_parameters()


        cparams.tile_size_on = opj2.TRUE
        cparams.cp_tdx = tile_width
        cparams.cp_tdy = tile_height

        cparams.cblockw_init, cparams.cblockh_init = cblockw_init, cblockh_init

        cparams.irreversible = 1 if irreversible else 0

        cparams.numresolution = numresolution
        cparams.prog_order = core.PROGRESSION_ORDER['LRCP']

        # greyscale so no mct
        cparams.tcp_mct = 0

        # comptparms == l_params
        comptparms = (opj2.ImageComptParmType * num_comps)()
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
            #image = opj2.image_create(self._comptparms, self._colorspace)
            #stack.callback(opj2.image_destroy, image)

            #self._populate_image_struct(image, img_array)

            codec = opj2.create_compress(opj2.CODEC_J2K)
            stack.callback(opj2.destroy_codec, codec)

            info_handler = _INFO_CALLBACK

            opj2.set_info_handler(codec, info_handler)
            opj2.set_warning_handler(codec, _WARNING_CALLBACK)
            opj2.set_error_handler(codec, _ERROR_CALLBACK)

            # l_params == comptparms
            # l_image == tile
            image = opj2.image_tile_create(comptparms, opj2.CLRSPC_GRAY)
            stack.callback(opj2.image_destroy, image)

            image.contents.x0, image.contents.y0 = 0, 0
            image.contents.x1, image.contents.y1 = image_width, image_height
            image.contents.color_space = opj2.CLRSPC_GRAY

            opj2.setup_encoder(codec, cparams, image)

            strm = opj2.stream_create_default_file_stream(filename, False)
            stack.callback(opj2.stream_destroy, strm)

            opj2.start_compress(codec, image, strm)

            opj2.write_tile(codec, 0, img[0:256, 0:256].copy(), strm)
            opj2.write_tile(codec, 1, img[0:256, 256:512].copy(), strm)
            opj2.write_tile(codec, 2, img[256:512, 0:256].copy(), strm)
            opj2.write_tile(codec, 3, img[256:512, 256:512].copy(), strm)

            opj2.end_compress(codec, strm)
