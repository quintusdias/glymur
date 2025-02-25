# standard library imports
import importlib.metadata as im
import logging
import unittest
import warnings

# 3rd party library imports
import numpy as np
import skimage

# Local imports
from glymur import Jp2k, JPEG2JP2
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):

    @classmethod
    def setUpClass(cls):

        files = im.files('scikit-image')

        jpeg = next(filter(lambda x: 'retina' in x.name, files), None)
        cls.retina = jpeg.locate()

        jpeg = next(
            filter(lambda x: 'hubble_deep_field' in x.name, files),
            None
        )
        cls.hubble = jpeg.locate()

    def test_smoke(self):
        """
        SCENARIO:  Convert JPEG without metadata to JP2

        EXPECTED RESULT:  data matches, it's just one big tile
        """

        with JPEG2JP2(self.retina, self.temp_jp2_filename) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        expected = skimage.data.retina()

        # data matches
        np.testing.assert_array_equal(actual, expected)

        [h, w, _] = actual.shape

        # it's one big tile
        c = j.get_codestream()
        self.assertEqual(c.segment[1].xtsiz, w)
        self.assertEqual(c.segment[1].ytsiz, h)

    def test_tilesize(self):
        """
        SCENARIO:  Convert JPEG without metadata to JP2, specify tile size

        EXPECTED RESULT:  data matches, tile size matches
        """

        kwargs = {'tilesize': (512, 512)}
        with JPEG2JP2(self.retina, self.temp_jp2_filename, **kwargs) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)

        actual = jp2[:]
        expected = skimage.data.retina()

        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xtsiz, 512)
        self.assertEqual(c.segment[1].ytsiz, 512)

    def test_exif(self):
        """
        SCENARIO:  Convert JPEG with EXIF metadata to JP2

        EXPECTED RESULT:  data matches, there is an EXIF UUID box
        """

        with JPEG2JP2(self.hubble, self.temp_jp2_filename) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        expected = skimage.data.hubble_deep_field()

        np.testing.assert_array_equal(actual, expected)

        box = next(filter(lambda x: x.box_id == 'uuid', j.box), None)
        self.assertIsNotNone(box)

    def test_verbosity(self):
        """
        SCENARIO:  Convert JPEG to JP2, use WARN log level.

        EXPECTED RESULT:  data matches, one message detected at WARN level
        """
        with (
            JPEG2JP2(
                self.retina, self.temp_jp2_filename, verbosity=logging.INFO
            ) as p,
            self.assertLogs(logger='tiff2jp2', level=logging.WARN) as cm
        ):
            p.run()

            self.assertEqual(len(cm.output), 1)

    def test_psnr(self):
        """
        SCENARIO:  Convert JPEG file to JP2 with the psnr keyword argument

        EXPECTED RESULT:  data matches
        """
        with JPEG2JP2(
            self.retina, self.temp_jp2_filename, psnr=(30, 35, 40, 0)
        ) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        d = {}
        for layer in range(4):
            j.layer = layer
            d[layer] = j[:]

        truth = skimage.io.imread(self.retina)

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
