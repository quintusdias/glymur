# standard library imports
import logging
import shutil
import unittest
import uuid
import warnings

# 3rd party library imports
import numpy as np
import skimage

# Local imports
import glymur
from glymur import Jp2k, JPEG2JP2
from glymur.core import SRGB
from . import fixtures
from .fixtures import (
    OPENJPEG_NOT_AVAILABLE,
    OPENJPEG_NOT_AVAILABLE_MSG,
    CANNOT_USE_IMPORTLIB_METADATA
)


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(
    CANNOT_USE_IMPORTLIB_METADATA,
    'missing importlib.metadata.files ?'
)
class TestSuite(fixtures.TestJPEGCommon):

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

    def test_default_action_icc_profile(self):
        """
        SCENARIO:  Convert JPEG with an ICC profile using default options

        EXPECTED RESULT:  data matches, it's just one big tile.  The JP2 has
        no ICC profile information.  There is a warning about a skipped ICC
        profile.
        """

        with (
            JPEG2JP2(self.rocket, self.temp_jp2_filename) as p,
            self.assertLogs(logger='tiff2jp2', level=logging.INFO) as cm,
        ):
            p.run()

        self.assertEqual(sum('ICC profile' in msg for msg in cm.output), 1)

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        expected = skimage.data.rocket()

        # data matches
        np.testing.assert_array_equal(actual, expected)

        # The colour specification box does not have the profile
        colr = j.box[2].box[1]
        self.assertEqual(colr.method, glymur.core.ENUMERATED_COLORSPACE)
        self.assertEqual(colr.precedence, 0)
        self.assertEqual(colr.approximation, 0)
        self.assertEqual(colr.colorspace, SRGB)
        self.assertIsNone(colr.icc_profile)

    def test_embed_icc_profile(self):
        """
        SCENARIO:  Convert JPEG with an ICC profile, specifically setting the
        include_icc_profile keyword.

        EXPECTED RESULT:  The JP2 has ICC profile information.
        """

        with (
            JPEG2JP2(
                self.rocket,
                self.temp_jp2_filename,
                include_icc_profile=True
            ) as p,
        ):
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        expected = skimage.data.rocket()

        # data matches
        np.testing.assert_array_equal(actual, expected)

        # The colour specification box has the profile
        self.assertIsNotNone(j.box[2].box[1].icc_profile)

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

    def test_exif_xmp(self):
        """
        SCENARIO:  Convert JPEG with EXIF and XMP metadata to JP2

        EXPECTED RESULT:  data matches, there is an EXIF UUID box, there is an
        XMP UUID box
        """

        with JPEG2JP2(self.hubble, self.temp_jp2_filename) as p:
            p.run()

        j = Jp2k(self.temp_jp2_filename)

        actual = j[:]
        expected = skimage.data.hubble_deep_field()

        np.testing.assert_array_equal(actual, expected)

        def exif_predicate(x):
            exif_uuid = uuid.UUID(bytes=b"JpgTiffExif->JP2")
            if x.box_id == 'uuid' and x.uuid == exif_uuid:
                return True
            else:
                return False

        box = next(filter(exif_predicate, j.box), None)
        self.assertIsNotNone(box)

        def xmp_predicate(x):
            xmp_uuid = uuid.UUID("be7acfcb-97a9-42e8-9c71-999491e3afac")
            if x.box_id == 'uuid' and x.uuid == xmp_uuid:
                return True
            else:
                return False

        box = next(filter(xmp_predicate, j.box), None)
        self.assertIsNotNone(box)

    def test_verbosity(self):
        """
        SCENARIO:  Convert JPEG to JP2, use WARN log level.

        EXPECTED RESULT:  data matches, one message detected at INFO level
        """
        with (
            JPEG2JP2(
                self.retina, self.temp_jp2_filename, verbosity=logging.INFO
            ) as p,
            self.assertLogs(logger='tiff2jp2', level=logging.INFO) as cm
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

    def test_existing_file(self):
        """
        Scenario:  provide an existing JP2 file as the output file

        Expected Result:  RuntimeError
        """
        shutil.copyfile(glymur.data.nemo(), self.temp_jp2_filename)
        with (
            self.assertRaises(FileExistsError),
            JPEG2JP2(self.retina, self.temp_jp2_filename) as p,
        ):
            p.run()
