# standard library imports
import importlib.metadata as im
import unittest

# 3rd party library imports
import numpy as np
import skimage

# Local imports
from glymur import Jp2k, JPEG2JP2
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):

    def test_smoke(self):
        """
        SCENARIO:  Convert JPEG without metadata to JP2

        EXPECTED RESULT:  data matches
        """
        files = im.files('scikit-image')
        jpeg = next(filter(lambda x: 'retina' in x.name, files), None)
        jpeg = jpeg.locate()

        with JPEG2JP2(jpeg, self.temp_jp2_filename) as j:
            j.run()

        actual = Jp2k(self.temp_jp2_filename)[:]
        expected = skimage.data.retina()

        np.testing.assert_array_equal(actual, expected)

    def test_tilesize(self):
        """
        SCENARIO:  Convert JPEG without metadata to JP2, specify tile size

        EXPECTED RESULT:  data matches, tile size matches
        """
        files = im.files('scikit-image')
        jpeg = next(filter(lambda x: 'retina' in x.name, files), None)
        jpeg = jpeg.locate()

        kwargs = {'tilesize': (512, 512)}
        with JPEG2JP2(jpeg, self.temp_jp2_filename, **kwargs) as j:
            j.run()

        jp2 = Jp2k(self.temp_jp2_filename)

        actual = jp2[:]
        expected = skimage.data.retina()

        np.testing.assert_array_equal(actual, expected)

        c = jp2.get_codestream()
        self.assertEqual(c.segment[1].xtsiz, 512)
        self.assertEqual(c.segment[1].ytsiz, 512)
