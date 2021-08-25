# 3rd party library imports
import skimage.io
import numpy as np

# local imports
from glymur import Jp2k
from . import fixtures


class TestSuite(fixtures.TestCommon):
    """
    Test suite for writing with tiles.
    """
    def test_astronaut(self):
        """
        SCENARIO:  construct a j2k file by tiling an image in a 2x2 grid.

        EXPECTED RESULT:  the written image validates
        """
        j2k_data = skimage.data.astronaut()
        data = [
            j2k_data[:256, :256, :],
            j2k_data[:256, 256:512, :],
            j2k_data[256:512, :256, :],
            j2k_data[256:512, 256:512, :],
        ]

        shape = j2k_data.shape
        tilesize = 256, 256

        j = Jp2k(self.temp_j2k_filename, shape=shape, tilesize=tilesize)
        for idx, tw in enumerate(j.get_tilewriters()):
            tw[:] = data[idx]

        new_j = Jp2k(self.temp_j2k_filename)
        actual = new_j[:]
        expected = j2k_data
        np.testing.assert_array_equal(actual, expected)

    def test_smoke(self):
        """
        SCENARIO:  construct a jp2 file by repeating an image in a 2x2 grid.

        EXPECTED RESULT:  the written image matches the 2x2 grid
        """
        j2k_data = skimage.data.astronaut()

        shape = (
            j2k_data.shape[0] * 2, j2k_data.shape[1] * 2, j2k_data.shape[2]
        )
        tilesize = (j2k_data.shape[0], j2k_data.shape[1])

        j = Jp2k(self.temp_j2k_filename, shape=shape, tilesize=tilesize)
        for tw in j.get_tilewriters():
            tw[:] = j2k_data

        new_j = Jp2k(self.temp_j2k_filename)
        actual = new_j[:]
        expected = np.tile(j2k_data, (2, 2, 1))
        np.testing.assert_array_equal(actual, expected)

    def test_tile_slice_has_non_none_elements(self):
        pass

    def test_use_verbose_option(self):
        pass
