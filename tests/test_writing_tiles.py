# local imports
from glymur import Jp2k
from . import fixtures

class TestSuite(fixtures.TestCommon):
    """
    Test suite for writing with tiles.
    """
    def test_smoke(self):
        """
        SCENARIO:  construct a jp2 file by repeating an image in a 2x2 grid.

        EXPECTED RESULT:  the written image matches the 2x2 grid
        """
        j2k_data = Jp2k(self.j2kfile)

        shape = j2k_data.shape * 2
        tilesize = shape
        j = Jp2k(self.temp_j2k_filename, shape=shape, tilesize=tilesize)
        for tw in j.get_tilewriters():
            tw[:] = j2k_data

        new_j = Jp2k(self.temp_j2k_filename)
        actual = new_j[:]
        expected = np.tile(j2k_data, (2, 2))
        np.testing.assert_array_equal(actual, expected) 
