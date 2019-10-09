import numpy as np

from glymur import Jp2k
from . import fixtures


class TestSuite(fixtures.TestCommon):

    def test_jp2_toxarray(self):
        """
        SCENARIO:  The toxarray method is invoked upon a regular JP2 file.

        EXPECTED RESULT:  A DataArray is returned with the expected dims,
        coords, and data values.
        """
        j = Jp2k(self.jp2file)
        da = j.toxarray()

        self.assertEqual(da.dims, ('y', 'x', 'bands'))

        self.assertEqual(len(da.coords), 3)
        np.testing.assert_array_equal(da.coords['y'].data,
                                      np.array(range(1456), dtype=np.float32))
        np.testing.assert_array_equal(da.coords['x'].data,
                                      np.array(range(2592), dtype=np.float32))
        np.testing.assert_array_equal(da.coords['bands'].data,
                                      np.array(range(3), dtype=np.float32))

        np.testing.assert_array_equal(j[:], da.data)
