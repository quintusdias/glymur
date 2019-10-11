# Standard library
try:
    import importlib.resources as ir
except ImportError:  # pragma:  no cover
    # before 3.7
    import importlib_resources as ir

# 3rd party library imports
import numpy as np

# Local imports
from glymur import Jp2k
from . import fixtures


class TestSuite(fixtures.TestCommon):

    def test_jp2_toxarray(self):
        """
        SCENARIO:  The toxarray method is invoked upon a 3-channel JP2 file.

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

    def test_2D_j2k_toxarray(self):
        """
        SCENARIO:  The toxarray method is invoked upon a 2D J2K file.

        EXPECTED RESULT:  A DataArray is returned with the expected dims,
        coords, and data values.
        """
        with ir.path('tests.data', 'p0_03.j2k') as p:
            j = Jp2k(p)
        da = j.toxarray()

        self.assertEqual(da.dims, ('y', 'x'))

        self.assertEqual(len(da.coords), 2)
        np.testing.assert_array_equal(da.coords['y'].data,
                                      np.array(range(256), dtype=np.float32))
        np.testing.assert_array_equal(da.coords['x'].data,
                                      np.array(range(256), dtype=np.float32))

        np.testing.assert_array_equal(j[:], da.data)
