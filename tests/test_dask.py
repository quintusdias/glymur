"""
The dtype attribute was introduced solely for the purpose of facilitating dask
access.
"""

# Third party library imports
import dask.array as da
import numpy as np

# Local imports
from glymur import Jp2k
from .fixtures import TestCommon


class TestSuite(TestCommon):

    def test_dask_array_values(self):
        """
        SCENARIO:  Create a dask array from a Jp2k.

        EXPECTED RESULT:  The dask array values are the same as the Jp2k image
        values.
        """
        j = Jp2k(self.jp2file)
        expected = j[:]
        x = da.from_array(j, chunks=(728, 1296, 3))
        actual = x.compute()
        np.testing.assert_array_equal(actual, expected)

    def test_bad_datatype(self):
        """
        SCENARIO:  The SIZ segment bitdepths are not the same.

        EXPECTED RESULT:  A Runtime exception is issued when the Jp2k dtype
        attribute (property) is accessed.
        """
        j = Jp2k(self.jp2file)

        # Change the bitdepths
        j.codestream.segment[1].bitdepth = (8, 8, 12)

        with self.assertRaises(RuntimeError):
            j.dtype

    def test_datatype_uint8(self):
        """
        SCENARIO:  The SIZ segment is consistent with uint8

        EXPECTED RESULT:  The dtype attribute specifies uint8.
        """
        j = Jp2k(self.jp2file)

        # Change the bitdepths
        j.codestream.segment[1].bitdepth = (8, 8, 8)
        j.codestream.segment[1].signed = (False, False, False)

        self.assertEqual(j.dtype, np.uint8)

    def test_datatype_uint16(self):
        """
        SCENARIO:  The SIZ segment is consistent with uint16

        EXPECTED RESULT:  The dtype attribute specifies uint16.
        """
        j = Jp2k(self.jp2file)

        # Change the bitdepths
        j.codestream.segment[1].bitdepth = (16, 16, 16)

        self.assertEqual(j.dtype, np.uint16)

    def test_datatype_int8(self):
        """
        SCENARIO:  The SIZ segment is consistent with int8.

        EXPECTED RESULT:  The dtype attribute specifies uint8.
        """
        j = Jp2k(self.jp2file)

        # Change the bitdepths
        j.codestream.segment[1].bitdepth = (8, 8, 8)
        j.codestream.segment[1].signed = (True, True, True)

        self.assertEqual(j.dtype, np.int8)

    def test_datatype_int16(self):
        """
        SCENARIO:  The SIZ segment is consistent with int16.

        EXPECTED RESULT:  The dtype attribute specifies int16.
        """
        j = Jp2k(self.jp2file)

        # Change the bitdepths
        j.codestream.segment[1].bitdepth = (16, 16, 16)
        j.codestream.segment[1].signed = (True, True, True)

        self.assertEqual(j.dtype, np.int16)
