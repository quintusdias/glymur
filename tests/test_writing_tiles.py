# standard library imports
import unittest

# 3rd party library imports
import numpy as np

# local imports
import glymur
from glymur import Jp2k
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(
    not fixtures.HAVE_SCIKIT_IMAGE, fixtures.HAVE_SCIKIT_IMAGE_MSG
)
@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):
    """
    Test suite for writing with tiles.
    """
    def test_astronaut(self):
        """
        SCENARIO:  construct a j2k file by tiling an image in a 2x2 grid.

        EXPECTED RESULT:  the written image validates
        """
        j2k_data = fixtures.skimage.data.astronaut()
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
        SCENARIO:  construct a j2k file by repeating a 3D image in a 2x2 grid.

        EXPECTED RESULT:  the written image matches the 2x2 grid
        """
        j2k_data = fixtures.skimage.data.astronaut()

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

    def test_moon(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 3x2 grid.

        EXPECTED RESULT:  the written image matches the 3x2 grid
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 3, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(self.temp_jp2_filename, shape=shape, tilesize=tilesize)
        for tw in j.get_tilewriters():
            tw[:] = jp2_data

        new_j = Jp2k(self.temp_jp2_filename)
        actual = new_j[:]
        expected = np.tile(jp2_data, (3, 2))
        np.testing.assert_array_equal(actual, expected)

    def test_tile_slice_has_non_none_elements(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 2x2 grid,
        but the tile writer does not receive a degenerate slice object.

        EXPECTED RESULT:  RuntimeError
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 2, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(self.temp_jp2_filename, shape=shape, tilesize=tilesize)
        with self.assertRaises(RuntimeError):
            for tw in j.get_tilewriters():
                tw[:256, :256] = jp2_data

    def test_tile_slice_is_ellipsis(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 2x2 grid,
        but the tile writer does not receive a degenerate slice object.

        EXPECTED RESULT:  RuntimeError
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 2, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(self.temp_jp2_filename, shape=shape, tilesize=tilesize)
        with self.assertRaises(RuntimeError):
            for tw in j.get_tilewriters():
                tw[...] = jp2_data

    def test_too_much_data_for_slice(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 2x2 grid,
        but the tile writer does not receive a degenerate slice object.

        EXPECTED RESULT:  RuntimeError
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 2, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(self.temp_jp2_filename, shape=shape, tilesize=tilesize)
        with self.assertRaises(glymur.lib.openjp2.OpenJPEGLibraryError):
            for tw in j.get_tilewriters():
                tw[:] = np.tile(jp2_data, (2, 2))

    def test_write_with_different_compression_ratios(self):
        """
        SCENARIO:  construct a jp2 file by repeating a 2D image in a 2x2 grid.

        EXPECTED RESULT:  There are three layers.
        """
        jp2_data = fixtures.skimage.data.moon()

        shape = jp2_data.shape[0] * 2, jp2_data.shape[1] * 2
        tilesize = (jp2_data.shape[0], jp2_data.shape[1])

        j = Jp2k(
            self.temp_jp2_filename, shape=shape, tilesize=tilesize,
            cratios=[20, 5, 1]
        )

        for tw in j.get_tilewriters():
            tw[:] = jp2_data

        codestream = j.get_codestream()
        self.assertEqual(codestream.segment[2].layers, 3)  # layers = 3

    def test_capture_resolution_and_tiled_writing(self):
        """
        SCENARIO:  Use the capture_resolution keyword.

        EXPECTED RESULT:  The resolution superbox, along with a capture
        box, is inserted into the jp2 header box.
        """
        j2k_data = fixtures.skimage.data.astronaut()

        shape = (
            j2k_data.shape[0] * 2, j2k_data.shape[1] * 2, j2k_data.shape[2]
        )
        tilesize = (j2k_data.shape[0], j2k_data.shape[1])

        vresc, hresc = 0.1, 0.2

        j = glymur.Jp2k(
            self.temp_jp2_filename, shape=shape, tilesize=tilesize,
            capture_resolution=[vresc, hresc],
        )

        for tw in j.get_tilewriters():
            tw[:] = j2k_data

        self.assertEqual(j.box[2].box[2].box_id, 'res ')

        self.assertEqual(j.box[2].box[2].box[0].box_id, 'resc')
        self.assertEqual(j.box[2].box[2].box[0].vertical_resolution, vresc)
        self.assertEqual(j.box[2].box[2].box[0].horizontal_resolution, hresc)

    def test_plt_for_tiled_writing(self):
        """
        SCENARIO:  Use the plt keyword.

        EXPECTED RESULT:  Plt segment is detected.
        """
        j2k_data = fixtures.skimage.data.astronaut()

        shape = (
            j2k_data.shape[0] * 2, j2k_data.shape[1] * 2, j2k_data.shape[2]
        )
        tilesize = (j2k_data.shape[0], j2k_data.shape[1])

        j = Jp2k(
            self.temp_j2k_filename, shape=shape, tilesize=tilesize,
            plt=True
        )
        for tw in j.get_tilewriters():
            tw[:] = j2k_data

        codestream = j.get_codestream(header_only=False)

        at_least_one_plt = any(
            isinstance(seg, glymur.codestream.PLTsegment)
            for seg in codestream.segment
        )
        self.assertTrue(at_least_one_plt)
