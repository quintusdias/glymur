# Standard library imports ...
import importlib.resources as ir
import os
import sys
import time
import unittest
from unittest.mock import patch
import warnings

# Third party library imports ...
import numpy as np

# Local imports
import glymur
from glymur import Jp2k
from glymur import command_line

from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG

from . import fixtures


@unittest.skipIf(os.cpu_count() < 2, "makes no sense if 2 cores not there")
@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(glymur.version.openjpeg_version < '2.4.0',
                 "Requires as least v2.4.0")
class TestSuite(fixtures.TestCommon):
    """Test behavior when multiple threads are possible."""

    def setUp(self):
        super().setUp()
        glymur.reset_option('all')

    def tearDown(self):
        glymur.reset_option('all')

    @unittest.skipIf(os.cpu_count() < 4, "makes no sense if 4 cores not there")
    def test_thread_support_for_reading(self):
        """
        SCENARIO:  Set a non-default thread support value.

        EXPECTED RESULTS:  Using more threads speeds up a full read.
        """
        jp2 = Jp2k(self.jp2file)
        t0 = time.time()
        jp2[:]
        t1 = time.time()
        delta0 = t1 - t0

        glymur.set_option('lib.num_threads', 4)
        t0 = time.time()
        jp2[:]
        t1 = time.time()
        delta1 = t1 - t0

        self.assertTrue(delta1 < delta0)

    def test_thread_support_on_openjpeg_lt_220(self):
        """
        SCENARIO:  Set number of threads on openjpeg < 2.2.0

        EXPECTED RESULTS:  RuntimeError
        """
        with patch('glymur.jp2k.version.openjpeg_version', new='2.1.0'):
            with self.assertRaises(RuntimeError):
                glymur.set_option('lib.num_threads', 2)

    @patch('glymur.lib.openjp2.has_thread_support')
    def test_thread_support_not_compiled_into_library(self, mock_ts):
        """
        SCENARIO:  Set number of threads on openjpeg >= 2.2.0, but openjpeg
        has not been compiled with thread support.

        EXPECTED RESULTS:  RuntimeError
        """
        mock_ts.return_value = False
        with patch('glymur.jp2k.version.openjpeg_version', new='2.2.0'):
            with self.assertRaises(RuntimeError):
                glymur.set_option('lib.num_threads', 2)

    @unittest.skipIf(
        glymur.version.openjpeg_version < '2.4.0', "Requires as least v2.4.0"
    )
    def test_threads_write_support__ge_2p4(self):
        """
        SCENARIO:  Attempt to encode with threading support.  This feature is
        new as of openjpeg library version 2.4.0.

        EXPECTED RESULT:  No errors.
        issued.
        """
        glymur.set_option('lib.num_threads', 2)

        with warnings.catch_warnings(record=True) as w:
            Jp2k(
                self.temp_jp2_filename,
                data=np.zeros((128, 128), dtype=np.uint8)
            )
            if glymur.version.openjpeg_version >= '2.4.0':
                self.assertEqual(len(w), 0)
            else:
                self.assertEqual(len(w), 1)

    def test_threads_write_support__eq_2p3(self):
        """
        SCENARIO:  Attempt to encode with threading support.  This feature is
        new as of openjpeg library version 2.4.0.

        EXPECTED RESULT:  In library versions prior to 2.4.0, a warning is
        issued.
        """
        with patch('glymur.jp2k.version.openjpeg_version', new='2.3.0'):
            glymur.set_option('lib.num_threads', 2)

            with warnings.catch_warnings(record=True) as w:
                Jp2k(
                    self.temp_jp2_filename,
                    data=np.zeros((128, 128), dtype=np.uint8)
                )
                self.assertEqual(len(w), 1)

    def test_tiff2jp2_num_threads(self):
        """
        Scenario:  The --num-threads option is given on the command line.

        Expected Result.  No errors.  If openjpeg is earlier than 2.5.0, there
        will be a warning.
        """
        path = ir.files('tests.data').joinpath('basn6a08.tif')

        sys.argv = [
            '', str(path), str(self.temp_jp2_filename), '--num-threads', '2',
        ]
        with warnings.catch_warnings(record=True) as w:
            command_line.tiff2jp2()
            if glymur.version.openjpeg_version < '2.4.0':
                self.assertEqual(len(w), 1)

        Jp2k(self.temp_jp2_filename)

        self.assertTrue(True)

    def test_openjpeg_library_too_old_for_threaded_tile_writing(self):
        """
        SCENARIO:  Try to create a jp2 file via writing tiles, but while the
        openjpeg library is not too old for writing, it's too old for threaded
        writing.  In other words, it's version 2.3.0

        EXPECTED RESULT:  There is a warning, but the image is created.
        """
        expected = fixtures.skimage.data.astronaut()

        shape = (
            expected.shape[0] * 2, expected.shape[1] * 2, expected.shape[2]
        )
        tilesize = (expected.shape[0], expected.shape[1])

        glymur.set_option('lib.num_threads', 2)
        j = Jp2k(
            self.temp_j2k_filename, shape=shape, tilesize=tilesize,
        )

        with patch('glymur.version.openjpeg_version', new='2.3.0'):
            with self.assertWarns(UserWarning):
                for tw in j.get_tilewriters():
                    tw[:] = expected

        expected = np.concatenate((expected, expected), axis=0)
        expected = np.concatenate((expected, expected), axis=1)
        actual = j[:]

        np.testing.assert_array_equal(actual, expected)
