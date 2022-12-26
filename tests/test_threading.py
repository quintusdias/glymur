# Standard library imports ...
import os
import time
import unittest
from unittest.mock import patch
import warnings

# Third party library imports ...
import numpy as np

# Local imports
import glymur
from glymur import Jp2k

from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG

from . import fixtures


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(glymur.version.openjpeg_version < '2.3.0',
                 "Requires as least v2.3.0")
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

    @unittest.skipIf(os.cpu_count() < 4, "makes no sense if 4 cores not there")
    def test_thread_support_on_openjpeg_lt_220(self):
        """
        SCENARIO:  Set number of threads on openjpeg < 2.2.0

        EXPECTED RESULTS:  RuntimeError
        """
        with patch('glymur.jp2k.version.openjpeg_version', new='2.1.0'):
            with self.assertRaises(RuntimeError):
                glymur.set_option('lib.num_threads', 4)

    @unittest.skipIf(os.cpu_count() < 4, "makes no sense if 4 cores not there")
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
                glymur.set_option('lib.num_threads', 4)

    @unittest.skipIf(os.cpu_count() < 2, "makes no sense if 2 cores not there")
    def test_threads_write_support(self):
        """
        SCENARIO:  Attempt to encode with threading support.  This feature is
        new as of openjpeg library version 2.4.0.

        EXPECTED RESULT:  In library versions prior to 2.4.0, a warning is
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
