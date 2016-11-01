"""
Tests for writing cinema 2k/4k files
"""
# Standard library imports ...
import os
import re
import sys
import tempfile
import unittest
import warnings
if sys.hexversion >= 0x03030000:
    from unittest.mock import patch
else:
    from mock import patch

# Third party library imports ...
import numpy as np

# Local imports
import glymur
from glymur import Jp2k
from glymur.codestream import SIZsegment

from .fixtures import WINDOWS_TMP_FILE_MSG
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG

from . import fixtures


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(os.name == "nt", WINDOWS_TMP_FILE_MSG)
@unittest.skipIf(re.match(r'''(1|2.0.0)''',
                          glymur.version.openjpeg_version) is not None,
                 "Uses features not supported until 2.0.1")
class TestSuite(fixtures.MetadataBase):

    @classmethod
    def setUpClass(cls):
        cls.jp2file = glymur.data.nemo()
        cls.jp2_data = glymur.Jp2k(cls.jp2file)[:]

    def verify_cinema_cod(self, cod_segment):

        self.assertFalse(cod_segment.scod & 2)  # no sop
        self.assertFalse(cod_segment.scod & 4)  # no eph
        self.assertEqual(cod_segment.prog_order, glymur.core.CPRL)
        self.assertEqual(cod_segment.layers, 1)
        self.assertEqual(cod_segment.mct, 1)
        self.assertEqual(cod_segment.num_res, 5)  # levels
        self.assertEqual(tuple(cod_segment.code_block_size), (32, 32))

    def check_cinema4k_codestream(self, codestream, image_size):

        kwargs = {'rsiz': 4, 'xysiz': image_size, 'xyosiz': (0, 0),
                  'xytsiz': image_size, 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12), 'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(codestream.segment[1], SIZsegment(**kwargs))

        self.verify_cinema_cod(codestream.segment[2])

    def check_cinema2k_codestream(self, codestream, image_size):

        kwargs = {'rsiz': 3, 'xysiz': image_size, 'xyosiz': (0, 0),
                  'xytsiz': image_size, 'xytosiz': (0, 0),
                  'bitdepth': (12, 12, 12), 'signed': (False, False, False),
                  'xyrsiz': [(1, 1, 1), (1, 1, 1)]}
        self.verifySizSegment(codestream.segment[1], SIZsegment(**kwargs))

        self.verify_cinema_cod(codestream.segment[2])

    def test_NR_ENC_X_6_2K_24_FULL_CBR_CIRCLE_000_tif_17_encode(self):
        """
        Original test file was

            input/nonregression/X_6_2K_24_FULL_CBR_CIRCLE_000.tif

        """
        # Need to provide the proper size image
        data = np.concatenate((self.jp2_data, self.jp2_data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:1080, :2048, :]

        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with warnings.catch_warnings():
                # Ignore a warning issued by the library.
                warnings.simplefilter('ignore')
                j = Jp2k(tfile.name, data=data, cinema2k=24)

            codestream = j.get_codestream()
            self.check_cinema2k_codestream(codestream, (2048, 1080))

    def test_NR_ENC_X_6_2K_24_FULL_CBR_CIRCLE_000_tif_20_encode(self):
        """
        Original test file was

            input/nonregression/X_6_2K_24_FULL_CBR_CIRCLE_000.tif

        """
        # Need to provide the proper size image
        data = np.concatenate((self.jp2_data, self.jp2_data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:1080, :2048, :]

        with warnings.catch_warnings():
            # Ignore a warning issued by the library.
            warnings.simplefilter('ignore')
            with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
                j = Jp2k(tfile.name, data=data, cinema2k=48)
                codestream = j.get_codestream()
                self.check_cinema2k_codestream(codestream, (2048, 1080))

    def test_NR_ENC_ElephantDream_4K_tif_21_encode(self):
        """
        Verify basic cinema4k write

        Original test file is input/nonregression/ElephantDream_4K.tif
        """
        # Need to provide the proper size image
        data = np.concatenate((self.jp2_data, self.jp2_data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:2160, :4096, :]

        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with warnings.catch_warnings():
                # Ignore a warning issued by the library.
                warnings.simplefilter('ignore')
                j = Jp2k(tfile.name, data=data, cinema4k=True)

            codestream = j.get_codestream()
            self.check_cinema4k_codestream(codestream, (4096, 2160))

    def test_cinema_mode_with_too_old_version_of_openjpeg(self):
        """
        Cinema mode not allowed for anything less than 2.0.1

        Origin file tested was

            input/nonregression/X_4_2K_24_185_CBR_WB_000.tif

        """
        data = np.zeros((857, 2048, 3), dtype=np.uint8)
        versions = ["1.5.0", "2.0.0"]
        for version in versions:
            with patch('glymur.version.openjpeg_version', new=version):
                with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
                    with self.assertRaises(IOError):
                        Jp2k(tfile.name, data=data, cinema2k=48)

    def test_cinema2K_with_others(self):
        """
        Can't specify cinema2k with any other options.

        Original test file was
        input/nonregression/X_5_2K_24_235_CBR_STEM24_000.tif
        """
        data = np.zeros((857, 2048, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cinema2k=48, cratios=[200, 100, 50])

    def test_cinema4K_with_others(self):
        """
        Can't specify cinema4k with any other options.

        Original test file was input/nonregression/ElephantDream_4K.tif
        """
        data = np.zeros((4096, 2160, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.j2k') as tfile:
            with self.assertRaises(IOError):
                Jp2k(tfile.name, data=data,
                     cinema4k=True, cratios=[200, 100, 50])
