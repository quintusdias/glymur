"""
Module for tests specifically devoted to cinema profile.
"""

# Standard library imports
import unittest
import warnings

# 3rd party library imports
import numpy as np

# Local imports
import glymur
from glymur import Jp2k
from glymur.codestream import SIZsegment
from . import fixtures


class CinemaBase(fixtures.MetadataBase):

    def verify_cinema_cod(self, cod_segment):

        self.assertFalse(cod_segment.scod & 2)  # no sop
        self.assertFalse(cod_segment.scod & 4)  # no eph
        self.assertEqual(cod_segment.prog_order, glymur.core.CPRL)
        self.assertEqual(cod_segment.layers, 1)
        self.assertEqual(cod_segment.mct, 1)
        self.assertEqual(cod_segment.num_res, 5)  # levels
        self.assertEqual(tuple(cod_segment.code_block_size), (32, 32))

    def check_cinema4k_codestream(self, codestream, image_size):

        kwargs = {
            'rsiz': 4,
            'xysiz': image_size,
            'xyosiz': (0, 0),
            'xytsiz': image_size,
            'xytosiz': (0, 0),
            'bitdepth': (12, 12, 12),
            'signed': (False, False, False),
            'xyrsiz': [(1, 1, 1), (1, 1, 1)]
        }
        self.verifySizSegment(codestream.segment[1], SIZsegment(**kwargs))

        self.verify_cinema_cod(codestream.segment[2])

    def check_cinema2k_codestream(self, codestream, image_size):

        kwargs = {
            'rsiz': 3,
            'xysiz': image_size,
            'xyosiz': (0, 0),
            'xytsiz': image_size,
            'xytosiz': (0, 0),
            'bitdepth': (12, 12, 12),
            'signed': (False, False, False),
            'xyrsiz': [(1, 1, 1), (1, 1, 1)]
        }
        self.verifySizSegment(codestream.segment[1], SIZsegment(**kwargs))

        self.verify_cinema_cod(codestream.segment[2])


@unittest.skipIf(
    fixtures.OPENJPEG_NOT_AVAILABLE, fixtures.OPENJPEG_NOT_AVAILABLE_MSG
)
class WriteCinema(CinemaBase):

    @classmethod
    def setUpClass(cls):
        cls.jp2_data = glymur.Jp2k(glymur.data.nemo())[:]

    def test_cinema2K_bad_frame_rate(self):
        """
        SCENARIO:  The cinema2k frame rate is not either 24 or 48.

        EXPECTED RESULT:  ValueError
        """
        with open(self.temp_j2k_filename, mode='wb') as tfile:
            with self.assertRaises(ValueError):
                Jp2k(tfile.name, data=self.jp2_data, cinema2k=36)

    def test_NR_ENC_X_6_2K_24_FULL_CBR_CIRCLE_000_tif_17_encode(self):
        """
        SCENARIO:  create JP2 file with cinema2k profile at 24 fps

        EXPECTED RESULT:  JP2 file has cinema2k profile

        The openjpeg test suite used the following input file for this test,
        input/nonregression/X_6_2K_24_FULL_CBR_CIRCLE_000.tif
        """
        # Need to provide the proper size image
        data = np.concatenate((self.jp2_data, self.jp2_data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:1080, :2048, :]

        with warnings.catch_warnings():
            # Ignore a warning issued by the library.
            warnings.simplefilter('ignore')
            j = Jp2k(self.temp_jp2_filename, data=data, cinema2k=24)

        codestream = j.get_codestream()
        self.check_cinema2k_codestream(codestream, (2048, 1080))

    def test_NR_ENC_X_6_2K_24_FULL_CBR_CIRCLE_000_tif_20_encode(self):
        """
        SCENARIO:  create JP2 file with cinema2k profile at 48 fps

        EXPECTED RESULT:  JP2 file has cinema2k profile

        The openjpeg test suite used the following input file for this test,
        input/nonregression/X_6_2K_24_FULL_CBR_CIRCLE_000.tif
        """
        # Need to provide the proper size image
        data = np.concatenate((self.jp2_data, self.jp2_data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:1080, :2048, :]

        with warnings.catch_warnings():
            # Ignore a warning issued by the library.
            warnings.simplefilter('ignore')
            j = Jp2k(self.temp_j2k_filename, data=data, cinema2k=48)

        codestream = j.get_codestream()
        self.check_cinema2k_codestream(codestream, (2048, 1080))

    def test_NR_ENC_ElephantDream_4K_tif_21_encode(self):
        """
        SCENARIO:  create JP2 file with cinema4k profile

        EXPECTED RESULT:  JP2 file has cinema4k profile

        The openjpeg test suite used the following input file for this test,
        input/nonregression/ElephantDream_4K.tif
        """
        # Need to provide the proper size image
        data = np.concatenate((self.jp2_data, self.jp2_data), axis=0)
        data = np.concatenate((data, data), axis=1).astype(np.uint16)
        data = data[:2160, :4096, :]

        with warnings.catch_warnings():
            # Ignore a warning issued by the library.
            warnings.simplefilter('ignore')
            j = Jp2k(self.temp_j2k_filename, data=data, cinema4k=True)

        codestream = j.get_codestream()
        self.check_cinema4k_codestream(codestream, (4096, 2160))
