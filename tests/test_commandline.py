"""
Test command line interface to JPEG2JP2
"""

# standard library imports
import importlib.metadata as im
import unittest
from unittest.mock import patch

# 3rd party library imports

# Local imports
from glymur import JPEG2JP2, command_line
from . import fixtures
from .fixtures import OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
class TestSuite(fixtures.TestCommon):

    @classmethod
    def setUpClass(cls):

        files = im.files('scikit-image')

        jpeg = next(filter(lambda x: 'retina' in x.name, files), None)
        cls.retina = str(jpeg.locate())

    def test_smoke(self):
        """
        SCENARIO:  no special options

        EXPECTED RESULT:  no errors
        """
        new = ['', self.retina, str(self.temp_jp2_filename)]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_verbosity(self):
        """
        SCENARIO:  verbosity is specified on the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', self.retina, str(self.temp_jp2_filename),
            '--verbosity', 'info'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_tilesize(self):
        """
        SCENARIO:  tilesize is specified on the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', self.retina, str(self.temp_jp2_filename),
            '--tilesize', '512', '512'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_psnr(self):
        """
        SCENARIO:  specify psnr via the command line

        EXPECTED RESULT:  data matches
        """
        new = [
            '', self.retina, str(self.temp_jp2_filename),
            '--psnr', '30', '35', '40', '0'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()
