"""
Test command line interface to JPEG2JP2
"""

# standard library imports
import unittest
from unittest.mock import patch

# 3rd party library imports

# Local imports
from glymur import JPEG2JP2, command_line, reset_option
from . import fixtures
from .fixtures import (
    OPENJPEG_NOT_AVAILABLE,
    OPENJPEG_NOT_AVAILABLE_MSG,
    CANNOT_USE_IMPORTLIB_METADATA
)


@unittest.skipIf(OPENJPEG_NOT_AVAILABLE, OPENJPEG_NOT_AVAILABLE_MSG)
@unittest.skipIf(
    CANNOT_USE_IMPORTLIB_METADATA,
    'missing importlib.metadata.files ?'
)
class TestSuite(fixtures.TestJPEGCommon):

    def test_smoke(self):
        """
        SCENARIO:  no special options

        EXPECTED RESULT:  no errors
        """
        new = ['', str(self.retina), str(self.temp_jp2_filename)]
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
            '', str(self.retina), str(self.temp_jp2_filename),
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
            '', str(self.retina), str(self.temp_jp2_filename),
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

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--psnr', '30', '35', '40', '0'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_irreversible(self):
        """
        SCENARIO:  specify the irreversible transform via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--irreversible'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_plt(self):
        """
        SCENARIO:  specify the PLT markers via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--plt'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_eph(self):
        """
        SCENARIO:  specify the EPH markers via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--eph'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_sop(self):
        """
        SCENARIO:  specify the SOP markers via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--sop'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_progression_order(self):
        """
        SCENARIO:  specify the procession order via the command line

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--prog', 'rlcp'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_number_of_resolutions(self):
        """
        SCENARIO:  specify resolution

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--numres', '6'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_num_threads(self):
        """
        SCENARIO:  specify number of threads to use

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--num-threads', '4'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

        reset_option('all')

    def test_icc_profile(self):
        """
        SCENARIO:  specify to include an ICC profile

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.rocket), str(self.temp_jp2_filename),
            '--include-icc-profile',
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_layers(self):
        """
        SCENARIO:  specify compression ratios

        EXPECTED RESULT:  no errors
        """
        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--cratio', '200', '50', '10'
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()

    def test_resolution_boxes(self):
        """
        SCENARIO:  specify capture and display resolution

        EXPECTED RESULT:  no errors
        """
        vresc, hresc = 0.1, 0.2
        vresd, hresd = 0.3, 0.4

        new = [
            '', str(self.retina), str(self.temp_jp2_filename),
            '--capture-resolution', str(vresc), str(hresc),
            '--display-resolution', str(vresd), str(hresd),
        ]
        with (
            patch('sys.argv', new=new),
            patch.object(JPEG2JP2, 'run', new=lambda x: None)
        ):
            command_line.jpeg2jp2()
