"""
Test fixtures common to more than one test point.
"""

# Standard library imports
import importlib.resources as ir
import pathlib
import shutil
import sys
import tempfile
import unittest

# are we anaconda?
if 'conda' in sys.version.lower():
    ANACONDA = True
else:
    ANACONDA = False

# are we macports?
if sys.executable.startswith('/opt/local/Library/Frameworks/Python.framework'):
    MACPORTS = True
else:
    MACPORTS = False

# 3rd party library imports
try:
    from osgeo import gdal  # noqa : F401

    # ok, but is the proper jpeg2000 built in?
    if gdal.GetDriverByName("JP2OpenJPEG") is None:
        # macports default port?
        raise ImportError("GDAL not built to handle OpenJPEG")

    # otherwise, hunky dory
    _HAVE_GDAL = True

except (ImportError, ModuleNotFoundError):
    _HAVE_GDAL = False

# Local imports
import glymur

# Require at least a certain version of openjpeg for running most tests.
if glymur.version.openjpeg_version < "2.2.0":  # pragma: no cover
    OPENJPEG_NOT_AVAILABLE = True
    OPENJPEG_NOT_AVAILABLE_MSG = (
        "A version of OPENJPEG of at least v2.2.0 must be installed."
    )
else:
    OPENJPEG_NOT_AVAILABLE = False
    OPENJPEG_NOT_AVAILABLE_MSG = None

if glymur.version.tiff_version < "4.0.0":
    TIFF_NOT_AVAILABLE = True
    TIFF_NOT_AVAILABLE_MSG = "A version of TIFF of at least v4.0.0 must be installed."  # noqa : E501
else:
    TIFF_NOT_AVAILABLE = False
    TIFF_NOT_AVAILABLE_MSG = None


class TestCommon(unittest.TestCase):
    """
    Common setup for many if not all tests.
    """

    def setUp(self):
        # Supply paths to these three shipping example files.
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()
        self.jpxfile = glymur.data.jpxfile()

        # Create a temporary directory to be cleaned up following each test, as
        # well as names for a JP2 and a J2K file.
        self.test_dir = tempfile.mkdtemp()
        self.test_dir_path = pathlib.Path(self.test_dir)
        self.temp_jp2_filename = self.test_dir_path / "test.jp2"
        self.temp_j2k_filename = self.test_dir_path / "test.j2k"
        self.temp_jpx_filename = self.test_dir_path / "test.jpx"
        self.temp_tiff_filename = self.test_dir_path / "test.tif"

    def tearDown(self):
        shutil.rmtree(self.test_dir)


class TestJPEGCommon(TestCommon):

    @classmethod
    def setUpClass(cls):
        """
        Use some files supplied by scikit-image for our tests.
        """

        cls.retina = ir.files('skimage') / 'data' / 'retina.jpg'
        cls.hubble = ir.files('skimage') / 'data' / 'hubble_deep_field.jpg'
        cls.rocket = ir.files('skimage') / 'data' / 'rocket.jpg'

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
