"""
Test fixtures common to more than one test point.
"""
import os
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import glymur

# If openjpeg is not installed, many tests cannot be run.
if glymur.version.openjpeg_version < '2.1.0':
    OPENJPEG_NOT_AVAILABLE = True
    OPENJPEG_NOT_AVAILABLE_MSG = 'OpenJPEG library not installed'
else:
    OPENJPEG_NOT_AVAILABLE = False
    OPENJPEG_NOT_AVAILABLE_MSG = None


def low_memory_linux_machine():
    """
    Detect if the current machine is low-memory (< 2.5GB)

    This is primarily aimed at Digital Ocean VMs running linux.  Don't bother
    on mac or windows.

    Returns
    -------
    bool
        True if <2GB, False otherwise
    """
    if not sys.platform.startswith('linux'):
        return False
    cmd1 = "free -m"
    cmd2 = "tail -n +2"
    cmd3 = "awk '{sum += $2} END {print sum}'"
    p1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(cmd2, shell=True,
                          stdin=p1.stdout, stdout=subprocess.PIPE)
    p3 = subprocess.Popen(cmd3, shell=True,
                          stdin=p2.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    p2.stdout.close()
    stdout, stderr = p3.communicate()
    nbytes = int(stdout.decode('utf-8').strip())
    return nbytes < 2000


class TestCommon(unittest.TestCase):
    """
    Common setup for many if not all tests.
    """
    def setUp(self):
        # Supply paths to these two shipping example files.
        self.jp2file = glymur.data.nemo()
        self.j2kfile = glymur.data.goodstuff()
        self.jpxfile = glymur.data.jpxfile()

        # Create a temporary directory to be cleaned up following each test, as
        # well as names for a JP2 and a J2K file.
        self.test_dir = tempfile.mkdtemp()
        self.temp_jp2_filename = os.path.join(self.test_dir, 'test.jp2')
        self.temp_j2k_filename = os.path.join(self.test_dir, 'test.j2k')
        self.temp_jpx_filename = os.path.join(self.test_dir, 'test.jpx')

    def tearDown(self):
        shutil.rmtree(self.test_dir)


class MetadataBase(TestCommon):
    """
    Base class for testing metadata.

    This class has helper routines defined for testing metadata so that it can
    be subclassed and used easily.
    """

    def verify_codeblock_style(self, actual, style):
        """
        Verify the code-block style for SPcod and SPcoc parameters.

        This information is stored in a single byte.  Please reference
        Table A-17 in FCD15444-1
        """
        expected = 0
        if style[0]:
            # Selective arithmetic coding bypass
            expected |= 0x01
        if style[1]:
            # Reset context probabilities
            expected |= 0x02
        if style[2]:
            # Termination on each coding pass
            expected |= 0x04
        if style[3]:
            # Vertically causal context
            expected |= 0x08
        if style[4]:
            # Predictable termination
            expected |= 0x10
        if style[5]:
            # Segmentation symbols
            expected |= 0x20
        self.assertEqual(actual, expected)

    def verifySignatureBox(self, box):
        """
        The signature box is a constant.
        """
        self.assertEqual(box.signature, (13, 10, 135, 10))

    def verify_filetype_box(self, actual, expected):
        """
        All JP2 files should have a brand reading 'jp2 ' and just a single
        entry in the compatibility list, also 'jp2 '.  JPX files can have more
        compatibility items.
        """
        self.assertEqual(actual.brand, expected.brand)
        self.assertEqual(actual.minor_version, expected.minor_version)
        self.assertEqual(actual.minor_version, 0)
        for cl in expected.compatibility_list:
            self.assertIn(cl, actual.compatibility_list)

    def verifyRGNsegment(self, actual, expected):
        """
        verify the fields of a RGN segment
        """
        self.assertEqual(actual.crgn, expected.crgn)  # 0 = component
        self.assertEqual(actual.srgn, expected.srgn)  # 0 = implicit
        self.assertEqual(actual.sprgn, expected.sprgn)

    def verifySOTsegment(self, actual, expected):
        """
        verify the fields of a SOT (start of tile) segment
        """
        self.assertEqual(actual.isot, expected.isot)
        self.assertEqual(actual.psot, expected.psot)
        self.assertEqual(actual.tpsot, expected.tpsot)
        self.assertEqual(actual.tnsot, expected.tnsot)

    def verifyCMEsegment(self, actual, expected):
        """
        verify the fields of a CME (comment) segment
        """
        self.assertEqual(actual.rcme, expected.rcme)
        self.assertEqual(actual.ccme, expected.ccme)

    def verifySizSegment(self, actual, expected):
        """
        Verify the fields of the SIZ segment.
        """
        for field in ['rsiz', 'xsiz', 'ysiz', 'xosiz', 'yosiz', 'xtsiz',
                      'ytsiz', 'xtosiz', 'ytosiz', 'bitdepth',
                      'xrsiz', 'yrsiz']:
            self.assertEqual(getattr(actual, field), getattr(expected, field))

    def verifyImageHeaderBox(self, box1, box2):
        self.assertEqual(box1.height, box2.height)
        self.assertEqual(box1.width, box2.width)
        self.assertEqual(box1.num_components, box2.num_components)
        self.assertEqual(box1.bits_per_component, box2.bits_per_component)
        self.assertEqual(box1.signed, box2.signed)
        self.assertEqual(box1.compression, box2.compression)
        self.assertEqual(box1.colorspace_unknown, box2.colorspace_unknown)
        self.assertEqual(box1.ip_provided, box2.ip_provided)

    def verifyColourSpecificationBox(self, actual, expected):
        """
        Does not currently check icc profiles.
        """
        self.assertEqual(actual.method, expected.method)
        self.assertEqual(actual.precedence, expected.precedence)
        self.assertEqual(actual.approximation, expected.approximation)

        if expected.colorspace is None:
            self.assertIsNone(actual.colorspace)
            self.assertIsNotNone(actual.icc_profile)
        else:
            self.assertEqual(actual.colorspace, expected.colorspace)
            self.assertIsNone(actual.icc_profile)


# Do we have gdal?
try:
    import gdal  # noqa: F401
    HAVE_GDAL = True
except ImportError:
    HAVE_GDAL = False


def mse(amat, bmat):
    """Mean Square Error"""
    diff = amat.astype(np.double) - bmat.astype(np.double)
    err = np.mean(diff**2)
    return err


def load_test_data(name):
    basename = name + '.txt'
    path = pathlib.Path(__file__).parent / 'data' / basename

    # Have to use str for python < 3.6
    with open(str(path), mode='rt') as f:
        return f.read().rstrip('\n')


DECOMPRESSION_PARAMETERS_TYPE = load_test_data('decompression_parameters_type')

id = 'default_compression_parameters_type'
DEFAULT_COMPRESSION_PARAMETERS_TYPE = load_test_data(id)

id = 'default_progression_order_changes_type'
DEFAULT_PROGRESSION_ORDER_CHANGES_TYPE = load_test_data(id)

FILE1_XML = load_test_data('file1_xml')
FILE1_XML_BOX = load_test_data('file1_xml_box')
GEOTIFF_UUID = load_test_data('geotiff_uuid')
GEOTIFF_UUID_WITHOUT_GDAL = load_test_data('geotiff_uuid_without_gdal')
GOODSTUFF_CODESTREAM_HEADER = load_test_data('goodstuff_codestream_header')
GOODSTUFF_WITH_FULL_HEADER = load_test_data('goodstuff_with_full_header')
ISSUE186_PROGRESSION_ORDER = load_test_data('issue186_progression_order')
MULTIPLE_PRECINCT_SIZE = load_test_data('multiple_precinct_size')
NEMO = load_test_data('nemo')
NEMO_DUMP_NO_XML = load_test_data('nemo_dump_no_xml')
NEMO_DUMP_NO_CODESTREAM = load_test_data('nemo_dump_no_codestream')

data = load_test_data('nemo_dump_no_codestream_no_xml')
NEMO_DUMP_NO_CODESTREAM_NO_XML = data

NEMO_DUMP_SHORT = load_test_data('nemo_dump_short')
NEMO_XMP_BOX = load_test_data('nemo_xmp_box')
SIMPLE_RDF = load_test_data('simple_rdf')
TEXT_GBR_34 = load_test_data('text_gbr_34')
TEXT_GBR_35 = load_test_data('text_gbr_35')
TEXT_GBR_RREQ = load_test_data('text_GBR_rreq')
