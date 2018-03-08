"""
Test fixtures common to more than one test point.
"""
import pathlib
import unittest

import numpy as np

import glymur

# Require at least a certain version of openjpeg for running most tests.
if glymur.version.openjpeg_version < '2.2.0':  # pragma: no cover
    OPENJPEG_NOT_AVAILABLE = True
    OPENJPEG_NOT_AVAILABLE_MSG = ('A version of OPENJPEG of at least v2.2.0 '
                                  'must be installed.')
else:
    OPENJPEG_NOT_AVAILABLE = False
    OPENJPEG_NOT_AVAILABLE_MSG = None

# Cannot reopen a named temporary file in windows.
WINDOWS_TMP_FILE_MSG = "cannot use NamedTemporaryFile like this in windows"


class MetadataBase(unittest.TestCase):
    """
    Base class for testing metadata.

    This class has helper routines defined for testing metadata so that it can
    be subclassed and used easily.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def verify_codeblock_style(self, actual, styles):
        """
        Verify the code-block style for SPcod and SPcoc parameters.

        This information is stored in a single byte.  Please reference
        Table A-17 in FCD15444-1
        """
        expected = 0
        masks = [
            0x01,  # Selective arithmetic coding bypass
            0x02,  # Reset context probabilities
            0x04,  # Termination on each coding pass
            0x08,  # Vertically causal context
            0x10,  # Predictable termination
            0x20,  # Segmentation symbols
        ]
        for style, mask in zip(styles, masks):
            if style:
                expected |= mask

        self.assertEqual(actual, expected)

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

    def verifySizSegment(self, actual, expected):
        """
        Verify the fields of the SIZ segment.
        """
        for field in ['rsiz', 'xsiz', 'ysiz', 'xosiz', 'yosiz', 'xtsiz',
                      'ytsiz', 'xtosiz', 'ytosiz', 'bitdepth',
                      'xrsiz', 'yrsiz']:
            self.assertEqual(getattr(actual, field), getattr(expected, field))


# Do we have gdal?
try:
    import gdal  # noqa: F401
except ImportError:  # pragma: no cover
    pass


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
P1_07 = load_test_data('p1_07')
