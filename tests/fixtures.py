"""
Test fixtures common to more than one test point.
"""
import pathlib
import subprocess
import sys
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

# Cannot reopen a named temporary file in windows.
WINDOWS_TMP_FILE_MSG = "cannot use NamedTemporaryFile like this in windows"


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


NEMO_XMP_BOX = load_test_data('nemo_xmp_box')

SIMPLE_RDF = load_test_data('simple_rdf')

TEXT_GBR_34 = load_test_data('text_gbr_34')
TEXT_GBR_35 = load_test_data('text_gbr_35')

NEMO_DUMP_SHORT = load_test_data('nemo_dump_short')

NEMO_DUMP_NO_CODESTREAM = load_test_data('nemo_dump_no_codestream')

data = load_test_data('nemo_dump_no_codestream_no_xml')
NEMO_DUMP_NO_CODESTREAM_NO_XML = data

NEMO = load_test_data('nemo')
NEMO_DUMP_NO_XML = load_test_data('nemo_dump_no_xml')
TEXT_GBR_RREQ = load_test_data('text_GBR_rreq')
FILE1_XML = load_test_data('file1_xml')
FILE1_XML_BOX = load_test_data('file1_xml_box')

# Progression order is invalid.
ISSUE186_PROGRESSION_ORDER = load_test_data('issue186_progression_order')

GOODSTUFF_CODESTREAM_HEADER = load_test_data('goodstuff_codestream_header')
GOODSTUFF_WITH_FULL_HEADER = load_test_data('goodstuff_with_full_header')

GEOTIFF_UUID = load_test_data('geotiff_uuid')

geotiff_uuid_without_gdal = load_test_data('geotiff_uuid_without_gdal')
multiple_precinct_size = load_test_data('multiple_precinct_size')

decompression_parameters_type = """<class 'glymur.lib.openjp2.DecompressionParametersType'>:
    cp_reduce: 0
    cp_layer: 0
    infile: b''
    outfile: b''
    decod_format: -1
    cod_format: -1
    DA_x0: 0
    DA_x1: 0
    DA_y0: 0
    DA_y1: 0
    m_verbose: 0
    tile_index: 0
    nb_tile_to_decode: 0
    jpwl_correct: 0
    jpwl_exp_comps: 0
    jpwl_max_tiles: 0
    flags: 0"""

default_progression_order_changes_type = """<class 'glymur.lib.openjp2.PocType'>:
    resno0: 0
    compno0: 0
    layno1: 0
    resno1: 0
    compno1: 0
    layno0: 0
    precno0: 0
    precno1: 0
    prg1: 0
    prg: 0
    progorder: b''
    tile: 0
    tx0: 0
    tx1: 0
    ty0: 0
    ty1: 0
    layS: 0
    resS: 0
    compS: 0
    prcS: 0
    layE: 0
    resE: 0
    compE: 0
    prcE: 0
    txS: 0
    txE: 0
    tyS: 0
    tyE: 0
    dx: 0
    dy: 0
    lay_t: 0
    res_t: 0
    comp_t: 0
    prec_t: 0
    tx0_t: 0
    ty0_t: 0"""

default_compression_parameters_type = """<class 'glymur.lib.openjp2.CompressionParametersType'>:
    tile_size_on: 0
    cp_tx0: 0
    cp_ty0: 0
    cp_tdx: 0
    cp_tdy: 0
    cp_disto_alloc: 0
    cp_fixed_alloc: 0
    cp_fixed_quality: 0
    cp_matrice: None
    cp_comment: None
    csty: 0
    prog_order: 0
    numpocs: 0
    numpocs: 0
    tcp_numlayers: 0
    tcp_rates: []
    tcp_distoratio: []
    numresolution: 6
    cblockw_init: 64
    cblockh_init: 64
    mode: 0
    irreversible: 0
    roi_compno: -1
    roi_shift: 0
    res_spec: 0
    prch_init: []
    prcw_init: []
    infile: b''
    outfile: b''
    index_on: 0
    index: b''
    image_offset_x0: 0
    image_offset_y0: 0
    subsampling_dx: 1
    subsampling_dy: 1
    decod_format: -1
    cod_format: -1
    jpwl_epc_on: 0
    jpwl_hprot_mh: 0
    jpwl_hprot_tph_tileno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_hprot_tph: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_pprot_tileno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_pprot_packno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_pprot: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_sens_size: 0
    jpwl_sens_addr: 0
    jpwl_sens_range: 0
    jpwl_sens_mh: 0
    jpwl_sens_tph_tileno: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    jpwl_sens_tph: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cp_cinema: 0
    max_comp_size: 0
    cp_rsiz: 0
    tp_on: 0
    tp_flag: 0
    tcp_mct: 0
    jpip_on: 0
    mct_data: None
    max_cs_size: 0
    rsiz: 0"""

default_image_component_parameters = """<class 'glymur.lib.openjp2.ImageComptParmType'>:
    dx: 0
    dy: 0
    w: 0
    h: 0
    x0: 0
    y0: 0
    prec: 0
    bpp: 0
    sgnd: 0"""

# The "icc_profile_buf" field is problematic as it is a pointer value, i.e.
#
#     icc_profile_buf: <glymur.lib.openjp2.LP_c_ubyte object at 0x7f28cd5d5d90>
#
# Have to treat it as a regular expression.
default_image_type = """<class 'glymur.lib.openjp2.ImageType'>:
    x0: 0
    y0: 0
    x1: 0
    y1: 0
    numcomps: 0
    color_space: 0
    icc_profile_buf: <glymur.lib.openjp2.LP_c_ubyte object at 0x[0-9A-Fa-f]*>
    icc_profile_len: 0"""
