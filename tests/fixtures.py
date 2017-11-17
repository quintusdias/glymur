"""
Test fixtures common to more than one test point.
"""
import pathlib
import subprocess
import sys
import textwrap
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


nemo_xmp = load_test_data('nemo_xmp')

nemo_xmp_box = """UUID Box (uuid) @ (77, 3146)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
    UUID Data:
{0}""".format(textwrap.indent(nemo_xmp, '    '))

SimpleRDF = """<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
  <rdf:Description rdf:about='Test:XMPCoreCoverage/kSimpleRDF'
                   xmlns:ns1='ns:test1/' xmlns:ns2='ns:test2/'>

    <ns1:SimpleProp>Simple value</ns1:SimpleProp>

    <ns1:Distros>
      <rdf:Bag>
        <rdf:li>Suse</rdf:li>
        <rdf:li>Fedora</rdf:li>
      </rdf:Bag>
    </ns1:Distros>

  </rdf:Description>
</rdf:RDF>"""

text_gbr_34 = load_test_data('text_gbr_34')
text_gbr_35 = load_test_data('text_gbr_35')

codestream_header = '''SOC marker segment @ (3231, 0)
SIZ marker segment @ (3233, 47)
    Profile:  no profile
    Reference Grid Height, Width:  (1456 x 2592)
    Vertical, Horizontal Reference Grid Offset:  (0 x 0)
    Reference Tile Height, Width:  (1456 x 2592)
    Vertical, Horizontal Reference Tile Offset:  (0 x 0)
    Bitdepth:  (8, 8, 8)
    Signed:  (False, False, False)
    Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))
COD marker segment @ (3282, 12)
    Coding style:
        Entropy coder, without partitions
        SOP marker segments:  False
        EPH marker segments:  False
    Coding style parameters:
        Progression order:  LRCP
        Number of layers:  2
        Multiple component transformation usage:  reversible
        Number of resolutions:  2
        Code block height, width:  (64 x 64)
        Wavelet transform:  5-3 reversible
        Precinct size:  (32768, 32768)
        Code block context:
            Selective arithmetic coding bypass:  False
            Reset context probabilities on coding pass boundaries:  False
            Termination on each coding pass:  False
            Vertically stripe causal context:  False
            Predictable termination:  False
            Segmentation symbols:  False
QCD marker segment @ (3296, 7)
    Quantization style:  no quantization, 2 guard bits
    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
CME marker segment @ (3305, 37)
    "Created by OpenJPEG version 2.0.0"'''

codestream_trailer = """SOT marker segment @ (3344, 10)
    Tile part index:  0
    Tile part length:  1132173
    Tile part instance:  0
    Number of tile parts:  1
COC marker segment @ (3356, 9)
    Associated component:  1
    Coding style for this component:  Entropy coder, PARTITION = 0
    Coding style parameters:
        Number of resolutions:  2
        Code block height, width:  (64 x 64)
        Wavelet transform:  5-3 reversible
        Precinct size:  (32768, 32768)
        Code block context:
            Selective arithmetic coding bypass:  False
            Reset context probabilities on coding pass boundaries:  False
            Termination on each coding pass:  False
            Vertically stripe causal context:  False
            Predictable termination:  False
            Segmentation symbols:  False
QCC marker segment @ (3367, 8)
    Associated Component:  1
    Quantization style:  no quantization, 2 guard bits
    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
COC marker segment @ (3377, 9)
    Associated component:  2
    Coding style for this component:  Entropy coder, PARTITION = 0
    Coding style parameters:
        Number of resolutions:  2
        Code block height, width:  (64 x 64)
        Wavelet transform:  5-3 reversible
        Precinct size:  (32768, 32768)
        Code block context:
            Selective arithmetic coding bypass:  False
            Reset context probabilities on coding pass boundaries:  False
            Termination on each coding pass:  False
            Vertically stripe causal context:  False
            Predictable termination:  False
            Segmentation symbols:  False
QCC marker segment @ (3388, 8)
    Associated Component:  2
    Quantization style:  no quantization, 2 guard bits
    Step size:  [(0, 8), (0, 9), (0, 9), (0, 10)]
SOD marker segment @ (3398, 0)
EOC marker segment @ (1135517, 0)"""

codestream = '\n'.join([codestream_header, codestream_trailer])

nemo_with_codestream_header = load_test_data('nemo_with_codestream_header')

nemo_dump_short = r"""JPEG 2000 Signature Box (jP  ) @ (0, 12)
File Type Box (ftyp) @ (12, 20)
JP2 Header Box (jp2h) @ (32, 45)
    Image Header Box (ihdr) @ (40, 22)
    Colour Specification Box (colr) @ (62, 15)
UUID Box (uuid) @ (77, 3146)
Contiguous Codestream Box (jp2c) @ (3223, 1132296)"""

nemo_dump_no_codestream = load_test_data('nemo_dump_no_codestream')

nemo_dump_no_codestream_no_xml = r"""JPEG 2000 Signature Box (jP  ) @ (0, 12)
    Signature:  0d0a870a
File Type Box (ftyp) @ (12, 20)
    Brand:  jp2
    Compatibility:  ['jp2 ']
JP2 Header Box (jp2h) @ (32, 45)
    Image Header Box (ihdr) @ (40, 22)
        Size:  [1456 2592 3]
        Bitdepth:  8
        Signed:  False
        Compression:  wavelet
        Colorspace Unknown:  False
    Colour Specification Box (colr) @ (62, 15)
        Method:  enumerated colorspace
        Precedence:  0
        Colorspace:  sRGB
UUID Box (uuid) @ (77, 3146)
    UUID:  be7acfcb-97a9-42e8-9c71-999491e3afac (XMP)
Contiguous Codestream Box (jp2c) @ (3223, 1132296)"""

nemo = load_test_data('nemo')
nemo_dump_no_xml = load_test_data('nemo_dump_no_xml')
text_GBR_rreq = load_test_data('text_GBR_rreq')
file1_xml = load_test_data('file1_xml')
file1_xml_box = load_test_data('file1_xml_box')

issue_182_cmap = """Component Mapping Box (cmap) @ (130, 24)
    Component 0 ==> palette column 0
    Component 0 ==> palette column 1
    Component 0 ==> palette column 2
    Component 0 ==> palette column 3"""

issue_183_colr = """Colour Specification Box (colr) @ (62, 12)
    Method:  restricted ICC profile
    Precedence:  0
    ICC Profile:  None"""

# Progression order is invalid.
issue_186_progression_order = """COD marker segment @ (174, 12)
    Coding style:
        Entropy coder, without partitions
        SOP marker segments:  False
        EPH marker segments:  False
    Coding style parameters:
        Progression order:  33 (invalid)
        Number of layers:  1
        Multiple component transformation usage:  reversible
        Number of resolutions:  6
        Code block height, width:  (32 x 32)
        Wavelet transform:  9-7 irreversible
        Precinct size:  (32768, 32768)
        Code block context:
            Selective arithmetic coding bypass:  False
            Reset context probabilities on coding pass boundaries:  False
            Termination on each coding pass:  False
            Vertically stripe causal context:  False
            Predictable termination:  False
            Segmentation symbols:  False"""

# Cinema 2K profile
cinema2k_profile = """SIZ marker segment @ (2, 47)
    Profile:  Cinema 2K
    Reference Grid Height, Width:  (1080 x 1920)
    Vertical, Horizontal Reference Grid Offset:  (0 x 0)
    Reference Tile Height, Width:  (1080 x 1920)
    Vertical, Horizontal Reference Tile Offset:  (0 x 0)
    Bitdepth:  (12, 12, 12)
    Signed:  (False, False, False)
    Vertical, Horizontal Subsampling:  ((1, 1), (1, 1), (1, 1))"""

jplh_color_group_box = r"""Compositing Layer Header Box (jplh) @ (314227, 31)
    Colour Group Box (cgrp) @ (314235, 23)
        Colour Specification Box (colr) @ (314243, 15)
            Method:  enumerated colorspace
            Precedence:  0
            Colorspace:  sRGB"""

goodstuff_codestream_header = load_test_data('goodstuff_codestream_header')
goodstuff_with_full_header = load_test_data('goodstuff_with_full_header')

# manually verified via gdalinfo
geotiff_uuid = """UUID Box (uuid) @ (149, 523)
    UUID:  b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03 (GeoTIFF)
    UUID Data:  Coordinate System =
        PROJCS["Equirectangular MARS",
            GEOGCS["GCS_MARS",
                DATUM["unknown",
                    SPHEROID["unnamed",3396190,0]],
                PRIMEM["Greenwich",0],
                UNIT["degree",0.0174532925199433]],
            PROJECTION["Equirectangular"],
            PARAMETER["latitude_of_origin",0],
            PARAMETER["central_meridian",180],
            PARAMETER["standard_parallel_1",0],
            PARAMETER["false_easting",0],
            PARAMETER["false_northing",0],
            UNIT["metre",1,
                AUTHORITY["EPSG","9001"]]]
    Origin = (-2523306.125000000000000,-268608.875000000000000)
    Pixel Size = (0.250000000000000,-0.250000000000000)
    Corner Coordinates:
    Upper Left  (-2523306.125, -268608.875) (137d25'49.08"E,  4d31'53.74"S
    Lower Left  (-2523306.125, -268609.125) (137d25'49.08"E,  4d31'53.75"S
    Upper Right (-2523305.875, -268608.875) (137d25'49.09"E,  4d31'53.74"S
    Lower Right (-2523305.875, -268609.125) (137d25'49.09"E,  4d31'53.75"S
    Center      (-2523306.000, -268609.000) (137d25'49.09"E,  4d31'53.75"S"""

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
