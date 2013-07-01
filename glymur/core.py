# Progression order
LRCP = 0
RLCP = 1
RPCL = 2
PCRL = 3
CPRL = 4

_progression_order_display = {
    LRCP: 'LRCP',
    RLCP: 'RLCP',
    RPCL: 'RPCL',
    PCRL: 'PCRL',
    CPRL: 'CPRL'}

progression_order = {
    'LRCP': LRCP,
    'RLCP': RLCP,
    'RPCL': RPCL,
    'PCRL': PCRL,
    'CPRL': CPRL}

WAVELET_TRANSFORM_9x7_IRREVERSIBLE = 0
WAVELET_TRANSFORM_5x3_REVERSIBLE = 1

_wavelet_transform_display = {
    WAVELET_TRANSFORM_9x7_IRREVERSIBLE: '9-7 irreversible',
    WAVELET_TRANSFORM_5x3_REVERSIBLE: '5-3 reversible'}

ENUMERATED_COLORSPACE = 1
RESTRICTED_ICC_PROFILE = 2
ANY_ICC_PROFILE = 3
VENDOR_COLOR_METHOD = 4

_method_display = {
    ENUMERATED_COLORSPACE: 'enumerated colorspace',
    RESTRICTED_ICC_PROFILE: 'restricted ICC profile',
    ANY_ICC_PROFILE: 'any ICC profile',
    VENDOR_COLOR_METHOD: 'vendor color method'}

_ = {1: 'accurately represents correct colorspace definition',
     2: 'approximates correct colorspace definition, exceptional quality',
     3: 'approximates correct colorspace definition, reasonable quality',
     4: 'approximates correct colorspace definition, poor quality'}
_approximation_display = _

# Registration values for comment markers.
RCME_BINARY = 0      # binary value comments
RCME_ISO_8859_1 = 1  # comments in latin-1 codec

# enumerated colorspaces
CMYK = 12
SRGB = 16
GREYSCALE = 17
YCC = 18
E_SRGB = 20
ROMM_RGB = 21

_colorspace_map_display = {
    CMYK:  'CMYK',
    SRGB:  'sRGB',
    GREYSCALE:  'greyscale',
    YCC:  'YCC',
    E_SRGB:  'e-sRGB',
    ROMM_RGB:  'ROMM-RGB'}

# enumerated color channel types
_COLOR = 0
_OPACITY = 1
_PRE_MULTIPLIED_OPACITY = 2
_UNSPECIFIED = 65535
_color_type_map_display = {
    _COLOR:  'color',
    _OPACITY:  'opacity',
    _PRE_MULTIPLIED_OPACITY:  'pre-multiplied opacity',
    _UNSPECIFIED:  'unspecified'}

# color channel definitions.
RED = 1
GREEN = 2
BLUE = 3
GREY = 1

# enumerated color channel associations
_rgb_colorspace = {"R": 1, "G": 2, "B": 3}
_greyscale_colorspace = {"Y": 1}
_ycbcr_colorspace = {"Y": 1, "Cb": 2, "Cr": 3}
_colorspace = {SRGB: _rgb_colorspace,
               GREYSCALE: _greyscale_colorspace,
               YCC: _ycbcr_colorspace,
               E_SRGB: _rgb_colorspace,
               ROMM_RGB: _rgb_colorspace}

# How to display the codestream profile.
_capabilities_display = {
    0: '2',
    1: '0',
    2: '1',
    3: '3'}

# Reader requirements
#RREQ_UNRESTRICTED_JPEG2000_PART_1 = 5
#RREQ_UNRESTRICTED_JPEG2000_PART_2 = 6
#RREQ_CMYK_ENUMERATED_COLORSPACE = 55
#RREQ_E_SRGB_ENUMERATED_COLORSPACE = 60
#RREQ_ROMM_RGB_ENUMERATED_COLORSPACE = 61
_reader_requirements_display = {
    0:  'File not completely understood',
    1:  'Deprecated',
    2:  'Contains multiple composition layers',
    3:  'Deprecated',
    4:  'JPEG 2000 Part 1 Profile 1 codestream',
    5:  'Unrestricted JPEG 2000 Part 1 codestream, ITU-T Rec. T.800 '
        + '| ISO/IEC 15444-1',
    6:  'Unrestricted JPEG 2000 Part 2 codestream',
    7:  'JPEG codestream as defined in ISO/IEC 10918-1',
    8:  'Deprecated',
    9:  'Non-premultiplied opacity channel',
    10:  'Premultiplied opacity channel',
    11:  'Chroma-key based opacity',
    12:  'Deprecated',
    13:  'Fragmented codestream where all fragments are in file and in order',
    14:  'Fragmented codestream where all fragments are in file '
         + 'but are out of order',
    15:  'Fragmented codestream where not all fragments are within the file '
         + 'but are all in locally accessible files',
    16:  'Fragmented codestream where some fragments may be accessible '
         + 'only through a URL specified network connection',
    17:  'Compositing required to produce rendered result from multiple '
         + 'compositing layers',
    18:  'Deprecated',
    19:  'Deprecated',
    20:  'Deprecated',
    21:  'At least one compositing layer consists of multiple codestreams',
    22:  'Deprecated',
    23:  'Colourspace transformations are required to combine compositing '
         + 'layers; not all compositing layers are in the same colourspace',
    24:  'Deprecated',
    25:  'Deprecated',
    26:  'First animation layer does not cover entire rendered result',
    27:  'Deprecated',
    28:  'Reuse of animation layers',
    29:  'Deprecated',
    30:  'Some animated frames are non-persistent',
    31:  'Deprecated',
    32:  'Rendered result involves scaling within a layer',
    33:  'Rendered result involves scaling between layers',
    34:  'ROI metadata',
    35:  'IPR metadata',
    36:  'Content metadata',
    37:  'History metadata',
    38:  'Creation metadata',
    39:  'JPX digital signatures',
    40:  'JPX checksums',
    41:  'Desires Graphics Arts Reproduction specified',
    42:  'Deprecated',
    43:  '(Deprecated) compositing layer uses restricted ICC profile',
    44:  'Compositing layer uses Any ICC profile',
    45:  'Deprecated',
    46:  'Deprecated',
    47:  'BiLevel 1 enumerated colourspace',
    48:  'BiLevel 2 enumerated colourspace',
    49:  'YCbCr 1 enumerated colourspace',
    50:  'YCbCr 2 enumerated colourspace',
    51:  'YCbCr 3 enumerated colourspace',
    52:  'PhotoYCC enumerated colourspace',
    53:  'YCCK enumerated colourspace',
    54:  'CMY enumerated colourspace',
    55: 'CMYK enumerated colorspace',
    56:  'CIELab enumerated colourspace with default parameters',
    57:  'CIELab enumerated colourspace with non-default parameters',
    58:  'CIEJab enumerated colourspace with default parameters',
    59:  'CIEJab enumerated colourspace with non-default parameters',
    60: 'e-sRGB enumerated colorspace',
    61: 'ROMM_RGB enumerated colorspace',
    62:  'Non-square samples',
    63:  'Deprecated',
    64:  'Deprecated',
    65:  'Deprecated',
    66:  'Deprecated',
    67:  'GIS metadata XML box',
    68:  'JPSEC extensions in codestream as specified by ISO/IEC 15444-8',
    69:  'JP3D extensions in codestream as specified by ISO/IEC 15444-10',
    70:  'Deprecated',
    71:  'e-sYCC enumerated colourspace',
    72:  'JPEG 2000 Part 2 codestream as restricted by baseline conformance '
         + 'requirements in M.9.2.3',
    73:  'YPbPr(1125/60) enumerated colourspace',
    74:  'YPbPr(1250/50) enumerated colourspace'}

