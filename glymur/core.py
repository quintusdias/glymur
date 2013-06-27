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
RREQ_UNRESTRICTED_JPEG2000_PART_1 = 5
RREQ_UNRESTRICTED_JPEG2000_PART_2 = 6
RREQ_CMYK_ENUMERATED_COLORSPACE = 55
RREQ_CMYK_ENUMERATED_COLORSPACE = 55
RREQ_E_SRGB_ENUMERATED_COLORSPACE = 60
RREQ_ROMM_RGB_ENUMERATED_COLORSPACE = 61
_reader_requirements_display = {
    0:  'File not completely understood',
    1:  'Deprecated',
    2:  'Contains multiple composition layers',
    3:  'Deprecated',
    4:  'JPEG 2000 Part 1 Profile 1 codestream',
    RREQ_UNRESTRICTED_JPEG2000_PART_1:
        'Unrestricted JPEG 2000 Part 1 codestream, ITU-T Rec. T.800 '
        + '| ISO/IEC 15444-1',
    RREQ_UNRESTRICTED_JPEG2000_PART_2:
        'Unrestricted JPEG 2000 Part 2 codestream',
    7: 'JPEG codestream as defined in ISO/IEC 10918-1',
    8:  'Deprecated',
    9:  'Non-premultiplied opacity channel',
    10:  'Premultiplied opacity channel',
    12:  'Deprecated',
    18:  'Deprecated',
    43:  '(Deprecated) compositing layer uses restricted ICC profile',
    44:  'Compositing layer uses Any ICC profile',
    45:  'Deprecated',
    RREQ_CMYK_ENUMERATED_COLORSPACE: 'CMYK enumerated colorspace',
    RREQ_E_SRGB_ENUMERATED_COLORSPACE: 'e-sRGB enumerated colorspace',
    RREQ_ROMM_RGB_ENUMERATED_COLORSPACE: 'ROMM_RGB enumerated colorspace'}
