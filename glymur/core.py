"""Core definitions to be shared amongst the modules.
"""
# Progression order
LRCP = 0
RLCP = 1
RPCL = 2
PCRL = 3
CPRL = 4

PROGRESSION_ORDER = {
    'LRCP': LRCP,
    'RLCP': RLCP,
    'RPCL': RPCL,
    'PCRL': PCRL,
    'CPRL': CPRL}

WAVELET_XFORM_9X7_IRREVERSIBLE = 0
WAVELET_XFORM_5X3_REVERSIBLE = 1

ENUMERATED_COLORSPACE = 1
RESTRICTED_ICC_PROFILE = 2
ANY_ICC_PROFILE = 3
VENDOR_COLOR_METHOD = 4

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

_COLORSPACE_MAP_DISPLAY = {
    CMYK:  'CMYK',
    SRGB:  'sRGB',
    GREYSCALE:  'greyscale',
    YCC:  'YCC',
    E_SRGB:  'e-sRGB',
    ROMM_RGB:  'ROMM-RGB'}

# enumerated color channel types
COLOR = 0
OPACITY = 1
PRE_MULTIPLIED_OPACITY = 2
_UNSPECIFIED = 65535
_COLOR_TYPE_MAP_DISPLAY = {
    COLOR:  'color',
    OPACITY:  'opacity',
    PRE_MULTIPLIED_OPACITY:  'pre-multiplied opacity',
    _UNSPECIFIED:  'unspecified'}

# color channel definitions.
RED = 1
GREEN = 2
BLUE = 3
GREY = 1
WHOLE_IMAGE = 0

# enumerated color channel associations
_COLORSPACE = {SRGB: {"R": 1, "G": 2, "B": 3},
               GREYSCALE: {"Y": 1},
               YCC: {"Y": 1, "Cb": 2, "Cr": 3},
               E_SRGB: {"R": 1, "G": 2, "B": 3},
               ROMM_RGB: {"R": 1, "G": 2, "B": 3}}

# How to display the codestream profile.
_CAPABILITIES_DISPLAY = {
    0: '2',
    1: '0',
    2: '1',
    3: '3'}
