"""Core definitions to be shared amongst the modules.
"""
import collections
import copy
import lxml.etree as ET

class _Keydefaultdict(collections.defaultdict):
    """Unlisted keys help form their own error message.

    Normally defaultdict uses a factory function with no input arguments, but
    that's not quite the behavior we want.
    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

# Progression order
LRCP = 0
RLCP = 1
RPCL = 2
PCRL = 3
CPRL = 4

STD = 0
CINEMA2K = 3
CINEMA4K = 4

RSIZ = {
    'STD': STD,
    'CINEMA2K': CINEMA2K,
    'CINEMA4K': CINEMA4K}

OFF = 0
CINEMA2K_24 = 1
CINEMA2K_48 = 2
CINEMA4K_24 = 3

CINEMA_MODE = {
    'off': OFF,
    'cinema2k_24': CINEMA2K_24,
    'cinema2k_48': CINEMA2K_48,
    'cinema4k_24': CINEMA4K_24, }

PROFILE = {
        # no profile, conform to 15444-1
        'none': 0x0000,
        # profile 0 as described in 15444-1,Table A.45
        '0': 0x0001,
        'zero': 0x0001,
        0: 0x0001,
        # profile 1 as described in 15444-1,Table A.45
        '1': 0x0002,
        'one': 0x0002,
        1: 0x0002,
        # at least 1 extension defined in 15444-2 (Part-2)
        '2': 0x8000,
        2: 0x8000,
        'two': 0x8000,
        # 2K cinema profile defined in 15444-1 AMD1
        'cinema_2k': 0x0003,
        # 4K cinema profile defined in 15444-1 AMD1
        'cinema_4k': 0x0004,
        # scalable 2K cinema profile defined in 15444-1 AMD2
        'cinema_s2k': 0x0005,
        # scalable 4K cinema profile defined in 15444-1 AMD2
        'cinema_s4k': 0x0006,
        # long term storage cinema profile defined in 15444-1 AMD2
        'cinema_lts': 0x0007,
        # Single Multi Tile Broadcast profile defined in 15444-1 AMD3
        'bc_single': 0x0100,
        # Multi Tile Broadcast profile defined in 15444-1 AMD3
        'bc_multi': 0x0200,
        # Multi Tile Reversible Broadcast profile defined in 15444-1 AMD3
        'bc_multi_r': 0x0300,
        # 2K Single Tile Lossy IMF profile defined in 15444-1 AMD 8
        'imf_2k': 0x0400,
        # 4K Single Tile Lossy IMF profile defined in 15444-1 AMD 8
        'imf_4k': 0x0401,
        # 8K Single Tile Lossy IMF profile defined in 15444-1 AMD 8
        'imf_8k': 0x0402,
        # 2K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8
        'imf_2k_r': 0x0403,
        # 4K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8
        'imf_4k_r': 0x0800,
        # 8K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8
        'imf_8k_r': 0x0801,
        }

# JPEG 2000 codestream and component size limits in cinema profiles
#
# Maximum codestream length for 24fps
OPJ_CINEMA_24_CS = 1302083
# Maximum codestream length for 48fps
OPJ_CINEMA_48_CS = 651041
# Maximum size per color component for 2K & 4K @ 24fps
OPJ_CINEMA_24_COMP = 1041666
# Maximum size per color component for 2K @ 48fps
OPJ_CINEMA_48_COMP = 520833


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

_factory = lambda x:  '{0} (unrecognized)'.format(x)
_COLORSPACE_MAP_DISPLAY = _Keydefaultdict(_factory,
        { CMYK:  'CMYK',
          SRGB:  'sRGB',
          GREYSCALE:  'greyscale',
          YCC:  'YCC',
          E_SRGB:  'e-sRGB',
          ROMM_RGB:  'ROMM-RGB'} )

# enumerated color channel types
COLOR = 0
OPACITY = 1
PRE_MULTIPLIED_OPACITY = 2
_UNSPECIFIED = 65535
_factory = lambda x:  '{0} (invalid)'.format(x)
_COLOR_TYPE_MAP_DISPLAY = _Keydefaultdict(_factory,
        { COLOR:  'color',
          OPACITY:  'opacity',
          PRE_MULTIPLIED_OPACITY:  'pre-multiplied opacity',
          _UNSPECIFIED:  'unspecified'})

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

