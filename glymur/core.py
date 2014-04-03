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

