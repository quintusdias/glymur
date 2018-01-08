# Standard library imports ...
from collections import OrderedDict
import datetime
import struct

# Third party library imports ...
import numpy as np


class _ICCProfile(object):
    """
    Container for ICC profile information.
    """
    profile_class = {
        b'scnr': 'input device profile',
        b'mntr': 'display device profile',
        b'prtr': 'output device profile',
        b'link': 'devicelink profile',
        b'spac': 'colorspace conversion profile',
        b'abst': 'abstract profile',
        b'nmcl': 'name colour profile'
    }

    colour_space_dict = {
        b'XYZ ': 'XYZ',
        b'Lab ': 'Lab',
        b'Luv ': 'Luv',
        b'YCbr': 'YCbCr',
        b'Yxy ': 'Yxy',
        b'RGB ': 'RGB',
        b'GRAY': 'gray',
        b'HSV ': 'hsv',
        b'HLS ': 'hls',
        b'CMYK': 'CMYK',
        b'CMY ': 'cmy',
        b'2CLR': '2colour',
        b'3CLR': '3colour',
        b'4CLR': '4colour',
        b'5CLR': '5colour',
        b'6CLR': '6colour',
        b'7CLR': '7colour',
        b'8CLR': '8colour',
        b'9CLR': '9colour',
        b'ACLR': '10colour',
        b'BCLR': '11colour',
        b'CCLR': '12colour',
        b'DCLR': '13colour',
        b'ECLR': '14colour',
        b'FCLR': '15colour'
    }

    rendering_intent_dict = {
        0: 'perceptual',
        1: 'media-relative colorimetric',
        2: 'saturation',
        3: 'ICC-absolute colorimetric'
    }

    def __init__(self, read_buffer):
        self._raw_buffer = read_buffer
        header = OrderedDict()

        data = struct.unpack('>IIBB', self._raw_buffer[0:10])
        header['Size'] = data[0]
        header['Preferred CMM Type'] = data[1]
        major = data[2]
        minor = (data[3] & 0xf0) >> 4
        bugfix = (data[3] & 0x0f)
        header['Version'] = f'{major}.{minor}.{bugfix}'

        header['Device Class'] = self.profile_class[self._raw_buffer[12:16]]
        header['Color Space'] = self.colour_space_dict[self._raw_buffer[16:20]]
        data = self.colour_space_dict[self._raw_buffer[20:24]]
        header['Connection Space'] = data

        data = struct.unpack('>HHHHHH', self._raw_buffer[24:36])
        try:
            header['Datetime'] = datetime.datetime(*data)
        except ValueError:
            header['Datetime'] = None
        header['File Signature'] = read_buffer[36:40].decode('utf-8')
        if read_buffer[40:44] == b'\x00\x00\x00\x00':
            header['Platform'] = 'unrecognized'
        else:
            header['Platform'] = read_buffer[40:44].decode('utf-8')

        fval, = struct.unpack('>I', read_buffer[44:48])
        header['Flags'] = (
            f"{'' if fval & 0x01 else 'not '}embedded, "
            f"{'cannot' if fval & 0x02 else 'can'} be used independently"
        )

        header['Device Manufacturer'] = read_buffer[48:52].decode('utf-8')
        if read_buffer[52:56] == b'\x00\x00\x00\x00':
            device_model = ''
        else:
            device_model = read_buffer[52:56].decode('utf-8')
        header['Device Model'] = device_model

        val, = struct.unpack('>Q', read_buffer[56:64])
        attr = (
            f"{'transparency' if val & 0x01 else 'reflective'}, "
            f"{'matte' if val & 0x02 else 'glossy'}, "
            f"{'negative' if val & 0x04 else 'positive'} media polarity, "
            f"{'black and white' if val & 0x08 else 'color'} media"
        )
        header['Device Attributes'] = attr

        rval, = struct.unpack('>I', read_buffer[64:68])
        try:
            header['Rendering Intent'] = self.rendering_intent_dict[rval]
        except KeyError:
            header['Rendering Intent'] = 'unknown'

        data = struct.unpack('>iii', read_buffer[68:80])
        header['Illuminant'] = np.array(data, dtype=np.float64) / 65536

        if read_buffer[80:84] == b'\x00\x00\x00\x00':
            creator = 'unrecognized'
        else:
            creator = read_buffer[80:84].decode('utf-8')
        header['Creator'] = creator

        if header['Version'][0] == '4':
            header['Profile Id'] = read_buffer[84:100]

        # Final 27 bytes are reserved.
        self.header = header
