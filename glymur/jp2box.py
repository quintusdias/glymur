"""Classes for individual JPEG 2000 boxes.

References
----------
.. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
   15444-1:2004 - Information technology -- JPEG 2000 image coding system:
   Core coding system

.. [JP2K15444-2m] International Organization for Standardication.  ISO/IEC
   15444-2:2004 - Information technology -- JPEG 2000 image coding system:
   Extensions
"""

import collections
import copy
import datetime
import math
import os
import pprint
import struct
import sys
import uuid
import warnings
import xml.etree.cElementTree as ET

import numpy as np

from .codestream import Codestream
from .core import _approximation_display
from .core import _colorspace_map_display
from .core import _color_type_map_display
from .core import _method_display
from .core import _reader_requirements_display


class Jp2kBox(object):
    """Superclass for JPEG 2000 boxes.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    """

    def __init__(self, id='', offset=0, length=0, longname=''):
        self.id = id
        self.length = length
        self.offset = offset
        self.longname = longname

    def __str__(self):
        msg = "{0} Box ({1})".format(self.longname, self.id)
        msg += " @ ({0}, {1})".format(self.offset, self.length)
        return msg

    def _parse_superbox(self, f):
        """Parse a superbox (box consisting of nothing but other boxes.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        List of top-level boxes in the JPEG 2000 file.
        """

        superbox = []

        start = f.tell()

        while True:

            # Are we at the end of the superbox?
            if start >= self.offset + self.length:
                break

            buffer = f.read(8)
            (L, T) = struct.unpack('>I4s', buffer)
            if sys.hexversion >= 0x03000000:
                T = T.decode('utf-8')

            if L == 0:
                # The length of the box is presumed to last until the end of
                # the file.  Compute the effective length of the box.
                num_bytes = self._file_size - f.tell() + 8

            elif L == 1:
                # The length of the box is in the XL field, a 64-bit value.
                buffer = f.read(8)
                num_bytes, = struct.unpack('>Q', buffer)

            else:
                num_bytes = L

            # Call the proper parser for the given box with ID "T".
            try:
                box = _box_with_id[T]._parse(f, T, start, num_bytes)
            except KeyError:
                msg = 'Unrecognized box ({0}) encountered.'.format(T)
                warnings.warn(msg)
                box = Jp2kBox(T, offset=start, length=num_bytes,
                              longname='Unknown box')

            superbox.append(box)

            # Position to the start of the next box.
            if num_bytes > self.length:
                # Length of the current box goes past the end of the
                # enclosing superbox.
                msg = '{0} box has incorrect box length ({1})'
                msg = msg.format(T, num_bytes)
                warnings.warn(msg)
            elif f.tell() > start + num_bytes:
                # The box must be invalid somehow, as the file pointer is
                # positioned past the end of the box.
                msg = '{0} box may be invalid, the file pointer is positioned '
                msg += '{1} bytes past the end of the box.'
                msg = msg.format(T, f.tell() - (start + num_bytes))
                warnings.warn(msg)
            f.seek(start + num_bytes)

            start += num_bytes

        return superbox


class ColourSpecificationBox(Jp2kBox):
    """Container for JPEG 2000 color specification box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        Length of the box in bytes.
    offset : int
        Offset of the box from the start of the file.
    longname : str
        More verbose description of the box.
    method : int
        Method for defining the colorspace.
    precedence : int
        How this box ranks in priority compared to other color specification
        boxes.
    approximation : int
        Measure of colorspace accuracy.
    colorspace : int or None
        Enumerated colorspace, corresponds to one of 'sRGB', 'greyscale', or
        'YCC'.  If not None, then icc_profile must be None.
    icc_profile : dict
        ICC profile header according to ICC profile specification.  If
        colorspace is not None, then icc_profile must be empty.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Colour Specification')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)

        msg += '\n    Method:  {0}'.format(_method_display[self.method])
        msg += '\n    Precedence:  {0}'.format(self.precedence)

        if self.approximation is not 0:
            x = _approximation_display[self.approximation]
            msg += '\n    Approximation:  {0}'.format(x)
        if self.colorspace is not None:
            x = _colorspace_map_display[self.colorspace]
            msg += '\n    Colorspace:  {0}'.format(x)
        else:
            # 2.7 has trouble pretty-printing ordered dicts so we just have
            # to print as a regular dict in this case.
            if sys.hexversion < 0x03000000:
                icc_profile = dict(self.icc_profile)
            else:
                icc_profile = self.icc_profile
            x = pprint.pformat(icc_profile)
            lines = [' ' * 8 + y for y in x.split('\n')]
            msg += '\n    ICC Profile:\n{0}'.format('\n'.join(lines))

        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse JPEG 2000 color specification box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary
            dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        # Read the brand, minor version.
        buffer = f.read(3)
        (method, precedence, approximation) = struct.unpack('>BBB', buffer)
        kwargs['method'] = method
        kwargs['precedence'] = precedence
        kwargs['approximation'] = approximation

        if method == 1:
            # enumerated colour space
            buffer = f.read(4)
            kwargs['colorspace'], = struct.unpack('>I', buffer)
            kwargs['icc_profile'] = None

        else:
            # ICC profile
            kwargs['colorspace'] = None
            n = offset + length - f.tell()
            if n < 128:
                msg = "ICC profile header is corrupt, length is "
                msg += "only {0} instead of 128."
                warnings.warn(msg.format(n), UserWarning)
                kwargs['icc_profile'] = None
            else:
                icc_profile = _ICCProfile(f.read(n))
                kwargs['icc_profile'] = icc_profile.header

        box = ColourSpecificationBox(**kwargs)
        return box


class _ICCProfile(object):
    """
    """
    profile_class = {b'scnr': 'input device profile',
                     b'mntr': 'display device profile',
                     b'prtr': 'output device profile',
                     b'link': 'devicelink profile',
                     b'spac': 'colorspace conversion profile',
                     b'abst': 'abstract profile',
                     b'nmcl': 'name colour profile'}

    colour_space_dict = {b'XYZ ': 'XYZ',
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
                         b'FCLR': '15colour'}

    rendering_intent_dict = {0: 'perceptual',
                             1: 'media-relative colorimetric',
                             2: 'saturation',
                             3: 'ICC-absolute colorimetric'}

    def __init__(self, buffer):
        self._raw_buffer = buffer
        header = collections.OrderedDict()

        data = struct.unpack('>IIBB', self._raw_buffer[0:10])
        header['Size'] = data[0]
        header['Preferred CMM Type'] = data[1]
        major = data[2]
        minor = (data[3] & 0xf0) >> 4
        bugfix = (data[3] & 0x0f)
        header['Version'] = '{0}.{1}.{2}'.format(major, minor, bugfix)

        header['Device Class'] = self.profile_class[self._raw_buffer[12:16]]
        header['Color Space'] = self.colour_space_dict[self._raw_buffer[16:20]]
        data = self.colour_space_dict[self._raw_buffer[20:24]]
        header['Connection Space'] = data

        data = struct.unpack('>HHHHHH', self._raw_buffer[24:36])
        header['Datetime'] = datetime.datetime(*data)
        header['File Signature'] = buffer[36:40].decode('utf-8')
        if buffer[40:44] == b'\x00\x00\x00\x00':
            header['Platform'] = 'unrecognized'
        else:
            header['Platform'] = buffer[40:44].decode('utf-8')

        x, = struct.unpack('>I', buffer[44:48])
        y = 'embedded, ' if x & 0x01 else 'not embedded, '
        y += 'cannot ' if x & 0x02 else 'can '
        y += 'be used independently'
        header['Flags'] = y

        header['Device Manufacturer'] = buffer[48:52].decode('utf-8')
        if buffer[52:56] == b'\x00\x00\x00\x00':
            device_model = ''
        else:
            device_model = buffer[52:56].decode('utf-8')
        header['Device Model'] = device_model

        x, = struct.unpack('>Q', buffer[56:64])
        y = 'transparency, ' if x & 0x01 else 'reflective, '
        y += 'matte, ' if x & 0x02 else 'glossy, '
        y += 'negative ' if x & 0x04 else 'positive '
        y += 'media polarity, '
        y += 'black and white media' if x & 0x08 else 'color media'
        header['Device Attributes'] = y

        x, = struct.unpack('>I', buffer[64:68])
        try:
            header['Rendering Intent'] = self.rendering_intent_dict[x]
        except KeyError:
            header['Rendering Intent'] = 'unknown'

        data = struct.unpack('>iii', buffer[68:80])
        header['Illuminant'] = np.array(data, dtype=np.float64) / 65536

        if buffer[80:84] == b'\x00\x00\x00\x00':
            creator = 'unrecognized'
        else:
            creator = buffer[80:84].decode('utf-8')
        header['Creator'] = creator

        if header['Version'][0] == '4':
            header['Profile Id'] = buffer[84:100]

        # Final 27 bytes are reserved.

        self.header = header


class ComponentDefinitionBox(Jp2kBox):
    """Container for component definition box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : numeric scalar
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    component_number : int
        number of the component
    component_type : int
        type of the component
    association : int
        number of the associated color
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Component Definition')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        for j in range(len(self.association)):
            color_type_string = _color_type_map_display[self.component_type[j]]
            if self.association[j] == 0:
                assn = 'whole image'
            else:
                assn = str(self.association[j])
            msg += '\n    Component {0} ({1}) ==> ({2})'
            msg = msg.format(self.component_number[j], color_type_string, assn)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse component definition box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        # Read the number of components.
        buffer = f.read(2)
        N, = struct.unpack('>H', buffer)

        component_number = []
        component_type = []
        association = []

        buffer = f.read(N * 6)
        data = struct.unpack('>' + 'HHH' * N, buffer)
        kwargs['component_number'] = data[0:N * 6:3]
        kwargs['component_type'] = data[1:N * 6:3]
        kwargs['association'] = data[2:N * 6:3]

        box = ComponentDefinitionBox(**kwargs)
        return box


class ComponentMappingBox(Jp2kBox):
    """Container for channel identification information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : numeric scalar
        Length of the box in bytes.
    offset : int
        Offset of the box from the start of the file.
    longname : str
        Verbose description of the box.
    component_index : int
        Index of component in codestream that is mapped to this channel.
    mapping_type : int
        mapping type, either direct use (0) or palette (1)
    palette_index : int
        Index component from palette
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Component Mapping')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)

        for k in range(len(self.component_index)):
            if self.mapping_type[k] == 1:
                msg += '\n    Component {0} ==> palette column {1}'
                msg = msg.format(self.component_index[k],
                                 self.palette_index[k])
            else:
                msg += '\n    Component %d ==> %d'
                msg = msg.format(self.component_index[k], k)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse component mapping box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        N = offset + length - f.tell()
        num_components = int(N/4)

        buffer = f.read(N)
        data = struct.unpack('>' + 'HBB' * num_components, buffer)

        kwargs['component_index'] = data[0:N:num_components]
        kwargs['mapping_type'] = data[1:N:num_components]
        kwargs['palette_index'] = data[2:N:num_components]

        box = ComponentMappingBox(**kwargs)
        return box


class ContiguousCodestreamBox(Jp2kBox):
    """Container for JPEG2000 codestream information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    main_header : list
        List of segments in the codestream header.
    """

    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='jp2c', longname='Contiguous Codestream')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        msg += '\n    Main header:'
        for segment in self.main_header.segment:
            segstr = str(segment)

            # Add indentation.
            strs = [('\n        ' + x) for x in segstr.split('\n')]
            msg += ''.join(strs)
        return msg

    @staticmethod
    def _parse(f, id='jp2c', offset=0, length=0):
        """Parse a codestream box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        kwargs['main_header'] = Codestream(f, header_only=True)

        box = ContiguousCodestreamBox(**kwargs)
        return box


class FileTypeBox(Jp2kBox):
    """Container for JPEG 2000 file type box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    brand: str
        Specifies the governing recommendation or standard upon which this
        file is based.
    minor_version: int
        Minor version number identifying the JP2 specification used.
    compatibility_list: list
        List of file conformance profiles.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='File Type')
        self.__dict__.update(**kwargs)

    def __str__(self):
        lst = [Jp2kBox.__str__(self),
               '    Brand:  {0}',
               '    Compatibility:  {1}']
        msg = '\n'.join(lst)
        msg = msg.format(self.brand, self.compatibility_list)

        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse JPEG 2000 file type box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns:
            kwargs:  dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        # Read the brand, minor version.
        buffer = f.read(8)
        (brand, minor_version) = struct.unpack('>4sI', buffer)
        if sys.hexversion < 0x030000:
            kwargs['brand'] = brand
        else:
            kwargs['brand'] = brand.decode('utf-8')
        kwargs['minor_version'] = minor_version

        # Read the compatibility list.  Each entry has 4 bytes.
        current_pos = f.tell()
        n = (offset + length - current_pos) / 4
        buffer = f.read(int(n) * 4)
        compatibility_list = []
        for j in range(int(n)):
            CL, = struct.unpack('>4s', buffer[4*j:4*(j+1)])
            if sys.hexversion >= 0x03000000:
                CL = CL.decode('utf-8')
            compatibility_list.append(CL)

        kwargs['compatibility_list'] = compatibility_list

        box = FileTypeBox(**kwargs)
        return box


class ImageHeaderBox(Jp2kBox):
    """Container for JPEG 2000 image header box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    height, width :  int
        Height and width of image.
    num_components : int
        Number of image channels.
    bits_per_component : int
        Bits per component.
    signed : bool
        False if the image components are unsigned.
    compression : nt
        The compression type, should be 7 if JP2.
    cspace_unknown : int
        0 if the color space is known and correctly specified.
    ip_provided : int
        0 if the file does not contain intellectual propery rights information.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Image Header')
        self.__dict__.update(**kwargs)

    def __str__(self):
        lst = [Jp2kBox.__str__(self)]
        lst.append('Size:  [{0} {1} {2}]'.format(self.height, self.width,
                                                 self.num_components))
        lst.append('Bitdepth:  {0}'.format(self.bits_per_component))
        lst.append('Signed:  {0}'.format(self.signed))
        if self.compression == 7:
            lst.append('Compression:  wavelet')
        if self.cspace_unknown:
            lst.append('Colorspace Unknown:  True')
        else:
            lst.append('Colorspace Unknown:  False')
        return '\n    '.join(lst)

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse JPEG 2000 image header box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns:
            kwargs:  dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        # Read the box information
        buffer = f.read(14)
        params = struct.unpack('>IIHBBBB', buffer)
        kwargs['height'] = params[0]
        kwargs['width'] = params[1]
        kwargs['num_components'] = params[2]
        kwargs['bits_per_component'] = (params[3] & 0x7f) + 1
        kwargs['signed'] = (params[3] & 0x80) > 1
        kwargs['compression'] = params[4]
        kwargs['cspace_unknown'] = params[5]
        kwargs['ip_provided'] = params[6]

        box = ImageHeaderBox(**kwargs)
        return box


class AssociationBox(Jp2kBox):
    """Container for Association box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    box : list
        List of boxes contained in this superbox.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Association')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        for box in self.box:
            boxstr = str(box)

            # Add indentation.
            strs = [('\n    ' + x) for x in boxstr.split('\n')]
            msg += ''.join(strs)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse association box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values, for now just a single
            4-byte tuple.
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        box = AssociationBox(**kwargs)

        # The JP2 header box is a superbox, so go ahead and parse its child
        # boxes.
        box.box = box._parse_superbox(f)

        return box


class JP2HeaderBox(Jp2kBox):
    """Container for JP2 header box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    box : list
        List of boxes contained in this superbox.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='JP2 Header')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        for box in self.box:
            boxstr = str(box)

            # Add indentation.
            strs = [('\n    ' + x) for x in boxstr.split('\n')]
            msg += ''.join(strs)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse JPEG 2000 header box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values, for now just a single
            4-byte tuple.
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        box = JP2HeaderBox(**kwargs)

        # The JP2 header box is a superbox, so go ahead and parse its child
        # boxes.
        box.box = box._parse_superbox(f)

        return box


class JPEG2000SignatureBox(Jp2kBox):
    """Container for JPEG 2000 signature box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    signature : byte
        Four-byte tuple identifying the file as JPEG 2000.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='JPEG 2000 Signature')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        msg += '\n    Signature:  {:02x}{:02x}{:02x}{:02x}'
        msg = msg.format(*self.signature)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse JPEG 2000 signature box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns:
            kwargs:  dictionary of parameter values, for now just a single
                4-byte tuple.
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        buffer = f.read(4)
        kwargs['signature'] = struct.unpack('>BBBB', buffer)

        box = JPEG2000SignatureBox(**kwargs)
        return box


class PaletteBox(Jp2kBox):
    """Container for palette box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    palette : list
        Colormap represented as list of 1D arrays, one per color component.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Palette')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        msg += '\n    Size:  ({0} x {1})'.format(len(self.palette[0]),
                                                 len(self.palette))
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse palette box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs:  dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        # Get the size of the palette.
        buffer = f.read(3)
        (NE, NC) = struct.unpack('>HB', buffer)

        # Need to determine bps and signed or not
        buffer = f.read(NC)
        data = struct.unpack('>' + 'B' * NC, buffer)
        bps = [((x & 0x07f) + 1) for x in data]
        signed = [((x & 0x80) > 1) for x in data]
        kwargs['bits_per_component'] = bps
        kwargs['signed'] = signed

        # Form the format string so that we can intelligently unpack the
        # colormap.  We have to do this because it is possible that the
        # colormap columns could have different datatypes.
        #
        # This means that we store the palette as a list of 1D arrays,
        # which reverses the usual indexing scheme.
        palette = []
        fmt = '>'
        bytes_per_row = 0
        for j in range(NC):
            if bps[j] <= 8:
                fmt += 'B'
                bytes_per_row += 1
                palette.append(np.zeros(NE, dtype=np.uint8))
            elif bps[j] <= 16:
                fmt += 'H'
                bytes_per_row += 2
                palette.append(np.zeros(NE, dtype=np.uint16))
            elif bps[j] <= 32:
                fmt += 'I'
                bytes_per_row += 4
                palette.append(np.zeros(NE, dtype=np.uint32))
            else:
                msg = 'Unsupported palette bitdepth (%d).'
                raise IOError(msg)
        n = NE * bytes_per_row
        buffer = f.read(n)

        for j in range(NE):
            row_buffer = buffer[(bytes_per_row * j):(bytes_per_row * (j + 1))]
            row = struct.unpack(fmt, row_buffer)
            for k in range(NC):
                palette[k][j] = row[k]

        kwargs['palette'] = palette
        box = PaletteBox(**kwargs)
        return box


class ReaderRequirementsBox(Jp2kBox):
    """Container for reader requirements box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    FUAM : int
        Fully Understand Aspects mask.
    DCM : int
        Decode completely mask.
    standard_flag : list
        Integers specifying standard features.
    standard_mask : list
        Specifies  the compatibility mask for each corresponding standard
        flag.
    vendor_feature : list
        Each item is a UUID corresponding to a vendor defined feature.
    vendor_mask : list
        Specifies the compatibility mask for each corresponding vendor
        feature.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Reader Requirements')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)

        # TODO:  include FUAM, DCM

        msg += '\n    Standard Features:'
        # TODO:  include each standard mask
        for j in range(len(self.standard_flag)):
            sfl = self.standard_flag[j]
            rrdisp = _reader_requirements_display[self.standard_flag[j]]
            msg += '\n        Feature {0:03d}:  {1}'.format(sfl, rrdisp)

        # TODO:  include the vendor mask
        msg += '\n    Vendor Features:'
        for j in range(len(self.vendor_feature)):
            msg += '\n        UUID {0}'.format(self.vendor_feature[j])

        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse reader requirements box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        ReaderRequirementsBox.
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        buffer = f.read(1)
        mask_length, = struct.unpack('>B', buffer)
        if mask_length == 1:
            mask_format = 'B'
        elif mask_length == 2:
            mask_format = 'H'
        elif mask_length == 4:
            mask_format = 'I'
        else:
            msg = 'Unhandled reader requirements box mask length (%d).'
            msg %= mask_length
            raise RuntimeError(msg)

        # Fully Understands Aspect Mask
        # Decodes Completely Mask
        buffer = f.read(2 * mask_length)
        data = struct.unpack('>' + mask_format * 2, buffer)
        kwargs['FUAM'] = data[0]
        kwargs['DCM'] = data[1]

        buffer = f.read(2)
        num_standard_flags, = struct.unpack('>H', buffer)

        # Read in standard flags and standard masks.  Each standard flag should
        # be two bytes, but the standard mask flag is as long as specified by
        # the mask length.
        buffer = f.read(num_standard_flags * (2 + mask_length))
        data = struct.unpack('>' + ('H' + mask_format) * num_standard_flags,
                             buffer)
        kwargs['standard_flag'] = data[0:num_standard_flags * 2:2]
        kwargs['standard_mask'] = data[1:num_standard_flags * 2:2]

        # Vendor features
        buffer = f.read(2)
        num_vendor_features, = struct.unpack('>H', buffer)

        # Each vendor feature consists of a 16-byte UUID plus a mask whose
        # length is specified by, you guessed it, "mask_length".
        entry_length = 16 + mask_length
        buffer = f.read(num_vendor_features * entry_length)
        vendor_feature = []
        vendor_mask = []
        for j in range(num_vendor_features):
            ubuffer = buffer[j * entry_length:(j + 1) * entry_length]
            vendor_feature.append(uuid.UUID(bytes=ubuffer[0:16]))

            vm = struct.unpack('>' + mask_format, ubuffer[16:])
            vendor_mask.append(vm)

        kwargs['vendor_feature'] = vendor_feature
        kwargs['vendor_mask'] = vendor_mask

        box = ReaderRequirementsBox(**kwargs)
        return box


class ResolutionBox(Jp2kBox):
    """Container for Resolution superbox information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    box : list
        List of boxes contained in this superbox.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Resolution')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        for box in self.box:
            boxstr = str(box)

            # Add indentation.
            strs = [('\n    ' + x) for x in boxstr.split('\n')]
            msg += ''.join(strs)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse Resolution box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values, for now just a single
                 4-byte tuple.
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        box = ResolutionBox(**kwargs)

        # The JP2 header box is a superbox, so go ahead and parse its child
        # boxes.
        box.box = box._parse_superbox(f)

        return box


class CaptureResolutionBox(ResolutionBox):
    """Container for Capture resolution box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    VR, HR : float
        Vertical, horizontal resolution.
    """
    def __init__(self, **kwargs):
        ResolutionBox.__init__(self, id='', longname='Capture Resolution')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        msg += '\n    VCR:  {0}'.format(self.VR)
        msg += '\n    HCR:  {0}'.format(self.HR)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse Resolution box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values, for now just a single
                 4-byte tuple.
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        buffer = f.read(10)
        (RN1, RD1, RN2, RD2, RE1, RE2) = struct.unpack('>HHHHBB', buffer)
        kwargs['VR'] = RN1 / RD1 * math.pow(10, RE1)
        kwargs['HR'] = RN2 / RD2 * math.pow(10, RE2)

        box = CaptureResolutionBox(**kwargs)

        return box


class DisplayResolutionBox(ResolutionBox):
    """Container for Display resolution box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    VR, HR : float
        Vertical, horizontal resolution.
    """
    def __init__(self, **kwargs):
        ResolutionBox.__init__(self, id='', longname='Display Resolution')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        msg += '\n    VDR:  {0}'.format(self.VR)
        msg += '\n    HDR:  {0}'.format(self.HR)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse Resolution box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values, for now just a single
                 4-byte tuple.
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        buffer = f.read(10)
        (RN1, RD1, RN2, RD2, RE1, RE2) = struct.unpack('>HHHHBB', buffer)
        kwargs['VR'] = RN1 / RD1 * math.pow(10, RE1)
        kwargs['HR'] = RN2 / RD2 * math.pow(10, RE2)

        box = DisplayResolutionBox(**kwargs)

        return box


class LabelBox(Jp2kBox):
    """Container for Label box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    label : str
        Label
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Label')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        msg += '\n    Label:  {0}'.format(self.label)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse Label box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        num_bytes = offset + length - f.tell()
        buffer = f.read(num_bytes)
        kwargs['label'] = buffer.decode('utf-8')
        box = LabelBox(**kwargs)
        return box


class XMLBox(Jp2kBox):
    """Container for XML box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    xml : ElementTree.Element
        XML section.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='XML')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        xml = self.xml
        if self.xml is not None:
            msg += _pretty_print_xml(self.xml)
        else:
            msg += '\n    {0}'.format(xml)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse XML box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        num_bytes = offset + length - f.tell()
        buffer = f.read(num_bytes)
        text = buffer.decode('utf-8')

        # Strip out any trailing nulls.
        text = text.rstrip('\0')

        try:
            kwargs['xml'] = ET.fromstring(text)
        except ET.ParseError as e:
            msg = 'A problem was encountered while parsing an XML box:  "{0}"'
            msg = msg.format(str(e))
            warnings.warn(msg, UserWarning)
            kwargs['xml'] = None

        box = XMLBox(**kwargs)
        return box


class UUIDListBox(Jp2kBox):
    """Container for JPEG 2000 UUID list box.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    ulst : list
        List of UUIDs.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='UUID List')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        for enumerated_item in enumerate(self.ulst):
            msg += '\n    UUID[{0}]:  {1}'.format(*enumerated_item)
        return(msg)

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse UUIDList box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values
        """
        buffer = f.read(2)
        N, = struct.unpack('>H', buffer)

        ulst = []
        for j in range(N):
            buffer = f.read(16)
            ulst.append(uuid.UUID(bytes=buffer))

        kwargs = {}
        kwargs['id'] = id
        kwargs['offset'] = offset
        kwargs['length'] = length
        kwargs['ulst'] = ulst
        box = UUIDListBox(**kwargs)
        return(box)


class UUIDInfoBox(Jp2kBox):
    """Container for JPEG 2000 UUID Info superbox.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    box : list
        List of boxes contained in this superbox.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='UUIDInfo')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)

        for box in self.box:
            box_str = str(box)

            # Add indentation.
            lst = [('\n    ' + x) for x in box_str.split('\n')]
            msg += ''.join(lst)

        return(msg)

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse UUIDInfo super box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        box = UUIDInfoBox(**kwargs)

        # The UUIDInfo box is a superbox, so go ahead and parse its child
        # boxes.
        box.box = box._parse_superbox(f)

        return box


class DataEntryURLBox(Jp2kBox):
    """Container for data entry URL box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    version : byte
        Must be 0 for JP2.
    flag : bytes
        Particular attributes of this box, consists of three bytes.
    URL : str
        Associated URL.
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='Data Entry URL')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Jp2kBox.__str__(self)
        msg += '\n    '

        lines = ['Version:  {0}',
                 'Flag:  {1} {2} {3}',
                 'URL:  "{4}"']
        msg += '\n    '.join(lines)
        msg = msg.format(self.version,
                         self.flag[0], self.flag[1], self.flag[2],
                         self.URL)
        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse Data Entry URL box.

        Parameters
        ----------
        f : file
            Open file object.
        id : byte
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        kwargs : dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        buffer = f.read(4)
        data = struct.unpack('>BBBB', buffer)
        kwargs['version'] = data[0]
        kwargs['flag'] = data[1:4]

        n = offset + length - f.tell()
        buffer = f.read(n)
        kwargs['URL'] = buffer.decode('utf-8')
        box = DataEntryURLBox(**kwargs)
        return box


class UUIDBox(Jp2kBox):
    """Container for UUID box information.

    Attributes
    ----------
    id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    uuid : uuid.UUID
        16-byte UUID
    data : bytes or dict or ElementTree.Element
        Vendor-specific data.  Exif UUIDs are interpreted as dictionaries.
        XMP UUIDs are interpreted as standard XML.

    References
    ----------
    .. [XMP] International Organization for Standardication.  ISO/IEC
       16684-1:2012 - Graphic technology -- Extensible metadata platform (XMP)
       specification -- Part 1:  Data model, serialization and core properties
    """
    def __init__(self, **kwargs):
        Jp2kBox.__init__(self, id='', longname='UUID')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = '{0}\n'
        msg += '    UUID:  {1}{2}\n'
        msg += '    UUID Data:  {3}'

        if self.uuid == uuid.UUID('be7acfcb-97a9-42e8-9c71-999491e3afac'):
            uuid_type = ' (XMP)'
            uuid_data = _pretty_print_xml(self.data)
        elif self.uuid.bytes == b'JpgTiffExif->JP2':
            uuid_type = ' (Exif)'
            # 2.7 has trouble pretty-printing ordered dicts, so print them
            # as regular dicts.  Not ideal, but at least it's good on 3.3+.
            x = self.data if sys.hexversion >= 0x03000000 else dict(self.data)
            uuid_data = '\n' + pprint.pformat(x)
        else:
            uuid_type = ''
            uuid_data = '{0} bytes'.format(len(self.data))

        msg = msg.format(Jp2kBox.__str__(self),
                         self.uuid,
                         uuid_type,
                         uuid_data)

        return msg

    @staticmethod
    def _parse(f, id, offset, length):
        """Parse JPEG 2000 signature box.

        Parameters
        ----------
        f : file
            Open file object.
        id : str
            4-byte unique identifier for this box.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns:
            kwargs:  dictionary of parameter values
        """
        kwargs = {}
        kwargs['id'] = id
        kwargs['length'] = length
        kwargs['offset'] = offset

        buffer = f.read(16)
        kwargs['uuid'] = uuid.UUID(bytes=buffer)

        n = offset + length - f.tell()
        buffer = f.read(n)
        if kwargs['uuid'] == uuid.UUID('be7acfcb-97a9-42e8-9c71-999491e3afac'):
            # XMP data.  Parse as XML.  Seems to be a difference between
            # ElementTree in version 2.7 and 3.3.
            if sys.hexversion < 0x03000000:
                parser = ET.XMLParser(encoding='utf-8')
                kwargs['data'] = ET.fromstringlist(buffer, parser=parser)
            else:
                text = buffer.decode('utf-8')
                kwargs['data'] = ET.fromstring(text)
        elif kwargs['uuid'].bytes == b'JpgTiffExif->JP2':
            e = Exif(buffer)
            d = collections.OrderedDict()
            d['Image'] = e.exif_image
            d['Photo'] = e.exif_photo
            d['GPSInfo'] = e.exif_gpsinfo
            d['Iop'] = e.exif_iop
            kwargs['data'] = d
        else:
            kwargs['data'] = buffer
        box = UUIDBox(**kwargs)
        return box


class Exif(object):
    """
    Attributes
    ----------
    buffer : bytes
        Raw byte stream consisting of the UUID data.
    endian : str
        Either '<' for big-endian, or '>' for little-endian.
    """

    def __init__(self, buffer):
        """Interpret raw buffer consisting of Exif IFD.
        """
        self.exif_image = None
        self.exif_photo = None
        self.exif_gpsinfo = None
        self.exif_iop = None

        self.buffer = buffer

        # Ignore the first six bytes.
        # Next 8 should be (73, 73, 42, 8)
        data = struct.unpack('<BBHI', buffer[6:14])
        if data[0] == 73 and data[1] == 73:
            # little endian
            self.endian = '<'
        else:
            # big endian
            self.endian = '>'
        offset = data[3]

        # This is the 'Exif Image' portion.
        exif = _ExifImageIfd(self.endian, buffer[6:], offset)
        self.exif_image = exif.processed_ifd

        if 'ExifTag' in self.exif_image.keys():
            offset = self.exif_image['ExifTag']
            photo = _ExifPhotoIfd(self.endian, buffer[6:], offset)
            self.exif_photo = photo.processed_ifd

            if 'InteroperabilityTag' in self.exif_photo.keys():
                offset = self.exif_photo['InteroperabilityTag']
                interop = _ExifInteroperabilityIfd(self.endian,
                                                   buffer[6:],
                                                   offset)
                self.iop = interop.processed_ifd

        if 'GPSTag' in self.exif_image.keys():
            offset = self.exif_image['GPSTag']
            gps = _ExifGPSInfoIfd(self.endian, buffer[6:], offset)
            self.exif_gpsinfo = gps.processed_ifd


class _Ifd(object):
    """
    Attributes
    ----------
    buffer : bytes
        Raw byte stream consisting of the UUID data.
    datatype2fmt : dictionary
        Class attribute, maps the TIFF enumerated datatype to the python
        datatype and data width.
    endian : str
        Either '<' for big-endian, or '>' for little-endian.
    num_tags : int
        Number of tags in the IFD.
    raw_ifd : dictionary
        Maps tag number to "mildly-interpreted" tag value.
    processed_ifd : dictionary
        Maps tag name to "mildly-interpreted" tag value.
    """
    datatype2fmt = {1: ('B', 1),
                    2: ('B', 1),
                    3: ('H', 2),
                    4: ('I', 4),
                    5: ('II', 8),
                    7: ('B', 1),
                    9: ('i', 4),
                    10: ('ii', 8)}

    def __init__(self, endian, buffer, offset):
        self.endian = endian
        self.buffer = buffer
        self.processed_ifd = collections.OrderedDict()

        self.num_tags, = struct.unpack(endian + 'H',
                                       buffer[offset:offset + 2])

        fmt = self.endian + 'HHII' * self.num_tags
        ifd_buffer = buffer[offset + 2:offset + 2 + self.num_tags * 12]
        data = struct.unpack(fmt, ifd_buffer)
        self.raw_ifd = collections.OrderedDict()
        for j, tag in enumerate(data[0::4]):
            # The offset to the tag offset/payload is the offset to the IFD
            # plus 2 bytes for the number of tags plus 12 bytes for each
            # tag entry plus 8 bytes to the offset/payload itself.
            toffp = buffer[offset + 10 + j * 12:offset + 10 + j * 12 + 4]
            tag_data = self.parse_tag(data[j * 4 + 1],
                                      data[j * 4 + 2],
                                      toffp)
            self.raw_ifd[tag] = tag_data

    def parse_tag(self, dtype, count, offset_buf):
        """Interpret an Exif image tag data payload.
        """
        fmt = self.datatype2fmt[dtype][0] * count
        payload_size = self.datatype2fmt[dtype][1] * count

        if payload_size <= 4:
            # Interpret the payload from the 4 bytes in the tag entry.
            target_buffer = offset_buf[:payload_size]
        else:
            # Interpret the payload at the offset specified by the 4 bytes in
            # the tag entry.
            offset, = struct.unpack(self.endian + 'I', offset_buf)
            target_buffer = self.buffer[offset:offset + payload_size]

        if dtype == 2:
            # ASCII
            if sys.hexversion < 0x03000000:
                payload = target_buffer.rstrip('\x00')
            else:
                payload = target_buffer.decode('utf-8').rstrip('\x00')

        else:
            payload = struct.unpack(self.endian + fmt, target_buffer)
            if dtype == 5 or dtype == 10:
                # Rational or Signed Rational.  Construct the list of values.
                rational_payload = []
                for j in range(count):
                    value = float(payload[j * 2]) / float(payload[j * 2 + 1])
                    rational_payload.append(value)
                payload = rational_payload
            if count == 1:
                # If just a single value, then return a scalar instead of a
                # tuple.
                payload = payload[0]

        return payload

    def post_process(self, tagnum2name):
        for tag, value in self.raw_ifd.items():
            try:
                tag_name = tagnum2name[tag]
            except KeyError:
                # Ok, we don't recognize this tag.  Just use the numeric id.
                msg = 'Unrecognized Exif tag "{0}".'.format(tag)
                warnings.warn(msg, UserWarning)
                tag_name = tag
            self.processed_ifd[tag_name] = value


class _ExifImageIfd(_Ifd):
    """
    Attributes
    ----------
    tagnum2name : dict
        Maps Exif image tag numbers to the tag names.
    ifd : dict
        Maps tag names to tag values.
    """
    tagnum2name = {11: 'ProcessingSoftware',
                   254: 'NewSubfileType',
                   255: 'SubfileType',
                   256: 'ImageWidth',
                   257: 'ImageLength',
                   258: 'BitsPerSample',
                   259: 'Compression',
                   262: 'PhotometricInterpretation',
                   263: 'Threshholding',
                   264: 'CellWidth',
                   265: 'CellLength',
                   266: 'FillOrder',
                   269: 'DocumentName',
                   270: 'ImageDescription',
                   271: 'Make',
                   272: 'Model',
                   273: 'StripOffsets',
                   274: 'Orientation',
                   277: 'SamplesPerPixel',
                   278: 'RowsPerStrip',
                   279: 'StripByteCounts',
                   282: 'XResolution',
                   283: 'YResolution',
                   284: 'PlanarConfiguration',
                   290: 'GrayResponseUnit',
                   291: 'GrayResponseCurve',
                   292: 'T4Options',
                   293: 'T6Options',
                   296: 'ResolutionUnit',
                   301: 'TransferFunction',
                   305: 'Software',
                   306: 'DateTime',
                   315: 'Artist',
                   316: 'HostComputer',
                   317: 'Predictor',
                   318: 'WhitePoint',
                   319: 'PrimaryChromaticities',
                   320: 'ColorMap',
                   321: 'HalftoneHints',
                   322: 'TileWidth',
                   323: 'TileLength',
                   324: 'TileOffsets',
                   325: 'TileByteCounts',
                   330: 'SubIFDs',
                   332: 'InkSet',
                   333: 'InkNames',
                   334: 'NumberOfInks',
                   336: 'DotRange',
                   337: 'TargetPrinter',
                   338: 'ExtraSamples',
                   339: 'SampleFormat',
                   340: 'SMinSampleValue',
                   341: 'SMaxSampleValue',
                   342: 'TransferRange',
                   343: 'ClipPath',
                   344: 'XClipPathUnits',
                   345: 'YClipPathUnits',
                   346: 'Indexed',
                   347: 'JPEGTables',
                   351: 'OPIProxy',
                   512: 'JPEGProc',
                   513: 'JPEGInterchangeFormat',
                   514: 'JPEGInterchangeFormatLength',
                   515: 'JPEGRestartInterval',
                   517: 'JPEGLosslessPredictors',
                   518: 'JPEGPointTransforms',
                   519: 'JPEGQTables',
                   520: 'JPEGDCTables',
                   521: 'JPEGACTables',
                   529: 'YCbCrCoefficients',
                   530: 'YCbCrSubSampling',
                   531: 'YCbCrPositioning',
                   532: 'ReferenceBlackWhite',
                   700: 'XMLPacket',
                   18246: 'Rating',
                   18249: 'RatingPercent',
                   32781: 'ImageID',
                   33421: 'CFARepeatPatternDim',
                   33422: 'CFAPattern',
                   33423: 'BatteryLevel',
                   33432: 'Copyright',
                   33434: 'ExposureTime',
                   33437: 'FNumber',
                   33723: 'IPTCNAA',
                   34377: 'ImageResources',
                   34665: 'ExifTag',
                   34675: 'InterColorProfile',
                   34850: 'ExposureProgram',
                   34852: 'SpectralSensitivity',
                   34853: 'GPSTag',
                   34855: 'ISOSpeedRatings',
                   34856: 'OECF',
                   34857: 'Interlace',
                   34858: 'TimeZoneOffset',
                   34859: 'SelfTimerMode',
                   36867: 'DateTimeOriginal',
                   37122: 'CompressedBitsPerPixel',
                   37377: 'ShutterSpeedValue',
                   37378: 'ApertureValue',
                   37379: 'BrightnessValue',
                   37380: 'ExposureBiasValue',
                   37381: 'MaxApertureValue',
                   37382: 'SubjectDistance',
                   37383: 'MeteringMode',
                   37384: 'LightSource',
                   37385: 'Flash',
                   37386: 'FocalLength',
                   37387: 'FlashEnergy',
                   37388: 'SpatialFrequencyResponse',
                   37389: 'Noise',
                   37390: 'FocalPlaneXResolution',
                   37391: 'FocalPlaneYResolution',
                   37392: 'FocalPlaneResolutionUnit',
                   37393: 'ImageNumber',
                   37394: 'SecurityClassification',
                   37395: 'ImageHistory',
                   37396: 'SubjectLocation',
                   37397: 'ExposureIndex',
                   37398: 'TIFFEPStandardID',
                   37399: 'SensingMethod',
                   40091: 'XPTitle',
                   40092: 'XPComment',
                   40093: 'XPAuthor',
                   40094: 'XPKeywords',
                   40095: 'XPSubject',
                   50341: 'PrintImageMatching',
                   50706: 'DNGVersion',
                   50707: 'DNGBackwardVersion',
                   50708: 'UniqueCameraModel',
                   50709: 'LocalizedCameraModel',
                   50710: 'CFAPlaneColor',
                   50711: 'CFALayout',
                   50712: 'LinearizationTable',
                   50713: 'BlackLevelRepeatDim',
                   50714: 'BlackLevel',
                   50715: 'BlackLevelDeltaH',
                   50716: 'BlackLevelDeltaV',
                   50717: 'WhiteLevel',
                   50718: 'DefaultScale',
                   50719: 'DefaultCropOrigin',
                   50720: 'DefaultCropSize',
                   50721: 'ColorMatrix1',
                   50722: 'ColorMatrix2',
                   50723: 'CameraCalibration1',
                   50724: 'CameraCalibration2',
                   50725: 'ReductionMatrix1',
                   50726: 'ReductionMatrix2',
                   50727: 'AnalogBalance',
                   50728: 'AsShotNeutral',
                   50729: 'AsShotWhiteXY',
                   50730: 'BaselineExposure',
                   50731: 'BaselineNoise',
                   50732: 'BaselineSharpness',
                   50733: 'BayerGreenSplit',
                   50734: 'LinearResponseLimit',
                   50735: 'CameraSerialNumber',
                   50736: 'LensInfo',
                   50737: 'ChromaBlurRadius',
                   50738: 'AntiAliasStrength',
                   50739: 'ShadowScale',
                   50740: 'DNGPrivateData',
                   50741: 'MakerNoteSafety',
                   50778: 'CalibrationIlluminant1',
                   50779: 'CalibrationIlluminant2',
                   50780: 'BestQualityScale',
                   50781: 'RawDataUniqueID',
                   50827: 'OriginalRawFileName',
                   50828: 'OriginalRawFileData',
                   50829: 'ActiveArea',
                   50830: 'MaskedAreas',
                   50831: 'AsShotICCProfile',
                   50832: 'AsShotPreProfileMatrix',
                   50833: 'CurrentICCProfile',
                   50834: 'CurrentPreProfileMatrix',
                   50879: 'ColorimetricReference',
                   50931: 'CameraCalibrationSignature',
                   50932: 'ProfileCalibrationSignature',
                   50934: 'AsShotProfileName',
                   50935: 'NoiseReductionApplied',
                   50936: 'ProfileName',
                   50937: 'ProfileHueSatMapDims',
                   50938: 'ProfileHueSatMapData1',
                   50939: 'ProfileHueSatMapData2',
                   50940: 'ProfileToneCurve',
                   50941: 'ProfileEmbedPolicy',
                   50942: 'ProfileCopyright',
                   50964: 'ForwardMatrix1',
                   50965: 'ForwardMatrix2',
                   50966: 'PreviewApplicationName',
                   50967: 'PreviewApplicationVersion',
                   50968: 'PreviewSettingsName',
                   50969: 'PreviewSettingsDigest',
                   50970: 'PreviewColorSpace',
                   50971: 'PreviewDateTime',
                   50972: 'RawImageDigest',
                   50973: 'OriginalRawFileDigest',
                   50974: 'SubTileBlockSize',
                   50975: 'RowInterleaveFactor',
                   50981: 'ProfileLookTableDims',
                   50982: 'ProfileLookTableData',
                   51008: 'OpcodeList1',
                   51009: 'OpcodeList2',
                   51022: 'OpcodeList3',
                   51041: 'NoiseProfile'}

    def __init__(self, endian, buffer, offset):
        _Ifd.__init__(self, endian, buffer, offset)
        self.post_process(self.tagnum2name)


class _ExifPhotoIfd(_Ifd):
    tagnum2name = {33434: 'ExposureTime',
                   33437: 'FNumber',
                   34850: 'ExposureProgram',
                   34852: 'SpectralSensitivity',
                   34855: 'ISOSpeedRatings',
                   34856: 'OECF',
                   34864: 'SensitivityType',
                   34865: 'StandardOutputSensitivity',
                   34866: 'RecommendedExposureIndex',
                   34867: 'ISOSpeed',
                   34868: 'ISOSpeedLatitudeyyy',
                   34869: 'ISOSpeedLatitudezzz',
                   36864: 'ExifVersion',
                   36867: 'DateTimeOriginal',
                   36868: 'DateTimeDigitized',
                   37121: 'ComponentsConfiguration',
                   37122: 'CompressedBitsPerPixel',
                   37377: 'ShutterSpeedValue',
                   37378: 'ApertureValue',
                   37379: 'BrightnessValue',
                   37380: 'ExposureBiasValue',
                   37381: 'MaxApertureValue',
                   37382: 'SubjectDistance',
                   37383: 'MeteringMode',
                   37384: 'LightSource',
                   37385: 'Flash',
                   37386: 'FocalLength',
                   37396: 'SubjectArea',
                   37500: 'MakerNote',
                   37510: 'UserComment',
                   37520: 'SubSecTime',
                   37521: 'SubSecTimeOriginal',
                   37522: 'SubSecTimeDigitized',
                   40960: 'FlashpixVersion',
                   40961: 'ColorSpace',
                   40962: 'PixelXDimension',
                   40963: 'PixelYDimension',
                   40964: 'RelatedSoundFile',
                   40965: 'InteroperabilityTag',
                   41483: 'FlashEnergy',
                   41484: 'SpatialFrequencyResponse',
                   41486: 'FocalPlaneXResolution',
                   41487: 'FocalPlaneYResolution',
                   41488: 'FocalPlaneResolutionUnit',
                   41492: 'SubjectLocation',
                   41493: 'ExposureIndex',
                   41495: 'SensingMethod',
                   41728: 'FileSource',
                   41729: 'SceneType',
                   41730: 'CFAPattern',
                   41985: 'CustomRendered',
                   41986: 'ExposureMode',
                   41987: 'WhiteBalance',
                   41988: 'DigitalZoomRatio',
                   41989: 'FocalLengthIn35mmFilm',
                   41990: 'SceneCaptureType',
                   41991: 'GainControl',
                   41992: 'Contrast',
                   41993: 'Saturation',
                   41994: 'Sharpness',
                   41995: 'DeviceSettingDescription',
                   41996: 'SubjectDistanceRange',
                   42016: 'ImageUniqueID',
                   42032: 'CameraOwnerName',
                   42033: 'BodySerialNumber',
                   42034: 'LensSpecification',
                   42035: 'LensMake',
                   42036: 'LensModel',
                   42037: 'LensSerialNumber'}

    def __init__(self, endian, buffer, offset):
        _Ifd.__init__(self, endian, buffer, offset)
        self.post_process(self.tagnum2name)


class _ExifGPSInfoIfd(_Ifd):
    tagnum2name = {0: 'GPSVersionID',
                   1: 'GPSLatitudeRef',
                   2: 'GPSLatitude',
                   3: 'GPSLongitudeRef',
                   4: 'GPSLongitude',
                   5: 'GPSAltitudeRef',
                   6: 'GPSAltitude',
                   7: 'GPSTimeStamp',
                   8: 'GPSSatellites',
                   9: 'GPSStatus',
                   10: 'GPSMeasureMode',
                   11: 'GPSDOP',
                   12: 'GPSSpeedRef',
                   13: 'GPSSpeed',
                   14: 'GPSTrackRef',
                   15: 'GPSTrack',
                   16: 'GPSImgDirectionRef',
                   17: 'GPSImgDirection',
                   18: 'GPSMapDatum',
                   19: 'GPSDestLatitudeRef',
                   20: 'GPSDestLatitude',
                   21: 'GPSDestLongitudeRef',
                   22: 'GPSDestLongitude',
                   23: 'GPSDestBearingRef',
                   24: 'GPSDestBearing',
                   25: 'GPSDestDistanceRef',
                   26: 'GPSDestDistance',
                   27: 'GPSProcessingMethod',
                   28: 'GPSAreaInformation',
                   29: 'GPSDateStamp',
                   30: 'GPSDifferential'}

    def __init__(self, endian, buffer, offset):
        _Ifd.__init__(self, endian, buffer, offset)
        self.post_process(self.tagnum2name)


class _ExifInteroperabilityIfd(_Ifd):
    tagnum2name = {1: 'InteroperabilityIndex',
                   2: 'InteroperabilityVersion',
                   4096: 'RelatedImageFileFormat',
                   4097: 'RelatedImageWidth',
                   4098: 'RelatedImageLength'}

    def __init__(self, endian, buffer, offset):
        _Ifd.__init__(self, endian, buffer, offset)
        self.post_process(self.tagnum2name)


# Map each box ID to the corresponding class.
_box_with_id = {
    'asoc': AssociationBox,
    'cdef': ComponentDefinitionBox,
    'cmap': ComponentMappingBox,
    'colr': ColourSpecificationBox,
    'jP  ': JPEG2000SignatureBox,
    'ftyp': FileTypeBox,
    'ihdr': ImageHeaderBox,
    'jp2h': JP2HeaderBox,
    'jp2c': ContiguousCodestreamBox,
    'lbl ': LabelBox,
    'pclr': PaletteBox,
    'res ': ResolutionBox,
    'resc': CaptureResolutionBox,
    'resd': DisplayResolutionBox,
    'rreq': ReaderRequirementsBox,
    'uinf': UUIDInfoBox,
    'ulst': UUIDListBox,
    'url ': DataEntryURLBox,
    'uuid': UUIDBox,
    'xml ': XMLBox}


def _indent(elem, level=0):
    """Recipe for pretty printing XML.  Please see

    http://effbot.org/zone/element-lib.htm#prettyprint
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _pretty_print_xml(xml, level=0):
    """Pretty print XML data.
    """
    xml = copy.deepcopy(xml)
    _indent(xml, level=level)
    xmltext = ET.tostring(xml).decode('utf-8')

    # Indent it a bit.
    lst = [('    ' + x) for x in xmltext.split('\n')]
    xml = '\n'.join(lst)
    return '\n{0}'.format(xml)
