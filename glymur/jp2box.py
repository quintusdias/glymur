"""Classes for individual JPEG 2000 boxes.

References
----------
.. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
   15444-1:2004 - Information technology -- JPEG 2000 image coding system:
   Core coding system"

.. [JP2K15444-2m] International Organization for Standardication.  ISO/IEC
   15444-2:2004 - Information technology -- JPEG 2000 image coding system:
   Extensions
"""

import copy
import math
import os
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


class Jp2kBox:
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
    color_space : int or None
        Enumerated colorspace, corresponds to one of 'sRGB', 'greyscale', or
        'YCC'.  If not None, then icc_profile must be None.
    icc_profile : byte array or None
        ICC profile according to ICC profile specification.  If not None, then
        color_space must be None.
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
        if self.color_space is not None:
            x = _colorspace_map_display[self.color_space]
            msg += '\n    Colorspace:  {0}'.format(x)
        else:
            x = len(self.icc_profile)
            msg += '\n    ICC Profile:  {0} bytes'.format(x)

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
            kwargs['color_space'], = struct.unpack('>I', buffer)
            kwargs['icc_profile'] = None

        else:
            # ICC profile
            kwargs['color_space'] = None
            n = offset + length - f.tell()
            kwargs['icc_profile'] = f.read(n)

        box = ColourSpecificationBox(**kwargs)
        return box


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
            segstr = segment.__str__()

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
            boxstr = box.__str__()

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
            boxstr = box.__str__()

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
            boxstr = box.__str__()

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
            box_str = box.__str__()

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
    data : bytes or ElementTree.Element
        Vendor-specific UUID data.  XMP UUIDs are interpreted as standard XML.
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
            kwargs['data'] = _parse_exif(buffer)
        else:
            kwargs['data'] = buffer
        box = UUIDBox(**kwargs)
        return box


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

_tagnum2name = {271: 'Make', 272: 'Model',
         282: 'XResolution', 283:  'YResolution',
         296: 'ResolutionUnit',
         531: 'YCbCrPositioning',
         34665: 'ExifTag',
         34853: 'GPSTag'}

def _parse_exif(buffer):
    """Interpret raw buffer consisting of Exif IFD.
    """
    # Ignore the first six bytes.
    # Next six should be (73, 73, 42, 8, numtags)
    data = struct.unpack('<BBHIH', buffer[6:16])
    num_tags = data[4]

    fmt = '<' + 'HHII' * num_tags
    data = struct.unpack(fmt, buffer[16:16 + num_tags * 12])
    exif = {}
    for j, tag in enumerate(data[0::4]):
        offset_bytes = buffer[16 + j * 12 + 8:16 + j * 12 + 8 + 4]
        exif[_tagnum2name[tag]] = _parse_exif_image_tag(data[j * 4 + 1],
                                                        data[j * 4 + 2],
                                                        offset_bytes,
                                                        buffer)
    print(exif)
    return exif 

# Map the TIFF enumerated datatype to the python datatype and data width.
_datatype2fmt = {1: ('B', 1),
                 2: ('B', 1),
                 3: ('H', 2),
                 4: ('I', 4),
                 5: ('II', 8),
                 7: ('B', 1),
                 9: ('i', 4),
                 10: ('ii', 8)}

def _parse_exif_image_tag(datatype, count, offset_buffer, exif_buffer):
    """Interpret an Exif image tag data payload.
    """
    fmt = _datatype2fmt[datatype][0] * count
    payload_size = _datatype2fmt[datatype][1] * count

    if payload_size <= 4:
        # Interpret the payload from the 4 bytes in the tag entry.
        target_buffer = offset_buffer[:payload_size]
    else:
        # Interpret the payload at the offset specified by the 4 bytes in the
        # tag entry.
        offset, = struct.unpack('<I', offset_buffer)
        target_buffer = exif_buffer[6 + offset:6 + offset + payload_size]

    if datatype == 2:
        payload = target_buffer.decode('utf-8').rstrip()
    else:
        payload = struct.unpack('<' + fmt, target_buffer)
        if datatype == 5:
            rational_payload = []
            for j in range(count):
                value = float(payload[j * 2]) / float(payload[j * 2 + 1])
                rational_payload.append(value)
            payload = rational_payload
        if count == 1:
            payload = payload[0]

    return payload                                            

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
