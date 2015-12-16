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

from collections import OrderedDict
import datetime
import io
import math
import os
import pprint
import struct
import sys
import textwrap
from uuid import UUID
import warnings

import lxml.etree as ET
import numpy as np
try:
    import gdal
    import osr
    _HAVE_GDAL = True
except ImportError:
    _HAVE_GDAL = False


from .codestream import Codestream
from .core import (_COLORSPACE_MAP_DISPLAY, _COLOR_TYPE_MAP_DISPLAY,
                   SRGB, GREYSCALE, YCC,
                   ENUMERATED_COLORSPACE, RESTRICTED_ICC_PROFILE,
                   ANY_ICC_PROFILE, VENDOR_COLOR_METHOD)

from . import _uuid_io


_METHOD_DISPLAY = {
    ENUMERATED_COLORSPACE: 'enumerated colorspace',
    RESTRICTED_ICC_PROFILE: 'restricted ICC profile',
    ANY_ICC_PROFILE: 'any ICC profile',
    VENDOR_COLOR_METHOD: 'vendor color method',
}


_APPROX_DISPLAY = {
    1: 'accurately represents correct colorspace definition',
    2: 'approximates correct colorspace definition, exceptional quality',
    3: 'approximates correct colorspace definition, reasonable quality',
    4: 'approximates correct colorspace definition, poor quality',
}

# Three different UUIDs are given special treatment.
_GEOTIFF_UUID = UUID('b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03')
_EXIF_UUID = UUID(bytes=b'JpgTiffExif->JP2')
_XMP_UUID = UUID('be7acfcb-97a9-42e8-9c71-999491e3afac')


class Jp2kBox(object):
    """Superclass for JPEG 2000 boxes.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    box : list
        List of JPEG 2000 boxes.
    """

    def __init__(self, offset=0, length=0):
        self.length = length
        self.offset = offset
        self.box = []

    def __repr__(self):
        msg = "glymur.jp2box.Jp2kBox(box_id='{id}', offset={offset}, "
        msg += "length={length}, longname='{longname}')"
        msg = msg.format(id=self.box_id, offset=self.offset,
                         length=self.length, longname=self.longname)
        return msg

    def __str__(self):
        msg = "{0} Box ({1})".format(self.longname, self.box_id)
        msg += " @ ({0}, {1})".format(self.offset, self.length)
        return msg

    def _dispatch_validation_error(self, msg, writing=False):
        """Issue either a warning or an error depending on circumstance.

        If writing to file, then error out, as we do not wish to create bad
        JP2 files.  If reading, then we should be more lenient and just warn.
        """
        if writing:
            raise IOError(msg)
        else:
            warnings.warn(msg)

    def write(self, _):
        """Must be implemented in a subclass.
        """
        msg = "Not supported for {0} box.".format(self.longname)
        raise NotImplementedError(msg)

    def _str_superbox(self):
        """__str__ method for all superboxes."""
        msg = Jp2kBox.__str__(self)
        for box in self.box:
            boxstr = str(box)
            # Indent the child boxes to make the association clear.
            msg += '\n' + self._indent(boxstr)
        return msg

    def _indent(self, textstr, indent_level=4):
        """
        Indent a string.

        Textwrap's indent method only exists for 3.3 or above.  In 2.7 we have
        to fake it.

        Parameters
        ----------
        textstring : str
            String to be indented.
        indent_level : str
            Number of spaces of indentation to add.

        Returns
        -------
        str
            Possibly multi-line string indented by the specified amount.
        """
        if sys.hexversion >= 0x03030000:
            return textwrap.indent(textstr, ' ' * indent_level)
        else:
            lst = [(' ' * indent_level + x) for x in textstr.split('\n')]
            return '\n'.join(lst)

    def _write_superbox(self, fptr, box_id):
        """Write a superbox.

        Parameters
        ----------
        fptr : file or file object
            Superbox (box of boxes) to be written to this file.
        box_id : bytes
            4-byte sequence that identifies the superbox.
        """
        # Write the contained boxes, then come back and write the length.
        orig_pos = fptr.tell()
        fptr.write(struct.pack('>I4s', 0, box_id))
        for box in self.box:
            box.write(fptr)

        end_pos = fptr.tell()
        fptr.seek(orig_pos)
        fptr.write(struct.pack('>I', end_pos - orig_pos))
        fptr.seek(end_pos)

    def _parse_this_box(self, fptr, box_id, start, num_bytes):
        """Parse the current box.

        Parameters
        ----------
        fptr : file
            Open file object, currently points to start of box payload, not the
            start of the box.
        box_id : str
            4-letter identifier for the current box.
        start, num_bytes : int
            Byte offset and length of the current box.

        Returns
        -------
        Jp2kBox
            Object corresponding to the current box.
        """
        try:
            parser = _BOX_WITH_ID[box_id].parse

        except KeyError:
            # We don't recognize the box ID, so create an UnknownBox and be
            # done with it.
            msg = ('Unrecognized box ({box_id}) encountered at byte offset '
                   '{offset}.')
            msg = msg.format(box_id=box_id, offset=fptr.tell() - 8)
            warnings.warn(msg, UserWarning)
            box = UnknownBox(box_id, offset=start, length=num_bytes,
                             longname='Unknown')

            return box

        try:
            box = parser(fptr, start, num_bytes)
        except ValueError as err:
            msg = ("Encountered an unrecoverable ValueError while parsing a "
                   "{box_id} box at byte offset {offset}.  The original error "
                   "message was \"{original_error_message}\".")
            msg = msg.format(box_id=_BOX_WITH_ID[box_id].longname,
                             offset=start,
                             original_error_message=str(err))
            warnings.warn(msg, UserWarning)
            box = UnknownBox(box_id.decode('utf-8'),
                             length=num_bytes,
                             offset=start, longname='Unknown')

        return box

    def parse_superbox(self, fptr):
        """Parse a superbox (box consisting of nothing but other boxes.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        list
            List of top-level boxes in the JPEG 2000 file.
        """

        superbox = []

        start = fptr.tell()

        while True:

            # Are we at the end of the superbox?
            if start >= self.offset + self.length:
                break

            read_buffer = fptr.read(8)
            if len(read_buffer) < 8:
                msg = "Extra bytes at end of file ignored."
                warnings.warn(msg, UserWarning)
                return superbox

            (box_length, box_id) = struct.unpack('>I4s', read_buffer)
            if box_length == 0:
                # The length of the box is presumed to last until the end of
                # the file.  Compute the effective length of the box.
                num_bytes = os.path.getsize(fptr.name) - fptr.tell() + 8

            elif box_length == 1:
                # The length of the box is in the XL field, a 64-bit value.
                read_buffer = fptr.read(8)
                num_bytes, = struct.unpack('>Q', read_buffer)

            else:
                # The box_length value really is the length of the box!
                num_bytes = box_length

            box = self._parse_this_box(fptr, box_id, start, num_bytes)

            superbox.append(box)

            # Position to the start of the next box.
            if num_bytes > self.length:
                # Length of the current box goes past the end of the
                # enclosing superbox.
                msg = '{0} box has incorrect box length ({1})'
                msg = msg.format(box_id, num_bytes)
                warnings.warn(msg)
            elif fptr.tell() > start + num_bytes:
                # The box must be invalid somehow, as the file pointer is
                # positioned past the end of the box.
                msg = ('{box_id} box may be invalid, the file pointer is '
                       'positioned {num_bytes} bytes past the end of the box.')
                msg = msg.format(box_id=box_id,
                                 num_bytes=fptr.tell() - (start + num_bytes))
                warnings.warn(msg, UserWarning)
            fptr.seek(start + num_bytes)

            start += num_bytes

        return superbox


class ColourSpecificationBox(Jp2kBox):
    """Container for JPEG 2000 color specification box information.

    Attributes
    ----------
    box_id : str
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
    longname = 'Colour Specification'
    box_id = 'colr'

    def __init__(self, method=ENUMERATED_COLORSPACE, precedence=0,
                 approximation=0, colorspace=None, icc_profile=None,
                 length=0, offset=-1):
        Jp2kBox.__init__(self)

        self.method = method
        self.precedence = precedence
        self.approximation = approximation

        self.colorspace = colorspace
        self.icc_profile = icc_profile
        self.length = length
        self.offset = offset

        self._validate(writing=False)

    def _validate(self, writing=False):
        """Verify that the box obeys the specifications."""
        if self.colorspace is not None and self.icc_profile is not None:
            msg = ("Colorspace and icc_profile cannot both be set when "
                   "creating a ColourSpecificationBox.")
            self._dispatch_validation_error(msg, writing=writing)
        if self.method not in (1, 2, 3, 4):
            msg = "Invalid colorspace method value ({method})."
            msg = msg.format(method=self.method)
            if writing:
                raise IOError(msg)
            else:
                warnings.warn(msg, UserWarning)
        if self.approximation not in (0, 1, 2, 3, 4):
            msg = "Invalid colr approximation value ({approx})."
            msg = msg.format(approx=self.approximation)
            if writing:
                raise IOError(msg)
            else:
                warnings.warn(msg, UserWarning)

    def _write_validate(self):
        """In addition to constructor validation steps, run validation steps
        for writing."""
        if self.colorspace is None:
            msg = ("Writing colr boxes without enumerated "
                   "colorspaces is not supported at this time.")
            self._dispatch_validation_error(msg, writing=True)

        if self.icc_profile is None:
            if self.colorspace not in [SRGB, GREYSCALE, YCC]:
                msg = "Colorspace should correspond to one of SRGB, "
                msg += "GREYSCALE, or YCC."
                self._dispatch_validation_error(msg, writing=True)

        self._validate(writing=True)

    def __repr__(self):
        msg = "glymur.jp2box.ColourSpecificationBox("
        msg += "method={0}, precedence={1}, approximation={2}, "
        msg += "colorspace={3}, "
        msg += "icc_profile={4})"
        msg = msg.format(self.method,
                         self.precedence,
                         self.approximation,
                         self.colorspace,
                         self.icc_profile)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        text = 'Method:  {0}'.format(_METHOD_DISPLAY[self.method])
        lst.append(text)
        text = 'Precedence:  {0}'.format(self.precedence)
        lst.append(text)

        if self.approximation is not 0:
            dispvalue = _APPROX_DISPLAY[self.approximation]
            text = 'Approximation:  {0}'.format(dispvalue)
            lst.append(text)

        if self.colorspace is not None:
            try:
                dispvalue = _COLORSPACE_MAP_DISPLAY[self.colorspace]
            except KeyError:
                dispvalue = '{colorspace} (unrecognized)'
                dispvalue = dispvalue.format(colorspace=self.colorspace)
            text = 'Colorspace:  {0}'.format(dispvalue)
        else:
            # 2.7 has trouble pretty-printing ordered dicts so we just have
            # to print as a regular dict in this case.
            if self.icc_profile is None:
                text = 'ICC Profile:  None'
            else:
                if sys.hexversion < 0x03000000:
                    icc_profile = dict(self.icc_profile)
                else:
                    icc_profile = self.icc_profile
                text = pprint.pformat(icc_profile)
                text = self._indent(text)
                text = '\n'.join(['ICC Profile:', text])

        lst.append(text)

        text = '\n'.join(lst)

        text = '\n'.join([title, self._indent(text)])

        return text

    def write(self, fptr):
        """Write an Colour Specification box to file.
        """
        self._write_validate()
        length = 15 if self.icc_profile is None else 11 + len(self.icc_profile)
        fptr.write(struct.pack('>I4s', length, b'colr'))

        read_buffer = struct.pack('>BBBI',
                                  self.method,
                                  self.precedence,
                                  self.approximation,
                                  self.colorspace)
        fptr.write(read_buffer)

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse JPEG 2000 color specification box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        ColourSpecificationBox
            Instance of the current colour specification box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)

        lst = struct.unpack_from('>BBB', read_buffer, offset=0)
        method, precedence, approximation = lst

        if method == 1:
            # enumerated colour space
            colorspace, = struct.unpack_from('>I', read_buffer, offset=3)
            if colorspace not in _COLORSPACE_MAP_DISPLAY.keys():
                msg = "Unrecognized colorspace ({0}).".format(colorspace)
                warnings.warn(msg, UserWarning)
            icc_profile = None

        else:
            # ICC profile
            colorspace = None
            if (num_bytes - 3) < 128:
                msg = ("ICC profile header is corrupt, length is "
                       "only {length} when it should be at least 128.")
                warnings.warn(msg.format(length=num_bytes - 3), UserWarning)
                icc_profile = None
            else:
                profile = _ICCProfile(read_buffer[3:])
                icc_profile = profile.header

        return cls(method=method,
                   precedence=precedence,
                   approximation=approximation,
                   colorspace=colorspace,
                   icc_profile=icc_profile,
                   length=length,
                   offset=offset)


class _ICCProfile(object):
    """
    Container for ICC profile information.
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

    def __init__(self, read_buffer):
        self._raw_buffer = read_buffer
        header = OrderedDict()

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
        header['Datetime'] = datetime.datetime(data[0], data[1], data[2],
                                               data[3], data[4], data[5])
        header['File Signature'] = read_buffer[36:40].decode('utf-8')
        if read_buffer[40:44] == b'\x00\x00\x00\x00':
            header['Platform'] = 'unrecognized'
        else:
            header['Platform'] = read_buffer[40:44].decode('utf-8')

        fval, = struct.unpack('>I', read_buffer[44:48])
        flags = "{0}embedded, {1} be used independently"
        header['Flags'] = flags.format('' if fval & 0x01 else 'not ',
                                       'cannot' if fval & 0x02 else 'can')

        header['Device Manufacturer'] = read_buffer[48:52].decode('utf-8')
        if read_buffer[52:56] == b'\x00\x00\x00\x00':
            device_model = ''
        else:
            device_model = read_buffer[52:56].decode('utf-8')
        header['Device Model'] = device_model

        val, = struct.unpack('>Q', read_buffer[56:64])
        attr = "{0}, {1}, {2} media polarity, {3} media"
        attr = attr.format('transparency' if val & 0x01 else 'reflective',
                           'matte' if val & 0x02 else 'glossy',
                           'negative' if val & 0x04 else 'positive',
                           'black and white' if val & 0x08 else 'color')
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


class ChannelDefinitionBox(Jp2kBox):
    """Container for component definition box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : numeric scalar
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    index : list
        number of the channel.  Defaults to monotonically increasing sequence,
        i.e. [0, 1, 2, ...]
    channel_type : list
        type of the channel
    association : list
        index of the associated color
    """
    box_id = 'cdef'
    longname = 'Channel Definition'

    def __init__(self, channel_type, association, index=None, **kwargs):
        Jp2kBox.__init__(self)

        if index is None:
            self.index = tuple(range(len(channel_type)))
        else:
            self.index = tuple(index)

        self.channel_type = tuple(channel_type)
        self.association = tuple(association)
        self.__dict__.update(**kwargs)
        self._validate(writing=False)

    def _validate(self, writing=False):
        """Verify that the box obeys the specifications."""
        # channel type and association must be specified.
        if not ((len(self.index) == len(self.channel_type)) and
                (len(self.channel_type) == len(self.association))):
            msg = ("The length of the index ({index}), channel_type "
                   "({channel_type}), and association ({association}) inputs "
                   "must be the same.")
            msg = msg.format(index=len(self.index),
                             channel_type=len(self.channel_type),
                             association=len(self.association))
            self._dispatch_validation_error(msg, writing=writing)

        # channel types must be one of 0, 1, 2, 65535
        if any(x not in [0, 1, 2, 65535] for x in self.channel_type):
            msg = ("channel_type specified as {channel_type}, but all values "
                   "must be in the set of\n\n"
                   "    0     - colour image data for associated color\n"
                   "    1     - opacity\n"
                   "    2     - premultiplied opacity\n"
                   "    65535 - unspecified\n")
            msg = msg.format(channel_type=self.channel_type)
            self._dispatch_validation_error(msg, writing=writing)

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        for j in range(len(self.association)):
            color_type_string = _COLOR_TYPE_MAP_DISPLAY[self.channel_type[j]]
            if self.association[j] == 0:
                assn = 'whole image'
            else:
                assn = str(self.association[j])
            text = 'Channel {0} ({1}) ==> ({2})'.format(self.index[j],
                                                        color_type_string,
                                                        assn)
            lst.append(text)

        text = '\n'.join(lst)
        text = self._indent(text)
        text = '\n'.join([title, text])
        return text

    def __repr__(self):
        msg = "glymur.jp2box.ChannelDefinitionBox("
        msg += "index={0}, channel_type={1}, association={2})"
        msg = msg.format(self.index, self.channel_type, self.association)
        return msg

    def write(self, fptr):
        """Write a channel definition box to file.
        """
        self._validate(writing=True)
        num_components = len(self.association)
        fptr.write(struct.pack('>I4s', 8 + 2 + num_components * 6, b'cdef'))
        fptr.write(struct.pack('>H', num_components))
        for j in range(num_components):
            fptr.write(struct.pack('>' + 'H' * 3,
                                   self.index[j],
                                   self.channel_type[j],
                                   self.association[j]))

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse component definition box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        ComponentDefinitionBox
            Instance of the current component definition box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)

        # Read the number of components.
        num_components, = struct.unpack_from('>H', read_buffer)

        data = struct.unpack_from('>' + 'HHH' * num_components, read_buffer,
                                  offset=2)
        index = data[0:num_components * 6:3]
        channel_type = data[1:num_components * 6:3]
        association = data[2:num_components * 6:3]

        return cls(index=tuple(index),
                   channel_type=tuple(channel_type),
                   association=tuple(association),
                   length=length, offset=offset)


class CodestreamHeaderBox(Jp2kBox):
    """Container for codestream header box information.

    Attributes
    ----------
    box_id : str
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
    box_id = 'jpch'
    longname = 'Codestream Header'

    def __init__(self, box=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset
        self.box = box if box is not None else []

    def __repr__(self):
        msg = "glymur.jp2box.CodestreamHeaderBox(box={0})".format(self.box)
        return msg

    def __str__(self):
        msg = self._str_superbox()
        return msg

    def write(self, fptr):
        """Write a codestream header box to file.
        """
        self._write_superbox(fptr, b'jpch')

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse codestream header box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        CodestreamHeaderBox
            Instance of the current codestream header box.
        """
        box = cls(length=length, offset=offset)

        # The codestream header box is a superbox, so go ahead and parse its
        # child boxes.
        box.box = box.parse_superbox(fptr)

        return box


class ColourGroupBox(Jp2kBox):
    """Container for colour group box information.

    Attributes
    ----------
    box_id : str
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
    box_id = 'cgrp'
    longname = 'Colour Group'

    def __init__(self, box=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset
        self.box = box if box is not None else []

    def __repr__(self):
        msg = "glymur.jp2box.ColourGroupBox(box={0})".format(self.box)
        return msg

    def __str__(self):
        msg = self._str_superbox()
        return msg

    def _validate(self, writing=True):
        """Verify that the box obeys the specifications."""
        if any([box.box_id != 'colr' for box in self.box]):
            msg = "Colour group boxes can only contain colour specification "
            msg += "boxes."
            self._dispatch_validation_error(msg, writing=writing)

    def write(self, fptr):
        """Write a colour group box to file.
        """
        self._validate(writing=True)
        self._write_superbox(fptr, b'cgrp')

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse colour group box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        ColourGroupBox
            Instance of the current colour group box.
        """
        box = cls(length=length, offset=offset)

        # The colour group box is a superbox, so go ahead and parse its
        # child boxes.
        box.box = box.parse_superbox(fptr)

        return box


class CompositingLayerHeaderBox(Jp2kBox):
    """Container for compositing layer header box information.

    Attributes
    ----------
    box_id : str
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
    box_id = 'jplh'
    longname = 'Compositing Layer Header'

    def __init__(self, box=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset
        self.box = box if box is not None else []

    def __repr__(self):
        msg = "glymur.jp2box.CompositingLayerHeaderBox(box={0})"
        msg = msg.format(self.box)
        return msg

    def __str__(self):
        msg = self._str_superbox()
        return msg

    def write(self, fptr):
        """Write a compositing layer header box to file.
        """
        self._write_superbox(fptr, b'jplh')

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse compositing layer header box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        CompositingLayerHeaderBox
            Instance of the current compositing layer header box.
        """
        box = cls(length=length, offset=offset)

        # This box is a superbox, so go ahead and parse its # child boxes.
        box.box = box.parse_superbox(fptr)

        return box


class ComponentMappingBox(Jp2kBox):
    """Container for component mapping information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : numeric scalar
        Length of the box in bytes.
    offset : int
        Offset of the box from the start of the file.
    longname : str
        Verbose description of the box.
    component_index : tuple
        Index of component in codestream that is mapped to this channel.
    mapping_type : tuple
        mapping type, either direct use (0) or palette (1)
    palette_index : tuple
        Index component from palette
    """
    box_id = 'cmap'
    longname = 'Component Mapping'

    def __init__(self, component_index, mapping_type, palette_index,
                 length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.component_index = component_index
        self.mapping_type = mapping_type
        self.palette_index = palette_index
        self.length = length
        self.offset = offset

    def __repr__(self):
        msg = "glymur.jp2box.ComponentMappingBox("
        msg += "component_index={0}, mapping_type={1}, palette_index={2})"
        msg = msg.format(self.component_index,
                         self.mapping_type,
                         self.palette_index)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        for k in range(len(self.component_index)):
            if self.mapping_type[k] == 1:
                text = 'Component {0} ==> palette column {1}'
                text = text.format(self.component_index[k],
                                   self.palette_index[k])
            else:
                text = 'Component {0} ==> {1}'
                text = text.format(self.component_index[k], k)
            lst.append(text)

        text = '\n'.join(lst)
        text = self._indent(text)
        text = '\n'.join([title, text])

        return text

    def write(self, fptr):
        """Write a Component Mapping box to file.
        """
        length = 8 + 4 * len(self.component_index)
        write_buffer = struct.pack('>I4s', length, b'cmap')
        fptr.write(write_buffer)

        for j in range(len(self.component_index)):
            write_buffer = struct.pack('>HBB',
                                       self.component_index[j],
                                       self.mapping_type[j],
                                       self.palette_index[j])
            fptr.write(write_buffer)

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse component mapping box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        ComponentMappingBox
            Instance of the current component mapping box.
        """
        num_bytes = offset + length - fptr.tell()
        num_components = int(num_bytes / 4)

        read_buffer = fptr.read(num_bytes)
        data = struct.unpack('>' + 'HBB' * num_components, read_buffer)

        component_index = data[0:num_bytes:3]
        mapping_type = data[1:num_bytes:3]
        palette_index = data[2:num_bytes:3]

        return cls(component_index, mapping_type, palette_index,
                   length=length, offset=offset)


class ContiguousCodestreamBox(Jp2kBox):
    """Container for JPEG2000 codestream information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    codestream : Codestream object
        Contains list of codestream marker/segments.  By default, only the main
        header is retrieved.
    main_header_offset : int
        offset of main header from start of file
    """
    box_id = 'jp2c'
    longname = 'Contiguous Codestream'

    def __init__(self, codestream=None, main_header_offset=None, length=0,
                 offset=-1):
        Jp2kBox.__init__(self)
        self._codestream = codestream
        self.length = length
        self.offset = offset
        self.main_header_offset = main_header_offset

        # The filename can be set if lazy loading is desired.
        self._filename = None

    @property
    def codestream(self):
        if _parseoptions['full_codestream'] is True:
            header_only = False
        else:
            header_only = True
        if self._codestream is None:
            if self._filename is not None:
                with open(self._filename, 'rb') as fptr:
                    fptr.seek(self.main_header_offset)
                    codestream = Codestream(fptr, self._length,
                                            header_only=header_only)
                    self._codestream = codestream
        return self._codestream

    def __repr__(self):
        msg = "glymur.jp2box.ContiguousCodeStreamBox(codestream={0})"
        return msg.format(repr(self.codestream))

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title
        if _printoptions['codestream'] is False:
            return title

        lst = []
        for segment in self.codestream.segment:
            lst.append(str(segment))

        text = '\n'.join(lst)
        text = self._indent(text)
        text = '\n'.join([title, text])
        return text

    @classmethod
    def parse(cls, fptr, offset=0, length=0):
        """Parse a codestream box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        ContiguousCodestreamBox
            Instance of the current contiguous codestream box.
        """
        main_header_offset = fptr.tell()
        if _parseoptions['full_codestream'] is True:
            codestream = Codestream(fptr, length, header_only=False)
        else:
            codestream = None
        box = cls(codestream, main_header_offset=main_header_offset,
                  length=length, offset=offset)
        box._filename = fptr.name
        box._length = length
        return box


class DataReferenceBox(Jp2kBox):
    """Container for Data Reference box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    DR : list
        Data Entry URL boxes.
    """
    box_id = 'dtbl'
    longname = 'Data Reference'

    def __init__(self, data_entry_url_boxes=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        if data_entry_url_boxes is None:
            self.DR = []
        else:
            self.DR = data_entry_url_boxes
        self.length = length
        self.offset = offset
        self._validate(writing=False)

    def _validate(self, writing=False):
        """Verify that the box obeys the specifications."""
        for box in self.DR:
            if box.box_id != 'url ':
                msg = ('Child boxes of a data reference box can only be data '
                       'entry URL boxes.')
                self._dispatch_validation_error(msg, writing=writing)

    def _write_validate(self):
        """Verify that the box obeys the specifications for writing.
        """
        if len(self.DR) == 0:
            msg = ("A data reference box cannot be empty when written to a "
                   "file.")
            self._dispatch_validation_error(msg, writing=True)
        self._validate(writing=True)

    def write(self, fptr):
        """Write a Data Reference box to file.
        """
        self._write_validate()

        # Very similar to the way a superbox is written.
        orig_pos = fptr.tell()
        fptr.write(struct.pack('>I4s', 0, b'dtbl'))

        # Write the number of data entry url boxes.
        write_buffer = struct.pack('>H', len(self.DR))
        fptr.write(write_buffer)

        for box in self.DR:
            box.write(fptr)

        end_pos = fptr.tell()
        fptr.seek(orig_pos)
        fptr.write(struct.pack('>I', end_pos - orig_pos))
        fptr.seek(end_pos)

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        for box in self.DR:
            lst.append(str(box))
        text = '\n'.join(lst)
        text = self._indent(text)

        text = '\n'.join([title, text])
        return text

    def __repr__(self):
        msg = 'glymur.jp2box.DataReferenceBox()'
        return msg

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse data reference box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        DataReferenceBox
            Instance of the current data reference box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)

        # Read the number of data references
        ndr, = struct.unpack_from('>H', read_buffer, offset=0)

        # Need to keep track of where the next url box starts.
        box_offset = 2

        data_entry_url_box_list = []
        for j in range(ndr):

            # Create an in-memory binary stream for each URL box.
            box_fptr = io.BytesIO(read_buffer[box_offset:])
            box_buffer = box_fptr.read(8)
            (box_length, box_id) = struct.unpack_from('>I4s', box_buffer,
                                                      offset=0)
            box = DataEntryURLBox.parse(box_fptr, 0, box_length)

            # Need to adjust the box start to that of the "real" file.
            box.offset = offset + 8 + box_offset
            data_entry_url_box_list.append(box)

            # Point to the next embedded URL box.
            box_offset += box_length

        return cls(data_entry_url_box_list, length=length, offset=offset)


class FileTypeBox(Jp2kBox):
    """Container for JPEG 2000 file type box information.

    Attributes
    ----------
    box_id : str
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
    box_id = 'ftyp'
    longname = 'File Type'
    _valid_cls = ['jp2 ', 'jpx ', 'jpxb']

    def __init__(self, brand='jp2 ', minor_version=0,
                 compatibility_list=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.brand = brand
        self.minor_version = minor_version
        if compatibility_list is None:
            self.compatibility_list = ['jp2 ']
        else:
            self.compatibility_list = compatibility_list
        self.length = length
        self.offset = offset
        self._validate(writing=False)

    def __repr__(self):
        msg = "glymur.jp2box.FileTypeBox(brand='{0}', minor_version={1}, "
        msg += "compatibility_list={2})"
        msg = msg.format(self.brand, self.minor_version,
                         self.compatibility_list)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        text = 'Brand:  {0}'.format(self.brand)
        lst.append(text)
        text = 'Compatibility:  {0}'.format(self.compatibility_list)
        lst.append(text)

        text = '\n'.join(lst)
        text = self._indent(text)

        text = '\n'.join([title, text])

        return text

    def _validate(self, writing=False):
        """
        Validate the box before writing to file.
        """
        if self.brand not in ['jp2 ', 'jpx ']:
            msg = ("The file type brand was '{brand}'.  "
                   "It should be either 'jp2 ' or 'jpx '.")
            msg = msg.format(brand=self.brand)
            if writing:
                raise IOError(msg)
            else:
                warnings.warn(msg, UserWarning)
        for item in self.compatibility_list:
            if item not in self._valid_cls:
                msg = ("The file type compatibility list {items} is "
                       "not valid.  All items should be members of "
                       "{valid_entries}.")
                msg = msg.format(items=self.compatibility_list,
                                 valid_entries=self._valid_cls)
                if writing:
                    raise IOError(msg)
                else:
                    warnings.warn(msg, UserWarning)

    def write(self, fptr):
        """Write a File Type box to file.
        """
        self._validate(writing=True)
        length = 16 + 4 * len(self.compatibility_list)
        fptr.write(struct.pack('>I4s', length, b'ftyp'))
        fptr.write(self.brand.encode())
        fptr.write(struct.pack('>I', self.minor_version))

        for item in self.compatibility_list:
            fptr.write(item.encode())

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse JPEG 2000 file type box.

        Parameters
        ----------
        f : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        FileTypeBox
            Instance of the current file type box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)
        # Extract the brand, minor version.
        (brand, minor_version) = struct.unpack_from('>4sI', read_buffer, 0)
        if sys.hexversion >= 0x030000:
            brand = brand.decode('utf-8')

        # Extract the compatibility list.  Each entry has 4 bytes.
        num_entries = int((length - 16) / 4)
        compatibility_list = []
        for j in range(int(num_entries)):
            entry, = struct.unpack_from('>4s', read_buffer, 8 + j * 4)
            if sys.hexversion >= 0x03000000:
                try:
                    entry = entry.decode('utf-8')
                except UnicodeDecodeError:
                    # The entry is invalid, but we've got code to catch this
                    # later on.
                    pass

            compatibility_list.append(entry)

        return cls(brand=brand, minor_version=minor_version,
                   compatibility_list=compatibility_list,
                   length=length, offset=offset)


class FragmentListBox(Jp2kBox):
    """Container for JPX fragment list box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    """
    box_id = 'flst'
    longname = 'Fragment List'

    def __init__(self, fragment_offset, fragment_length, data_reference,
                 length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.fragment_offset = fragment_offset
        self.fragment_length = fragment_length
        self.data_reference = data_reference
        self.length = length
        self.offset = offset
        self._validate(writing=False)

    def _validate(self, writing=False):
        """Validate internal correctness."""
        if (((len(self.fragment_offset) != len(self.fragment_length)) or
             (len(self.fragment_length) != len(self.data_reference)))):
            msg = ("The lengths of the fragment offsets ({len_offsets}), "
                   "fragment lengths ({len_fragments}), and "
                   "data reference items ({len_drefs}) must be the same.")
            msg = msg.format(len_offsets=len(self.fragment_offset),
                             len_fragments=len(self.fragment_length),
                             len_drefs=len(self.data_reference))
            self._dispatch_validation_error(msg, writing=writing)
        if any([x <= 0 for x in self.fragment_offset]):
            msg = "Fragment offsets must all be positive."
            self._dispatch_validation_error(msg, writing=writing)
        if any([x <= 0 for x in self.fragment_length]):
            msg = "Fragment lengths must all be positive."
            self._dispatch_validation_error(msg, writing=writing)

    def __repr__(self):
        msg = "glymur.jp2box.FragmentListBox({0}, {1}, {2})"
        msg = msg.format(str(self.fragment_offset),
                         str(self.fragment_length),
                         str(self.data_reference))
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        for j in range(len(self.fragment_offset)):
            text = "Offset {0}:  {1}".format(j, self.fragment_offset[j])
            lst.append(text)
            text = "Fragment Length {0}:  {1}".format(j,
                                                      self.fragment_length[j])
            lst.append(text)
            text = "Data Reference {0}:  {1}".format(j, self.data_reference[j])
            lst.append(text)

        text = '\n'.join(lst)
        text = self._indent(text)
        text = '\n'.join([title, text])
        return text

    def write(self, fptr):
        """Write a fragment list box to file.
        """
        self._validate(writing=True)
        num_items = len(self.fragment_offset)
        length = 8 + 2 + num_items * 14
        fptr.write(struct.pack('>I4s', length, b'flst'))
        fptr.write(struct.pack('>H', num_items))
        for j in range(num_items):
            write_buffer = struct.pack('>QIH',
                                       self.fragment_offset[j],
                                       self.fragment_length[j],
                                       self.data_reference[j])
            fptr.write(write_buffer)

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse JPX free box.

        Parameters
        ----------
        f : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        FragmentListBox
            Instance of the current fragment list box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)
        num_fragments, = struct.unpack_from('>H', read_buffer, offset=0)

        lst = struct.unpack_from('>' + 'QIH' * num_fragments,
                                 read_buffer,
                                 offset=2)
        frag_offset = lst[0::3]
        frag_len = lst[1::3]
        data_reference = lst[2::3]
        return cls(frag_offset, frag_len, data_reference,
                   length=length, offset=offset)


class FragmentTableBox(Jp2kBox):
    """Container for JPX fragment table box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    """
    box_id = 'ftbl'
    longname = 'Fragment Table'

    def __init__(self, box=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset
        self.box = box if box is not None else []

    def __repr__(self):
        msg = "glymur.jp2box.FragmentTableBox(box={0})"
        if len(self.box) == 0:
            msg = msg.format(None)
        else:
            msg = msg.format(self.box)
        return msg

    def __str__(self):
        msg = self._str_superbox()
        return msg

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse JPX fragment table superbox box.

        Parameters
        ----------
        f : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        FragmentTableBox
            Instance of the current fragment table box.
        """
        box = cls(length=length, offset=offset)

        # The FragmentTable box is a superbox, so go ahead and parse its child
        # boxes.
        box.box = box.parse_superbox(fptr)

        return box

    def _validate(self, writing=False):
        """Self-validate the box before writing."""
        box_ids = [box.box_id for box in self.box]
        if len(box_ids) != 1 or box_ids[0] != 'flst':
            msg = "Fragment table boxes must have a single fragment list "
            msg += "box as a child box."
            self._dispatch_validation_error(msg, writing=writing)

    def write(self, fptr):
        """Write a fragment table box to file.
        """
        self._validate(writing=True)
        self._write_superbox(fptr, b'ftbl')


class FreeBox(Jp2kBox):
    """Container for JPX free box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    """
    box_id = 'free'
    longname = 'Free'

    def __init__(self, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset

    def __repr__(self):
        msg = "glymur.jp2box.FreeBox()"
        return msg

    def __str__(self):
        return Jp2kBox.__str__(self)

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse JPX free box.

        Parameters
        ----------
        f : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        FreeBox
            Instance of the current free box.
        """
        return cls(length=length, offset=offset)


class ImageHeaderBox(Jp2kBox):
    """Container for JPEG 2000 image header box information.

    Attributes
    ----------
    box_id : str
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
    compression : int
        The compression type, should be 7 if JP2.
    colorspace_unknown : bool
        False if the color space is known and correctly specified.
    ip_provided : bool
        False if the file does not contain intellectual propery rights
        information.
    """
    box_id = 'ihdr'
    longname = 'Image Header'

    def __init__(self, height, width, num_components=1, signed=False,
                 bits_per_component=8, compression=7, colorspace_unknown=False,
                 ip_provided=False, length=0, offset=-1):
        """
        Examples
        --------
        >>> import glymur
        >>> box = glymur.jp2box.ImageHeaderBox(height=512, width=256)
        """
        Jp2kBox.__init__(self)
        self.height = height
        self.width = width
        self.num_components = num_components
        self.signed = signed
        self.bits_per_component = bits_per_component
        self.compression = compression
        self.colorspace_unknown = colorspace_unknown
        self.ip_provided = ip_provided
        self.length = length
        self.offset = offset

    def __repr__(self):
        msg = "glymur.jp2box.ImageHeaderBox("
        msg += "{height}, {width}, num_components={num_components}, "
        msg += "signed={signed}, bits_per_component={bits_per_component}, "
        msg += "compression={compression}, "
        msg += "colorspace_unknown={colorspace_unknown}, "
        msg += "ip_provided={ip_provided})"
        msg = msg.format(height=self.height, width=self.width,
                         num_components=self.num_components,
                         signed=self.signed,
                         bits_per_component=self.bits_per_component,
                         compression=self.compression,
                         colorspace_unknown=self.colorspace_unknown,
                         ip_provided=self.ip_provided)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []

        text = 'Size:  [{0} {1} {2}]'
        text = text.format(self.height, self.width, self.num_components)
        lst.append(text)

        text = 'Bitdepth:  {0}'.format(self.bits_per_component)
        lst.append(text)

        text = 'Signed:  {0}'.format(self.signed)
        lst.append(text)

        text = 'Compression:  {0}'
        text = text.format('wavelet' if self.compression == 7 else 'unknown')
        lst.append(text)

        text = 'Colorspace Unknown:  {0}'.format(self.colorspace_unknown)
        lst.append(text)

        text = '\n'.join(lst)
        text = self._indent(text)
        text = '\n'.join([title, text])

        return text

    def write(self, fptr):
        """Write an Image Header box to file.
        """
        fptr.write(struct.pack('>I4s', 22, b'ihdr'))

        # signedness and bps are stored together in a single byte
        bit_depth_signedness = 0x80 if self.signed else 0x00
        bit_depth_signedness |= self.bits_per_component - 1
        read_buffer = struct.pack('>IIHBBBB',
                                  self.height,
                                  self.width,
                                  self.num_components,
                                  bit_depth_signedness,
                                  self.compression,
                                  1 if self.colorspace_unknown else 0,
                                  1 if self.ip_provided else 0)
        fptr.write(read_buffer)

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse JPEG 2000 image header box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        ImageHeaderBox
            Instance of the current image header box.
        """
        # Read the box information
        read_buffer = fptr.read(14)
        params = struct.unpack('>IIHBBBB', read_buffer)
        height = params[0]
        width = params[1]
        num_components = params[2]
        bits_per_component = (params[3] & 0x7f) + 1
        signed = (params[3] & 0x80) > 1
        compression = params[4]
        colorspace_unknown = True if params[5] else False
        ip_provided = True if params[6] else False

        return cls(height, width, num_components=num_components,
                   bits_per_component=bits_per_component,
                   signed=signed,
                   compression=compression,
                   colorspace_unknown=colorspace_unknown,
                   ip_provided=ip_provided,
                   length=length, offset=offset)


class AssociationBox(Jp2kBox):
    """Container for Association box information.

    Attributes
    ----------
    box_id : str
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
    box_id = 'asoc'
    longname = 'Association'

    def __init__(self, box=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset
        self.box = box if box is not None else []

    def __repr__(self):
        msg = "glymur.jp2box.AssociationBox(box={0})".format(self.box)
        return msg

    def __str__(self):
        msg = self._str_superbox()
        return msg

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse association box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        AssociationBox
            Instance of the current association box.
        """
        box = cls(length=length, offset=offset)

        # The Association box is a superbox, so go ahead and parse its child
        # boxes.
        box.box = box.parse_superbox(fptr)

        return box

    def write(self, fptr):
        """Write an association box to file.
        """
        self._write_superbox(fptr, b'asoc')


class BitsPerComponentBox(Jp2kBox):
    """Container for bits per component box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    bpc : list
        bits per component for each component
    signed : list
        True if signed, false if not, for each component
    """
    box_id = 'bpcc'
    longname = 'Bits Per Component'

    def __init__(self, bpc, signed, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset
        self.bpc = bpc
        self.signed = signed

    def __repr__(self):
        msg = "glymur.jp2box.BitsPerComponentBox(box={0})".format(self.box)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        body = 'Bits per component:  ['
        body += ', '.join(str(x) for x in self.bpc)
        body += ']'
        body += '\n'
        body += 'Signed:  [' + ', '.join(str(x) for x in self.signed) + ']'

        body = self._indent(body)

        text = '\n'.join([title, body])
        return text

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse bits per component box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        BitsPerComponent
            Instance of the current bits per component box.
        """
        nbytes = length - 8
        data = fptr.read(nbytes)
        bpc = tuple(((x & 0x7f) + 1) for x in bytearray(data))
        signed = tuple(((x & 0x80) > 0) for x in bytearray(data))

        return cls(bpc, signed, length=length, offset=offset)


class JP2HeaderBox(Jp2kBox):
    """Container for JP2 header box information.

    Attributes
    ----------
    box_id : str
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
    box_id = 'jp2h'
    longname = 'JP2 Header'

    def __init__(self, box=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset
        self.box = box if box is not None else []

    def __repr__(self):
        msg = "glymur.jp2box.JP2HeaderBox(box={0})".format(self.box)
        return msg

    def __str__(self):
        msg = self._str_superbox()
        return msg

    def write(self, fptr):
        """Write a JP2 Header box to file.
        """
        self._write_superbox(fptr, b'jp2h')

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse JPEG 2000 header box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        JP2HeaderBox
            Instance of the current JP2 header box.
        """
        box = cls(length=length, offset=offset)

        # The JP2 header box is a superbox, so go ahead and parse its child
        # boxes.
        box.box = box.parse_superbox(fptr)

        return box


class JPEG2000SignatureBox(Jp2kBox):
    """Container for JPEG 2000 signature box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    signature : tuple
        Four-byte tuple identifying the file as JPEG 2000.
    """
    box_id = 'jP  '
    longname = 'JPEG 2000 Signature'

    def __init__(self, signature=(13, 10, 135, 10), length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.signature = signature
        self.length = length
        self.offset = offset

    def __repr__(self):
        return 'glymur.jp2box.JPEG2000SignatureBox()'

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        body = 'Signature:  {0:02x}{1:02x}{2:02x}{3:02x}'
        body = body.format(self.signature[0], self.signature[1],
                           self.signature[2], self.signature[3])
        body = self._indent(body)
        text = '\n'.join([title, body])
        return text

    def write(self, fptr):
        """Write a JPEG 2000 Signature box to file.
        """
        fptr.write(struct.pack('>I4s', 12, b'jP  '))
        fptr.write(struct.pack('>BBBB', *self.signature))

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse JPEG 2000 signature box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        JPEG2000SignatureBox
            Instance of the current JPEG2000 signature box.
        """
        read_buffer = fptr.read(4)
        signature = struct.unpack('>BBBB', read_buffer)

        return cls(signature=signature, length=length, offset=offset)


class PaletteBox(Jp2kBox):
    """Container for palette box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    palette : ndarray
        Colormap array.
    """
    longname = 'Palette'
    box_id = 'pclr'

    def __init__(self, palette, bits_per_component, signed, length=0,
                 offset=-1):
        Jp2kBox.__init__(self)
        self.palette = palette
        self.bits_per_component = bits_per_component
        self.signed = signed
        self.length = length
        self.offset = offset
        self._validate(writing=False)

    def _validate(self, writing=False):
        """Verify that the box obeys the specifications."""
        if ((len(self.bits_per_component) != len(self.signed)) or
                (len(self.signed) != self.palette.shape[1])):
            msg = "The length of the 'bits_per_component' and the 'signed' "
            msg += "members must equal the number of columns of the palette."
            self._dispatch_validation_error(msg, writing=writing)
        bps = self.bits_per_component
        if writing and not all(b == bps[0] for b in bps):
            # We don't support writing palettes with bit depths that are
            # different.
            msg = "Writing palettes with varying bit depths is not supported."
            self._dispatch_validation_error(msg, writing=writing)

    def __repr__(self):
        msg = "glymur.jp2box.PaletteBox({0}, bits_per_component={1}, "
        msg += "signed={2})"
        msg = msg.format(repr(self.palette), self.bits_per_component,
                         self.signed)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        body = 'Size:  ({0} x {1})'.format(*self.palette.shape)
        body = self._indent(body)

        text = '\n'.join([title, body])
        return text

    def write(self, fptr):
        """Write a Palette box to file.
        """
        self._validate(writing=True)
        bytes_per_row = sum(self.bits_per_component) / 8
        bytes_per_palette = bytes_per_row * self.palette.shape[0]
        box_length = 8 + 3 + self.palette.shape[1] + bytes_per_palette

        # Write the usual header.
        write_buffer = struct.pack('>I4s', int(box_length), b'pclr')
        fptr.write(write_buffer)

        write_buffer = struct.pack('>HB', self.palette.shape[0],
                                   self.palette.shape[1])
        fptr.write(write_buffer)

        bps_signed = [x - 1 for x in self.bits_per_component]
        for j, _ in enumerate(bps_signed):
            if self.signed[j]:
                bps_signed[j] |= 0x80
        write_buffer = struct.pack('>' + 'B' * self.palette.shape[1],
                                   *bps_signed)
        fptr.write(write_buffer)

        # All components are the same.  Writing is straightforward.
        if self.bits_per_component[0] <= 8:
            write_buffer = memoryview(self.palette.astype(np.uint8))
        elif self.bits_per_component[0] <= 16:
            write_buffer = memoryview(self.palette.astype(np.uint16))
        elif self.bits_per_component[0] <= 32:
            write_buffer = memoryview(self.palette.astype(np.uint32))
        fptr.write(write_buffer)

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse palette box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        PaletteBox
            Instance of the current palette box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)
        nrows, ncols = struct.unpack_from('>HB', read_buffer, offset=0)

        bps_signed = struct.unpack_from('>' + 'B' * ncols, read_buffer,
                                        offset=3)
        bps = [((x & 0x7f) + 1) for x in bps_signed]
        signed = [((x & 0x80) > 1) for x in bps_signed]

        if all(b == bps_signed[0] for b in bps_signed):
            # Ok the palette has the same datatype for all columns.  We should
            # be able to efficiently read it.
            if bps[0] <= 8:
                dtype = np.uint8
            elif bps[0] <= 16:
                dtype = np.uint16
            elif bps[0] <= 32:
                dtype = np.uint32

            palette = np.frombuffer(read_buffer[3 + ncols:], dtype=dtype)
            palette = np.reshape(palette, (nrows, ncols))

        else:
            # General case where the columns may not be the same width.
            fmt = '>'
            for bits in bps:
                if bits <= 8:
                    fmt += 'B'
                elif bits <= 16:
                    fmt += 'H'
                elif bits <= 32:
                    fmt += 'I'

            # Each palette component is padded out to the next largest byte.
            # That means a list comprehension does this in one shot.
            row_nbytes = sum([int(math.ceil(x / 8.0)) for x in bps])

            palette = np.zeros((nrows, ncols), dtype=np.int32)
            for j in range(nrows):
                poff = 3 + ncols + j * row_nbytes
                palette[j] = struct.unpack_from(fmt, read_buffer, offset=poff)

        return cls(palette, bps, signed, length=length, offset=offset)


# Map rreq codes to display text.
_READER_REQUIREMENTS_DISPLAY = {
    0: 'File not completely understood',
    1: 'Deprecated - contains no extensions',
    2: 'Contains multiple composition layers',
    3: ('Deprecated - codestream is compressed using JPEG 2000 and requires '
        'at least a Profile 0 decoder as defind in ITU-T Rec. T.800 '
        '| ISO/IEC 15444-1, A.10 Table A.45'),
    4: 'JPEG 2000 Part 1 Profile 1 codestream',
    5: ('Unrestricted JPEG 2000 Part 1 codestream, ITU-T Rec. T.800 '
        '| ISO/IEC 15444-1'),
    6: 'Unrestricted JPEG 2000 Part 2 codestream',
    7: 'JPEG codestream as defined in ISO/IEC 10918-1',
    8: 'Deprecated - does not contain opacity',
    9: 'Non-premultiplied opacity channel',
    10: 'Premultiplied opacity channel',
    11: 'Chroma-key based opacity',
    12: 'Deprecated - codestream is contiguous',
    13: 'Fragmented codestream where all fragments are in file and in order',
    14: ('Fragmented codestream where all fragments are in file '
         'but are out of order'),
    15: ('Fragmented codestream where not all fragments are within the file '
         'but are all in locally accessible files'),
    16: ('Fragmented codestream where some fragments may be accessible '
         'only through a URL specified network connection'),
    17: ('Compositing required to produce rendered result from multiple '
         'compositing layers'),
    18: 'Deprecated - support for compositing is not required',
    19: ('Deprecated - contains multiple, discrete layers that should not '
         'be combined through either animation or compositing'),
    20: ('Deprecated - compositing layers each contain only a single '
         'codestream'),
    21: 'At least one compositing layer consists of multiple codestreams',
    22: 'Deprecated - all compositing layers are in the same colourspace',
    23: ('Colourspace transformations are required to combine compositing '
         'layers; not all compositing layers are in the same colourspace'),
    24: 'Deprecated - rendered result created without using animation',
    25: ('Deprecated - animated, but first layer covers entire area and is '
         'opaque'),
    26: 'First animation layer does not cover entire rendered result',
    27: 'Deprecated - animated, and no layer is reused',
    28: 'Reuse of animation layers',
    29: 'Deprecated - animated, but layers are reused',
    30: 'Some animated frames are non-persistent',
    31: 'Deprecated - rendered result created without using scaling',
    32: 'Rendered result involves scaling within a layer',
    33: 'Rendered result involves scaling between layers',
    34: 'ROI metadata',
    35: 'IPR metadata',
    36: 'Content metadata',
    37: 'History metadata',
    38: 'Creation metadata',
    39: 'JPX digital signatures',
    40: 'JPX checksums',
    41: 'Desires Graphics Arts Reproduction specified',
    42: 'Deprecated - compositing layer uses palettized colour',
    43: 'Deprecated - compositing layer uses restricted ICC profile',
    44: 'Compositing layer uses Any ICC profile',
    45: 'Deprecated - compositing layer uses sRGB enumerated colourspace',
    46: 'Deprecated - compositing layer uses sRGB-grey enumerated colourspace',
    47: 'BiLevel 1 enumerated colourspace',
    48: 'BiLevel 2 enumerated colourspace',
    49: 'YCbCr 1 enumerated colourspace',
    50: 'YCbCr 2 enumerated colourspace',
    51: 'YCbCr 3 enumerated colourspace',
    52: 'PhotoYCC enumerated colourspace',
    53: 'YCCK enumerated colourspace',
    54: 'CMY enumerated colourspace',
    55: 'CMYK enumerated colorspace',
    56: 'CIELab enumerated colourspace with default parameters',
    57: 'CIELab enumerated colourspace with non-default parameters',
    58: 'CIEJab enumerated colourspace with default parameters',
    59: 'CIEJab enumerated colourspace with non-default parameters',
    60: 'e-sRGB enumerated colorspace',
    61: 'ROMM_RGB enumerated colorspace',
    62: 'Non-square samples',
    63: 'Deprecated - compositing layers have labels',
    64: 'Deprecated - codestreams have labels',
    65: 'Deprecated - compositing layers have different colour spaces',
    66: 'Deprecated - compositing layers have different metadata',
    67: 'GIS metadata XML box',
    68: 'JPSEC extensions in codestream as specified by ISO/IEC 15444-8',
    69: 'JP3D extensions in codestream as specified by ISO/IEC 15444-10',
    70: 'Deprecated - compositing layer uses sYCC enumerated colour space',
    71: 'e-sYCC enumerated colourspace',
    72: ('JPEG 2000 Part 2 codestream as restricted by baseline conformance '
         'requirements in M.9.2.3'),
    73: 'YPbPr(1125/60) enumerated colourspace',
    74: 'YPbPr(1250/50) enumerated colourspace'}


class ReaderRequirementsBox(Jp2kBox):
    """Container for reader requirements box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    fuam : int
        Fully Understand Aspects mask.
    dcm : int
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
    box_id = 'rreq'
    longname = 'Reader Requirements'

    def __init__(self, fuam, dcm, standard_flag, standard_mask, vendor_feature,
                 vendor_mask, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.fuam = fuam
        self.dcm = dcm
        self.standard_flag = tuple(standard_flag)
        self.standard_mask = tuple(standard_mask)
        self.vendor_feature = tuple(vendor_feature)
        self.vendor_mask = tuple(vendor_mask)
        self.length = length
        self.offset = offset

    def __repr__(self):
        msg = "glymur.jp2box.ReaderRequirementsBox(fuam={fuam}, dcm={dcm}, "
        msg += "standard_flag={standard_flag}, standard_mask={standard_mask}, "
        msg += "vendor_feature={vendor_feature}, vendor_mask={vendor_mask})"
        msg = msg.format(fuam=self.fuam,
                         dcm=self.dcm,
                         standard_flag=self.standard_flag,
                         standard_mask=self.standard_mask,
                         vendor_feature=self.vendor_feature,
                         vendor_mask=self.vendor_mask)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []

        text = 'Fully Understands Aspect Mask:  0x{fuam:x}'
        lst.append(text.format(fuam=self.fuam))

        text = 'Display Completely Mask:  0x{dcm:x}'.format(dcm=self.dcm)
        lst.append(text)

        text = 'Standard Features and Masks:'
        lst.append(text)

        lst2 = []
        text = 'Feature {flag:03d}:  0x{mask:x} {decoded}'
        for j in range(len(self.standard_flag)):
            kwargs = {
                'flag': self.standard_flag[j],
                'mask': self.standard_mask[j],
                'decoded': _READER_REQUIREMENTS_DISPLAY[self.standard_flag[j]],
            }
            lst2.append(text.format(**kwargs))
        text = '\n'.join(lst2)
        text = self._indent(text)
        lst.append(text)

        text = 'Vendor Features:'
        lst.append(text)

        lst2 = []
        for j in range(len(self.vendor_feature)):
            text = 'UUID {0}'.format(self.vendor_feature[j])
            lst2.append(text)
        text = '\n'.join(lst2)
        text = self._indent(text)
        lst.append(text)

        text = '\n'.join(lst)
        text = self._indent(text)
        text = '\n'.join([title, text])

        return text

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse reader requirements box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        ReaderRequirementsBox
            Instance of the current reader requirements box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)
        mask_length, = struct.unpack_from('>B', read_buffer, offset=0)

        if mask_length == 3:
            return _parse_rreq3(read_buffer, length, offset)

        # Fully Understands Aspect Mask
        # Decodes Completely Mask
        fuam = dcm = standard_flag = standard_mask = []
        vendor_feature = vendor_mask = []

        # The mask length tells us the format string to use when unpacking
        # from the buffer read from file.
        try:
            mask_format = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}[mask_length]
            fuam, dcm = struct.unpack_from('>' + mask_format * 2, read_buffer,
                                           offset=1)
            std_flg_offset = 1 + 2 * mask_length
            data = _parse_standard_flag(read_buffer[std_flg_offset:],
                                        mask_length)
            standard_flag, standard_mask = data

            nflags = len(standard_flag)
            vndr_offset = 1 + 2 * mask_length + 2 + (2 + mask_length) * nflags
            data = _parse_vendor_features(read_buffer[vndr_offset:],
                                          mask_length)
            vendor_feature, vendor_mask = data

        except KeyError:
            msg = ('The ReaderRequirements box (rreq) has a mask length of '
                   '{length} bytes, but only values of 1, 2, 4, or 8 are '
                   'supported.  The box contents will not be interpreted.')
            warnings.warn(msg.format(length=mask_length), UserWarning)

        return cls(fuam, dcm, standard_flag, standard_mask,
                   vendor_feature, vendor_mask,
                   length=length, offset=offset)


def _parse_rreq3(read_buffer, length, offset):
    """Parse a reader requirements box.  Special case when mask length is 3."""
    # Fully Understands Aspect Mask
    # Decodes Completely Mask
    fuam = dcm = standard_flag = standard_mask = []
    vendor_feature = vendor_mask = []

    # The mask length tells us the format string to use when unpacking
    # from the buffer read from file.
    lst = struct.unpack_from('>BBBBBB', read_buffer, offset=1)
    fuam = lst[0] << 16 | lst[1] << 8 | lst[2]
    dcm = lst[3] << 16 | lst[4] << 8 | lst[5]

    num_standard_features, = struct.unpack_from('>H', read_buffer, offset=7)

    fmt = '>' + 'HBBB' * num_standard_features
    lst = struct.unpack_from(fmt, read_buffer, offset=9)

    standard_flag = lst[0::4]
    standard_mask = []
    for j in range(num_standard_features):
        items = lst[slice(j * 4 + 1, j * 4 + 4)]
        mask = items[0] << 16 | items[1] << 8 | items[2]
        standard_mask.append(mask)

    boffset = 9 + num_standard_features * 5
    num_vendor_features, = struct.unpack_from('>H', read_buffer,
                                              offset=boffset)

    fmt = '>' + 'HBBB' * num_vendor_features
    buffer_offset = 11 + num_standard_features * 5
    lst = struct.unpack_from(fmt, read_buffer, offset=buffer_offset)

    # Each vendor feature consists of a 16-byte UUID plus a mask whose
    # length is specified by, you guessed it, "mask_length".
    entry_length = 16 + 3
    vendor_feature = []
    vendor_mask = []
    read_buffer = read_buffer[9 + num_standard_features * 10:]
    for j in range(num_vendor_features):
        uslice = slice(j * entry_length, (j + 1) * entry_length)
        ubuffer = read_buffer[uslice]
        vendor_feature.append(UUID(bytes=ubuffer[0:16]))

        lst = struct.unpack('>BBB', ubuffer[16:])
        vmask = lst[0] << 16 | lst[1] << 8 | lst[2]
        vendor_mask.append(vmask)

    box = ReaderRequirementsBox(fuam, dcm, standard_flag, standard_mask,
                                vendor_feature, vendor_mask,
                                length=length, offset=offset)
    return box


def _parse_standard_flag(read_buffer, mask_length):
    """Construct standard flag, standard mask data from the file.

    Specifically working on Reader Requirements box.

    Parameters
    ----------
    fptr : file object
        File object for JP2K file.
    mask_length : int
        Length of standard mask flag
    """
    # The mask length tells us the format string to use when unpacking
    # from the buffer read from file.
    mask_format = {1: 'B', 2: 'H', 4: 'I'}[mask_length]

    num_standard_flags, = struct.unpack_from('>H', read_buffer, offset=0)

    # Read in standard flags and standard masks.  Each standard flag should
    # be two bytes, but the standard mask flag is as long as specified by
    # the mask length.
    fmt = '>' + ('H' + mask_format) * num_standard_flags
    data = struct.unpack_from(fmt, read_buffer, offset=2)

    standard_flag = data[0:num_standard_flags * 2:2]
    standard_mask = data[1:num_standard_flags * 2:2]

    return standard_flag, standard_mask


def _parse_vendor_features(read_buffer, mask_length):
    """Construct vendor features, vendor mask data from the file.

    Specifically working on Reader Requirements box.

    Parameters
    ----------
    fptr : file object
        File object for JP2K file.
    mask_length : int
        Length of vendor mask flag
    """
    # The mask length tells us the format string to use when unpacking
    # from the buffer read from file.
    mask_format = {1: 'B', 2: 'H', 4: 'I'}[mask_length]

    num_vendor_features, = struct.unpack_from('>H', read_buffer)

    # Each vendor feature consists of a 16-byte UUID plus a mask whose
    # length is specified by, you guessed it, "mask_length".
    entry_length = 16 + mask_length
    vendor_feature = []
    vendor_mask = []
    for j in range(num_vendor_features):
        uslice = slice(2 + j * entry_length, 2 + (j + 1) * entry_length)
        ubuffer = read_buffer[uslice]
        vendor_feature.append(UUID(bytes=ubuffer[0:16]))

        vmask = struct.unpack('>' + mask_format, ubuffer[16:])
        vendor_mask.append(vmask)

    return vendor_feature, vendor_mask


class ResolutionBox(Jp2kBox):
    """Container for Resolution superbox information.

    Attributes
    ----------
    box_id : str
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
    box_id = 'res '
    longname = 'Resolution'

    def __init__(self, box=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset
        self.box = box if box is not None else []

    def __repr__(self):
        msg = "glymur.jp2box.ResolutionBox(box={0})"
        msg = msg.format(self.box)
        return msg

    def __str__(self):
        msg = self._str_superbox()
        return msg

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse Resolution box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        ResolutionBox
            Instance of the current resolution box.
        """
        box = cls(length=length, offset=offset)

        # The JP2 header box is a superbox, so go ahead and parse its child
        # boxes.
        box.box = box.parse_superbox(fptr)

        return box


class CaptureResolutionBox(Jp2kBox):
    """Container for Capture resolution box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    vertical_resolution, horizontal_resolution : float
        Vertical, horizontal resolution.
    """
    box_id = 'resc'
    longname = 'Capture Resolution'

    def __init__(self, vertical_resolution, horizontal_resolution, length=0,
                 offset=-1):
        Jp2kBox.__init__(self)
        self.vertical_resolution = vertical_resolution
        self.horizontal_resolution = horizontal_resolution
        self.length = length
        self.offset = offset

    def __repr__(self):
        msg = "glymur.jp2box.CaptureResolutionBox({0}, {1})"
        msg = msg.format(self.vertical_resolution, self.horizontal_resolution)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        text = 'VCR:  {0}'.format(self.vertical_resolution)
        lst.append(text)
        text = 'HCR:  {0}'.format(self.horizontal_resolution)
        lst.append(text)

        text = '\n'.join(lst)
        body = self._indent(text)

        text = '\n'.join([title, body])
        return text

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse CaptureResolutionBox.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        CaptureResolutionBox
            Instance of the current capture resolution box.
        """
        read_buffer = fptr.read(10)
        (rn1, rd1, rn2, rd2, re1, re2) = struct.unpack('>HHHHBB', read_buffer)
        vres = rn1 / rd1 * math.pow(10, re1)
        hres = rn2 / rd2 * math.pow(10, re2)

        return cls(vres, hres, length=length, offset=offset)


class DisplayResolutionBox(Jp2kBox):
    """Container for Display resolution box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    vertical_resolution, horizontal_resolution : float
        Vertical, horizontal resolution.
    """
    box_id = 'resd'
    longname = 'Display Resolution'

    def __init__(self, vertical_resolution, horizontal_resolution,
                 length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.vertical_resolution = vertical_resolution
        self.horizontal_resolution = horizontal_resolution
        self.length = length
        self.offset = offset

    def __repr__(self):
        msg = "glymur.jp2box.DisplayResolutionBox({0}, {1})"
        msg = msg.format(self.vertical_resolution, self.horizontal_resolution)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        text = 'VDR:  {0}'.format(self.vertical_resolution)
        lst.append(text)
        text = 'HDR:  {0}'.format(self.horizontal_resolution)
        lst.append(text)

        text = '\n'.join(lst)
        body = self._indent(text)

        text = '\n'.join([title, body])
        return text

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse display resolution box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        DisplayResolutionBox
            Instance of the current display resolution box.
        """

        read_buffer = fptr.read(10)
        (rn1, rd1, rn2, rd2, re1, re2) = struct.unpack('>HHHHBB', read_buffer)
        vres = rn1 / rd1 * math.pow(10, re1)
        hres = rn2 / rd2 * math.pow(10, re2)

        return cls(vres, hres, length=length, offset=offset)


class LabelBox(Jp2kBox):
    """Container for Label box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    label : str
        Textual label.
    """
    box_id = 'lbl '
    longname = 'Label'

    def __init__(self, label, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.label = label
        self.length = length
        self.offset = offset

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        text = 'Label:  {0}'.format(self.label)
        body = self._indent(text)

        text = '\n'.join([title, body])
        return text

    def __repr__(self):
        msg = 'glymur.jp2box.LabelBox("{0}")'.format(self.label)
        return msg

    def write(self, fptr):
        """Write a Label box to file.
        """
        length = 8 + len(self.label.encode())
        fptr.write(struct.pack('>I4s', length, b'lbl '))
        fptr.write(self.label.encode())

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse Label box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        LabelBox
            Instance of the current label box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)
        label = read_buffer.decode('utf-8')
        return cls(label, length=length, offset=offset)


class NumberListBox(Jp2kBox):
    """Container for Number List box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    AN : list
        Descriptors of an entity with which the data contained within the same
        Association box is associated.
    """
    box_id = 'nlst'
    longname = 'Number List'

    def __init__(self, associations, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.associations = associations
        self.length = length
        self.offset = offset

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        for j, association in enumerate(self.associations):
            text = 'Association[{0}]:  '.format(j)
            if association == 0:
                text += 'the rendered result'
            elif (association >> 24) == 1:
                idx = association & 0x00FFFFFF
                text += 'codestream {0}'.format(idx)
            elif (association >> 24) == 2:
                idx = association & 0x00FFFFFF
                text += 'compositing layer {0}'.format(idx)
            else:
                text += 'unrecognized'
            lst.append(text)

        body = '\n'.join(lst)
        body = self._indent(body)

        text = '\n'.join([title, body])
        return text

    def __repr__(self):
        msg = 'glymur.jp2box.NumberListBox(associations={0})'
        msg = msg.format(self.associations)
        return msg

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse number list box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        LabelBox
            Instance of the current number list box.
        """
        num_bytes = offset + length - fptr.tell()
        raw_data = fptr.read(num_bytes)
        num_associations = int(len(raw_data) / 4)
        lst = struct.unpack('>' + 'I' * num_associations, raw_data)
        return cls(lst, length=length, offset=offset)

    def write(self, fptr):
        """Write a NumberList box to file.
        """
        fptr.write(struct.pack('>I4s',
                               len(self.associations) * 4 + 8, b'nlst'))

        fmt = '>' + 'I' * len(self.associations)
        write_buffer = struct.pack(fmt, *self.associations)
        fptr.write(write_buffer)


class XMLBox(Jp2kBox):
    """Container for XML box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    xml : ElementTree object
        XML section.
    """
    box_id = 'xml '
    longname = 'XML'

    def __init__(self, xml=None, filename=None, length=0, offset=-1):
        """
        Parameters
        ----------
        xml : ElementTree
            An ElementTree object already existing in python.
        filename : str
            File from which to read XML.  If filename is not None, then the xml
            keyword argument must be None.
        """
        Jp2kBox.__init__(self)
        if filename is not None and xml is not None:
            msg = ("Only one of either a filename or an ElementTree instance "
                   "should be provided.")
            raise IOError(msg)
        if filename is not None:
            self.xml = ET.parse(filename)
        else:
            self.xml = xml
        self.length = length
        self.offset = offset

    def __repr__(self):
        return "glymur.jp2box.XMLBox(xml={xml})".format(xml=self.xml)

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title
        if _printoptions['xml'] is False:
            return title

        if self.xml is not None:
            body = ET.tostring(self.xml,
                               encoding='utf-8',
                               pretty_print=True).decode('utf-8')
        else:
            body = 'None'
        body = self._indent(body)

        text = '\n'.join([title, body])
        return text

    def write(self, fptr):
        """Write an XML box to file.
        """
        read_buffer = ET.tostring(self.xml, encoding='utf-8')
        fptr.write(struct.pack('>I4s', len(read_buffer) + 8, b'xml '))
        fptr.write(read_buffer)

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse XML box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        XMLBox
            Instance of the current XML box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)
        try:
            text = read_buffer.decode('utf-8')
        except UnicodeDecodeError as err:
            # Possibly bad string of bytes to begin with.
            # Try to search for <?xml and go from there.
            decl_start = read_buffer.find(b'<?xml')
            if decl_start <= -1:
                # Nope, that's not it.  All is lost.
                msg = ('A problem was encountered while parsing an XML box:'
                       '\n\n\t"{error}"\n\nNo XML was retrieved.')
                warnings.warn(msg.format(error=str(err)), UserWarning)
                return XMLBox(xml=None, length=length, offset=offset)

            text = read_buffer[decl_start:].decode('utf-8')

            # Let the user know that the XML box was problematic.
            msg = ('A UnicodeDecodeError was encountered parsing an XML box '
                   'at byte position {offset:d} ({reason}), but the XML was '
                   'still recovered.')
            msg = msg.format(offset=offset, reason=err.reason)
            warnings.warn(msg, UserWarning)

        # Strip out any trailing nulls, as they can foul up XML parsing.
        text = text.rstrip(chr(0))

        # Remove any byte order markers.
        if u'\ufeff' in text:
            msg = ('An illegal BOM (byte order marker) was detected and '
                   'removed from the XML contents in the box starting at byte '
                   'offset {offset:d}.')
            msg = msg.format(offset=offset)
            warnings.warn(msg, UserWarning)
            text = text.replace(u'\ufeff', '')

        # Remove any encoding declaration.
        if text.startswith('<?xml version="1.0" encoding="UTF-8"?>'):
            text = text[38:]

        try:
            elt = ET.fromstring(text)
            xml = ET.ElementTree(elt)
        except ET.ParseError as err:
            msg = ('A problem was encountered while parsing an XML box:'
                   '\n\n\t"{reason}"\n\nNo XML was retrieved.')
            msg = msg.format(reason=str(err))
            warnings.warn(msg, UserWarning)
            xml = None

        return cls(xml=xml, length=length, offset=offset)


class UUIDListBox(Jp2kBox):
    """Container for JPEG 2000 UUID list box.

    Attributes
    ----------
    box_id : str
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
    box_id = 'ulst'
    longname = 'UUID List'

    def __init__(self, ulst, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.ulst = ulst
        self.length = length
        self.offset = offset

    def __repr__(self):
        msg = "glymur.jp2box.UUIDListBox({0})".format(self.ulst)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = []
        for j, uuid_item in enumerate(self.ulst):
            text = 'UUID[{item_no}]:  {uuid}'.format(item_no=j, uuid=uuid_item)
            lst.append(text)
        body = '\n'.join(lst)
        body = self._indent(body)

        text = '\n'.join([title, body])
        return text

    def write(self, fptr):
        """Write a UUID list box to file.
        """
        num_uuids = len(self.ulst)
        length = 4 + 4 + 2 + num_uuids * 16
        write_buffer = struct.pack('>I4sH', length, b'ulst', num_uuids)
        fptr.write(write_buffer)

        for j in range(num_uuids):
            fptr.write(self.ulst[j].bytes)

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse UUIDList box.

        Parameters
        ----------
        f : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        UUIDListBox
            Instance of the current UUID list box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)

        num_uuids, = struct.unpack_from('>H', read_buffer)

        ulst = []
        for j in range(num_uuids):
            uuid_buffer = read_buffer[2 + j * 16:2 + (j + 1) * 16]
            ulst.append(UUID(bytes=uuid_buffer))

        return cls(ulst, length=length, offset=offset)


class UUIDInfoBox(Jp2kBox):
    """Container for JPEG 2000 UUID Info superbox.

    Attributes
    ----------
    box_id : str
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
    box_id = 'uinf'
    longname = 'UUIDInfo'

    def __init__(self, box=None, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.length = length
        self.offset = offset
        self.box = box if box is not None else []

    def __repr__(self):
        msg = "glymur.jp2box.UUIDInfoBox(box={0})".format(self.box)
        return msg

    def __str__(self):
        msg = self._str_superbox()
        return msg

    def write(self, fptr):
        """Write a UUIDInfo box to file.
        """
        self._write_superbox(fptr, b'uinf')

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse UUIDInfo super box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        UUIDInfoBox
            Instance of the current UUID information box.
        """

        box = cls(length=length, offset=offset)

        # The UUIDInfo box is a superbox, so go ahead and parse its child
        # boxes.
        box.box = box.parse_superbox(fptr)

        return box


class DataEntryURLBox(Jp2kBox):
    """Container for data entry URL box information.

    Attributes
    ----------
    box_id : str
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
    box_id = 'url '
    longname = 'Data Entry URL'

    def __init__(self, version, flag, url, length=0, offset=-1):
        Jp2kBox.__init__(self)
        self.version = version
        self.flag = flag
        self.url = url
        self.length = length
        self.offset = offset

    def write(self, fptr):
        """Write a data entry url box to file.
        """
        # Make sure it is written out as null-terminated.
        url = self.url
        if self.url[-1] != chr(0):
            url = url + chr(0)
        url = url.encode()

        length = 8 + 1 + 3 + len(url)
        write_buffer = struct.pack('>I4sBBBB',
                                   length, b'url ',
                                   self.version,
                                   self.flag[0], self.flag[1], self.flag[2])
        fptr.write(write_buffer)
        fptr.write(url)

    def __repr__(self):
        msg = "glymur.jp2box.DataEntryURLBox({version}, {flag}, '{url}')"
        msg = msg.format(version=self.version, flag=self.flag, url=self.url)
        return msg

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        lst = ['Version:  {version}',
               'Flag:  {flag0} {flag1} {flag2}',
               'URL:  "{url}"']
        body = '\n'.join(lst)
        body = body.format(version=self.version,
                           flag0=self.flag[0],
                           flag1=self.flag[1],
                           flag2=self.flag[2],
                           url=self.url)
        body = self._indent(body)

        text = '\n'.join([title, body])
        return text

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse data entry URL box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        DataEntryURLbox
            Instance of the current data entry URL box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)
        data = struct.unpack_from('>BBBB', read_buffer)
        version = data[0]
        flag = data[1:4]

        url = read_buffer[4:].decode('utf-8').rstrip(chr(0))
        return cls(version, flag, url, length=length, offset=offset)


class UnknownBox(Jp2kBox):
    """Container for unrecognized boxes.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    """
    def __init__(self, box_id, length=0, offset=-1, longname=''):
        Jp2kBox.__init__(self)
        self.longname = longname
        self.box_id = box_id
        self.length = length
        self.offset = offset

    def __repr__(self):
        msg = "glymur.jp2box.UnknownBox({0})".format(self.box_id)
        return msg

    def __str__(self):
        if len(self.box) > 0:
            msg = self._str_superbox()
        else:
            msg = Jp2kBox.__str__(self)
        return msg


class UUIDBox(Jp2kBox):
    """Container for UUID box information.

    Attributes
    ----------
    box_id : str
        4-character identifier for the box.
    length : int
        length of the box in bytes.
    offset : int
        offset of the box from the start of the file.
    longname : str
        more verbose description of the box.
    uuid : uuid.UUID
        16-byte UUID
    raw_data : byte array
        Sequence of uninterpreted bytes as read from the file.
    data : object
        Specific to each type of UUID.  There are handlers for XMP, Exif, and
        generic (unknown) UUIDs.  In the case of XMP and Exif UUIDs, this is
        the interpreted version of raw_data.

    References
    ----------
    .. [XMP] International Organization for Standardication.  ISO/IEC
       16684-1:2012 - Graphic technology -- Extensible metadata platform (XMP)
       specification -- Part 1:  Data model, serialization and core properties
    """
    box_id = 'uuid'
    longname = 'UUID'

    def __init__(self, the_uuid, raw_data, length=0, offset=-1):
        """
        Parameters
        ----------
        the_uuid : uuid.UUID
            Identifies the type of UUID box.
        raw_data : byte array
            Sequence of uninterpreted bytes as read from the UUID box.
        length : int
            length of the box in bytes.
        offset : int
            offset of the box from the start of the file.
        """
        Jp2kBox.__init__(self)
        self.uuid = the_uuid
        self.raw_data = raw_data
        self.length = length
        self.offset = offset
        self.data = None

        try:
            self._parse_raw_data()
        except KeyError as error:
            # Such as when an Exif tag is unrecognized.
            warnings.warn(str(error))
        except IOError as error:
            # Such as when Exif byte order is unrecognized.
            warnings.warn(str(error))

    def _parse_raw_data(self):
        """
        Private function for parsing UUID payloads if possible.
        """
        if self.uuid == UUID('be7acfcb-97a9-42e8-9c71-999491e3afac'):
            self.data = _uuid_io.xml(self.raw_data)
        elif self.uuid == _GEOTIFF_UUID:
            self.data = _uuid_io.tiff_header(self.raw_data)
        elif self.uuid == _EXIF_UUID:
            # Cut off 'EXIF\0\0' part.
            self.data = _uuid_io.tiff_header(self.raw_data[6:])
        else:
            self.data = self.raw_data

    def __repr__(self):
        msg = "glymur.jp2box.UUIDBox(the_uuid={0}, "
        msg += "raw_data=<byte array {1} elements>)"
        return msg.format(repr(self.uuid), len(self.raw_data))

    def __str__(self):
        title = Jp2kBox.__str__(self)
        if _printoptions['short'] is True:
            return title

        text = 'UUID:  {0}'.format(self.uuid)
        if self.uuid == UUID('be7acfcb-97a9-42e8-9c71-999491e3afac'):
            text += ' (XMP)'
        elif self.uuid == UUID('b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03'):
            text += ' (GeoTIFF)'
        elif self.uuid.bytes == b'JpgTiffExif->JP2':
            text += ' (EXIF)'
        else:
            text += ' (unknown)'

        lst = [text]

        if (((_printoptions['xml'] is False) and
             (self.uuid == UUID('be7acfcb-97a9-42e8-9c71-999491e3afac')))):
            # If it's an XMP UUID, don't print the XML contents.
            pass

        elif self.uuid == UUID('be7acfcb-97a9-42e8-9c71-999491e3afac'):
            line = 'UUID Data:\n{0}'
            xmlstring = ET.tostring(self.data,
                                    encoding='utf-8',
                                    pretty_print=True).decode('utf-8').rstrip()
            text = line.format(xmlstring)
            lst.append(text)
        elif self.uuid.bytes == b'JpgTiffExif->JP2':
            text = 'UUID Data:  {0}'.format(str(self.data))
            lst.append(text)
        elif self.uuid == UUID('b14bf8bd-083d-4b43-a5ae-8cd7d5a6ce03'):
            if _HAVE_GDAL:
                txt = self._print_geotiff()
            else:
                txt = 'UUID Data:  {0}'.format(str(self.data))
            lst.append(txt)
        else:
            text = 'UUID Data:  {0} bytes'.format(len(self.raw_data))
            lst.append(text)

        body = '\n'.join(lst)
        body = self._indent(body)

        text = '\n'.join([title, body])
        return text

    def _print_geotiff(self):
        """
        Print geotiff information.  Shamelessly ripped off from gdalinfo.py
        """
        lst = []
        in_mem_name = '/vsimem/geo.tif'
        gdal.FileFromMemBuffer(in_mem_name, self.raw_data)
        gtif = gdal.Open(in_mem_name)

        # Report projection
        proj_ref = gtif.GetProjectionRef()
        sref = osr.SpatialReference()
        lst.append("Coordinate System =")
        if sref.ImportFromWkt(proj_ref) == gdal.CE_None:
            psz_pretty_wkt = sref.ExportToPrettyWkt(False)
        else:
            psz_pretty_wkt = proj_ref
        lst.append(self._indent(psz_pretty_wkt))

        # report geotransform
        geo_transform = gtif.GetGeoTransform(can_return_null=True)
        if geo_transform is not None:

            if geo_transform[2] == 0.0 and geo_transform[4] == 0.0:
                fmt = ('Origin = ({:.15f},{:.15f})\n'
                       'Pixel Size = ({:.15f},{:.15f})')
                txt = fmt.format(geo_transform[0], geo_transform[3],
                                 geo_transform[1], geo_transform[5])
            else:
                fmt = ('GeoTransform =   '
                       '{:.16g}, {:16g}, {:.16g}, {:.16g}, {:16g} {:16g}')
                txt = fmt.format(geo_transform[0], geo_transform[1],
                                 geo_transform[2], geo_transform[3],
                                 geo_transform[4], geo_transform[5])
            lst.append(txt)

        # setup projected to lat/long transform if appropriate
        if proj_ref is not None and len(proj_ref) > 0:
            hProj = osr.SpatialReference(proj_ref)
            if hProj is not None:
                hLatLong = hProj.CloneGeogCS()

            if hLatLong is not None:
                gdal.PushErrorHandler('CPLQuietErrorHandler')
                hTransform = osr.CoordinateTransformation(hProj, hLatLong)
                gdal.PopErrorHandler()
                msg = 'Unable to load PROJ.4 library'
                if gdal.GetLastErrorMsg().find(msg) != -1:
                    hTransform = None

        # report corners
        lst.append("Corner Coordinates:")
        txt = self.GDALInfoReportCorner(gtif, hTransform, "Upper Left", 0, 0)
        lst.append(txt)

        txt = self.GDALInfoReportCorner(gtif, hTransform, "Lower Left",
                                        0, gtif.RasterYSize)
        lst.append(txt)

        txt = self.GDALInfoReportCorner(gtif, hTransform, "Upper Right",
                                        gtif.RasterXSize, 0)
        lst.append(txt)

        txt = self.GDALInfoReportCorner(gtif, hTransform, "Lower Right",
                                        gtif.RasterXSize, gtif.RasterYSize)
        lst.append(txt)

        txt = self.GDALInfoReportCorner(gtif, hTransform, "Center",
                                        gtif.RasterXSize / 2.0,
                                        gtif.RasterYSize / 2.0)
        lst.append(txt)

        gdal.Unlink(in_mem_name)
        return '\n'.join(lst)

    def GDALInfoReportCorner(self, hDataset, hTransform, corner_name, x, y):
        line = '{:<11s} '.format(corner_name)

        # transform the point into georeferenced coordinates
        geo_transform = hDataset.GetGeoTransform(can_return_null=True)
        if geo_transform is not None:
            dfGeoX = (geo_transform[0] + geo_transform[1] * x +
                      geo_transform[2] * y)
            dfGeoY = geo_transform[3] + geo_transform[4] * x
            dfGeoY += geo_transform[5] * y
        else:
            line += '({:12.7f},{:12.7f})'.format(x, y)
            return line

        # report the georeferenced coordinates
        if abs(dfGeoX) < 181 and abs(dfGeoY) < 91:
            line += '({:12.7f},{:12.7f}) '.format(dfGeoX, dfGeoY)
        else:
            line += '({:12.3f},{:12.3f}) '.format(dfGeoX, dfGeoY)

        # transform to latlong and report
        if hTransform is not None:
            point = hTransform.TransformPoint(dfGeoX, dfGeoY, 0)
            if point is not None:
                line += '({},{}'.format(gdal.DecToDMS(point[0], 'Long', 2),
                                        gdal.DecToDMS(point[1], 'Lat', 2))

        return line

    def write(self, fptr):
        """Write a UUID box to file.
        """
        length = 4 + 4 + 16 + len(self.raw_data)
        write_buffer = struct.pack('>I4s', length, b'uuid')
        fptr.write(write_buffer)
        fptr.write(self.uuid.bytes)
        fptr.write(self.raw_data)

    @classmethod
    def parse(cls, fptr, offset, length):
        """Parse UUID box.

        Parameters
        ----------
        fptr : file
            Open file object.
        offset : int
            Start position of box in bytes.
        length : int
            Length of the box in bytes.

        Returns
        -------
        UUIDBox
            Instance of the current UUID box.
        """
        num_bytes = offset + length - fptr.tell()
        read_buffer = fptr.read(num_bytes)
        the_uuid = UUID(bytes=read_buffer[0:16])
        return cls(the_uuid, read_buffer[16:], length=length, offset=offset)


# Map each box ID to the corresponding class.
_BOX_WITH_ID = {
    b'asoc': AssociationBox,
    b'bpcc': BitsPerComponentBox,
    b'cdef': ChannelDefinitionBox,
    b'cgrp': ColourGroupBox,
    b'cmap': ComponentMappingBox,
    b'colr': ColourSpecificationBox,
    b'dtbl': DataReferenceBox,
    b'ftyp': FileTypeBox,
    b'ihdr': ImageHeaderBox,
    b'jP  ': JPEG2000SignatureBox,
    b'jpch': CodestreamHeaderBox,
    b'jplh': CompositingLayerHeaderBox,
    b'jp2c': ContiguousCodestreamBox,
    b'free': FreeBox,
    b'flst': FragmentListBox,
    b'ftbl': FragmentTableBox,
    b'jp2h': JP2HeaderBox,
    b'lbl ': LabelBox,
    b'nlst': NumberListBox,
    b'pclr': PaletteBox,
    b'res ': ResolutionBox,
    b'resc': CaptureResolutionBox,
    b'resd': DisplayResolutionBox,
    b'rreq': ReaderRequirementsBox,
    b'uinf': UUIDInfoBox,
    b'ulst': UUIDListBox,
    b'url ': DataEntryURLBox,
    b'uuid': UUIDBox,
    b'xml ': XMLBox}

_parseoptions = {'full_codestream': False}


def set_parseoptions(full_codestream=True):
    """Set parsing options.

    These options determine the way JPEG 2000 boxes are parsed.

    Parameters
    ----------
    full_codestream : bool, defaults to True
        When False, only the codestream header is parsed for metadata.  This
        can results in faster JP2/JPX parsing.  When True, the entire
        codestream is parsed for metadata.

    See also
    --------
    get_parseoptions

    Examples
    --------
    To put back the default options, you can use:

    >>> import glymur
    >>> glymur.set_parseoptions(full_codestream=True)
    """
    _parseoptions['full_codestream'] = full_codestream


def get_parseoptions():
    """Return the current parsing options.

    Returns
    -------
    dict
        Dictionary of current print options with keys

          - codestream : bool

        For a full description of these options, see `set_parseoptions`.

    See also
    --------
    set_parseoptions
    """
    return _parseoptions

_printoptions = {'short': False, 'xml': True, 'codestream': True}


def set_printoptions(**kwargs):
    """Set printing options.

    These options determine the way JPEG 2000 boxes are displayed.

    Parameters
    ----------
    short : bool, optional
        When True, only the box ID, offset, and length are displayed.  Useful
        for displaying only the basic structure or skeleton of a JPEG 2000
        file.
    xml : bool, optional
        When False, printing of the XML contents of any XML boxes or UUID XMP
        boxes is suppressed.
    codestream : bool, optional
        When False, only the codestream header is printed.  When True, the
        entire codestream is printed.  This option has no effect when the
        'short' option is set to True.

    See also
    --------
    get_printoptions

    Examples
    --------
    To put back the default options, you can use:

    >>> import glymur
    >>> glymur.set_printoptions(short=False, xml=True, codestream=True)
    """
    for key, value in kwargs.items():
        if key not in ['short', 'xml', 'codestream']:
            raise TypeError('"{0}" not a valid keyword parameter.'.format(key))
        _printoptions[key] = value


def get_printoptions():
    """Return the current print options.

    Returns
    -------
    dict
        Dictionary of current print options with keys

          - short : bool
          - xml : bool
          - codestream : bool

        For a full description of these options, see `set_printoptions`.

    See also
    --------
    set_printoptions
    """
    return _printoptions
