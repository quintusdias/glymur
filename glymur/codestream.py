"""Codestream information.

The module contains classes used to store information parsed from JPEG 2000
codestreams.
"""

import math
import struct
import sys
import warnings

import numpy as np

from .core import _progression_order_display
from .core import _wavelet_transform_display
from .core import _capabilities_display
from .lib import openjp2 as opj2

# Need a catch-all list of valid markers.
# See table A-1 in ISO/IEC FCD15444-1.
_valid_markers = [0xff00, 0xff01, 0xfffe]
for _marker in range(0xffc0, 0xffe0):
    _valid_markers.append(_marker)
for _marker in range(0xfff0, 0xfff9):
    _valid_markers.append(_marker)
for _marker in range(0xff4f, 0xff70):
    _valid_markers.append(_marker)
for _marker in range(0xff90, 0xff94):
    _valid_markers.append(_marker)


class Codestream:
    """Container for codestream information.

    Attributes
    ----------
    segment : list of marker segments

    Raises
    ------
    IOError
        If the file does not parse properly.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, f, header_only=True):
        """
        Parameters
        ----------
        f : file
            Open file object.
        header_only : bool, optional
            If True, only marker segments in the main header are parsed.
            Supplying False may impose a large performance penalty.
        """

        self._parse_tile_part_bit_stream_flag = False

        self.segment = []

        # First two bytes are the SOC marker
        buffer = f.read(2)
        marker_id, = struct.unpack('>H', buffer)
        segment = SOCsegment(offset=f.tell() - 2, length=0)
        self.segment.append(segment)

        tile_offset = []
        tile_length = []

        while True:
            offset = f.tell()
            buffer = f.read(2)
            marker_id, = struct.unpack('>H', buffer)

            if marker_id >= 0xff30 and marker_id <= 0xff3f:
                the_id = '0x{0:x}'.format(marker_id)
                segment = Segment(id=the_id, offset=offset, length=0)

            elif marker_id == 0xff51:
                # Need to keep track of the number of components from SIZ for
                # other markers
                segment = self._parseSIZsegment(f)
                self._Csiz = len(segment.Ssiz)

            elif marker_id == 0xff52:
                segment = self._parseCODsegment(f)

            elif marker_id == 0xff53:
                segment = self._parseCOCsegment(f)

            elif marker_id == 0xff55:
                segment = self._parseTLMsegment(f)

            elif marker_id == 0xff58:
                segment = self._parsePLTsegment(f)

            elif marker_id == 0xff5c:
                segment = self._parseQCDsegment(f)

            elif marker_id == 0xff5d:
                segment = self._parseQCCsegment(f)

            elif marker_id == 0xff5e:
                segment = self._parseRGNsegment(f)

            elif marker_id == 0xff5f:
                segment = self._parsePODsegment(f)

            elif marker_id == 0xff60:
                segment = self._parsePPMsegment(f)

            elif marker_id == 0xff61:
                segment = self._parsePPTsegment(f)

            elif marker_id == 0xff63:
                segment = self._parseCRGsegment(f)

            elif marker_id == 0xff64:
                segment = self._parseCMEsegment(f)

            elif marker_id == 0xff90:
                # Need to keep easy access to tile offsets and lengths for when
                # we encounter start-of-data marker segments.
                if header_only:
                    # Stop parsing as soon as we hit the first Start Of Tile.
                    return

                segment = self._parseSOTsegment(f)
                if segment.offset not in tile_offset:
                    tile_offset.append(segment.offset)
                    tile_length.append(segment.Psot)
                else:
                    msg = "Inconsistent start-of-tile (SOT) marker segment "
                    msg += "encountered in tile with index {0}.  "
                    msg += "Codestream parsing terminated."
                    msg = msg.format(segment.Isot)
                    warnings.warn(msg)
                    return

            elif marker_id == 0xff93:
                # start of data.  Need to seek past the current tile part.
                # The last SOT marker segment has the info that we need.
                segment = self._parseSODsegment(f)

            elif marker_id == 0xffd9:
                # end of codestream
                segment = self._parseEOCsegment(f)
                self.segment.append(segment)
                break

            elif marker_id in _valid_markers:
                # It's a reserved marker that I don't know anything about.
                # See table A-1 in ISO/IEC FCD15444-1.
                segment = self._parseGenericSegment(f, marker_id)

            elif ((marker_id & 0xff00) >> 8) == 255:
                # Peek ahead to see if the next two bytes are a marker or not.
                # Then seek back.
                msg = "Unrecognized marker id:  0x{0:x}".format(marker_id)
                warnings.warn(msg)
                cpos = f.tell()
                buffer = f.read(2)
                next_item, = struct.unpack('>H', buffer)
                f.seek(cpos)
                if ((next_item & 0xff00) >> 8) == 255:
                    # No segment associated with this marker, so reset
                    # to two bytes after it.
                    segment = Segment(id='0x{0:x}'.format(marker_id),
                                      offset=offset, length=0)
                else:
                    segment = self._parseGenericSegment(f, marker_id)

            else:
                msg = 'Invalid marker id encountered at byte {0:d}'
                msg += 'in codestream:  "0x{1:x}"'
                msg = msg.format(offset, marker_id)
                raise IOError(msg)

            self.segment.append(segment)

            if marker_id == 0xff93:
                # If SOD, then we need to seek past the tile part bit stream.
                x = f.tell()
                if self._parse_tile_part_bit_stream_flag:
                    # But first parse the tile part bit stream for SOP and
                    # EPH segments.
                    self._parse_tile_part_bit_stream(f, segment,
                                                     tile_length[-1])

                f.seek(tile_offset[-1] + tile_length[-1])

    def _parse_tile_part_bit_stream(self, f, sod_marker, tile_length):
        """Parse the tile part bit stream for SOP, EPH marker segments."""
        buffer = f.read(tile_length)
        # The tile length could possibly be too large and extend past
        # the end of file.  We need to be a bit resilient.
        count = min(tile_length, len(buffer))
        packet = np.frombuffer(buffer, dtype=np.uint8, count=count)

        indices = np.where(packet == 0xff)
        for idx in indices[0]:
            try:
                if packet[idx+1] == 0x91 and (idx < (len(packet) - 5)):
                    kwargs = {}
                    kwargs['offset'] = sod_marker.offset + 2 + idx
                    kwargs['length'] = 4
                    nsop = packet[(idx + 4):(idx+6)].view('uint16')[0]
                    if sys.byteorder == 'little':
                        nsop = nsop.byteswap()
                    kwargs['Nsop'] = nsop
                    segment = SOPsegment(**kwargs)
                    self.segment.append(segment)
                elif packet[idx + 1] == 0x92:
                    kwargs = {}
                    kwargs['offset'] = sod_marker.offset + 2 + idx
                    kwargs['length'] = 0
                    segment = EPHsegment(**kwargs)
                    self.segment.append(segment)
            except IndexError:
                continue

    def __str__(self):
        msg = 'Codestream:\n'
        for segment in self.segment:
            strs = segment.__str__()

            # Add indentation
            strs = [('    ' + x + '\n') for x in strs.split('\n')]
            msg += ''.join(strs)
        return msg

    def _parseSIZsegment(self, f):
        """Parse the SIZ segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        SIZsegment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(38)
        data = struct.unpack('>HHIIIIIIIIH', buffer)

        kwargs['length'] = data[0]
        kwargs['Rsiz'] = data[1]

        Xsiz = data[2]
        Ysiz = data[3]
        XOsiz = data[4]
        YOsiz = data[5]
        XTsiz = data[6]
        YTsiz = data[7]
        XTOsiz = data[8]
        YTOsiz = data[9]

        num_tiles_x = (Xsiz - XOsiz) / (XTsiz - XTOsiz)
        num_tiles_y = (Ysiz - YOsiz) / (YTsiz - YTOsiz)
        numtiles = math.ceil(num_tiles_x) * math.ceil(num_tiles_y)
        if numtiles > 65535:
            msg = "Invalid number of tiles ({0}).".format(numtiles)
            warnings.warn(msg)

        kwargs['Xsiz'] = Xsiz
        kwargs['Ysiz'] = Ysiz
        kwargs['XOsiz'] = XOsiz
        kwargs['YOsiz'] = YOsiz
        kwargs['XTsiz'] = XTsiz
        kwargs['YTsiz'] = YTsiz
        kwargs['XTOsiz'] = XTOsiz
        kwargs['YTOsiz'] = YTOsiz

        num_components = data[10]
        buffer = f.read(num_components * 3)
        data = struct.unpack('>' + 'B' * num_components * 3, buffer)

        Ssiz = data[0::3]
        kwargs['Ssiz'] = Ssiz
        kwargs['_bitdepth'] = tuple(((x & 0x7f) + 1) for x in Ssiz)
        kwargs['_signed'] = tuple(((x & 0xb0) > 0) for x in Ssiz)

        ssf = []
        for j, subsampling in enumerate(list(zip(data[1::3], data[2::3]))):
            if 0 in subsampling:
                msg = "Invalid subsampling value for component {0}: "
                msg += "dx={1}, dy={2}."
                msg = msg.format(j, subsampling[0], subsampling[1])
                warnings.warn(msg)
        kwargs['XRsiz'] = data[1::3]
        kwargs['YRsiz'] = data[2::3]

        return SIZsegment(**kwargs)

    def _parseGenericSegment(self, f, marker_id):
        """Parse a generic marker segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        Segment instance.
        """
        kwargs = {}
        offset = f.tell() - 2

        buffer = f.read(2)
        length, = struct.unpack('>H', buffer)
        data = f.read(length-2)

        segment = Segment(id='0x{0:x}'.format(marker_id),
                          offset=offset, length=length)
        segment.data = data
        return segment

    def _parseCMEsegment(self, f):
        """Parse the CME marker segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        CME segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(4)
        data = struct.unpack('>HH', buffer)
        kwargs['length'] = data[0]
        kwargs['Rcme'] = data[1]
        kwargs['Ccme'] = f.read(kwargs['length'] - 4)

        return CMEsegment(**kwargs)

    def _parseCRGsegment(self, f):
        """Parse the CRG marker segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        CRG segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(2)
        length, = struct.unpack('>H', buffer)
        kwargs['length'] = length

        buffer = f.read(4 * self._Csiz)
        data = struct.unpack('>' + 'HH' * self._Csiz, buffer)
        kwargs['Xcrg'] = data[0::2]
        kwargs['Ycrg'] = data[1::2]

        return CRGsegment(**kwargs)

    def _parseEOCsegment(self, f):
        """Parse the EOC marker segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        EOC Segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2
        kwargs['length'] = 0

        return EOCsegment(**kwargs)

    def _parseCOCsegment(self, f):
        """Parse the COC marker segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        COC segment instance.
        """
        kwargs = {}
        offset = f.tell() - 2
        kwargs['offset'] = offset

        buffer = f.read(2)
        length, = struct.unpack('>H', buffer)
        kwargs['length'] = length

        if self._Csiz <= 255:
            buffer = f.read(1)
            component, = struct.unpack('>B', buffer)
        else:
            buffer = f.read(2)
            component, = struct.unpack('>H', buffer)
        kwargs['Ccoc'] = component

        buffer = f.read(1)
        kwargs['Scoc'], = struct.unpack('>B', buffer)

        n = offset + 2 + length - f.tell()
        buffer = f.read(n)
        SPcoc = np.frombuffer(buffer, dtype=np.uint8)
        kwargs['SPcoc'] = SPcoc

        e1 = SPcoc[1]
        e2 = SPcoc[2]
        _code_block_size = (4 * math.pow(2, e2), 4 * math.pow(2, e1))
        kwargs['_code_block_size'] = _code_block_size

        if len(SPcoc) > 5:
            kwargs['_precinct_size'] = _parse_precinct_size(SPcoc[5:])
        else:
            kwargs['_precinct_size'] = None

        return COCsegment(**kwargs)

    def _parseCODsegment(self, f):
        """Parse the COD segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        COD segment instance.
        """
        kwargs = {}
        offset = f.tell() - 2
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(3)
        length, Scod = struct.unpack('>HB', buffer)
        kwargs['length'] = length
        kwargs['Scod'] = Scod

        sop = (Scod & 2) > 0
        eph = (Scod & 4) > 0

        if sop or eph:
            self._parse_tile_part_bit_stream_flag = True
        else:
            self._parse_tile_part_bit_stream_flag = False

        n = offset + 2 + length - f.tell()
        SPcod = f.read(n)
        kwargs['SPcod'] = np.frombuffer(SPcod, dtype=np.uint8)

        params = struct.unpack('>BHBBBBBB', SPcod[0:9])
        kwargs['_layers'] = params[1]
        kwargs['_numresolutions'] = params[3]

        if params[3] > opj2._J2K_MAXRLVLS:
            msg = "Invalid number of resolutions ({0})."
            msg = msg.format(params[3] + 1)
            warnings.warn(msg)

        cblk_width = 4 * math.pow(2, params[4])
        cblk_height = 4 * math.pow(2, params[5])
        code_block_size = (cblk_height, cblk_width)
        kwargs['_code_block_size'] = code_block_size

        if len(SPcod) > 9:
            kwargs['_precinct_size'] = _parse_precinct_size(SPcod[9:])
        else:
            kwargs['_precinct_size'] = None

        return CODsegment(**kwargs)

    def _parsePODsegment(self, f):
        """Parse the POD segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        POD segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(2)
        length, = struct.unpack('>H', buffer)

        if self._Csiz < 257:
            n = int((length - 2) / 7)
            buffer = f.read(n * 7)
            fmt = '>' + 'BBHBBB' * n
        else:
            n = int((length - 2) / 9)
            buffer = f.read(n * 9)
            fmt = '>' + 'BHHBHB' * n

        data = struct.unpack(fmt, buffer)

        kwargs['length'] = length
        kwargs['RSpod'] = data[0::6]
        kwargs['CSpod'] = data[1::6]
        kwargs['LYEpod'] = data[2::6]
        kwargs['REpod'] = data[3::6]
        kwargs['CEpod'] = data[4::6]
        kwargs['Ppod'] = data[5::6]

        return PODsegment(**kwargs)

    def _parsePPMsegment(self, f):
        """Parse the PPM segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        PPM segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(3)
        length, zppm = struct.unpack('>HB', buffer)
        kwargs['length'] = length
        kwargs['Zppm'] = zppm

        n = length - 3
        kwargs['data'] = f.read(n)

        return PPMsegment(**kwargs)

    def _parsePLTsegment(self, f):
        """Parse the PLT segment.

        The packet headers are not parsed, i.e. they remain "uninterpreted"
        raw data beffers.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        PLT segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(3)
        length, zplt = struct.unpack('>HB', buffer)
        kwargs['length'] = length
        kwargs['Zplt'] = zplt

        n = length - 3
        buffer = f.read(n)
        iplt = np.frombuffer(buffer, dtype=np.uint8)

        packet_len = []
        plen = 0
        for x in iplt:
            plen |= (x & 0x7f)
            if x & 0x80:
                # Continue by or-ing in the next byte.
                plen <<= 7
            else:
                packet_len.append(plen)
                plen = 0

        kwargs['Iplt'] = packet_len

        return PLTsegment(**kwargs)

    def _parsePPTsegment(self, f):
        """Parse the PPT segment.

        The packet headers are not parsed, i.e. they remain "uninterpreted"
        raw data beffers.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        PPT segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(3)
        length, zppt = struct.unpack('>HB', buffer)
        kwargs['length'] = length
        kwargs['Zppt'] = zppt

        n = length - 3
        kwargs['Ippt'] = f.read(n)

        return PPTsegment(**kwargs)

    def _parseQuantization(self, buffer, sqcd):
        """Tease out the quantization values.

        Args:
            buffer:  sequence of bytes from the QCC and QCD segments.
        """
        n = len(buffer)

        exponent = []
        mantissa = []

        if sqcd & 0x1f == 0:  # no quantization
            data = struct.unpack('>' + 'B' * n, buffer)
            for j in range(len(data)):
                exponent.append(data[j] >> 3)
                mantissa.append(0)
        else:
            fmt = '>' + 'H' * int(n / 2)
            data = struct.unpack(fmt, buffer)
            for j in range(len(data)):
                exponent.append(data[j] >> 11)
                mantissa.append(data[j] & 0x07ff)

        return mantissa, exponent

    def _parseQCCsegment(self, f):
        """Parse the QCC segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        QCC Segment instance.
        """
        kwargs = {}
        offset = f.tell() - 2
        kwargs['offset'] = offset

        buffer = f.read(2)
        length, = struct.unpack('>H', buffer)
        kwargs['length'] = length

        if self._Csiz > 256:
            buffer = f.read(3)
            fmt = '>HB'
            n = length - 5
        else:
            buffer = f.read(2)
            fmt = '>BB'
            n = length - 4
        Cqcc, Sqcc = struct.unpack(fmt, buffer)
        if Cqcc >= self._Csiz:
            msg = "Invalid component number (%d), "
            msg += "number of components is only %d."
            msg = msg.format(Cqcc, self._Csiz)
            warnings.warn(msg)
        kwargs['Cqcc'] = Cqcc

        kwargs['Sqcc'] = Sqcc
        kwargs['_guardBits'] = (Sqcc & 0xe0) >> 5

        buffer = f.read(n)

        mantissa, exponent = self._parseQuantization(buffer, Sqcc)

        kwargs['SPqcc'] = buffer
        kwargs['_exponent'] = exponent
        kwargs['_mantissa'] = mantissa

        return QCCsegment(**kwargs)

    def _parseQCDsegment(self, f):
        """Parse the QCD segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        QCD Segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(3)
        length, sqcd = struct.unpack('>HB', buffer)
        kwargs['length'] = length
        kwargs['Sqcd'] = sqcd

        kwargs['_guardBits'] = (sqcd & 0xe0) >> 5

        buffer = f.read(length - 3)

        mantissa, exponent = self._parseQuantization(buffer, sqcd)

        kwargs['SPqcd'] = buffer
        kwargs['_exponent'] = exponent
        kwargs['_mantissa'] = mantissa

        return QCDsegment(**kwargs)

    def _parseRGNsegment(self, f):
        """Parse the RGN segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        RGN segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(2)
        length, = struct.unpack('>H', buffer)

        if self._Csiz < 257:
            buffer = f.read(3)
            data = struct.unpack('>BBB', buffer)
        else:
            buffer = f.read(4)
            data = struct.unpack('>HBB', buffer)

        kwargs['length'] = length
        kwargs['Crgn'] = data[0]
        kwargs['Srgn'] = data[1]
        kwargs['SPrgn'] = data[2]

        return RGNsegment(**kwargs)

    def _parseSODsegment(self, f):
        """Parse the SOD segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        SOD segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2
        kwargs['length'] = 0

        return SODsegment(**kwargs)

    def _parseSOTsegment(self, f):
        """Parse the SOT segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        SOT segment instance.
        """
        kwargs = {}
        kwargs['offset'] = f.tell() - 2

        buffer = f.read(10)
        data = struct.unpack('>HHIBB', buffer)

        kwargs['length'] = data[0]
        kwargs['Isot'] = data[1]
        kwargs['Psot'] = data[2]
        kwargs['TPsot'] = data[3]
        kwargs['TNsot'] = data[4]

        return SOTsegment(**kwargs)

    def _parseTLMsegment(self, f):
        """Parse the TLM segment.

        Parameters
        ----------
        f : file
            Open file object.

        Returns
        -------
        TLM segment instance.
        """
        kwargs = {}
        offset = f.tell() - 2
        kwargs['offset'] = offset

        buffer = f.read(2)
        length, = struct.unpack('>H', buffer)
        kwargs['length'] = length

        buffer = f.read(2)
        Ztlm, Stlm = struct.unpack('>BB', buffer)
        st = (Stlm >> 4) & 0x3
        sp = (Stlm >> 6) & 0x1

        nbytes = length - 4
        if st == 0:
            ntiles = nbytes / ((sp + 1) * 2)
        else:
            ntiles = nbytes / (st + (sp + 1) * 2)

        buffer = f.read(nbytes)
        if st == 0:
            Ttlm = None
            fmt = ''
        elif st == 1:
            fmt = 'B'
        elif st == 2:
            fmt = 'H'

        if sp == 0:
            fmt += 'H'
        else:
            fmt += 'I'

        data = struct.unpack('>' + fmt * int(ntiles), buffer)
        if st == 0:
            Ttlm = None
            Ptlm = data
        else:
            Ttlm = data[0::2]
            Ptlm = data[1::2]

        kwargs['Ztlm'] = Ztlm
        kwargs['Ttlm'] = Ttlm
        kwargs['Ptlm'] = Ptlm

        return TLMsegment(**kwargs)


class Segment:
    """Segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    """
    def __init__(self, id='', offset=-1, length=-1):
        self.id = id
        self.offset = offset
        self.length = length

    def __str__(self):
        msg = '{0} marker segment @ ({1}, {2})'.format(self.id,
                                                       self.offset,
                                                       self.length)
        return msg

    def _print_quantization_style(self, sqcc):
        """Only to be used with QCC and QCD segments."""

        msg = '\n    Quantization style:  '
        if sqcc & 0x1f == 0:
            msg += 'no quantization, '
        elif sqcc & 0x1f == 1:
            msg += 'scalar implicit, '
        elif sqcc & 0x1f == 2:
            msg += 'scalar explicit, '
        return msg


class COCsegment(Segment):
    """COC (Coding style Component) segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Ccoc : byte
        Index of associated component.
    Scoc : byte
        Coding style for this component.
    SPcoc : byte array
        Coding style parameters for this component.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='COC')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)

        msg += '\n    Associated component:  {0}'.format(self.Ccoc)

        msg += '\n    Coding style for this component:  '
        if self.Scoc == 0:
            msg += 'Entropy coder, PARTITION = 0'
        elif self.Scoc & 0x01:
            msg += 'Entropy coder, PARTITION = 1'

        msg += '\n    Coding style parameters:'
        msg += '\n        Number of resolutions:  {0}'
        msg += '\n        Code block height, width:  ({1} x {2})'
        msg += '\n        Wavelet transform:  {3}'
        msg = msg.format(self.SPcoc[0] + 1,
                         int(self._code_block_size[0]),
                         int(self._code_block_size[1]),
                         _wavelet_transform_display[self.SPcoc[4]])

        msg += '\n        '
        msg += _context_string(self.SPcoc[3])

        if self._precinct_size is not None:
            msg += '\n        Precinct size:  '
            for pps in self._precinct_size:
                msg += '(%d, %d)'.format(pps)

        return msg


class CODsegment(Segment):
    """COD segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Scod : int
        Default coding style.
    SPcod : bytes
        Coding style parameters, including quality layers, multicomponent
        transform usage, decomposition levels, code block size, style of code-
        block passes, and which wavelet transform is used.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='COD')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)

        msg += '\n    Coding style:'
        msg += '\n        Entropy coder, {0} partitions'
        msg += '\n        SOP marker segments:  {1}'
        msg += '\n        EPH marker segments:  {2}'
        msg = msg.format('with' if (self.Scod & 1) else 'without',
                         ((self.Scod & 2) > 0),
                         ((self.Scod & 4) > 0))

        if self.SPcod[3] == 0:
            mct = 'no transform specified'
        elif self.SPcod[3] & 0x01:
            mct = 'reversible'
        elif self.SPcod[3] & 0x02:
            mct = 'irreversible'
        else:
            mct = 'unknown'

        msg += '\n    '
        lines = ['Coding style parameters:',
                 '    Progression order:  {0}',
                 '    Number of layers:  {1}',
                 '    Multiple component transformation usage:  {2}',
                 '    Number of resolutions:  {3}',
                 '    Code block height, width:  ({4} x {5})',
                 '    Wavelet transform:  {6}']
        msg += '\n    '.join(lines)

        msg = msg.format(_progression_order_display[self.SPcod[0]],
                         self._layers,
                         mct,
                         self.SPcod[4] + 1,
                         int(self._code_block_size[0]),
                         int(self._code_block_size[1]),
                         _wavelet_transform_display[self.SPcod[8]])

        msg += '\n        Precinct size:  '
        if self._precinct_size is None:
            msg += 'default, 2^15 x 2^15'
        else:
            for pps in self._precinct_size:
                msg += '({0}, {1})'.format(pps[0], pps[1])

        msg += '\n        '
        msg += _context_string(self.SPcod[7])

        return msg


class CMEsegment(Segment):
    """CME (comment and  extention)  segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Rcme : int
        Registration value of the marker segment.  Zero means general binary
        values, otherwise probably a string encoded in latin-1.
    Ccme:  bytes
        Raw bytes representing the comment data.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='CME')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self) + '\n'
        if self.Rcme == 1:
            # latin-1 string
            msg += '    "{0}"'
            msg = msg.format(self.Ccme.decode('latin-1'))
        else:
            msg += "    binary data (Rcme = {0}):  {1} bytes"
            msg = msg.format(self.Rcme, len(self.Ccme))
        return msg


class CRGsegment(Segment):
    """CRG (component registration) segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Xcrg, Ycrg : int
        Horizontal, vertical offset for each component
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='CRG')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n    Vertical, Horizontal offset: '
        for j in range(len(self.Xcrg)):
            msg += ' ({0:.2f}, {1:.2f})'.format(self.Ycrg[j]/65535.0,
                                                self.Xcrg[j]/65535.0)
        return msg


class EOCsegment(Segment):
    """EOC segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker, making the length for this marker
        segment to be zero.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='EOC')
        self.__dict__.update(**kwargs)


class PODsegment(Segment):
    """Progression Order Default/Change (POD) segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    RSpod : tuple
        resolution indices for start of a progression
    CSpod : tuple
        component indices for start of a progression
    LYEpod : tuple
        layer indices for end of a progression
    REpod : tuple
        resolution indices for end of a progression
    CEpod : tuple
        component indices for end of a progression
    Ppod : tuple
        progression order for each change

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='POD')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)
        for j in range(len(self.RSpod)):

            msg += '\n    '
            lines = ['Progression change {0}:',
                     '    Resolution index start:  {1}',
                     '    Component index start:  {2}',
                     '    Layer index end:  {3}',
                     '    Resolution index end:  {4}',
                     '    Component index end:  {5}',
                     '    Progression order:  {6}']
            submsg = '\n    '.join(lines)
            msg += submsg.format(j,
                                 self.RSpod[j],
                                 self.CSpod[j],
                                 self.LYEpod[j],
                                 self.REpod[j],
                                 self.CEpod[j],
                                 _progression_order_display[self.Ppod[j]])

        return msg


class PLTsegment(Segment):
    """PLT segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Zplt : int
        Index of this segment relative to other PLT segments.
    Iplt : list
        Packet lengths.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='PLT')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)
        msg += "\n    Index:  {0}"
        msg += "\n    Iplt:  {1}"
        msg = msg.format(self.Zplt, self.Iplt)

        return msg


class PPMsegment(Segment):
    """PPM segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Zppm : int
        Index of this segment relative to other PPM segments.
    data: byte array
        Raw data buffer, constitutes both Nppm and Ippm fields.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='PPM')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n    Index:  {0}'
        msg += '\n    Data:  {1} uninterpreted bytes'
        msg = msg.format(self.Zppm, len(self.data))
        return msg


class PPTsegment(Segment):
    """PPT segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Zppt : int
        Index of this segment relative to other PPT segments
    Ippt : list
        Uninterpreted packet headers.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='PPT')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n    Index:  {0}'
        msg += '\n    Packet headers:  {1} uninterpreted bytes'
        msg = msg.format(self.Zppt, len(self.Ippt))
        return msg


class QCCsegment(Segment):
    """QCC segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Cqcc : int
        Index of associated component.
    Sqcc : int
        Quantization style for this component.
    SPqcc : iterable bytes
        Quantization value for each sub-band.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='QCC')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)

        msg += '\n    Associated Component:  {0}'.format(self.Cqcc)
        msg += self._print_quantization_style(self.Sqcc)
        msg += '{0} guard bits'.format(self._guardBits)

        step_size = zip(self._mantissa, self._exponent)
        msg += '\n    Step size:  ' + str(list(step_size))
        return msg


class QCDsegment(Segment):
    """QCD segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Sqcd : int
        Quantization style for all components.
    SPqcd : iterable bytes
        Quantization step size values (uninterpreted).

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='QCD')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)

        msg += self._print_quantization_style(self.Sqcd)

        msg += '{0} guard bits'.format(self._guardBits)

        step_size = zip(self._mantissa, self._exponent)
        msg += '\n    Step size:  ' + str(list(step_size))
        return msg


class RGNsegment(Segment):
    """RGN segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Crgn : int
        Associated component.
    Srgn : int
        ROI style.
    SPrgn : int
        Parameter for ROI style.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='RGN')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)

        msg += '\n    Associated component:  {0}'
        msg += '\n    ROI style:  {1}'
        msg += '\n    Parameter:  {2}'
        msg = msg.format(self.Crgn, self.Srgn, self.SPrgn)

        return msg


class SIZsegment(Segment):
    """Container for SIZ segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Rsiz : int
        Capabilities (profile) of codestream.
    Xsiz, Ysiz : int
        Width, height of reference grid.
    XOsiz, YOsiz : int
        Horizontal, vertical offset of reference grid.
    XTsiz, YTsiz : int
        Width and height of reference tile with respect to the reference grid.
    XTOsiz, YTOsiz : int
        Horizontal and vertical offsets of tile from origin of reference grid.
    Ssiz : iterable bytes
        Precision (depth) in bits and sign of each component.
    XRsiz, YRsiz : int
        Horizontal and vertical sample separations with respect to reference
        grid.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='SIZ')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n    '

        lines = ['Profile:  {0}',
                 'Reference Grid Height, Width:  ({1} x {2})',
                 'Vertical, Horizontal Reference Grid Offset:  ({3} x {4})',
                 'Reference Tile Height, Width:  ({5} x {6})',
                 'Vertical, Horizontal Reference Tile Offset:  ({7} x {8})',
                 'Bitdepth:  {9}',
                 'Signed:  {10}',
                 'Vertical, Horizontal Subsampling:  {11}']
        msg += '\n    '.join(lines)
        msg = msg.format(_capabilities_display[self.Rsiz],
                         self.Ysiz, self.Xsiz,
                         self.YOsiz, self.XOsiz,
                         self.YTsiz, self.XTsiz,
                         self.YTOsiz, self.XTOsiz,
                         self._bitdepth,
                         self._signed,
                         tuple(zip(self.YRsiz, self.XRsiz)))

        return msg


class SOCsegment(Segment):
    """SOC segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker, making the length for this marker
        segment to be zero.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='SOC')
        self.__dict__.update(**kwargs)


class SODsegment(Segment):
    """Container for Start of Data (SOD) segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker, making the length for this marker
        segment to be zero.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='SOD')
        self.__dict__.update(**kwargs)


class EPHsegment(Segment):
    """Container for End of Packet (EPH) header information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker, making the length for this marker
        segment to be zero.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='EPH')
        self.__dict__.update(**kwargs)


class SOPsegment(Segment):
    """Container for Start of Pata (SOP) segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='SOP')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n    Nsop:  {0}'.format(self.Nsop)
        return msg


class SOTsegment(Segment):
    """Container for Start of Tile (SOT) segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Isot : int
        Index of this particular tile.
    Psot : int
        Length, in bytes, from first byte of this SOT marker segment to the
        end of the data of that tile part.
    TPsot : int
        Tile part instance.
    TNsot : int
        Number of tile-parts of a tile in codestream.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='SOT')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n    '
        lines = ['Tile part index:  {0}',
                 'Tile part length:  {1}',
                 'Tile part instance:  {2}',
                 'Number of tile parts:  {3}']
        msg += '\n    '.join(lines)
        msg = msg.format(self.Isot,
                         self.Psot,
                         self.TPsot,
                         self.TNsot)
        return msg


class TLMsegment(Segment):
    """Container for TLM segment information.

    Attributes
    ----------
    id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    Ztlm : int
        index relative to other TML marksers
    Ttlm : int
        number of the ith tile-part
    Ptlm : int
        length in bytes from beginning of the SOT marker of the ith
        tile-part to the end of the data for that tile part

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, **kwargs):
        Segment.__init__(self, id='TLM')
        self.__dict__.update(**kwargs)

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n    '
        lines = ['Index:  {0}',
                 'Tile number:  {1}',
                 'Length:  {2}']
        msg += '\n    '.join(lines)
        msg = msg.format(self.Ztlm,
                         self.Ttlm,
                         self.Ptlm)

        return msg


def _parse_precinct_size(buffer):
    """Compute precinct size from SPcod or SPcoc."""
    SPcocd = np.frombuffer(buffer, dtype=np.uint8)
    precinct_size = []
    for x in SPcocd:
        ep2 = (x & 0xF0) >> 4
        ep1 = x & 0x0F
        precinct_size.append((2 ** ep1, 2 ** ep2))
    return precinct_size


def _context_string(context):
    """Produce a string to represent the code block context"""
    msg = 'Code block context:\n            '
    lines = ['Selective arithmetic coding bypass:  {0}',
             'Reset context probabilities on coding pass boundaries:  {1}',
             'Termination on each coding pass:  {2}',
             'Vertically stripe causal context:  {3}',
             'Predictable termination:  {4}',
             'Segmentation symbols:  {5}']
    msg += '\n            '.join(lines)
    msg = msg.format(((context & 0x01) > 0),
                     ((context & 0x02) > 0),
                     ((context & 0x04) > 0),
                     ((context & 0x08) > 0),
                     ((context & 0x10) > 0),
                     ((context & 0x20) > 0))
    return msg
