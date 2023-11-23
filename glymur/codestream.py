"""Codestream information.

The module contains classes used to store information parsed from JPEG 2000
codestreams.
"""

# standard library imports
import struct
import sys
import warnings

# 3rd party library imports
import numpy as np

# local imports
from .core import (
    LRCP, RLCP, RPCL, PCRL, CPRL,
    WAVELET_XFORM_9X7_IRREVERSIBLE, WAVELET_XFORM_5X3_REVERSIBLE
)
from .lib import openjp2 as opj2


_PROGRESSION_ORDER_DISPLAY = {
    LRCP: 'LRCP',
    RLCP: 'RLCP',
    RPCL: 'RPCL',
    PCRL: 'PCRL',
    CPRL: 'CPRL',
}

_WAVELET_XFORM_DISPLAY = {
    WAVELET_XFORM_9X7_IRREVERSIBLE: '9-7 irreversible',
    WAVELET_XFORM_5X3_REVERSIBLE: '5-3 reversible'
}

_NO_PROFILE = 0
_PROFILE_0 = 1
_PROFILE_1 = 2
_PROFILE_3 = 3
_PROFILE_4 = 4

# no profile, conform to 15444-1
_PROFILE_NONE = 0x0000

# Profile 0 as described in 15444-1,Table A.45
_PROFILE_0 = 0x0001

# Profile 1 as described in 15444-1,Table A.45
_PROFILE_1 = 0x0002

# At least 1 extension defined in 15444-2 (Part-2)
_PROFILE_PART2 = 0x8000

# 2K cinema profile defined in 15444-1 AMD1
_PROFILE_CINEMA_2K = 0x0003

# 4K cinema profile defined in 15444-1 AMD1
_PROFILE_CINEMA_4K = 0x0004

# Scalable 2K cinema profile defined in 15444-1 AMD2
_PROFILE_CINEMA_S2K = 0x0005

# Scalable 4K cinema profile defined in 15444-1 AMD2
_PROFILE_CINEMA_S4K = 0x0006

# Long term storage cinema profile defined in 15444-1 AMD2
_PROFILE_CINEMA_LTS = 0x0007

# Single Tile Broadcast profile defined in 15444-1 AMD3
_PROFILE_BC_SINGLE = 0x0100

# Multi Tile Broadcast profile defined in 15444-1 AMD3
_PROFILE_BC_MULTI = 0x0200

# Multi Tile Reversible Broadcast profile defined in 15444-1 AMD3
_PROFILE_BC_MULTI_R = 0x0300

# 2K Single Tile Lossy IMF profile defined in 15444-1 AMD 8
_PROFILE_IMF_2K = 0x0400

# 4K Single Tile Lossy IMF profile defined in 15444-1 AMD 8
_PROFILE_IMF_4K = 0x0500

# 8K Single Tile Lossy IMF profile defined in 15444-1 AMD 8
_PROFILE_IMF_8K = 0x0600

# 2K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8
_PROFILE_IMF_2K_R = 0x0700

# 4K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8
_PROFILE_IMF_4K_R = 0x0800

# 8K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8
_PROFILE_IMF_8K_R = 0x0900

# How to display the codestream profile.
_CAPABILITIES_DISPLAY = {
    _PROFILE_NONE: 'no profile',
    _PROFILE_0: '0',
    _PROFILE_1: '1',
    _PROFILE_PART2: 'at least 1 extension defined in 15444-2 (Part-2)',
    _PROFILE_CINEMA_2K: '2K cinema',
    _PROFILE_CINEMA_4K: '4K cinema',
    _PROFILE_CINEMA_S2K: 'scalable 2K cinema',
    _PROFILE_CINEMA_S4K: 'scalable 4K cinema',
    _PROFILE_CINEMA_LTS: 'long term storage cinema',
    _PROFILE_BC_SINGLE: 'single tile broadcast',
    _PROFILE_BC_MULTI: 'multi tile broadcast',
    _PROFILE_BC_MULTI_R: 'multi tile reversible broadcast',
    _PROFILE_IMF_2K: '2K single tile lossy IMF',
    _PROFILE_IMF_4K: '4K single tile lossy IMF',
    _PROFILE_IMF_8K: '8K single tile lossy IMF',
    _PROFILE_IMF_2K_R: '2K single/multi tile reversible IMF',
    _PROFILE_IMF_4K_R: '4K single/multi tile reversible IMF',
    _PROFILE_IMF_8K_R: '8K single/multi tile reversible IMF',
}

_KNOWN_PROFILES = _CAPABILITIES_DISPLAY.keys()


class J2KParseError(struct.error):
    pass


class Codestream(object):
    """Container for codestream information.

    Attributes
    ----------
    segment : iterable
        list of marker segments
    offset : int
        Offset of the codestream from start of the file in bytes.
    length : int
        Length of the codestream in bytes.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    # These two begin their lives as class attributes that usually become
    # instance attributes.  The reason why isn't important for users; it's only
    # important for testing purposes.
    _csiz = -1  # Number of components in the image.
    _parse_tpart_flag = False  # Do we parse the bit stream for SOP / EPH?

    def __init__(self, fptr, length, header_only=True):
        """
        Parameters
        ----------
        fptr : file
            Open file object.
        length : int
            Length of the codestream in bytes.
        header_only : bool, optional
            If True, only marker segments in the main header are parsed.
            Supplying False may impose a large performance penalty.
        """
        # Map each of the known markers to a method that processes them.
        # Consult table A-1 in ISO/IEC FCD15444-1 for the definitive list.
        #
        # Some markers are mentiond in the following specs:
        #     ITU-T Rec. T.81
        #     ITU-T Rec. T.84
        #     ITU-T Rec. T.87
        # We really don't know what to do with them, so they are treated as if
        # they are reserved markers.
        self.parse_marker_segment_fcn = {

            # The following are definitively reserved markers according to
            # table A-1 in ISO/IEC FCD15444-1.
            0xff30: self._parse_reserved_marker,
            0xff31: self._parse_reserved_marker,
            0xff32: self._parse_reserved_marker,
            0xff33: self._parse_reserved_marker,
            0xff34: self._parse_reserved_marker,
            0xff35: self._parse_reserved_marker,
            0xff36: self._parse_reserved_marker,
            0xff37: self._parse_reserved_marker,
            0xff38: self._parse_reserved_marker,
            0xff39: self._parse_reserved_marker,
            0xff3a: self._parse_reserved_marker,
            0xff3b: self._parse_reserved_marker,
            0xff3c: self._parse_reserved_marker,
            0xff3d: self._parse_reserved_marker,
            0xff3e: self._parse_reserved_marker,
            0xff3f: self._parse_reserved_marker,

            # 0xff4f:  SOC (already encountered by the time we get here)
            0xff50: self._parse_reserved_segment,
            0xff51: self._parse_siz_segment,
            0xff52: self._parse_cod_segment,
            0xff53: self._parse_coc_segment,
            0xff54: self._parse_reserved_segment,
            0xff55: self._parse_tlm_segment,
            0xff56: self._parse_reserved_segment,
            0xff57: self._parse_reserved_segment,
            0xff58: self._parse_plt_segment,
            0xff59: self._parse_reserved_segment,
            0xff5a: self._parse_reserved_segment,
            0xff5b: self._parse_reserved_segment,
            0xff5c: self._parse_qcd_segment,
            0xff5d: self._parse_qcc_segment,
            0xff5e: self._parse_rgn_segment,
            0xff5f: self._parse_pod_segment,
            0xff60: self._parse_ppm_segment,
            0xff61: self._parse_ppt_segment,
            0xff62: self._parse_reserved_segment,
            0xff63: self._parse_crg_segment,
            0xff64: self._parse_cme_segment,
            0xff65: self._parse_reserved_segment,
            0xff66: self._parse_reserved_segment,
            0xff67: self._parse_reserved_segment,
            0xff68: self._parse_reserved_segment,
            0xff69: self._parse_reserved_segment,
            0xff6a: self._parse_reserved_segment,
            0xff6b: self._parse_reserved_segment,
            0xff6c: self._parse_reserved_segment,
            0xff6d: self._parse_reserved_segment,
            0xff6e: self._parse_reserved_segment,
            0xff6f: self._parse_reserved_segment,
            0xff90: self._parse_sot_segment,
            # 0xff91:  SOP (only found in bit stream)
            # 0xff92:  EPH (only found in bit stream)
            0xff93: self._parse_sod_marker,
            0xffd9: self._parse_eoc_marker,

        }

        self.offset = fptr.tell()
        self.length = length
        self.header_only = header_only

        self.segment = []

        self._parse(fptr)

    def _parse(self, fptr):
        """
        Parse the codestream.
        """
        # First two bytes are the SOC marker.  We already know that.
        read_buffer = fptr.read(2)
        segment = SOCsegment(offset=fptr.tell() - 2, length=0)
        self.segment.append(segment)

        self._tile_offset = []
        self._tile_length = []

        while True:

            read_buffer = fptr.read(2)

            try:
                self._marker_id, = struct.unpack('>H', read_buffer)
            except struct.error as e:
                msg = (
                    f"Unable to read an expected marker in the codestream "
                    f"at byte offset {fptr.tell()}.  \"{e}\"."
                )
                raise J2KParseError(msg)

            if self._marker_id < 0xff00:
                offset = fptr.tell() - 2
                msg = (
                    f'Invalid codestream marker at byte offset {offset}.  It '
                    f'must be must be greater than 0xff00, but '
                    f'found 0x{self._marker_id:04x} instead.  Codestream '
                    f'parsing will cease.'
                )
                raise ValueError(msg)

            self._offset = fptr.tell() - 2

            if self._marker_id == 0xff90 and self.header_only:
                # Start-of-tile (SOT) means that we are out of the main header
                # and there is no need to go further.
                break

            try:
                segment = self.parse_marker_segment_fcn[self._marker_id](fptr)
            except KeyError:
                segment = self._parse_reserved_segment(fptr)

            self.segment.append(segment)

            if self._marker_id == 0xffd9:
                # end of codestream, should break.
                break

            # If the marker ID is SOD, then we need to seek past the tile part
            # bit stream.
            if self._marker_id == 0xff93:

                if self._parse_tpart_flag and not self.header_only:
                    # But first parse the tile part bit stream for SOP and
                    # EPH segments.
                    self._parse_tile_part_bit_stream(
                        fptr, segment, self._tile_length[-1]
                    )

                new_offset = self._tile_offset[-1] + self._tile_length[-1]
                fptr.seek(new_offset)

    def _parse_reserved_segment(self, fptr):
        """
        Parse valid marker segment, segment description is unknown.

        Parameters
        ----------
        fptr : file-like object
            The file to parse.

        Returns
        -------
        The current segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(2)
        length, = struct.unpack('>H', read_buffer)
        if length > 0:
            data = fptr.read(length - 2)
        else:
            data = None

        segment = Segment(marker_id=f'0x{self._marker_id:x}',
                          offset=offset, length=length, data=data)
        return segment

    def _parse_tile_part_bit_stream(self, fptr, sod_marker, tile_length):
        """Parse the tile part bit stream for SOP, EPH marker segments."""
        read_buffer = fptr.read(tile_length)
        # The tile length could possibly be too large and extend past
        # the end of file.  We need to be a bit resilient.
        count = min(tile_length, len(read_buffer))
        packet = np.frombuffer(read_buffer, dtype=np.uint8, count=count)

        indices = np.where(packet == 0xff)
        for idx in indices[0]:
            try:
                if packet[idx + 1] == 0x91 and (idx < (len(packet) - 5)):
                    offset = sod_marker.offset + 2 + idx
                    length = 4
                    nsop = packet[(idx + 4):(idx + 6)].view('uint16')[0]
                    if sys.byteorder == 'little':
                        nsop = nsop.byteswap()
                    segment = SOPsegment(nsop, length, offset)
                    self.segment.append(segment)
                elif packet[idx + 1] == 0x92:
                    offset = sod_marker.offset + 2 + idx
                    length = 0
                    segment = EPHsegment(length, offset)
                    self.segment.append(segment)
            except IndexError:
                continue

    def __str__(self):
        msg = 'Codestream:\n'
        for segment in self.segment:
            strs = str(segment)

            # Add indentation
            strs = [('    ' + x + '\n') for x in strs.split('\n')]
            msg += ''.join(strs)
        return msg.rstrip()

    def _parse_cme_segment(self, fptr):
        """Parse the CME marker segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        CMESegment
            The current CME segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(4)
        data = struct.unpack('>HH', read_buffer)
        length = data[0]
        rcme = data[1]
        ccme = fptr.read(length - 4)

        return CMEsegment(rcme, ccme, length, offset)

    def _parse_coc_segment(self, fptr):
        """Parse the COC marker segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        COCSegment
            The current COC segment.
        """
        kwargs = {}
        offset = fptr.tell() - 2
        kwargs['offset'] = offset

        read_buffer = fptr.read(2)
        length, = struct.unpack('>H', read_buffer)
        kwargs['length'] = length

        fmt = '>B' if self._csiz <= 255 else '>H'
        nbytes = 1 if self._csiz <= 255 else 2
        read_buffer = fptr.read(nbytes)
        ccoc, = struct.unpack(fmt, read_buffer)

        read_buffer = fptr.read(1)
        scoc, = struct.unpack('>B', read_buffer)

        numbytes = offset + 2 + length - fptr.tell()
        read_buffer = fptr.read(numbytes)
        spcoc = np.frombuffer(read_buffer, dtype=np.uint8)
        spcoc = spcoc

        return COCsegment(ccoc, scoc, spcoc, length, offset)

    @classmethod
    def _parse_cod_segment(cls, fptr):
        """Parse the COD segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        CODSegment
            The current COD segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(2)
        length, = struct.unpack('>H', read_buffer)

        read_buffer = fptr.read(length - 2)

        lst = struct.unpack_from('>BBHBBBBBB', read_buffer, offset=0)
        scod, prog, nlayers, mct, nr, xcb, ycb, cstyle, xform = lst

        if len(read_buffer) > 10:
            precinct_size = _parse_precinct_size(read_buffer[10:])
        else:
            precinct_size = None

        sop = (scod & 2) > 0
        eph = (scod & 4) > 0

        if sop or eph:
            cls._parse_tpart_flag = True
        else:
            cls._parse_tpart_flag = False

        pargs = (scod, prog, nlayers, mct, nr, xcb, ycb, cstyle, xform,
                 precinct_size)

        return CODsegment(*pargs, length=length, offset=offset)

    def _parse_crg_segment(self, fptr):
        """Parse the CRG marker segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        CRGSegment
            The current CRG segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(2)
        length, = struct.unpack('>H', read_buffer)

        read_buffer = fptr.read(4 * self._csiz)
        data = struct.unpack('>' + 'HH' * self._csiz, read_buffer)
        xcrg = data[0::2]
        ycrg = data[1::2]

        return CRGsegment(xcrg, ycrg, length, offset)

    def _parse_eoc_marker(self, fptr):
        """Parse the EOC (end-of-codestream) marker.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        EOCSegment
            The current EOC segment.
        """
        offset = fptr.tell() - 2
        length = 0

        return EOCsegment(length, offset)

    def _parse_plt_segment(self, fptr):
        """Parse the PLT segment.

        The packet headers are not parsed, i.e. they remain uninterpreted raw
        data buffers.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        PLTSegment
            The current PLT segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(3)
        length, zplt = struct.unpack('>HB', read_buffer)

        numbytes = length - 3
        read_buffer = fptr.read(numbytes)
        iplt = np.frombuffer(read_buffer, dtype=np.uint8)

        packet_len = []
        plen = 0
        for byte in iplt:
            plen |= (byte & 0x7f)
            if byte & 0x80:
                # Continue by or-ing in the next byte.
                plen <<= 7
            else:
                packet_len.append(plen)
                plen = 0

        iplt = packet_len

        return PLTsegment(zplt, iplt, length, offset)

    def _parse_pod_segment(self, fptr):
        """Parse the POD segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        PODSegment
            The current POD segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(2)
        length, = struct.unpack('>H', read_buffer)

        n = ((length - 2) / 7) if self._csiz < 257 else ((length - 2) / 9)
        n = int(n)
        nbytes = n * 7 if self._csiz < 257 else n * 9
        read_buffer = fptr.read(nbytes)
        fmt = '>' + 'BBHBBB' * n if self._csiz < 257 else '>' + 'BHHBHB' * n
        pod_params = struct.unpack(fmt, read_buffer)

        return PODsegment(pod_params, length, offset)

    def _parse_ppm_segment(self, fptr):
        """Parse the PPM segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        PPMSegment
            The current PPM segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(3)
        length, zppm = struct.unpack('>HB', read_buffer)

        numbytes = length - 3
        read_buffer = fptr.read(numbytes)

        return PPMsegment(zppm, read_buffer, length, offset)

    def _parse_ppt_segment(self, fptr):
        """Parse the PPT segment.

        The packet headers are not parsed, i.e. they remain "uninterpreted"
        raw data beffers.

        Parameters
        ----------
        fptr : file object
            The file to parse.

        Returns
        -------
        PPTSegment
            The current PPT segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(3)
        length, zppt = struct.unpack('>HB', read_buffer)
        length = length
        zppt = zppt

        numbytes = length - 3
        ippt = fptr.read(numbytes)

        return PPTsegment(zppt, ippt, length, offset)

    @classmethod
    def _parse_qcc_segment(cls, fptr):
        """Parse the QCC segment.

        Parameters
        ----------
        fptr : file object
            The file to parse.

        Returns
        -------
        QCCSegment
            The current QCC segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(2)
        length, = struct.unpack('>H', read_buffer)

        read_buffer = fptr.read(length - 2)
        fmt = '>HB' if cls._csiz > 256 else '>BB'
        mantissa_exponent_offset = 3 if cls._csiz > 256 else 2
        cqcc, sqcc = struct.unpack_from(fmt, read_buffer)
        if cqcc >= cls._csiz:
            msg = ("Invalid QCC component number ({cqcc}), "
                   "the actual number of components is only {cls._csiz}.")
            warnings.warn(msg, UserWarning)

        spqcc = read_buffer[mantissa_exponent_offset:]

        return QCCsegment(cqcc, sqcc, spqcc, length, offset)

    def _parse_qcd_segment(self, fptr):
        """Parse the QCD segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        QCDSegment
            The current QCD segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(3)
        length, sqcd = struct.unpack('>HB', read_buffer)
        spqcd = fptr.read(length - 3)

        return QCDsegment(sqcd, spqcd, length, offset)

    @classmethod
    def _parse_rgn_segment(cls, fptr):
        """Parse the RGN segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        RGNSegment
            The current RGN segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(2)
        length, = struct.unpack('>H', read_buffer)

        nbytes = 3 if cls._csiz < 257 else 4
        fmt = '>BBB' if cls._csiz < 257 else '>HBB'
        read_buffer = fptr.read(nbytes)
        data = struct.unpack(fmt, read_buffer)

        length = length
        crgn = data[0]
        srgn = data[1]
        sprgn = data[2]

        return RGNsegment(crgn, srgn, sprgn, length, offset)

    @classmethod
    def _parse_siz_segment(cls, fptr):
        """Parse the SIZ segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        SIZSegment
            The current SIZ segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(2)
        length, = struct.unpack('>H', read_buffer)

        read_buffer = fptr.read(length - 2)
        data = struct.unpack_from('>HIIIIIIIIH', read_buffer)

        rsiz = data[0]
        if rsiz not in _KNOWN_PROFILES:
            msg = f"Invalid profile: (Rsiz={rsiz})."
            warnings.warn(msg, UserWarning)

        xysiz = (data[1], data[2])
        xyosiz = (data[3], data[4])
        xytsiz = (data[5], data[6])
        xytosiz = (data[7], data[8])

        # Csiz is the number of components
        Csiz = data[9]

        data = struct.unpack_from('>' + 'B' * (length - 36 - 2),
                                  read_buffer, offset=36)

        bitdepth = tuple(((x & 0x7f) + 1) for x in data[0::3])
        signed = tuple(((x & 0x80) > 0) for x in data[0::3])
        xrsiz = data[1::3]
        yrsiz = data[2::3]

        for j, subsampling in enumerate(zip(xrsiz, yrsiz)):
            if 0 in subsampling:
                msg = (f"Invalid subsampling value for component {j}: "
                       f"dx={subsampling[0]}, dy={subsampling[1]}.")
                warnings.warn(msg, UserWarning)

        try:
            num_tiles_x = (xysiz[0] - xyosiz[0]) / (xytsiz[0] - xytosiz[0])
            num_tiles_y = (xysiz[1] - xyosiz[1]) / (xytsiz[1] - xytosiz[1])
        except ZeroDivisionError:
            msg = (
                f"Invalid tile specification:  "
                f"size of {xytsiz[1]} x {xytsiz[0]}, "
                f"offset of {xytosiz[1]} x {xytsiz[0]}."
            )
            warnings.warn(msg, UserWarning)
        else:
            numtiles = np.ceil(num_tiles_x) * np.ceil(num_tiles_y)
            if numtiles > 65535:
                msg = f"Invalid number of tiles: ({numtiles})."
                warnings.warn(msg, UserWarning)

        kwargs = {
            'rsiz': rsiz,
            'xysiz': xysiz,
            'xyosiz': xyosiz,
            'xytsiz': xytsiz,
            'xytosiz': xytosiz,
            'Csiz': Csiz,
            'bitdepth': bitdepth,
            'signed': signed,
            'xyrsiz': (xrsiz, yrsiz),
            'length': length,
            'offset': offset
        }
        segment = SIZsegment(**kwargs)

        # Need to keep track of the number of components from SIZ for
        # other segments.
        cls._csiz = Csiz

        return segment

    def _parse_sod_marker(self, fptr):
        """Parse the SOD (start-of-data) segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        SODSegment
            The current SOD segment.
        """
        offset = fptr.tell() - 2
        length = 0

        return SODsegment(length, offset)

    def _parse_sot_segment(self, fptr):
        """Parse the SOT segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        SOTSegment
            The current SOT segment.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(10)
        data = struct.unpack('>HHIBB', read_buffer)

        length = data[0]
        isot = data[1]
        psot = data[2]
        tpsot = data[3]
        tnsot = data[4]

        segment = SOTsegment(isot, psot, tpsot, tnsot, length, offset)

        # Need to keep easy access to tile offsets and lengths for when
        # we encounter start-of-data marker segments.

        self._tile_offset.append(segment.offset)
        if segment.psot == 0:
            tile_part_length = self.offset + self.length - segment.offset - 2
        else:
            tile_part_length = segment.psot
        self._tile_length.append(tile_part_length)

        return segment

    def _parse_tlm_segment(self, fptr):
        """Parse the TLM segment.

        Parameters
        ----------
        fptr : file
            Open file object.

        Returns
        -------
        TLMSegment
            The current TLM segment.

        See tables A-33 and A-34 in 15444.pdf.
        """
        offset = fptr.tell() - 2

        read_buffer = fptr.read(2)
        length, = struct.unpack('>H', read_buffer)

        read_buffer = fptr.read(length - 2)
        ztlm, stlm = struct.unpack_from('>BB', read_buffer)
        st = (stlm >> 4) & 0x3
        sp = (stlm >> 6) & 0x1

        nbytes = length - 4
        ntiles = nbytes / (st + (sp + 1) * 2)

        fmt = {0: '', 1: 'B', 2: 'H'}[st]
        fmt += {0: 'H', 1: 'I'}[sp]

        data = struct.unpack_from('>' + fmt * int(ntiles), read_buffer,
                                  offset=2)
        if st == 0:
            ttlm = None
            ptlm = data
        else:
            ttlm = data[0::2]
            ptlm = data[1::2]

        return TLMsegment(ztlm, ttlm, ptlm, length, offset)

    def _parse_reserved_marker(self, fptr):
        """Marker range between 0xff30 and 0xff39.
        """
        the_id = f'0x{self._marker_id:x}'
        segment = Segment(marker_id=the_id, offset=self._offset, length=0)
        return segment


class Segment(object):
    """Segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    data : bytes iterable or None
        Uninterpreted buffer of raw bytes, only used where a segment is not
        well understood.
    """
    def __init__(self, marker_id='', offset=-1, length=-1, data=None):
        self.marker_id = marker_id
        self.offset = offset
        self.length = length
        self.data = data

    def __str__(self):
        msg = (f'{self.marker_id} marker segment @ '
               f'({self.offset}, {self.length})')
        return msg


class COCsegment(Segment):
    """COC (Coding style Component) segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    ccoc : int
        Index of associated component.
    scoc : int
        Coding style for this component.
    spcoc : byte array
        Coding style parameters for this component.
    precinct_size : list of tuples
        Dimensions of precinct.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, ccoc, scoc, spcoc, length, offset):
        super().__init__(marker_id='COC')
        self.ccoc = ccoc
        self.scoc = scoc
        self.spcoc = spcoc

        self.code_block_size = (4 * 2 ** self.spcoc[2], 4 * 2 ** self.spcoc[1])

        if len(self.spcoc) > 5:
            self.precinct_size = _parse_precinct_size(self.spcoc[5:])
        else:
            self.precinct_size = ((32768, 32768))

        self.length = length
        self.offset = offset

    def __str__(self):
        msg = Segment.__str__(self)

        msg += '\n'

        partition = 0 if self.scoc == 0 else 1
        width = int(self.code_block_size[0])
        height = int(self.code_block_size[1])
        xform = _WAVELET_XFORM_DISPLAY[self.spcoc[4]]
        msg += (
            f'    Associated component:  {self.ccoc}\n'
            f'    Coding style for this component:  '
            f'Entropy coder, PARTITION = {partition}\n'
            f'    Coding style parameters:\n'
            f'        Number of decomposition levels:  {self.spcoc[0]}\n'
            f'        Code block height, width:  ({width} x {height})\n'
            f'        Wavelet transform:  {xform}\n'
            f'        Precinct size:  {self.precinct_size}\n'
            f'        {_context_string(self.spcoc[3])}'
        )

        return msg


class CODsegment(Segment):
    """COD segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment
    offset : int
        Offset of marker segment in bytes from beginning of file
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    scod : int
        Default coding style
    code_block_size : tuple
        Size of code block
    layers : int
        Number of decomposition levels
    progression_order : int
        Progression order
    mct : int
        Multiple component transform usage
    num_res : int
        Number of decomposition levels.  Zero implies no transformation.
    xform : int
        Wavelet transform used
    cstyle : int
        Style of the code-block passes
    precinct_size : list
        2-tuples of precinct sizes.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system

        pargs = (scod, prog, nlayers, mct, nr, xcb, ycb, cstyle, xform,
                 precinct_size)
    """
    def __init__(self, scod, prog_order, num_layers, mct, nr, xcb, ycb,
                 cstyle, xform, precinct_size, length=0, offset=0):
        super().__init__(marker_id='COD')
        self.scod = scod
        self.length = length
        self.offset = offset
        self.mct = mct
        self.cstyle = cstyle
        self.xform = xform

        self.layers = num_layers
        self._numresolutions = nr

        if nr > opj2.J2K_MAXRLVLS:
            msg = f"Invalid number of resolutions: ({nr + 1})."
            warnings.warn(msg, UserWarning)
        self.num_res = nr

        if prog_order not in [LRCP, RLCP, RPCL, PCRL, CPRL]:
            msg = f"Invalid progression order in COD segment: {prog_order}."
            warnings.warn(msg, UserWarning)
        self.prog_order = prog_order

        if xform not in [WAVELET_XFORM_9X7_IRREVERSIBLE,
                         WAVELET_XFORM_5X3_REVERSIBLE]:
            msg = f"Invalid wavelet transform in COD segment: {xform}."
            warnings.warn(msg, UserWarning)

        self.code_block_size = 4 * 2 ** ycb, 4 * 2 ** xcb

        if precinct_size is None:
            self.precinct_size = ((2 ** 15, 2 ** 15))
        else:
            self.precinct_size = precinct_size

    def __str__(self):
        msg = Segment.__str__(self)

        msg += '\n'
        msg += (
            '    Coding style:\n'
            '        Entropy coder, {with_without} partitions\n'
            '        SOP marker segments:  {sop}\n'
            '        EPH marker segments:  {eph}\n'
            '    Coding style parameters:\n'
            '        Progression order:  {prog}\n'
            '        Number of layers:  {num_layers}\n'
            '        Multiple component transformation usage:  {mct}\n'
            '        Number of decomposition levels:  {num_dlevels}\n'
            '        Code block height, width:  ({cbh} x {cbw})\n'
            '        Wavelet transform:  {xform}\n'
            '        Precinct size:  {precinct_size}\n'
            '        {code_block_context}'
        )

        if self.mct == 0:
            mct_str = 'no transform specified'
        elif self.mct & 0x01:
            mct_str = 'reversible'
        elif self.mct & 0x02:
            mct_str = 'irreversible'
        else:
            mct_str = 'unknown'

        try:
            progression_order = _PROGRESSION_ORDER_DISPLAY[self.prog_order]
        except KeyError:
            progression_order = f'{self.prog_order} (invalid)'
        try:
            xform = _WAVELET_XFORM_DISPLAY[self.xform]
        except KeyError:
            xform = f'{self.xform} (invalid)'
        msg = msg.format(
            with_without='with' if (self.scod & 1) else 'without',
            sop=((self.scod & 2) > 0),
            eph=((self.scod & 4) > 0),
            prog=progression_order,
            num_layers=self.layers,
            mct=mct_str,
            num_dlevels=self.num_res,
            cbh=int(self.code_block_size[0]),
            cbw=int(self.code_block_size[1]),
            xform=xform,
            precinct_size=self.precinct_size,
            code_block_context=_context_string(self.cstyle)
        )

        return msg


class CMEsegment(Segment):
    """CME (comment and  extention)  segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    rcme : int
        Registration value of the marker segment.  Zero means general binary
        values, otherwise probably a string encoded in latin-1.
    ccme:  bytes
        Raw bytes representing the comment data.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, rcme, ccme, length=-1, offset=-1):
        super().__init__(marker_id='CME')
        self.rcme = rcme
        self.ccme = ccme
        self.length = length
        self.offset = offset

    def __str__(self):
        msg = Segment.__str__(self) + '\n'
        if self.rcme == 1:
            # latin-1 string
            msg += '    "{ccme}"'.format(ccme=self.ccme.decode('latin-1'))
        else:
            msg += "    binary data (rcme = {rcme}):  {nbytes} bytes"
            msg = msg.format(rcme=self.rcme, nbytes=len(self.ccme))
        return msg


class CRGsegment(Segment):
    """CRG (component registration) segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    xcrg, ycrg : int sequences
        Horizontal, vertical offset for each component
    """
    def __init__(self, xcrg, ycrg, length, offset):
        super().__init__(marker_id='CRG')
        self.xcrg = xcrg
        self.ycrg = ycrg
        self.length = length
        self.offset = offset

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n    Vertical, Horizontal offset: '
        for j in range(len(self.xcrg)):
            msg += ' ({0:.2f}, {1:.2f})'.format(
                self.ycrg[j] / 65535.0, self.xcrg[j] / 65535.0
            )
        return msg


class EOCsegment(Segment):
    """EOC segment information.

    Attributes
    ----------
    marker_id : str
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
    def __init__(self, length, offset):
        super().__init__(marker_id='EOC')
        self.length = length
        self.offset = offset


class PODsegment(Segment):
    """Progression Order Default/Change (POD) segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    rspod : tuple
        resolution indices for start of a progression
    cspod : tuple
        component indices for start of a progression
    lyepod : tuple
        layer indices for end of a progression
    repod : tuple
        resolution indices for end of a progression
    cdpod : tuple
        component indices for end of a progression
    ppod : tuple
        progression order for each change

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, pod_params, length, offset):
        super().__init__(marker_id='POD')

        self.rspod = pod_params[0::6]
        self.cspod = pod_params[1::6]
        self.lyepod = pod_params[2::6]
        self.repod = pod_params[3::6]
        self.cdpod = pod_params[4::6]
        self.ppod = pod_params[5::6]
        self.length = length
        self.offset = offset

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n'

        submsg = ('    Progression change {0}:\n'
                  '        Resolution index start:  {1}\n'
                  '        Component index start:  {2}\n'
                  '        Layer index end:  {3}\n'
                  '        Resolution index end:  {4}\n'
                  '        Component index end:  {5}\n'
                  '        Progression order:  {6}\n')
        for j in range(len(self.rspod)):

            try:
                progorder = _PROGRESSION_ORDER_DISPLAY[self.ppod[j]]
            except KeyError:
                progorder = f'invalid value: {self.ppod[j]}'

            msg += submsg.format(
                j,
                self.rspod[j],
                self.cspod[j],
                self.lyepod[j],
                self.repod[j],
                self.cdpod[j],
                progorder
            )

        return msg.rstrip()


class PLTsegment(Segment):
    """PLT segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    zplt : int
        Index of this segment relative to other PLT segments.
    iplt : list
        Packet lengths.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, zplt, iplt, length, offset):
        super().__init__(marker_id='PLT')
        self.zplt = zplt
        self.iplt = iplt
        self.length = length
        self.offset = offset

    def __str__(self):
        msg = Segment.__str__(self)
        msg += (
            f"\n    Index:  {self.zplt}"
            f"\n    Iplt:  {self.iplt}"
        )

        return msg


class PPMsegment(Segment):
    """PPM segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    zppm : int
        Index of this segment relative to other PPM segments.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, zppm, data, length, offset):
        super().__init__(marker_id='PPM')
        self.zppm = zppm

        # both Nppm and Ippms information stored in data
        self.data = data

        self.length = length
        self.offset = offset

    def __str__(self):
        msg = Segment.__str__(self)
        msg += (
            f'\n    Index:  {self.zppm}'
            f'\n    Data:  {len(self.data)} uninterpreted bytes'
        )
        return msg


class PPTsegment(Segment):
    """PPT segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    zppt : int
        Index of this segment relative to other PPT segments
    ippt : list
        Uninterpreted packet headers.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, zppt, ippt, length, offset):
        super().__init__(marker_id='PPT')
        self.zppt = zppt
        self.ippt = ippt
        self.length = length
        self.offset = offset

    def __str__(self):
        msg = Segment.__str__(self)
        msg += (
            f'\n    Index:  {self.zppt}'
            f'\n    Packet headers:  {len(self.ippt)} uninterpreted bytes'
        )
        return msg


class QCCsegment(Segment):
    """QCC segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    cqcc : int
        Index of associated component.
    sqcc : int
        Quantization style for this component.
    spqcc : iterable bytes
        Quantization value for each sub-band.
    mantissa, exponent : iterable
        Defines quantization factors.
    guard_bits : int
        Number of guard bits.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, cqcc, sqcc, spqcc, length, offset):
        super().__init__(marker_id='QCC')
        self.cqcc = cqcc
        self.sqcc = sqcc
        self.spqcc = spqcc
        self.length = length
        self.offset = offset

        self.mantissa, self.exponent = parse_quantization(self.spqcc,
                                                          self.sqcc)
        self.guard_bits = (self.sqcc & 0xe0) >> 5

    def __str__(self):
        msg = Segment.__str__(self)

        msg += f'\n    Associated Component:  {self.cqcc}'
        msg += _print_quantization_style(self.sqcc)
        msg += f'{self.guard_bits} guard bits'

        step_size = zip(self.mantissa, self.exponent)
        msg += '\n    Step size:  ' + str(list(step_size))
        return msg


class QCDsegment(Segment):
    """QCD segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    sqcd : int
        Quantization style for all components.
    spqcd : iterable bytes
        Quantization step size values (uninterpreted).
    mantissa, exponent : iterable
        Defines quantization factors.
    guard_bits : int
        Number of guard bits.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, sqcd, spqcd, length, offset):
        super().__init__(marker_id='QCD')

        self.sqcd = sqcd
        self.spqcd = spqcd
        self.length = length
        self.offset = offset

        mantissa, exponent = parse_quantization(self.spqcd, self.sqcd)
        self.mantissa = mantissa
        self.exponent = exponent
        self.guard_bits = (self.sqcd & 0xe0) >> 5

    def __str__(self):
        msg = Segment.__str__(self)

        msg += _print_quantization_style(self.sqcd)

        msg += f'{self.guard_bits} guard bits'

        step_size = zip(self.mantissa, self.exponent)
        msg += '\n    Step size:  ' + str(list(step_size))
        return msg


class RGNsegment(Segment):
    """RGN segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    crgn : int
        Associated component.
    srgn : int
        ROI style.
    sprgn : int
        Parameter for ROI style.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, crgn, srgn, sprgn, length=-1, offset=-1):
        super().__init__(marker_id='RGN')
        self.length = length
        self.offset = offset
        self.crgn = crgn
        self.srgn = srgn
        self.sprgn = sprgn

    def __str__(self):
        msg = Segment.__str__(self)

        msg += (
            f'\n    Associated component:  {self.crgn}'
            f'\n    ROI style:  {self.srgn}'
            f'\n    Parameter:  {self.sprgn}'
        )

        return msg


class SIZsegment(Segment):
    """Container for SIZ segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    rsiz : int
        Capabilities (profile) of codestream.
    xsiz, ysiz : int
        Width, height of reference grid.
    xosiz, yosiz : int
        Horizontal, vertical offset of reference grid.
    xtsiz, ytsiz : int
        Width and height of reference tile with respect to the reference grid.
    xtosiz, ytosiz : int
        Horizontal and vertical offsets of tile from origin of reference grid.
    Csiz : int
        Number of components in image.
    bitdepth : iterable bytes
        Precision (depth) in bits of each component.
    signed : iterable bool
        Signedness of each component.
    xrsiz, yrsiz : int
        Horizontal and vertical sample separations with respect to reference
        grid.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, rsiz=-1, xysiz=None, xyosiz=-1, xytsiz=-1, xytosiz=-1,
                 Csiz=-1, bitdepth=None, signed=None, xyrsiz=-1, length=-1,
                 offset=-1):
        super().__init__(marker_id='SIZ', length=length, offset=offset)

        self.rsiz = rsiz
        self.xsiz, self.ysiz = xysiz
        self.xosiz, self.yosiz = xyosiz
        self.xtsiz, self.ytsiz = xytsiz
        self.xtosiz, self.ytosiz = xytosiz
        self.Csiz = Csiz
        self.bitdepth = bitdepth
        self.signed = signed
        self.xrsiz, self.yrsiz = xyrsiz

        # ssiz attribute to be removed in 1.0.0
        lst = []
        for bitdepth, signed in zip(self.bitdepth, self.signed):
            if signed:
                lst.append((bitdepth - 1) | 0x80)
            else:
                lst.append(bitdepth - 1)
        self.ssiz = tuple(lst)

    def __repr__(self):
        msg = (
            f"glymur.codestream.SIZsegment("
            f"rsiz={self.rsiz}, "
            f"xysiz={(self.xsiz, self.ysiz)}, "
            f"xyosiz={(self.xosiz, self.yosiz)}, "
            f"xytsiz={(self.xtsiz, self.ytsiz)}, "
            f"xytosiz={(self.xtosiz, self.ytosiz)}, "
            f"Csiz={self.Csiz}, "
            f"bitdepth={self.bitdepth}, "
            f"signed={self.signed}, "
            f"xyrsiz={(self.xrsiz, self.yrsiz)})"
        )
        return msg

    def __str__(self):
        msg = Segment.__str__(self)
        msg += '\n'

        msg += (
            '    Profile:  {profile}\n'
            '    Reference Grid Height, Width:  ({height} x {width})\n'
            '    Vertical, Horizontal Reference Grid Offset:  ({goy} x {gox})\n'  # noqa : E501
            '    Reference Tile Height, Width:  ({tileh} x {tilew})\n'
            '    Vertical, Horizontal Reference Tile Offset:  ({toy} x {tox})\n'  # noqa : E501
            '    Bitdepth:  {bitdepth}\n'
            '    Signed:  {signed}\n'
            '    Vertical, Horizontal Subsampling:  {subsampling}'
        )
        try:
            profile = _CAPABILITIES_DISPLAY[self.rsiz]
        except KeyError:
            profile = f'{self.rsiz} (invalid)'
        msg = msg.format(
            profile=profile,
            height=self.ysiz, width=self.xsiz,
            goy=self.yosiz, gox=self.xosiz,
            tileh=self.ytsiz, tilew=self.xtsiz,
            toy=self.ytosiz, tox=self.xtosiz,
            bitdepth=self.bitdepth,
            signed=self.signed,
            subsampling=tuple(zip(self.yrsiz, self.xrsiz))
        )

        return msg


class SOCsegment(Segment):
    """SOC segment information.

    Attributes
    ----------
    marker_id : str
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
        super().__init__(marker_id='SOC')
        self.__dict__.update(**kwargs)

    def __repr__(self):
        msg = "glymur.codestream.SOCsegment()"
        return msg


class SODsegment(Segment):
    """Container for Start of Data (SOD) segment information.

    Attributes
    ----------
    marker_id : str
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
    def __init__(self, length, offset):
        super().__init__(marker_id='SOD')
        self.length = length
        self.offset = offset


class EPHsegment(Segment):
    """Container for End of Packet (EPH) header information.

    Attributes
    ----------
    marker_id : str
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
    def __init__(self, length, offset):
        super().__init__(marker_id='EPH')
        self.length = length
        self.offset = offset


class SOPsegment(Segment):
    """Container for Start of Packet (SOP) segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    nsop : int
        Packet sequence number.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, nsop, length, offset):
        super().__init__(marker_id='SOP')
        self.nsop = nsop
        self.length = length
        self.offset = offset

    def __str__(self):
        msg = Segment.__str__(self)
        msg += f'\n    Nsop:  {self.nsop}'
        return msg


class SOTsegment(Segment):
    """Container for Start of Tile (SOT) segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    isot : int
        Index of this particular tile.
    psot : int
        Length, in bytes, from first byte of this SOT marker segment to the
        end of the data of that tile part.
    tpsot : int
        Tile part instance.
    tnsot : int
        Number of tile-parts of a tile in codestream.

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, isot, psot, tpsot, tnsot, length=-1, offset=-1):
        super().__init__(marker_id='SOT')
        self.isot = isot
        self.psot = psot
        self.tpsot = tpsot
        self.tnsot = tnsot
        self.length = length
        self.offset = offset

    def __str__(self):
        msg = Segment.__str__(self)
        msg += (
            f'\n    Tile part index:  {self.isot}'
            f'\n    Tile part length:  {self.psot}'
            f'\n    Tile part instance:  {self.tpsot}'
            f'\n    Number of tile parts:  {self.tnsot}'
        )
        return msg


class TLMsegment(Segment):
    """Container for TLM segment information.

    Attributes
    ----------
    marker_id : str
        Identifier for the segment.
    offset : int
        Offset of marker segment in bytes from beginning of file.
    length : int
        Length of marker segment in bytes.  This number does not include the
        two bytes constituting the marker.
    ztlm : int
        index relative to other TML marksers
    ttlm : int
        number of the ith tile-part
    ptlm : int
        length in bytes from beginning of the SOT marker of the ith
        tile-part to the end of the data for that tile part

    References
    ----------
    .. [JP2K15444-1i] International Organization for Standardication.  ISO/IEC
       15444-1:2004 - Information technology -- JPEG 2000 image coding system:
       Core coding system
    """
    def __init__(self, ztlm, ttlm, ptlm, length, offset):
        super().__init__(marker_id='TLM')
        self.length = length
        self.offset = offset
        self.ztlm = ztlm
        self.ttlm = np.array(ttlm)
        self.ptlm = np.array(ptlm)

    def __str__(self):
        msg = Segment.__str__(self)
        with np.printoptions(threshold=4):
            msg += (
                f'\n    Index:  {self.ztlm}'
                f'\n    Tile number:  {self.ttlm}'
                f'\n    Length:  {self.ptlm}'
            )

        return msg


def _parse_precinct_size(spcod):
    """Compute precinct size from SPcod or SPcoc."""
    spcod = np.frombuffer(spcod, dtype=np.uint8)
    precinct_size = []
    for item in spcod:
        ep2 = (item & 0xF0) >> 4
        ep1 = item & 0x0F
        precinct_size.append((2 ** ep1, 2 ** ep2))
    return tuple(precinct_size)


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
    msg = msg.format(
        ((context & 0x01) > 0),
        ((context & 0x02) > 0),
        ((context & 0x04) > 0),
        ((context & 0x08) > 0),
        ((context & 0x10) > 0),
        ((context & 0x20) > 0)
    )
    return msg


def parse_quantization(read_buffer, sqcd):
    """Tease out the quantization values.

    Parameters
    ----------
        read_buffer:  sequence of bytes from the QCC and QCD segments.

    Returns
    ------
    tuple
        Mantissa and exponents from quantization buffer.
    """
    numbytes = len(read_buffer)

    exponent = []
    mantissa = []

    if sqcd & 0x1f == 0:  # no quantization
        data = struct.unpack('>' + 'B' * numbytes, read_buffer)
        for j in range(len(data)):
            exponent.append(data[j] >> 3)
            mantissa.append(0)
    else:
        fmt = '>' + 'H' * int(numbytes / 2)
        data = struct.unpack(fmt, read_buffer)
        for j in range(len(data)):
            exponent.append(data[j] >> 11)
            mantissa.append(data[j] & 0x07ff)

    return mantissa, exponent


def _print_quantization_style(sqcc):
    """Only to be used with QCC and QCD segments."""

    msg = '\n    Quantization style:  '
    if sqcc & 0x1f == 0:
        msg += 'no quantization, '
    elif sqcc & 0x1f == 1:
        msg += 'scalar implicit, '
    elif sqcc & 0x1f == 2:
        msg += 'scalar explicit, '
    return msg
