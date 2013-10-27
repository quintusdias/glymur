# -*- coding: utf-8 -*-

"""
Handler for a UUID for XMP.
"""

import sys
from xml.etree import cElementTree as ET

from ..core import _pretty_print_xml

class UUIDXMP(object):
    """
    Handler for a UUID for XMP.

    Attributes
    ----------
    packet : ElementTree
        XML conforming to the XMP specifications.

    References
    ----------
    .. [XMP] International Organization for Standardication.  ISO/IEC
       16684-1:2012 - Graphic technology -- Extensible metadata platform (XMP)
       specification -- Part 1:  Data model, serialization and core properties
    """
    def __init__(self, read_buffer):
        """
        Parameters
        ----------
        read_buffer : byte array
            sequence of bytes that can be decoded into an XMP packet.
        """

        # XMP data.  Parse as XML.
        if sys.hexversion < 0x03000000:
            # 2.x strings same as bytes
            elt = ET.fromstring(read_buffer)
        else:
            # 3.x takes strings, not bytes.
            text = read_buffer.decode('utf-8')
            elt = ET.fromstring(text)
        self.packet = ET.ElementTree(elt)

    def __str__(self):
        return _pretty_print_xml(self.packet)
