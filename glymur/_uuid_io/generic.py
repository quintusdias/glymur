# -*- coding: utf-8 -*-

"""
Handler for a generic UUID.
"""

class UUIDGeneric(object):
    """
    Handler for a generic UUID that is not currently recognized.

    Attributes
    ----------
    data : byte array
        Sequence of uninterpreted bytes as read from the file.
    """
    def __init__(self, read_buffer):
        """
        Parameters
        ----------
        read_buffer : byte array
            sequence of bytes as read from the file.
        """
        self.data = read_buffer

    def __str__(self):
        return '{0} bytes'.format(len(self.data))

