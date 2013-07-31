"""
Wraps fopen and fclose functions in libc.
"""

import ctypes
import ctypes.util

LIBC_PATH = ctypes.util.find_library('c')
C_LIB = ctypes.CDLL(LIBC_PATH)


def fopen(filename, mode):
    """Opens the file with the specified mode.

    Parameters
    ----------
    filename : str
        Path to filename.
    mode : str
        Specifies how the file is to be opened.

    Returns
    -------
    fptr : ctypes.c_void_p
        File pointer.
    """
    C_LIB.fopen.restype = ctypes.c_void_p
    C_LIB.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    fptr = C_LIB.fopen(ctypes.c_char_p(filename.encode()),
                       ctypes.c_char_p(mode.encode()))
    return fptr


def fclose(fptr):
    """Closes a file stream.

    Parameters
    ----------
    fptr : ctypes.c_void_p
        File pointer.
    """
    C_LIB.fclose.argtypes = [ctypes.c_void_p]
    C_LIB.fclose.restype = ctypes.c_int32
    status = C_LIB.fclose(fptr)
    if status != 0:
        raise IOError("Unable to close file.")
