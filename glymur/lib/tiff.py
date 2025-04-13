"""
This module is deprecated.  No external use, please.
"""
# standard library imports
import warnings

msg = "glymur.lib.tiff is for internal use only.  Please do not use it."
warnings.warn(msg, DeprecationWarning)
from ._tiff import *

