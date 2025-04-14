"""
This module is deprecated.  No external use, please.
"""
# standard library imports
import warnings

from ._tiff import *  # noqa : F401, F403

msg = "glymur.lib.tiff is for internal use only.  Please do not use it."
warnings.warn(msg, DeprecationWarning)
