"""
This file is part of glymur, a Python interface for accessing JPEG 2000.

http://glymur.readthedocs.org

Copyright 2013 John Evans

License:  MIT
"""

# Standard library imports ...
import sys

# Third party library imports ...
from packaging.version import parse
import numpy as np

# Local imports ...
from .lib import openjp2 as opj2
from .lib import tiff

# Do not change the format of this next line!  Doing so risks breaking
# setup.py
version = "0.11.6post1"

version_tuple = parse(version).release

openjpeg_version = opj2.version()
openjpeg_version_tuple = parse(openjpeg_version).release

tiff_version = tiff.getVersion()

__doc__ = f"""\
This is glymur **{version}**

* OpenJPEG version:  **{openjpeg_version}**
"""

info = f"""\
Summary of glymur configuration
-------------------------------

glymur        {version}
OpenJPEG      {openjpeg_version}
Python        {sys.version}
sys.platform  {sys.platform}
sys.maxsize   {sys.maxsize}
numpy         {np.__version__}
"""
