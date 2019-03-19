"""
This file is part of glymur, a Python interface for accessing JPEG 2000.

http://glymur.readthedocs.org

Copyright 2013 John Evans

License:  MIT
"""

# Standard library imports ...
import sys

# Third party library imports ...
from distutils.version import LooseVersion
import numpy as np

# Local imports ...
from .lib import openjp2 as opj2

# Do not change the format of this next line!  Doing so risks breaking
# setup.py
version = "0.9.0"
_sv = LooseVersion(version)
version_tuple = _sv.version

openjpeg_version = opj2.version()

_sv = LooseVersion(openjpeg_version)
openjpeg_version_tuple = _sv.version

__doc__ = """\
This is glymur **{glymur_version}**

* OpenJPEG version:  **{openjpeg}**
""".format(glymur_version=version,
           openjpeg=openjpeg_version)

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
