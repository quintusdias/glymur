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
from .lib import openjpeg as opj, openjp2 as opj2

# Do not change the format of this next line!  Doing so risks breaking
# setup.py
version = "0.8.9"
_sv = LooseVersion(version)
version_tuple = _sv.version

if opj2.OPENJP2 is None and opj.OPENJPEG is not None:
    openjpeg_version = opj.version()
else:
    openjpeg_version = opj2.version()

_sv = LooseVersion(openjpeg_version)
openjpeg_version_tuple = _sv.version

__doc__ = """\
This is glymur **{glymur_version}**

* OpenJPEG version:  **{openjpeg}**
""".format(glymur_version=version,
           openjpeg=openjpeg_version)

info = """\
Summary of glymur configuration
-------------------------------

glymur        {glymur}
OpenJPEG      {openjpeg}
Python        {python}
sys.platform  {platform}
sys.maxsize   {maxsize}
numpy         {numpy}
"""

kwargs = {
    'glymur': version,
    'openjpeg': openjpeg_version,
    'python': sys.version,
    'platform': sys.platform,
    'maxsize': sys.maxsize,
    'numpy': np.__version__,
}

try:
    import lxml.etree
    info += "lxml.etree    {elxml}\n"
    kwargs['elxml'] = lxml.etree.__version__
except Exception:
    pass

info = info.format(**kwargs)
