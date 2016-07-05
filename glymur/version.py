"""
This file is part of glymur, a Python interface for accessing JPEG 2000.

http://glymur.readthedocs.org

Copyright 2013 John Evans

License:  MIT
"""

import sys
from distutils.version import LooseVersion

try:
    import lxml.etree
except ImportError:
    pass

import numpy as np

from .lib import openjpeg as opj, openjp2 as opj2

# Do not change the format of this next line!  Doing so risks breaking
# setup.py
version = "0.8.4"
_sv = LooseVersion(version)
version_tuple = _sv.version


if opj.OPENJPEG is None and opj2.OPENJP2 is None:
    openjpeg_version = '0.0.0'
elif opj2.OPENJP2 is None:
    openjpeg_version = opj.version()
else:
    openjpeg_version = opj2.version()

_sv = LooseVersion(openjpeg_version)
openjpeg_version_tuple = _sv.version

__doc__ = """\
This is glymur **{glymur_version}**

* OPENJPEG version:  **{openjpeg}**
""".format(glymur_version=version,
           openjpeg=openjpeg_version)

info = """\
Summary of glymur configuration
-------------------------------

glymur        {glymur}
OPENJPEG      {openjpeg}
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

if 'lxml' in sys.modules.keys():
    info += "lxml          {elxml}\n"
    kwargs['elxml'] = lxml.etree.__version__

info = info.format(**kwargs)
