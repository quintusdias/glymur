----------------------------------------
Glymur: a Python interface for JPEG 2000
----------------------------------------

**Glymur** is an interface to the OpenJPEG library
which allows one to read and write JPEG 2000 files from within Python.  
Glymur supports both reading and writing of JPEG 2000 images, but writing
JPEG 2000 images is currently limited to images that can fit in memory

Of particular focus is retrieval of metadata.  Reading Exif UUIDs is supported,
as is reading XMP UUIDs as the XMP data packet is just XML.  There is
some very limited support for reading JPX metadata.  For instance,
**asoc** and **labl** boxes are recognized, so GMLJP2 metadata can
be retrieved from such JPX files.

Glymur works on Python 2.6, 2.7, and 3.3.

OpenJPEG Installation
=====================
Glymur will read JPEG 2000 images with versions 1.3, 1.4, 1.5, 2.0,
and the trunk/development version of OpenJPEG.  Writing images is
only supported with the 1.5 or better, however, and the trunk/development
version is strongly recommended.  For more information about OpenJPEG,
please consult http://www.openjpeg.org.

If you use MacPorts or if you have a sufficiently recent version of
Linux, your package manager should already provide you with a version of
OpenJPEG 1.X which glymur can already use.  If your platform is windows,
I suggest using the windows installers provided to you by the OpenJPEG
folks at https://code.google.com/p/openjpeg/downloads/list .

Glymur Installation
===================
You can retrieve the source for Glymur from either of

    * https://pypi.python.org/pypi/Glymur/ (stable releases)
    * http://github.com/quintusdias/glymur (bleeding edge)

but you should also be able to install Glymur via pip ::

    $ pip install glymur

This will install the **jp2dump** script that can be used from the unix command
line, so you should adjust your **$PATH**
to take advantage of it.  For example, if you install with pip's
`--user` option on linux ::

    $ export PATH=$HOME/.local/bin:$PATH

