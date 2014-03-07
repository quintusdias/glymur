----------------------------------------
Glymur: a Python interface for JPEG 2000
----------------------------------------

**Glymur** is an interface to the OpenJPEG library
which allows one to read and write JPEG 2000 files from within Python.  
Glymur supports both reading and writing of JPEG 2000 images, but writing
JPEG 2000 images is currently limited to images that can fit in memory

In regards to metadata, most JP2 boxes are properly interpreted.
Certain optional JP2 boxes can also be written, including XML boxes and
XMP UUIDs.  There is some very limited support for reading JPX metadata.

Glymur 0.6 works on Python versions 2.7, 3.3 and 3.4.  If you have Python 2.6,
you should use the 0.5 series of Glymur.

OpenJPEG Installation
=====================
Glymur will read JPEG 2000 images with versions 1.3, 1.4, 1.5, 2.0,
and the trunk/development version of OpenJPEG.  Writing images is
only supported with the 1.5 or better, however, and the trunk/development
version of OpenJPEG is strongly recommended.  For more information about
OpenJPEG, please consult http://www.openjpeg.org.

If you use MacPorts or if you have a sufficiently recent version of
Linux, your package manager should already provide you with a version of
OpenJPEG 1.X which glymur can already use.  

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

