----------------------------------------
Glymur: a Python interface for JPEG 2000
----------------------------------------

**Glymur** is an interface to the OpenJPEG library
which allows one to read and write JPEG 2000 files from within Python.  
Glymur supports both reading and writing of JPEG 2000 images.  Writing
JPEG 2000 images is currently limited to images that can fit in memory,
however.

Of particular focus is retrieval of metadata.  Reading Exif UUIDs is supported,
as is reading XMP UUIDs as the XMP data packet is just XML.  There is
some very limited support for reading JPX metadata.  For instance,
**asoc** and **labl** boxes are recognized, so GMLJP2 metadata can
be retrieved from such JPX files.

Glymur works on Python 2.6, 2.7, and 3.3.  Python 3.3 is strongly recommended.

OpenJPEG Installation
=====================
The OpenJPEG library version must be either 1.4, 1.5, or the trunk/development
version of OpenJPEG.  Version 2.0.0 or versions earlier than 1.4.0
are not supported.  Furthermore, the 1.4 and 1.5 versions of OpenJPEG are
currently only utilized for read-only purposes.  For more information
about OpenJPEG, please consult http://www.openjpeg.org.

If you use MacPorts on the mac or if you have a sufficiently recent version of
Linux, your package manager should already provide you with at least version
1.5.1 of OpenJPEG, which means that glymur can be installed ready to read JPEG
2000 images.  If your platform is windows, I suggest using the 1.5.1 windows 
installer provided to you by the OpenJPEG folks at
https://code.google.com/p/openjpeg/downloads/list .

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

    $ export PYTHONPATH=$HOME/.local/lib/python3.3/site-packages
    $ export PATH=$HOME/.local/bin:$PATH

You can run the tests from within python as follows::

    >>> import glymur
    >>> glymur.runtests()

Many tests are currently skipped; in fact most of them are skipped if you 
are relying on OpenJPEG 1.4 or 1.5.  But the important thing, though, is whether or
not any tests fail.
